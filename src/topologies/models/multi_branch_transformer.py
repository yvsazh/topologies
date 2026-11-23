import math

import torch
from torch import nn
from torch.nn import functional as F


def _rms_norm(hidden, weight, eps):
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.float()).to(hidden.dtype)


def _router_logits(hidden, router):
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), router.weight.float())


def _router_topk(logits, expert_bias, top_k):
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if expert_bias.shape != (logits.shape[1],):
        raise ValueError("expert bias must match the router expert dimension")
    if type(top_k) is not int or not 0 < top_k < logits.shape[1]:
        raise ValueError("top_k must be in the range 1..num_experts-1")
    scores = logits.float()
    biased = scores + expert_bias.detach().float()
    top_values, top_indices = torch.topk(biased, top_k + 1, dim=-1, sorted=True)
    indices = top_indices[:, :top_k]
    selected_scores = torch.gather(scores, 1, indices)
    selected_weights = torch.sigmoid(selected_scores)
    selected_weights = selected_weights * (
        2.5 / (selected_weights.sum(dim=-1, keepdim=True) + 1e-9)
    )
    weights = torch.zeros_like(scores).scatter(1, indices, selected_weights)
    return weights, indices, top_values[:, top_k].detach()


def _apply_rope(values, rope_base):
    half = values.shape[-1] // 2
    positions = torch.arange(values.shape[-2], device=values.device, dtype=torch.float32)
    dimensions = torch.arange(half, device=values.device, dtype=torch.float32)
    frequencies = rope_base ** (-2.0 * dimensions / values.shape[-1])
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos().to(values.dtype)[None, None, :, :]
    sine = angles.sin().to(values.dtype)[None, None, :, :]
    first, second = values[..., :half], values[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def _make_block(width, num_experts, norm_eps, output_std, shared_kv):
    kv_width = width // 2
    hidden_moe = width // 2
    expert_width = width // 2
    block = nn.ModuleDict({
        "attention_norm": nn.RMSNorm(width, eps=norm_eps),
        "q_proj": nn.Linear(width, width, bias=False),
        "gate_proj": nn.Linear(width, width, bias=False),
        "q_norm": nn.RMSNorm(64, eps=norm_eps),
        "o_proj": nn.Linear(width, width, bias=False),
        "moe_norm": nn.RMSNorm(width, eps=norm_eps),
        "router": nn.Linear(width, num_experts, bias=False),
        "moe_down": nn.Linear(width, hidden_moe, bias=False),
        "moe_up": nn.Linear(hidden_moe, width, bias=False),
        "expert_gate": nn.ParameterList(
            nn.Parameter(torch.empty(expert_width, hidden_moe)) for _ in range(num_experts)
        ),
        "expert_value": nn.ParameterList(
            nn.Parameter(torch.empty(expert_width, hidden_moe)) for _ in range(num_experts)
        ),
        "expert_out": nn.ParameterList(
            nn.Parameter(torch.empty(hidden_moe, expert_width)) for _ in range(num_experts)
        ),
    })
    if not shared_kv:
        block["k_proj"] = nn.Linear(width, kv_width, bias=False)
        block["v_proj"] = nn.Linear(width, kv_width, bias=False)
        block["k_norm"] = nn.RMSNorm(64, eps=norm_eps)
    for name, layer in block.items():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=output_std if name in ("o_proj", "moe_up") else 0.02)
    for name in ("expert_gate", "expert_value", "expert_out"):
        for parameter in block[name]:
            nn.init.normal_(parameter, std=output_std if name == "expert_out" else 0.02)
    return block


def _quantile_bias(margins, top_k, num_experts, bins):
    if margins.ndim != 2 or margins.shape[1] != num_experts or margins.shape[0] == 0:
        raise ValueError("router margins must have shape [tokens, experts] with tokens present")
    values = margins.float()
    lower = values.min()
    upper = torch.maximum(values.max(), lower + 1e-6)
    lower_value = lower.item()
    upper_value = upper.item()
    histograms = torch.stack([
        torch.histc(values[:, expert], bins=bins, min=lower_value, max=upper_value)
        for expert in range(num_experts)
    ])
    totals = histograms.sum(dim=-1)
    quota = totals * (top_k / num_experts)
    reverse_cumulative = torch.flip(
        torch.cumsum(torch.flip(histograms, dims=[-1]), dim=-1), dims=[-1]
    )
    quota_bin = (reverse_cumulative >= quota.unsqueeze(-1)).sum(dim=-1).clamp(min=1) - 1
    counts = histograms.gather(1, quota_bin.unsqueeze(-1)).squeeze(-1)
    cumulative = reverse_cumulative.gather(1, quota_bin.unsqueeze(-1)).squeeze(-1)
    width = (upper - lower) / bins
    threshold = lower + quota_bin.float() * width
    threshold = threshold + width * ((cumulative - quota) / counts.clamp_min(1.0))
    bias = -threshold
    return bias - bias.mean()


class MultiBranchTransformer(nn.Module):
    architecture_id = "multi-branch-transformer"
    PRESETS = {
        "tiny": {
            "vocab_size": 64,
            "hidden_size": 128,
            "branch_width": 128,
            "prelude_blocks": 1,
            "branch_depths": (1, 1),
            "coda_blocks": 1,
            "num_experts": 64,
            "top_k": 8,
            "kv_mode": "independent",
            "norm_eps": 1e-5,
            "rope_base": 500000.0,
            "dropout": 0.0,
        },
        "small": {
            "vocab_size": 256,
            "hidden_size": 256,
            "branch_width": 256,
            "prelude_blocks": 1,
            "branch_depths": (2, 2),
            "coda_blocks": 1,
            "num_experts": 64,
            "top_k": 8,
            "kv_mode": "independent",
            "norm_eps": 1e-5,
            "rope_base": 500000.0,
            "dropout": 0.0,
        },
    }

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=128,
        prelude_blocks=1,
        branch_depths=(1, 1),
        coda_blocks=1,
        num_experts=64,
        top_k=8,
        kv_mode="independent",
        norm_eps=1e-5,
        rope_base=500000.0,
        dropout=0.0,
        branch_width=None,
    ):
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 2:
            raise ValueError("vocab_size must be an integer of at least 2")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if hidden_size % 128:
            raise ValueError("hidden_size must be divisible by 128 for GQA2 with 64-wide heads")
        if branch_width is None:
            branch_width = hidden_size
        if type(branch_width) is not int or branch_width <= 0:
            raise ValueError("branch_width must be a positive integer")
        if branch_width % 128:
            raise ValueError("branch_width must be divisible by 128 for GQA2 with 64-wide heads")
        try:
            branch_depths = tuple(branch_depths)
        except TypeError as error:
            raise ValueError("branch_depths must contain at least two branch depths") from error
        if len(branch_depths) < 2 or len(branch_depths) > 4:
            raise ValueError("branch_depths must configure at least two branches and at most four")
        if any(type(depth) is not int or depth <= 0 for depth in branch_depths):
            raise ValueError("branch_depths must contain positive integers")
        if type(prelude_blocks) is not int or prelude_blocks < 0:
            raise ValueError("prelude_blocks must be a nonnegative integer")
        if type(coda_blocks) is not int or coda_blocks < 0:
            raise ValueError("coda_blocks must be a nonnegative integer")
        if type(num_experts) is not int or num_experts < 2:
            raise ValueError("num_experts must be an integer of at least 2")
        if type(top_k) is not int or not 0 < top_k < num_experts:
            raise ValueError("top_k must be in the range 1..num_experts-1")
        if kv_mode not in ("independent", "joint"):
            raise ValueError("kv_mode must be independent or joint")
        if kv_mode == "joint" and len(set(branch_depths)) != 1:
            raise ValueError("Joint-KV requires equal-depth branches")
        if not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
            raise ValueError("norm_eps must be a finite positive number")
        if not isinstance(rope_base, (int, float)) or not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("rope_base must be a finite positive number")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.branch_width = branch_width
        self.branch_depths = branch_depths
        self.num_experts = num_experts
        self.top_k = top_k
        self.kv_mode = kv_mode
        self.norm_eps = float(norm_eps)
        self.rope_base = float(rope_base)
        self.dropout = float(dropout)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        self._router_bias_names = []
        self._pending_router_margins = []
        self._collect_router_margins = False

        physical_layers = prelude_blocks + sum(branch_depths) + coda_blocks
        output_std = 0.02 / math.sqrt(2 * physical_layers)

        def create_block(width, shared_kv=False):
            index = len(self._router_bias_names)
            name = f"_router_bias_{index}"
            self.register_buffer(name, torch.zeros(num_experts))
            self._router_bias_names.append(name)
            self._pending_router_margins.append([])
            return _make_block(width, num_experts, self.norm_eps, output_std, shared_kv), index

        self.prelude = nn.ModuleList()
        self._prelude_bias_indices = []
        for _ in range(prelude_blocks):
            block, index = create_block(hidden_size)
            self.prelude.append(block)
            self._prelude_bias_indices.append(index)

        self.branches = nn.ModuleList()
        self._branch_bias_indices = []
        for depth in branch_depths:
            stack = nn.ModuleList()
            stack_indices = []
            for _ in range(depth):
                block, index = create_block(branch_width, kv_mode == "joint")
                stack.append(block)
                stack_indices.append(index)
            self.branches.append(stack)
            self._branch_bias_indices.append(stack_indices)

        self.coda = nn.ModuleList()
        self._coda_bias_indices = []
        for _ in range(coda_blocks):
            block, index = create_block(hidden_size)
            self.coda.append(block)
            self._coda_bias_indices.append(index)

        self.width_bridge_norm = nn.RMSNorm(hidden_size, eps=self.norm_eps) if branch_width != hidden_size else None
        self.width_bridge = nn.Linear(hidden_size, branch_width, bias=False) if branch_width != hidden_size else None
        if self.width_bridge is not None:
            nn.init.normal_(self.width_bridge.weight, std=0.02)
        self.fusion_norms = nn.ModuleList(nn.RMSNorm(branch_width, eps=self.norm_eps) for _ in branch_depths)
        self.fusion_projection = nn.Linear(len(branch_depths) * branch_width, hidden_size, bias=False)
        nn.init.normal_(self.fusion_projection.weight, std=output_std)
        self.final_norm = nn.RMSNorm(hidden_size, eps=self.norm_eps)
        self.joint_kv_projections = nn.ModuleList()
        self.joint_kv_key_norms = nn.ModuleList()
        if kv_mode == "joint":
            for _ in range(branch_depths[0]):
                projection = nn.Linear(len(branch_depths) * branch_width, branch_width, bias=False)
                nn.init.normal_(projection.weight, std=0.02)
                self.joint_kv_projections.append(projection)
                self.joint_kv_key_norms.append(nn.RMSNorm(64, eps=self.norm_eps))

    def _attention(self, block, normalized, shared_kv=None):
        batch, sequence, width = normalized.shape
        query_heads = width // 64
        kv_heads = width // 128
        query = F.linear(normalized, block["q_proj"].weight)
        gate = F.linear(normalized, block["gate_proj"].weight)
        query = query.view(batch, sequence, query_heads, 64)
        gate = gate.view(batch, sequence, query_heads, 64)
        query = _rms_norm(query, block["q_norm"].weight, self.norm_eps)
        if shared_kv is None:
            key = F.linear(normalized, block["k_proj"].weight).view(batch, sequence, kv_heads, 64)
            value = F.linear(normalized, block["v_proj"].weight).view(batch, sequence, kv_heads, 64)
            key = _rms_norm(key, block["k_norm"].weight, self.norm_eps)
        else:
            key, value = shared_kv
        query = _apply_rope(query.transpose(1, 2), self.rope_base)
        key = _apply_rope(key.transpose(1, 2), self.rope_base)
        value = value.transpose(1, 2)
        if query_heads != kv_heads:
            repeat = query_heads // kv_heads
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attended = attended * torch.sigmoid(gate.transpose(1, 2))
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return F.linear(attended, block["o_proj"].weight)

    def _moe(self, block, normalized, bias_index):
        batch, sequence, _ = normalized.shape
        logits = _router_logits(normalized, block["router"])
        bias = getattr(self, self._router_bias_names[bias_index])
        weights, selected_indices, alpha = _router_topk(
            logits.reshape(-1, self.num_experts), bias, self.top_k
        )
        margins = logits.float().reshape(-1, self.num_experts) - alpha[:, None]
        if self.training:
            stored = margins.detach()
            if self._collect_router_margins:
                self._pending_router_margins[bias_index].append(stored)
            else:
                self._pending_router_margins[bias_index] = [stored]
        latent = F.linear(normalized, block["moe_down"].weight)
        flat_latent = latent.reshape(-1, latent.shape[-1])
        flat_indices = selected_indices.reshape(-1)
        flat_weights = torch.gather(weights, 1, selected_indices).reshape(-1)
        routed_tokens = torch.arange(flat_latent.shape[0], device=latent.device).repeat_interleave(
            self.top_k
        )
        combined = torch.zeros(
            flat_latent.shape[0],
            flat_latent.shape[-1],
            device=flat_latent.device,
            dtype=torch.float32,
        )
        for expert_index in range(self.num_experts):
            positions = torch.nonzero(flat_indices == expert_index, as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            token_indices = routed_tokens.index_select(0, positions)
            expert_input = flat_latent.index_select(0, token_indices)
            gate = F.linear(expert_input, block["expert_gate"][expert_index])
            value = F.linear(expert_input, block["expert_value"][expert_index])
            expert_hidden = F.silu(gate) * value
            expert_output = F.linear(expert_hidden, block["expert_out"][expert_index])
            routed_output = expert_output.float() * flat_weights.index_select(0, positions).unsqueeze(-1)
            combined.index_add_(0, token_indices, routed_output)
        combined = combined.view(batch, sequence, -1).to(block["moe_up"].weight.dtype)
        return F.linear(combined, block["moe_up"].weight)

    def _run_block(self, block, hidden, bias_index, shared_kv=None):
        normalized = _rms_norm(hidden, block["attention_norm"].weight, self.norm_eps)
        attention = self._attention(block, normalized, shared_kv)
        hidden = hidden + F.dropout(attention, p=self.dropout, training=self.training)
        normalized = _rms_norm(hidden, block["moe_norm"].weight, self.norm_eps)
        moe = self._moe(block, normalized, bias_index)
        return hidden + F.dropout(moe, p=self.dropout, training=self.training)

    def fuse_branches(self, *states):
        if len(states) != len(self.branches):
            raise ValueError("fusion input count must match the configured branch count")
        if any(state.shape != states[0].shape for state in states):
            raise ValueError("branch outputs must have matching shapes")
        if states[0].shape[-1] != self.branch_width:
            raise ValueError("branch output width must match branch_width")
        normalized = [
            _rms_norm(state, norm.weight, self.norm_eps)
            for state, norm in zip(states, self.fusion_norms)
        ]
        return F.linear(torch.cat(normalized, dim=-1), self.fusion_projection.weight)

    def project_joint_kv(self, depth, branch_inputs):
        if self.kv_mode != "joint":
            raise ValueError("project_joint_kv requires kv_mode='joint'")
        if type(depth) is not int or not 0 <= depth < len(self.joint_kv_projections):
            raise ValueError("joint-KV depth is outside the configured branch depth")
        if len(branch_inputs) != len(self.branches):
            raise ValueError("Joint-KV input count must match the configured branch count")
        if any(value.shape != branch_inputs[0].shape for value in branch_inputs):
            raise ValueError("Joint-KV inputs must have matching shapes")
        normalized = [
            _rms_norm(value, self.branches[index][depth]["attention_norm"].weight, self.norm_eps)
            for index, value in enumerate(branch_inputs)
        ]
        joint = torch.cat(normalized, dim=-1) * (1.0 / math.sqrt(len(self.branches)))
        projected = F.linear(joint, self.joint_kv_projections[depth].weight)
        batch, sequence, _ = projected.shape
        projected = projected.view(batch, sequence, self.branch_width // 128, 128)
        key, value = projected.split(64, dim=-1)
        key = _rms_norm(key, self.joint_kv_key_norms[depth].weight, self.norm_eps)
        return key, value.contiguous()

    def begin_quantile_update(self):
        self._pending_router_margins = [[] for _ in self._router_bias_names]
        self._collect_router_margins = True

    def update_quantile_biases(self, histogram_bins=10000):
        if type(histogram_bins) is not int or histogram_bins <= 0:
            raise ValueError("histogram_bins must be a positive integer")
        if not any(self._pending_router_margins):
            raise ValueError("no router margins are available for a quantile-bias update")
        with torch.no_grad():
            for index, chunks in enumerate(self._pending_router_margins):
                if not chunks:
                    continue
                margins = torch.cat(chunks, dim=0)
                bias = _quantile_bias(margins, self.top_k, self.num_experts, histogram_bins)
                getattr(self, self._router_bias_names[index]).copy_(bias)
        self._pending_router_margins = [[] for _ in self._router_bias_names]
        self._collect_router_margins = False

    @staticmethod
    def masked_next_token_loss(logits, targets, loss_mask=None):
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError("logits must be [batch, sequence, vocab] and targets [batch, sequence]")
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        if loss_mask is None:
            return losses.mean()
        if loss_mask.shape != targets.shape:
            raise ValueError("loss_mask must match the target shape")
        weights = loss_mask.to(device=losses.device, dtype=losses.dtype)
        denominator = weights.sum()
        if denominator.item() <= 0:
            raise ValueError("loss_mask must contain at least one active position")
        return (losses * weights).sum() / denominator

    def forward(self, input_ids=None, *, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            if input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must use an integer dtype")
            if input_ids.numel() == 0 or input_ids.min().item() < 0 or input_ids.max().item() >= self.vocab_size:
                raise ValueError("input_ids must be nonempty and within the configured vocabulary")
            hidden = self.token_embedding(input_ids)
        else:
            if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
                raise ValueError("inputs_embeds must have shape [batch, sequence, hidden_size]")
            if inputs_embeds.shape[-1] != self.hidden_size or inputs_embeds.shape[0] == 0 or inputs_embeds.shape[1] == 0:
                raise ValueError("inputs_embeds must be nonempty and match hidden_size")
            hidden = inputs_embeds

        for block, bias_index in zip(self.prelude, self._prelude_bias_indices):
            hidden = self._run_block(block, hidden, bias_index)

        branch_input = hidden
        if self.width_bridge is not None:
            branch_input = self.width_bridge(
                _rms_norm(hidden, self.width_bridge_norm.weight, self.norm_eps)
            )
        states = [branch_input for _ in self.branches]
        if self.kv_mode == "independent":
            states = [
                self._run_branch(stack, indices, state)
                for stack, indices, state in zip(self.branches, self._branch_bias_indices, states)
            ]
        else:
            for depth in range(len(self.joint_kv_projections)):
                shared_kv = self.project_joint_kv(depth, tuple(states))
                states = [
                    self._run_block(stack[depth], state, indices[depth], shared_kv)
                    for stack, indices, state in zip(self.branches, self._branch_bias_indices, states)
                ]

        hidden = self.fuse_branches(*states)
        for block, bias_index in zip(self.coda, self._coda_bias_indices):
            hidden = self._run_block(block, hidden, bias_index)
        hidden = _rms_norm(hidden, self.final_norm.weight, self.norm_eps)
        return F.linear(hidden, self.token_embedding.weight)

    def _run_branch(self, stack, indices, hidden):
        for block, bias_index in zip(stack, indices):
            hidden = self._run_block(block, hidden, bias_index)
        return hidden
