import math

import torch
from torch import nn
from torch.nn import functional as F


def _channels(value, multiplier):
    return max(1, round(value * multiplier))


def _make_stage(in_channels, planes, depth, stride, bottleneck):
    blocks = nn.ModuleList()
    expansion = 4 if bottleneck else 1
    out_channels = planes * expansion
    for index in range(depth):
        block_stride = stride if index == 0 else 1
        if bottleneck:
            block = nn.ModuleList([
                nn.Conv2d(in_channels, planes, 1, bias=False),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, planes, 3, stride=block_stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        else:
            block = nn.ModuleList([
                nn.Conv2d(in_channels, planes, 3, stride=block_stride, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.Conv2d(planes, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        if block_stride != 1 or in_channels != out_channels:
            block.extend([
                nn.Conv2d(in_channels, out_channels, 1, stride=block_stride, bias=False),
                nn.BatchNorm2d(out_channels),
            ])
        else:
            block.extend([nn.Identity(), nn.Identity()])
        blocks.append(block)
        in_channels = out_channels
    return blocks, out_channels


def _run_stage(x, blocks, bottleneck):
    for block in blocks:
        residual = x
        y = F.relu(block[1](block[0](x)))
        if bottleneck:
            y = F.relu(block[3](block[2](y)))
            y = block[5](block[4](y))
            projection_index = 6
        else:
            y = block[3](block[2](y))
            projection_index = 4
        if not isinstance(block[projection_index], nn.Identity):
            residual = block[projection_index + 1](block[projection_index](residual))
        x = F.relu(y + residual)
    return x


def _resnet_features(x, stem, stages, bottleneck):
    x = F.relu(stem[1](stem[0](x)))
    x = stem[2](x)
    f4 = _run_stage(x, stages[0], bottleneck)
    f8 = _run_stage(f4, stages[1], bottleneck)
    f16 = _run_stage(f8, stages[2], bottleneck)
    return f16, f8, f4


def _make_fusion_block(in_channels, out_channels):
    projection = nn.Conv2d(in_channels, out_channels, 3, padding=1) if in_channels != out_channels else nn.Identity()
    return nn.ModuleList([
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        projection,
    ])


def _run_fusion_block(x, block):
    residual = block[2](x)
    output = block[0](F.relu(x))
    output = block[1](F.relu(output))
    return output + residual


class XMem(nn.Module):
    architecture_id = "xmem"
    PRESETS = {
        "tiny": {
            "width_mult": 0.125,
            "key_dim": 8,
            "value_dim": 64,
            "hidden_dim": 8,
            "mem_every": 2,
            "min_mid_term_frames": 2,
            "max_mid_term_frames": 3,
            "num_prototypes": 2,
            "max_long_term_elements": 64,
            "top_k": 4,
        },
        "small": {
            "width_mult": 0.5,
            "key_dim": 32,
            "value_dim": 256,
            "hidden_dim": 32,
            "mem_every": 5,
            "min_mid_term_frames": 3,
            "max_mid_term_frames": 8,
            "num_prototypes": 8,
            "max_long_term_elements": 2048,
            "top_k": 8,
        },
        "base": {
            "width_mult": 1.0,
            "key_dim": 64,
            "value_dim": 512,
            "hidden_dim": 64,
            "mem_every": 5,
            "min_mid_term_frames": 5,
            "max_mid_term_frames": 10,
            "num_prototypes": 128,
            "max_long_term_elements": 10000,
            "top_k": 30,
        },
    }

    def __init__(
        self,
        width_mult=0.125,
        key_dim=8,
        value_dim=64,
        hidden_dim=8,
        mem_every=2,
        min_mid_term_frames=2,
        max_mid_term_frames=3,
        num_prototypes=2,
        max_long_term_elements=64,
        top_k=4,
    ):
        super().__init__()
        if not isinstance(width_mult, (int, float)) or isinstance(width_mult, bool) or not 0 < width_mult <= 1:
            raise ValueError("width_mult must be in (0, 1]")
        integers = {
            "key_dim": key_dim,
            "value_dim": value_dim,
            "hidden_dim": hidden_dim,
            "mem_every": mem_every,
            "min_mid_term_frames": min_mid_term_frames,
            "max_mid_term_frames": max_mid_term_frames,
            "num_prototypes": num_prototypes,
            "max_long_term_elements": max_long_term_elements,
            "top_k": top_k,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integers.values()):
            raise ValueError("XMem dimensions and memory limits must be positive integers")
        if min_mid_term_frames >= max_mid_term_frames:
            raise ValueError("min_mid_term_frames must be less than max_mid_term_frames")
        self.width_mult = float(width_mult)
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.mem_every = mem_every
        self.min_mid_term_frames = min_mid_term_frames
        self.max_mid_term_frames = max_mid_term_frames
        self.num_prototypes = num_prototypes
        self.max_long_term_elements = max_long_term_elements
        self.top_k = top_k

        key_stem_dim = _channels(64, self.width_mult)
        key_stem = nn.ModuleList([
            nn.Conv2d(3, key_stem_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(key_stem_dim),
            nn.MaxPool2d(3, stride=2, padding=1),
        ])
        key_stages = []
        current = key_stem_dim
        for planes, depth, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2)):
            stage, current = _make_stage(
                current, _channels(planes, self.width_mult), depth, stride, True
            )
            key_stages.append(stage)
        self.key_stem = key_stem
        self.key_stages = nn.ModuleList(key_stages)
        self.key_feature_dim = current
        self.key_projection = nn.Conv2d(current, key_dim, 3, padding=1)
        self.shrinkage_projection = nn.Conv2d(current, 1, 3, padding=1)
        self.selection_projection = nn.Conv2d(current, key_dim, 3, padding=1)

        value_stem_dim = _channels(64, self.width_mult)
        value_stem = nn.ModuleList([
            nn.Conv2d(4, value_stem_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(value_stem_dim),
            nn.MaxPool2d(3, stride=2, padding=1),
        ])
        value_stages = []
        current = value_stem_dim
        for planes, depth, stride in ((64, 2, 1), (128, 2, 2), (256, 2, 2)):
            stage, current = _make_stage(
                current, _channels(planes, self.width_mult), depth, stride, False
            )
            value_stages.append(stage)
        self.value_stem = value_stem
        self.value_stages = nn.ModuleList(value_stages)
        self.value_feature_dim = current
        self.value_fusion = nn.ModuleList([
            _make_fusion_block(self.key_feature_dim + current, value_dim),
            _make_fusion_block(value_dim, value_dim),
        ])
        attention_hidden = max(1, value_dim // 16)
        self.value_attention_reduce = nn.Linear(value_dim, attention_hidden)
        self.value_attention_expand = nn.Linear(attention_hidden, value_dim)
        self.value_attention_spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.deep_update = nn.Conv2d(value_dim + hidden_dim, hidden_dim * 3, 3, padding=1)

        decoder_dim = _channels(512, self.width_mult)
        decoder_dim = max(16, decoder_dim)
        decoder_mid = max(16, _channels(256, self.width_mult))
        decoder_low = max(16, _channels(128, self.width_mult))
        self.decoder_fuse = nn.ModuleList([
            _make_fusion_block(self.key_feature_dim + value_dim + hidden_dim, decoder_dim),
            _make_fusion_block(decoder_dim, decoder_dim),
        ])
        decoder_attention_hidden = max(1, decoder_dim // 16)
        self.decoder_attention_reduce = nn.Linear(decoder_dim, decoder_attention_hidden)
        self.decoder_attention_expand = nn.Linear(decoder_attention_hidden, decoder_dim)
        self.decoder_attention_spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.decoder_skip8 = nn.Conv2d(self.key_feature_dim // 2, decoder_dim, 3, padding=1)
        self.decoder_skip4 = nn.Conv2d(self.key_feature_dim // 4, decoder_mid, 3, padding=1)
        self.decoder_up8 = _make_fusion_block(decoder_dim, decoder_mid)
        self.decoder_up4 = _make_fusion_block(decoder_mid, decoder_low)
        sensory_mid = max(16, _channels(256, self.width_mult))
        self.sensory_f16 = nn.Conv2d(decoder_dim, sensory_mid, 1)
        self.sensory_f8 = nn.Conv2d(decoder_mid, sensory_mid, 1)
        self.sensory_f4 = nn.Conv2d(decoder_low + 1, sensory_mid, 1)
        self.sensory_update = nn.Conv2d(sensory_mid + hidden_dim, hidden_dim * 3, 3, padding=1)
        self.prediction = nn.Conv2d(decoder_low, 1, 3, padding=1)

    @staticmethod
    def anisotropic_similarity(memory_keys, shrinkage, query_keys, selection):
        if memory_keys.ndim != 3 or query_keys.ndim != 3:
            raise ValueError("memory_keys and query_keys must have shape [B,C,N]")
        if memory_keys.shape[:2] != query_keys.shape[:2]:
            raise ValueError("memory and query key batch and channel dimensions must match")
        if shrinkage is None or selection is None:
            raise ValueError("shrinkage and selection tensors are required")
        if shrinkage.shape != (memory_keys.shape[0], 1, memory_keys.shape[2]):
            raise ValueError("shrinkage must have shape [B,1,N_memory]")
        if selection.shape != query_keys.shape:
            raise ValueError("selection must match query_keys shape")
        delta = memory_keys.unsqueeze(-1) - query_keys.unsqueeze(-2)
        distances = (delta.square() * selection.unsqueeze(-2)).sum(dim=1)
        return -shrinkage.squeeze(1).unsqueeze(-1) * distances / math.sqrt(memory_keys.shape[1])

    @staticmethod
    def memory_readout(affinity, values):
        if affinity.ndim != 3 or values.ndim != 3 or affinity.shape[0] != values.shape[0]:
            raise ValueError("affinity and values must have shape [B,N,Q] and [B,C,N]")
        if affinity.shape[1] != values.shape[2]:
            raise ValueError("affinity memory length must match values")
        return torch.bmm(values, affinity).unsqueeze(2)

    @classmethod
    def consolidate_memory(cls, candidate_keys, candidate_shrinkage, candidate_selection,
                           candidate_values, usage, prototype_count):
        if candidate_keys.ndim != 3 or candidate_values.ndim != 3 or usage.ndim != 2:
            raise ValueError("candidate keys, values, and usage must have ranks 3, 3, and 2")
        if candidate_keys.shape[0] != candidate_values.shape[0] or usage.shape != (
            candidate_keys.shape[0], candidate_keys.shape[2]
        ):
            raise ValueError("candidate batch and memory dimensions must match")
        if prototype_count <= 0:
            raise ValueError("prototype_count must be positive")
        count = min(int(prototype_count), candidate_keys.shape[2])
        indices = torch.topk(usage, k=count, dim=-1, sorted=True).indices
        prototype_keys = torch.gather(
            candidate_keys, 2, indices.unsqueeze(1).expand(-1, candidate_keys.shape[1], -1)
        )
        prototype_selection = torch.gather(
            candidate_selection, 2, indices.unsqueeze(1).expand(-1, candidate_selection.shape[1], -1)
        ) if candidate_selection is not None else None
        similarity = cls.anisotropic_similarity(
            candidate_keys, candidate_shrinkage, prototype_keys, prototype_selection
        )
        affinity = torch.softmax(similarity, dim=1)
        prototype_values = cls.memory_readout(affinity, candidate_values).squeeze(2)
        prototype_shrinkage = cls.memory_readout(affinity, candidate_shrinkage).squeeze(2)
        return prototype_keys, prototype_values, prototype_shrinkage, indices

    @staticmethod
    def loss_components(logits, target, top_p=0.15, iteration=0, start_warm=20000, end_warm=70000):
        if logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("logits must have shape [B,2,H,W]")
        if target.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
            raise ValueError("target must have shape [B,H,W] aligned with logits")
        if not 0 < top_p <= 1 or end_warm <= start_warm:
            raise ValueError("top_p must be in (0,1] and end_warm must exceed start_warm")
        if iteration < start_warm:
            ce = F.cross_entropy(logits, target.long())
        else:
            fraction = min(1.0, max(0.0, (end_warm - iteration) / (end_warm - start_warm)))
            keep_fraction = top_p + (1.0 - top_p) * fraction
            pixel_losses = F.cross_entropy(logits, target.long(), reduction="none").reshape(-1)
            count = max(1, int(pixel_losses.numel() * keep_fraction))
            ce = torch.topk(pixel_losses, count, sorted=False).values.mean()
        probabilities = torch.softmax(logits, dim=1)[:, 1].flatten(start_dim=1)
        truth = target.eq(1).to(probabilities.dtype).flatten(start_dim=1)
        intersection = (probabilities * truth).sum(dim=1)
        dice = (1.0 - (2.0 * intersection + 1.0) /
                (probabilities.sum(dim=1) + truth.sum(dim=1) + 1.0)).mean()
        return ce + dice, ce, dice

    def _encode_keys(self, frames):
        batch, time = frames.shape[:2]
        flat = frames.flatten(0, 1)
        f16, f8, f4 = _resnet_features(flat, self.key_stem, self.key_stages, True)
        keys = self.key_projection(f16)
        shrinkage = self.shrinkage_projection(f16).square() + 1.0
        selection = torch.sigmoid(self.selection_projection(f16))
        shape = (batch, time)
        return tuple(value.view(*shape, *value.shape[1:]) for value in (keys, shrinkage, selection, f16, f8, f4))

    def _encode_values(self, frame, key_features, mask, hidden):
        encoded_input = torch.cat([frame, mask], dim=1)
        value_features, _, _ = _resnet_features(
            encoded_input, self.value_stem, self.value_stages, False
        )
        value = _run_fusion_block(torch.cat([key_features, value_features], dim=1), self.value_fusion[0])
        value = value + self._attention(
            value, self.value_attention_reduce, self.value_attention_expand,
            self.value_attention_spatial
        )
        value = _run_fusion_block(value, self.value_fusion[1])
        gates = self.deep_update(torch.cat([value, hidden], dim=1))
        forget, update, candidate = gates.chunk(3, dim=1)
        next_hidden = torch.sigmoid(forget) * hidden * (1.0 - torch.sigmoid(update)) + torch.sigmoid(update) * torch.tanh(candidate)
        return value, next_hidden

    def _decode(self, f16, f8, f4, readout, hidden, update_hidden=True):
        fused = _run_fusion_block(torch.cat([f16, readout, hidden], dim=1), self.decoder_fuse[0])
        fused = _run_fusion_block(
            fused + self._attention(
                fused, self.decoder_attention_reduce, self.decoder_attention_expand,
                self.decoder_attention_spatial
            ),
            self.decoder_fuse[1],
        )
        up8 = F.interpolate(fused, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        decoded8 = _run_fusion_block(up8 + self.decoder_skip8(f8), self.decoder_up8)
        up4 = F.interpolate(decoded8, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        decoded4 = _run_fusion_block(self.decoder_skip4(f4) + up4, self.decoder_up4)
        foreground_logit = self.prediction(F.relu(decoded4))
        sensory8 = F.interpolate(decoded8, size=fused.shape[-2:], mode="area")
        sensory4 = F.interpolate(
            torch.cat([decoded4, foreground_logit], dim=1), size=fused.shape[-2:], mode="area"
        )
        if update_hidden:
            sensory_features = self.sensory_f16(fused) + self.sensory_f8(sensory8) + self.sensory_f4(sensory4)
            gates = self.sensory_update(torch.cat([sensory_features, hidden], dim=1))
            forget, update, candidate = gates.chunk(3, dim=1)
            hidden = torch.sigmoid(forget) * hidden * (1.0 - torch.sigmoid(update)) + torch.sigmoid(update) * torch.tanh(candidate)
        return foreground_logit, hidden

    @staticmethod
    def _attention(features, reduce, expand, spatial):
        average = F.adaptive_avg_pool2d(features, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(features, 1).flatten(1)
        average_gate = expand(F.relu(reduce(average)))
        maximum_gate = expand(F.relu(reduce(maximum)))
        features = features * torch.sigmoid(average_gate + maximum_gate).unsqueeze(-1).unsqueeze(-1)
        spatial_input = torch.cat([
            features.max(dim=1, keepdim=True).values,
            features.mean(dim=1, keepdim=True),
        ], dim=1)
        return features * torch.sigmoid(spatial(spatial_input))

    def forward(self, frames, first_mask):
        if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if frames.shape[2] != 3 or frames.shape[0] <= 0 or frames.shape[1] <= 0:
            raise ValueError("frames must contain a positive batch and time dimension with 3 channels")
        if frames.shape[-2] <= 0 or frames.shape[-1] <= 0:
            raise ValueError("frame height and width must be positive")
        if not isinstance(first_mask, torch.Tensor) or first_mask.ndim != 4:
            raise ValueError("first_mask must have shape [B,1,H,W]")
        if first_mask.shape != (frames.shape[0], 1, frames.shape[-2], frames.shape[-1]):
            raise ValueError("first_mask batch and spatial dimensions must match frames")
        batch, time, _, height, width = frames.shape
        if time == 1:
            return frames.new_empty((batch, 0, 2, height, width))
        pad_h = (-height) % 16
        pad_w = (-width) % 16
        padded_frames = F.pad(frames.flatten(0, 1), (0, pad_w, 0, pad_h), mode="replicate")
        padded_frames = padded_frames.view(batch, time, 3, height + pad_h, width + pad_w)
        padded_mask = F.pad(first_mask, (0, pad_w, 0, pad_h))
        keys, shrinkages, selections, f16s, f8s, f4s = self._encode_keys(padded_frames)
        hidden = f16s.new_zeros((batch, self.hidden_dim, *f16s.shape[-2:]))
        initial_value, hidden = self._encode_values(
            padded_frames[:, 0], f16s[:, 0], padded_mask, hidden
        )
        work_keys = [keys[:, 0].flatten(2)]
        work_shrinkages = [shrinkages[:, 0].flatten(2)]
        work_selections = [selections[:, 0].flatten(2)]
        work_values = [initial_value.flatten(2)]
        work_usage = [keys.new_zeros((batch, keys.shape[-2] * keys.shape[-1]))]
        work_life = [keys.new_full((batch, keys.shape[-2] * keys.shape[-1]), 1e-7)]
        long_keys = keys.new_empty((batch, self.key_dim, 0))
        long_shrinkages = shrinkages.new_empty((batch, 1, 0))
        long_values = initial_value.new_empty((batch, self.value_dim, 0))
        long_usage = keys.new_empty((batch, 0))
        long_life = keys.new_empty((batch, 0))
        predictions = []
        for frame_index in range(1, time):
            query_key = keys[:, frame_index].flatten(2)
            query_selection = selections[:, frame_index].flatten(2)
            memory_keys = torch.cat([long_keys, *work_keys], dim=2)
            memory_shrinkage = torch.cat([long_shrinkages, *work_shrinkages], dim=2)
            memory_values = torch.cat([long_values, *work_values], dim=2)
            similarity = self.anisotropic_similarity(
                memory_keys, memory_shrinkage, query_key, query_selection
            )
            top_count = min(self.top_k, similarity.shape[1])
            top_values, top_indices = torch.topk(similarity, top_count, dim=1)
            affinity = torch.zeros_like(similarity).scatter(
                1, top_indices, torch.softmax(top_values, dim=1)
            )
            readout = self.memory_readout(affinity, memory_values).view(
                batch, self.value_dim, *keys.shape[-2:]
            )
            long_count = long_keys.shape[2]
            if long_count:
                long_usage = long_usage + affinity[:, :long_count].sum(dim=2).detach()
                long_life = long_life + 1.0
            work_offset = 0
            for index, work_key in enumerate(work_keys):
                count = work_key.shape[2]
                usage = affinity[:, long_count + work_offset:long_count + work_offset + count].sum(dim=2).detach()
                work_usage[index] = work_usage[index] + usage
                work_life[index] = work_life[index] + 1.0
                work_offset += count
            is_mem_frame = frame_index % self.mem_every == 0
            foreground_logit, hidden = self._decode(
                f16s[:, frame_index], f8s[:, frame_index], f4s[:, frame_index], readout, hidden,
                update_hidden=not is_mem_frame,
            )
            foreground_logit = F.interpolate(
                foreground_logit, size=(height + pad_h, width + pad_w),
                mode="bilinear", align_corners=False
            )
            predictions.append(torch.cat([
                -foreground_logit[..., :height, :width],
                foreground_logit[..., :height, :width],
            ], dim=1))
            if is_mem_frame:
                predicted_mask = torch.sigmoid(foreground_logit)
                value, hidden = self._encode_values(
                    padded_frames[:, frame_index], f16s[:, frame_index], predicted_mask, hidden
                )
                work_keys.append(keys[:, frame_index].flatten(2))
                work_shrinkages.append(shrinkages[:, frame_index].flatten(2))
                work_selections.append(selections[:, frame_index].flatten(2))
                work_values.append(value.flatten(2))
                work_usage.append(keys.new_zeros((batch, keys.shape[-2] * keys.shape[-1])))
                work_life.append(keys.new_full((batch, keys.shape[-2] * keys.shape[-1]), 1e-7))
                if len(work_keys) >= self.max_mid_term_frames:
                    compress_frames = max(1, len(work_keys) - self.min_mid_term_frames)
                    candidate_keys = torch.cat(work_keys[:compress_frames], dim=2)
                    candidate_shrinkage = torch.cat(work_shrinkages[:compress_frames], dim=2)
                    candidate_values = torch.cat(work_values[:compress_frames], dim=2)
                    candidate_selection = torch.cat(work_selections[:compress_frames], dim=2)
                    candidate_usage = torch.cat(work_usage[:compress_frames], dim=1) / torch.cat(
                        work_life[:compress_frames], dim=1
                    )
                    prototype_keys, prototype_values, prototype_shrinkage, _ = self.consolidate_memory(
                        candidate_keys, candidate_shrinkage, candidate_selection,
                        candidate_values, candidate_usage, self.num_prototypes
                    )
                    long_keys = torch.cat([long_keys, prototype_keys], dim=2)
                    long_shrinkages = torch.cat([long_shrinkages, prototype_shrinkage], dim=2)
                    long_values = torch.cat([long_values, prototype_values], dim=2)
                    long_usage = torch.cat([long_usage, keys.new_zeros((batch, prototype_keys.shape[2]))], dim=1)
                    long_life = torch.cat([long_life, keys.new_full((batch, prototype_keys.shape[2]), 1e-7)], dim=1)
                    work_keys = work_keys[compress_frames:]
                    work_shrinkages = work_shrinkages[compress_frames:]
                    work_selections = work_selections[compress_frames:]
                    work_values = work_values[compress_frames:]
                    work_usage = work_usage[compress_frames:]
                    work_life = work_life[compress_frames:]
                    if long_keys.shape[2] > self.max_long_term_elements:
                        keep_count = self.max_long_term_elements
                        keep = torch.topk(long_usage / long_life, keep_count, dim=1).indices
                        long_keys = torch.gather(long_keys, 2, keep.unsqueeze(1).expand(-1, self.key_dim, -1))
                        long_shrinkages = torch.gather(long_shrinkages, 2, keep.unsqueeze(1))
                        long_values = torch.gather(long_values, 2, keep.unsqueeze(1).expand(-1, self.value_dim, -1))
                        long_usage = torch.gather(long_usage, 1, keep)
                        long_life = torch.gather(long_life, 1, keep)
        if not predictions:
            return frames.new_empty((batch, 0, 2, height, width))
        return torch.stack(predictions, dim=1)
