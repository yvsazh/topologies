import torch
from torch import nn


class SLSTMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gates = nn.Linear(2 * d_model, 4 * d_model)

    def gate_values(self, x, previous_hidden):
        pre_i, pre_f, pre_z, pre_o = self.gates(torch.cat((x, previous_hidden), dim=-1)).chunk(4, dim=-1)
        return torch.exp(pre_i), torch.exp(pre_f), torch.tanh(pre_z), torch.sigmoid(pre_o)

    def forward(self, x):
        batch, length, width = x.shape
        hidden = x.new_zeros(batch, width)
        cell = x.new_zeros(batch, width)
        normalizer = x.new_zeros(batch, width)
        outputs = []
        for index in range(length):
            input_gate, forget_gate, update, output_gate = self.gate_values(x[:, index], hidden)
            cell = forget_gate * cell + input_gate * update
            normalizer = forget_gate * normalizer + input_gate
            hidden = output_gate * cell / normalizer.clamp_min(1e-6)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class MLSTMBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gates = nn.Linear(d_model, 2 * d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)

    def gate_values(self, x, previous_hidden=None):
        pre_i, pre_f = self.gates(x).chunk(2, dim=-1)
        return torch.exp(pre_i), torch.exp(pre_f)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        memory = x.new_zeros(batch, width, width)
        normalizer = x.new_zeros(batch, width)
        outputs = []
        for index in range(length):
            input_gate, forget_gate = self.gate_values(x[:, index])
            key = torch.tanh(k[:, index])
            value = torch.tanh(v[:, index])
            query = torch.tanh(q[:, index])
            memory = forget_gate.unsqueeze(-1) * memory + input_gate.unsqueeze(-1) * value.unsqueeze(-1) * key.unsqueeze(-2)
            normalizer = forget_gate * normalizer + input_gate * key
            read = torch.matmul(memory, query.unsqueeze(-1)).squeeze(-1)
            denominator = torch.abs((normalizer * query).sum(dim=-1, keepdim=True)).clamp_min(1.0)
            outputs.append(read / denominator)
        return torch.stack(outputs, dim=1)


class XLSTMBlock(nn.Module):
    def __init__(self, d_model, mode):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        self.recurrent = SLSTMBlock(d_model) if mode == "slstm" else MLSTMBlock(d_model)
        self.output = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.output(self.recurrent(self.norm(x)))
        return x + self.ff(self.ff_norm(x))


class XLSTM(nn.Module):
    architecture_id = "xlstm"
    PRESETS = {"tiny": {"d_model": 16, "num_layers": 1}, "base": {"d_model": 64, "num_layers": 2}}

    def __init__(self, vocab_size=128, context_length=128, d_model=32, num_layers=2, mode="slstm", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"slstm", "mlstm"}:
            raise ValueError("mode must be slstm or mlstm")
        if min(vocab_size, context_length, d_model, num_layers) <= 0 or vocab_size <= 1:
            raise ValueError("invalid xLSTM dimensions")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([XLSTMBlock(d_model, mode) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
