import torch
from torch import nn
import torch.nn.functional as F


class TransitionalText(nn.Module):
    architecture_id = "transitional-text"
    PRESETS = {
        "tiny": {"vocab_size": 64, "context_length": 16, "d_model": 16, "num_classes": 3, "sentence_width": 2, "mode": "multichannel"},
        "base": {"vocab_size": 256, "context_length": 32, "d_model": 32, "num_classes": 4, "sentence_width": 4, "mode": "multichannel"},
    }

    def __init__(self, vocab_size=128, context_length=32, d_model=32, num_classes=3, sentence_width=2, mode="multichannel", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"multichannel", "bilstm", "han", "elmo", "ulmfit"}:
            raise ValueError("mode must be multichannel, bilstm, han, elmo, or ulmfit")
        if min(vocab_size, context_length, d_model, num_classes, sentence_width) <= 0 or vocab_size <= 1:
            raise ValueError("invalid text dimensions")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_classes = num_classes
        self.sentence_width = sentence_width
        self.mode = mode
        self.embed = nn.Embedding(vocab_size, d_model)
        self.static_embed = nn.Embedding(vocab_size, d_model)
        self.static_embed.weight.requires_grad_(False)
        self.conv = nn.Conv1d(d_model, d_model, 3)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.attn = nn.Parameter(torch.zeros(d_model * 2))
        self.word_attn = nn.Parameter(torch.zeros(d_model))
        self.sent_attn = nn.Parameter(torch.zeros(d_model))
        self.layer_logits = nn.Parameter(torch.zeros(3))
        self.deep1 = nn.Linear(d_model, d_model, bias=False)
        self.deep2 = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, num_classes)
        self.attn.data[0] = 1.0
        self.word_attn.data[0] = 1.0
        self.sent_attn.data[0] = 1.0

    def _tokens(self, tokens):
        tokens = torch.as_tensor(tokens)
        if tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.shape[1] > self.context_length:
            raise ValueError("tokens must have shape (batch, length<=context_length)")
        return tokens

    def attention_pool(self, hidden, vector):
        weights = torch.softmax(hidden @ vector, dim=-1)
        return (weights.unsqueeze(-1) * hidden).sum(dim=-2), weights

    def layer_mix(self, tokens):
        tokens = self._tokens(tokens)
        first = self.embed(tokens)
        second = self.deep1(first)
        third = self.deep2(second)
        mix = torch.softmax(self.layer_logits, dim=0)
        return mix[0] * first + mix[1] * second + mix[2] * third

    def word_weights(self, tokens):
        tokens = self._tokens(tokens)
        if tokens.shape[1] % self.sentence_width != 0:
            raise ValueError("length must be divisible by sentence_width")
        words = self.embed(tokens).reshape(tokens.shape[0], tokens.shape[1] // self.sentence_width, self.sentence_width, self.d_model)
        return torch.softmax(words @ self.word_attn, dim=-1)

    def freeze_stage(self, stage):
        if stage not in (0, 1, 2):
            raise ValueError("stage must be 0, 1, or 2")
        self.embed.weight.requires_grad_(stage == 2)
        for parameter in self.lstm.parameters():
            parameter.requires_grad_(stage >= 1)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.static_embed.weight.requires_grad_(False)

    def forward(self, tokens):
        tokens = self._tokens(tokens)
        if self.mode == "multichannel":
            if tokens.shape[1] < 3:
                raise ValueError("multichannel mode needs at least 3 tokens")
            mixed = self.embed(tokens) + self.static_embed(tokens)
            pooled = F.relu(self.conv(mixed.transpose(1, 2))).max(dim=-1).values
            return self.head(pooled)
        if self.mode == "elmo":
            return self.head(self.layer_mix(tokens).mean(dim=1))
        if self.mode == "han":
            weights = self.word_weights(tokens)
            words = self.embed(tokens).reshape(tokens.shape[0], tokens.shape[1] // self.sentence_width, self.sentence_width, self.d_model)
            sentences = (weights.unsqueeze(-1) * words).sum(dim=2)
            document, _ = self.attention_pool(sentences, self.sent_attn)
            return self.head(document)
        hidden, _ = self.lstm(self.embed(tokens))
        pooled, _ = self.attention_pool(hidden, self.attn)
        return self.head(self.proj(pooled))
