import torch
from torch import nn


class GlowTTS(nn.Module):
    architecture_id = "glow-tts"

    def __init__(self, channels=4, text_vocab_size=32, text_embedding_dim=8, hidden_channels=8):
        super().__init__()
        if channels < 2:
            raise ValueError("channels must be at least 2")
        self.channels = channels
        self.text_embedding = nn.Embedding(text_vocab_size, text_embedding_dim)
        first = channels // 2
        second = channels - first
        self.conditioner = nn.Sequential(
            nn.Linear(first + text_embedding_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, second * 2),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def _coupling_parameters(self, unchanged, text):
        if text.ndim == 2:
            condition = self.text_embedding(text).mean(dim=1, keepdim=True)
        elif text.ndim == 3 and text.shape[-1] == self.text_embedding.embedding_dim:
            condition = text
        else:
            raise ValueError("text must contain token ids or embedded frame conditions")
        if condition.shape[1] == 1:
            condition = condition.expand(-1, unchanged.shape[1], -1)
        elif condition.shape[1] != unchanged.shape[1]:
            raise ValueError("embedded text condition must have one vector per mel frame")
        scale_shift = self.conditioner(torch.cat((unchanged, condition), dim=-1))
        shift, log_scale = scale_shift.chunk(2, dim=-1)
        return shift, 2.0 * torch.tanh(log_scale / 2.0)

    def coupling(self, mel, text, reverse=False):
        if mel.ndim != 3 or mel.shape[-1] != self.channels:
            raise ValueError("mel must have shape (batch, frames, channels)")
        unchanged, transformed = mel.split((self.channels // 2, self.channels - self.channels // 2), dim=-1)
        shift, log_scale = self._coupling_parameters(unchanged, text)
        if reverse:
            mapped = (transformed - shift) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=-1)
        else:
            mapped = transformed * torch.exp(log_scale) + shift
            log_det = log_scale.sum(dim=-1)
        return torch.cat((unchanged, mapped), dim=-1), log_det

    @staticmethod
    def monotonic_alignment(log_likelihood):
        squeeze = log_likelihood.ndim == 2
        if squeeze:
            log_likelihood = log_likelihood.unsqueeze(0)
        if log_likelihood.ndim != 3:
            raise ValueError("log_likelihood must have shape (text, frames) or (batch, text, frames)")
        batch, text_length, frame_length = log_likelihood.shape
        if text_length == 0 or frame_length < text_length:
            raise ValueError("MAS requires nonempty text and at least one frame per text position")

        alignments = []
        for batch_index in range(batch):
            scores = log_likelihood[batch_index]
            dp = torch.full_like(scores, float("-inf"))
            dp[0, 0] = scores[0, 0]
            for frame in range(1, frame_length):
                max_text = min(frame, text_length - 1)
                dp[0, frame] = dp[0, frame - 1] + scores[0, frame]
                for text_index in range(1, max_text + 1):
                    dp[text_index, frame] = torch.maximum(
                        dp[text_index, frame - 1], dp[text_index - 1, frame - 1]
                    ) + scores[text_index, frame]

            path = torch.empty(frame_length, dtype=torch.long, device=scores.device)
            text_index = text_length - 1
            path[-1] = text_index
            for frame in range(frame_length - 1, 0, -1):
                if text_index > 0 and dp[text_index - 1, frame - 1] > dp[text_index, frame - 1]:
                    text_index -= 1
                path[frame - 1] = text_index
            alignments.append(path)
        result = torch.stack(alignments)
        return result[0] if squeeze else result

    def align(self, text, log_likelihood):
        if text.ndim != 2 or text.shape[1] == 0:
            raise ValueError("text must have shape (batch, nonempty text length)")
        positions = self.monotonic_alignment(log_likelihood)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[0] != text.shape[0] or positions.shape[1] < text.shape[1]:
            raise ValueError("alignment scores must match the text batch and be surjective")
        return torch.gather(text, 1, positions)

    def forward(self, text, mel, log_likelihood):
        if text.ndim != 2 or mel.ndim != 3 or text.shape[0] != mel.shape[0]:
            raise ValueError("text and mel must have shapes (batch, text) and (batch, frames, channels)")
        aligned_positions = self.monotonic_alignment(log_likelihood)
        if aligned_positions.ndim == 1:
            aligned_positions = aligned_positions.unsqueeze(0)
        if aligned_positions.shape != mel.shape[:2]:
            raise ValueError("alignment scores must produce one text position per mel frame")
        text_embeddings = self.text_embedding(text)
        aligned_embeddings = text_embeddings.gather(
            1, aligned_positions.unsqueeze(-1).expand(-1, -1, text_embeddings.shape[-1])
        )
        return self.coupling(mel, aligned_embeddings)
