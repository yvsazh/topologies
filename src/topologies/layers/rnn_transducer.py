import torch
from torch import nn


class RNNTransducer(nn.Module):
    architecture_id = "rnn-transducer"
    component_id = "rnn-transducer"

    def __init__(self, input_size=4, hidden_size=4, vocab_size=3, blank_id=0):
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or vocab_size <= 1:
            raise ValueError("input_size and hidden_size must be positive; vocab_size must exceed one")
        if not 0 <= blank_id < vocab_size:
            raise ValueError("blank_id must index the vocabulary")
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.prediction = nn.Embedding(vocab_size, hidden_size)
        self.prediction_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.transcription_projection = nn.Linear(hidden_size, vocab_size)
        self.prediction_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_frames, prediction_tokens):
        if encoder_frames.ndim != 3 or prediction_tokens.ndim != 2:
            raise ValueError("expected encoder_frames [batch, time, features] and prediction_tokens [batch, time]")
        if encoder_frames.shape[0] != prediction_tokens.shape[0]:
            raise ValueError("encoder and prediction batches must match")
        if encoder_frames.shape[1] == 0 or prediction_tokens.shape[1] == 0:
            raise ValueError("encoder and prediction sequences must be nonempty")
        if torch.any(prediction_tokens < 0) or torch.any(prediction_tokens >= self.vocab_size):
            raise ValueError("prediction token id is outside the vocabulary")
        if torch.any(prediction_tokens == self.blank_id):
            raise ValueError("prediction_tokens must contain labels, not the blank symbol")
        encoded, _ = self.encoder(encoder_frames)
        label_embeddings = self.prediction(prediction_tokens)
        blank_prefix = label_embeddings.new_zeros(label_embeddings.shape[0], 1, label_embeddings.shape[-1])
        prediction_inputs = torch.cat((blank_prefix, label_embeddings), dim=1)
        predicted, _ = self.prediction_rnn(prediction_inputs)
        transcription_logits = self.transcription_projection(encoded)
        prediction_logits = self.prediction_projection(predicted)
        joint_logits = transcription_logits.unsqueeze(2) + prediction_logits.unsqueeze(1)
        return torch.softmax(joint_logits, dim=-1)
