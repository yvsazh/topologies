"""Offline event-codec, recurrent-LM motif training and seeded generation."""
import torch
from topologies.data.event_codec import EventCodec
from topologies.models.recurrent_models import RecurrentLM
from topologies.inference_tasks.music import sample_tokens

torch.manual_seed(11)
codec = EventCodec(ticks_per_beat=24, max_duration=16)
motifs = [[{"pitch": 60 + j, "duration_ticks": 6, "velocity": 80} for j in range(4)] for _ in range(8)]
encoded = [codec.encode(m) for m in motifs]
x = torch.tensor([seq[:-1] for seq in encoded]); y = torch.tensor([seq[1:] for seq in encoded])
model = RecurrentLM(vocab_size=len(codec.events), context_length=8, hidden_size=16)
opt = torch.optim.Adam(model.parameters(), lr=.005)
for _ in range(2):
    opt.zero_grad(); logits = model(x); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)); loss.backward(); opt.step()
generated = sample_tokens(model, x[:1, :1], 6, seed=5, top_k=4)
print(codec.metadata(model.context_length))
print("generated events", codec.decode(generated[0].tolist()))
