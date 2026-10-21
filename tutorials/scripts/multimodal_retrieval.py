"""Offline generated image-caption contrastive training and held-out retrieval."""
import torch
from topologies.models.image_text_dual_encoder import ImageTextDualEncoder
from topologies.training.recipes.contrastive import symmetric_info_nce, retrieval_recall

torch.manual_seed(7)
model = ImageTextDualEncoder(vocab_size=16, context_length=4, embedding_dim=8,
                             width=4, text_width=8, text_heads=2, text_layers=1)
# Synthetic colored-shape proxy: each image/caption pair has a stable identity.
images = torch.randn(8, 3, 16, 16)
captions = torch.tensor([[i + 1, 0] for i in range(8)])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(3):
    optimizer.zero_grad()
    image, text, _ = model(images[:4], captions[:4])
    loss = symmetric_info_nce(image, text, pair_ids=[f"train-{i}" for i in range(4)])
    loss.backward(); optimizer.step()
with torch.no_grad():
    image, text, _ = model(images[4:], captions[4:])
print("held-out retrieval", retrieval_recall(image, text, [f"test-{i}" for i in range(4)], ks=(1, 5)))
