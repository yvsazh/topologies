"""Small task helpers for paired retrieval without dataset/provider coupling."""
from topologies.training.recipes.contrastive import retrieval_recall


def evaluate_retrieval(model, batches, ks=(1, 5), forbidden_train_pair_ids=()):
    """Evaluate unique held-out identities; reject any identity seen in training."""
    images, texts, identities = [], [], []
    forbidden = set(forbidden_train_pair_ids)
    model.eval()
    import torch
    with torch.no_grad():
        for batch in batches:
            ids = list(batch["pair_ids"])
            if forbidden.intersection(ids): raise ValueError("held-out retrieval reuses training pair identity")
            images.append(model.encode_image(batch["images"]))
            texts.append(model.encode_text(batch["tokens"], batch.get("mask")))
            identities.extend(ids)
    if not images: raise ValueError("retrieval evaluation requires at least one batch")
    return retrieval_recall(torch.cat(images), torch.cat(texts), identities, ks)
