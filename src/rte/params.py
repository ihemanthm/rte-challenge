"""Parameter accounting for the model card: total, trainable, embedding vs. the rest."""

from __future__ import annotations

from torch import nn


def count_params(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    embedding = emb.weight.numel() if emb is not None else 0
    return {
        "total": total,
        "trainable": trainable,
        "word_embedding": embedding,
        "non_embedding": total - embedding,
    }


def fmt(n: int) -> str:
    return f"{n / 1e6:.1f}M"
