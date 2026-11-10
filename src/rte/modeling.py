"""Model construction: pretrained encoder + classifier, optionally with a low-rank word embedding.

The DeBERTa-v3 word embedding (128k vocab x hidden) is most of a small model's parameters
(49.2M of xsmall's 70.8M). `embedding_rank=r` replaces it with A (V x r) @ B (r x H), initialised
from the truncated SVD of the pretrained matrix, so the factorised model starts as the best
rank-r approximation of the original and is then fine-tuned end to end.

`prune_vocab` instead keeps only the embedding rows of a chosen token set (see rte.prune). The
tokenizer is untouched: a lookup buffer maps original ids to kept rows, and any other id goes to
[UNK]. On text made only of kept tokens the pruned model computes exactly what the full one does.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from rte import ID2LABEL, LABEL2ID


class FactorizedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, dim: int, rank: int, padding_idx: int | None = None):
        super().__init__()
        self.A = nn.Embedding(num_embeddings, rank, padding_idx=padding_idx)
        self.B = nn.Linear(rank, dim, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(input_ids))


@torch.no_grad()
def factorize_embeddings(model, rank: int, from_svd: bool = True) -> float:
    """Swap the input word embedding for a rank-`rank` factorisation. Returns the fraction of
    the pretrained matrix's squared Frobenius norm kept by the truncated SVD (1.0 if not SVD)."""
    old: nn.Embedding = model.get_input_embeddings()
    num, dim = old.weight.shape
    new = FactorizedEmbedding(num, dim, rank, padding_idx=old.padding_idx)
    kept = 1.0
    if from_svd:
        w = old.weight.detach().float().cpu()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        root = s[:rank].sqrt()
        new.A.weight.copy_(u[:, :rank] * root)
        new.B.weight.copy_((root[:, None] * vh[:rank]).T)
        kept = float((s[:rank] ** 2).sum() / (s**2).sum())
    model.set_input_embeddings(new.to(old.weight.device))
    model.config.embedding_rank = rank  # saved with the checkpoint, read by load_checkpoint
    return kept


class PrunedEmbedding(nn.Module):
    """Embedding over a subset of the vocabulary. `remap[orig_id]` is the row for that token;
    the buffer is a lookup table (int32), not a learned parameter."""

    def __init__(self, orig_vocab: int, kept: int, dim: int, padding_idx: int | None = None):
        super().__init__()
        self.emb = nn.Embedding(kept, dim, padding_idx=padding_idx)
        self.register_buffer("remap", torch.zeros(orig_vocab, dtype=torch.int32))

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(self.remap[input_ids].long())


@torch.no_grad()
def prune_vocab(model, keep_ids, unk_id: int) -> int:
    """Keep only the embedding rows in `keep_ids` (must include unk_id). Returns rows kept."""
    old: nn.Embedding = model.get_input_embeddings()
    num, dim = old.weight.shape
    keep = torch.tensor(sorted(set(int(i) for i in keep_ids) | {unk_id}), dtype=torch.long)
    remap = torch.empty(num, dtype=torch.int32)
    new_unk = int((keep == unk_id).nonzero())
    remap.fill_(new_unk)
    remap[keep] = torch.arange(len(keep), dtype=torch.int32)
    pad = old.padding_idx
    new = PrunedEmbedding(num, len(keep), dim, int(remap[pad]) if pad is not None else None)
    new.emb.weight.copy_(old.weight[keep])
    new.remap.copy_(remap)
    model.set_input_embeddings(new.to(old.weight.device))
    model.config.pruned_vocab_size = len(keep)  # read by load_checkpoint
    return len(keep)


def load_model(name: str, embedding_rank: int | None = None):
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # transformers>=5 defaults to the checkpoint dtype; DeBERTa-v3 ships fp16 weights, and
        # AdamW on fp16 params underflows (eps, v) and diverges to NaN after one step.
        # Master weights stay fp32; mixed precision is applied via autocast on CUDA only.
        dtype=torch.float32,
    )
    if embedding_rank:
        kept = factorize_embeddings(model, embedding_rank)
        print(f"factorised word embedding to rank {embedding_rank} (SVD keeps {kept:.1%} energy)")
    return model


def load_checkpoint(path: str | Path):
    """Load a fine-tuned checkpoint saved by rte.train, factorised or not."""
    conf = AutoConfig.from_pretrained(path)
    rank = getattr(conf, "embedding_rank", None)
    kept = getattr(conf, "pruned_vocab_size", None)
    if not rank and not kept:
        return AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
    from safetensors.torch import load_file

    model = AutoModelForSequenceClassification.from_config(conf, dtype=torch.float32)
    if rank:
        factorize_embeddings(model, rank, from_svd=False)
    else:
        old = model.get_input_embeddings()
        num, dim = old.weight.shape
        model.set_input_embeddings(PrunedEmbedding(num, kept, dim, old.padding_idx))
    model.load_state_dict(load_file(Path(path) / "model.safetensors"), strict=True)
    return model
