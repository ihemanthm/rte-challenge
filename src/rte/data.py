"""Data loading, splitting and batching. Labels: 0 = entailment, 1 = not_entailment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from rte import LABEL2ID


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([LABEL2ID[r["label"]] for r in rows])


def premise_groups(rows: list[dict]) -> np.ndarray:
    """Group id per row so that examples sharing a premise never straddle a split."""
    ids: dict[str, int] = {}
    return np.array([ids.setdefault(r["premise"], len(ids)) for r in rows])


def grouped_holdout(rows: list[dict], fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Row indices for (fit, val), split by premise."""
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit_idx, val_idx = next(gss.split(rows, groups=premise_groups(rows)))
    return fit_idx.tolist(), val_idx.tolist()


def grouped_kfold(rows: list[dict], k: int) -> list[tuple[list[int], list[int]]]:
    gkf = GroupKFold(n_splits=k)
    return [(a.tolist(), b.tolist()) for a, b in gkf.split(rows, groups=premise_groups(rows))]


class PairDataset(Dataset):
    """Tokenised premise/hypothesis pairs. `row_id` lets distillation look up teacher logits."""

    def __init__(self, rows: list[dict], tokenizer, max_length: int, row_ids=None):
        self.enc = tokenizer(
            [r["premise"] for r in rows],
            [r["hypothesis"] for r in rows],
            truncation="only_first",  # hypotheses are short; trim the premise if needed
            max_length=max_length,
        )
        self.labels = [LABEL2ID[r["label"]] for r in rows] if "label" in rows[0] else None
        self.row_ids = list(row_ids) if row_ids is not None else list(range(len(rows)))

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["row_id"] = self.row_ids[i]
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def make_collate(pad_id: int):
    """Dynamic padding to the longest sequence in the batch."""

    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        out: dict[str, list] = {}
        for key in batch[0]:
            if key in ("row_id", "labels"):
                out[key] = [b[key] for b in batch]
                continue
            fill = pad_id if key == "input_ids" else 0
            out[key] = [b[key] + [fill] * (width - len(b[key])) for b in batch]
        return {k: torch.tensor(v) for k, v in out.items()}

    return collate


def make_loader(ds: PairDataset, tokenizer, batch_size: int, shuffle: bool, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )
