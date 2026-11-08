from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rte.baselines import overlap
from rte.config import Config
from rte.data import grouped_holdout, grouped_kfold, labels_of, load_jsonl, make_collate
from rte.params import count_params
from rte.train import kd_loss

DATA = Path("data")
needs_data = pytest.mark.skipif(not (DATA / "train.jsonl").exists(), reason="run `make data`")


@needs_data
def test_splits_sizes_and_balance():
    train, dev = load_jsonl(DATA / "train.jsonl"), load_jsonl(DATA / "dev.jsonl")
    assert (len(train), len(dev)) == (2490, 277)
    assert abs(labels_of(train).mean() - 0.5) < 0.02
    assert not {(r["premise"], r["hypothesis"]) for r in train} & {
        (r["premise"], r["hypothesis"]) for r in dev
    }


@needs_data
def test_grouped_splits_never_share_a_premise():
    rows = load_jsonl(DATA / "train.jsonl")
    fit, val = grouped_holdout(rows, 0.1, seed=0)
    assert not {rows[i]["premise"] for i in fit} & {rows[i]["premise"] for i in val}
    folds = grouped_kfold(rows, 5)
    held = np.concatenate([np.array(h) for _, h in folds])
    assert sorted(held.tolist()) == list(range(len(rows)))  # every row scored exactly once


def test_collate_pads_dynamically():
    batch = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "row_id": 0, "labels": 0},
        {"input_ids": [1, 2], "attention_mask": [1, 1], "row_id": 1, "labels": 1},
    ]
    out = make_collate(pad_id=0)(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
    assert out["row_id"].tolist() == [0, 1]


def test_kd_loss_reduces_to_ce_when_alpha_is_one():
    s, t, y = torch.randn(4, 2), torch.randn(4, 2), torch.tensor([0, 1, 0, 1])
    ce = torch.nn.functional.cross_entropy(s, y)
    assert torch.allclose(kd_loss(s, t, y, alpha=1.0, t=2.0), ce)
    assert kd_loss(s, s, y, alpha=0.0, t=2.0).abs() < 1e-6  # identical teacher -> zero KL


def test_count_params_splits_embeddings():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.head = nn.Embedding(10, 4), nn.Linear(4, 2)

        def get_input_embeddings(self):
            return self.emb

    p = count_params(Tiny())
    assert p == {"total": 50, "trainable": 50, "word_embedding": 40, "non_embedding": 10}


def test_overlap_feature():
    rows = [{"premise": "a b c", "hypothesis": "a b"}, {"premise": "x", "hypothesis": "y z"}]
    assert overlap(rows).tolist() == [1.0, 0.0]


@pytest.mark.parametrize("name", sorted(p.name for p in Path("configs").glob("*.yaml")))
def test_configs_load(name):
    cfg = Config.load(Path("configs") / name)
    assert cfg.model_name.startswith("microsoft/deberta-v3")
