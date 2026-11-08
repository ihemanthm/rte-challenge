"""Non-neural sanity baselines: anything we train must clearly beat these.

python -m rte.baselines --train data/train.jsonl --dev data/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rte.data import labels_of, load_jsonl

_TOK = re.compile(r"\w+")


def _toks(s: str) -> set[str]:
    return set(_TOK.findall(s.lower()))


def overlap(rows: list[dict]) -> np.ndarray:
    """Fraction of hypothesis tokens that also appear in the premise."""
    return np.array(
        [
            len(_toks(r["hypothesis"]) & _toks(r["premise"])) / max(1, len(_toks(r["hypothesis"])))
            for r in rows
        ]
    )


def majority(train_y: np.ndarray, dev_y: np.ndarray) -> float:
    return float((dev_y == np.bincount(train_y).argmax()).mean())


def overlap_threshold(train: list[dict], dev: list[dict]) -> tuple[float, float]:
    """Predict entailment (0) when overlap >= t; t tuned on train only."""
    tr_o, dv_o = overlap(train), overlap(dev)
    tr_y, dv_y = labels_of(train), labels_of(dev)
    grid = np.linspace(0, 1, 41)
    t = max(grid, key=lambda t: ((tr_o < t).astype(int) == tr_y).mean())
    return float(((dv_o < t).astype(int) == dv_y).mean()), float(t)


def tfidf_lr(train: list[dict], dev: list[dict]) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit([r["premise"] for r in train] + [r["hypothesis"] for r in train])

    def feats(rows):
        p = vec.transform([r["premise"] for r in rows])
        h = vec.transform([r["hypothesis"] for r in rows])
        return hstack([h, p.multiply(h), csr_matrix(overlap(rows)[:, None])]).tocsr()

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    clf.fit(feats(train), labels_of(train))
    return float((clf.predict(feats(dev)) == labels_of(dev)).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--out", default="outputs/baselines/metrics.json")
    args = p.parse_args()

    train, dev = load_jsonl(args.train), load_jsonl(args.dev)
    ov_acc, t = overlap_threshold(train, dev)
    res = {
        "majority": majority(labels_of(train), labels_of(dev)),
        "word_overlap": ov_acc,
        "word_overlap_threshold": t,
        "tfidf_logreg": tfidf_lr(train, dev),
    }
    for k, v in res.items():
        print(f"{k:24s} {v:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
