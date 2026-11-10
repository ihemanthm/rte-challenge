"""Vocabulary pruning: drop the word-embedding rows a trained checkpoint will (almost) never use.

DeBERTa-v3's 128k-token embedding is 49.2M of xsmall's 70.8M parameters, but RTE text uses ~19k
tokens. The kept set is
    special tokens  U  every token in the training set  U  the top-N tokens of the tokenizer's
    own unigram model (its piece log-probabilities rank tokens by general-English frequency),
so unseen text is still covered without fitting anything to dev. Any other token maps to [UNK].

N is chosen on train only: split train in half by premise, and measure how much of half B's
tokens are covered by (half A's tokens U top-N); keep the smallest N reaching --target (98%).
Unseen RTE text is mostly rare names, so no N short of the full vocabulary is lossless; the
script also writes the whole params-vs-dev-accuracy curve over N (reported, not used to choose).

python -m rte.prune --checkpoint outputs/<run> --out outputs/<run>_pruned
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from rte import ID2LABEL
from rte.data import (
    PairDataset,
    grouped_holdout,
    labels_of,
    load_jsonl,
    make_loader,
    write_jsonl,
)
from rte.device import pick_device
from rte.modeling import load_checkpoint, prune_vocab
from rte.params import count_params
from rte.train import predict_logits

GRID = [0, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 60000]


def token_ids(rows: list[dict], tok, max_length: int) -> list[list[int]]:
    return PairDataset(rows, tok, max_length).enc["input_ids"]


def frequency_ranking(ckpt: Path) -> list[int]:
    """Token ids sorted by the SentencePiece unigram log-probability, most frequent first."""
    vocab = json.loads((ckpt / "tokenizer.json").read_text())["model"]["vocab"]
    return sorted(range(len(vocab)), key=lambda i: -vocab[i][1])


def coverage(seqs: list[list[int]], keep: set[int]) -> dict[str, float]:
    toks = [t for s in seqs for t in s]
    return {
        "token_coverage": float(np.mean([t in keep for t in toks])),
        "rows_fully_covered": float(np.mean([all(t in keep for t in s) for s in seqs])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="run dir or its model/ subdir")
    p.add_argument("--out", required=True, help="new run dir; the pruned model goes in out/model")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--dev", default="data/dev.jsonl")
    p.add_argument("--top-n", type=int, help="skip the search and keep this many frequent tokens")
    p.add_argument("--target", type=float, default=0.98, help="held-out token coverage to reach")
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if (ckpt / "model").is_dir():
        ckpt = ckpt / "model"
    out = Path(args.out)
    tok = AutoTokenizer.from_pretrained(ckpt)
    specials = set(tok.all_special_ids)
    ranking = frequency_ranking(ckpt)

    train, dev = load_jsonl(args.train), load_jsonl(args.dev)
    train_seqs = token_ids(train, tok, args.max_length)

    # choose N on train only: half A's vocabulary + top-N must cover half B
    a_idx, b_idx = grouped_holdout(train, 0.5, seed=0)
    a_vocab = specials | {t for i in a_idx for t in train_seqs[i]}
    b_seqs = [train_seqs[i] for i in b_idx]
    search = []
    for n in GRID:
        cov = coverage(b_seqs, a_vocab | set(ranking[:n]))
        search.append({"top_n": n, **cov})
        print(f"top_n {n:6d}: held-out token coverage {cov['token_coverage']:.4%}")
    top_n = args.top_n
    if top_n is None:
        top_n = next((s["top_n"] for s in search if s["token_coverage"] >= args.target), GRID[-1])
    print(f"keeping train vocabulary + top {top_n} frequent tokens")

    keep = specials | {t for s in train_seqs for t in s} | set(ranking[:top_n])
    dev_seqs = token_ids(dev, tok, args.max_length)
    dev_cov = coverage(dev_seqs, keep)

    device = pick_device()
    model = load_checkpoint(ckpt).to(device)
    before = count_params(model)
    dl = make_loader(PairDataset(dev, tok, args.max_length), tok, 64, shuffle=False)
    logits_full = predict_logits(model, dl, device)
    y = labels_of(dev)
    train_vocab = specials | {t for s in train_seqs for t in s}
    curve = []
    for n in GRID + [len(ranking)]:
        m = copy.deepcopy(model)
        k = train_vocab | set(ranking[:n])
        prune_vocab(m, k, tok.unk_token_id)
        acc = float((predict_logits(m, dl, device).argmax(-1) == y).mean())
        curve.append(
            {
                "top_n": n,
                "total_params": count_params(m)["total"],
                "dev_acc": acc,
                "dev_token_coverage": coverage(dev_seqs, k)["token_coverage"],
            }
        )
        print(f"curve top_n {n:6d}: {curve[-1]['total_params'] / 1e6:5.1f}M dev acc {acc:.4f}")
        del m
    kept = prune_vocab(model, keep, tok.unk_token_id)
    after = count_params(model)
    logits_pruned = predict_logits(model, dl, device)

    full_pred, pruned_pred = logits_full.argmax(-1), logits_pruned.argmax(-1)
    covered = np.array([all(t in keep for t in s) for s in dev_seqs])
    max_diff_covered = (
        float(np.abs(logits_full - logits_pruned)[covered].max()) if covered.any() else 0.0
    )

    (out / "model").mkdir(parents=True, exist_ok=True)
    model.cpu().save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    for f in ("selected_seed.txt",):
        if (ckpt / f).exists():
            shutil.copy(ckpt / f, out / "model" / f)

    probs = torch.softmax(torch.tensor(logits_pruned), -1).numpy()
    write_jsonl(
        [
            {**r, "pred": ID2LABEL[int(pr.argmax())], "p_entailment": float(pr[0])}
            for r, pr in zip(dev, probs, strict=True)
        ],
        out / "dev_predictions.jsonl",
    )
    report = {
        "source_checkpoint": str(ckpt),
        "top_n": top_n,
        "kept_rows": kept,
        "original_rows": before["word_embedding"] // model.get_input_embeddings().weight.shape[1],
        "top_n_search_on_train_halves": search,
        "dev_coverage": dev_cov,
        "dev_acc_full": float((full_pred == y).mean()),
        "dev_acc_pruned": float((pruned_pred == y).mean()),
        "dev_prediction_agreement": float((full_pred == pruned_pred).mean()),
        "max_logit_diff_on_fully_covered_rows": max_diff_covered,
        "params_full": before,
        "params": after,
        "curve_over_top_n": curve,
    }
    (out / "prune.json").write_text(json.dumps(report, indent=2))
    (out / "params.json").write_text(json.dumps(after, indent=2))
    print(
        json.dumps(
            {k: v for k, v in report.items() if not k.startswith(("top_n_search", "curve"))},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
