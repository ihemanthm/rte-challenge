"""Validation-only evidence for choosing the submitted models (accuracy vs. parameters).

For each candidate checkpoint, score its own seed's premise-grouped validation split while
keeping only: special tokens U tokens of the rows it was *fit* on U the top-N frequent tokens.
Any other token becomes [UNK], exactly as in the pruned model. (Keeping all train tokens would
include the validation rows themselves, so pruning would look free on validation.)

Dev is never read. Output: report/model_selection.json and a table on stdout.

python scripts/select_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from rte.data import PairDataset, grouped_holdout, labels_of, load_jsonl, make_loader
from rte.device import pick_device
from rte.modeling import load_checkpoint
from rte.prune import GRID, frequency_ranking
from rte.train import predict_logits

CANDIDATES = [  # label, checkpoint dir (model/), training run (its seeds/ hold the full-vocab val)
    (
        "masked KD, seed 21",
        "outputs/distill_masked_deberta_v3_xsmall_pruned/model",
        "outputs/distill_masked_deberta_v3_xsmall",
    ),
    (
        "plain, seed 21",
        "outputs/student_deberta_v3_xsmall/model",
        "outputs/student_deberta_v3_xsmall",
    ),
    (
        "plain KD, seed 100",
        "outputs/distill_deberta_v3_xsmall/model",
        "outputs/distill_deberta_v3_xsmall",
    ),
]
FULL_PARAMS = 70_830_722
DIM, NON_EMB = 384, 21_640_322  # DeBERTa-v3-xsmall hidden size and non-embedding parameters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out", default="report/model_selection.json")
    args = p.parse_args()

    train = load_jsonl(args.train)
    device = pick_device()
    results = []
    for label, ckpt, run in CANDIDATES:
        ckpt = Path(ckpt)
        seed = int((ckpt / "selected_seed.txt").read_text())
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = load_checkpoint(ckpt).to(device).eval()
        emb = model.get_input_embeddings()
        # ids the checkpoint still has a row for (all of them unless already pruned)
        if hasattr(emb, "remap"):
            remap = emb.remap.cpu().numpy()
            available = set(np.flatnonzero(remap != remap[tok.unk_token_id]).tolist())
            available.add(tok.unk_token_id)
        else:
            available = set(range(emb.weight.shape[0]))
        fit_idx, val_idx = grouped_holdout(train, args.val_fraction, seed)
        val = [train[i] for i in val_idx]
        y = labels_of(val)
        ds = PairDataset(val, tok, 256)
        ids = [list(s) for s in ds.enc["input_ids"]]
        specials = set(tok.all_special_ids)
        fit_vocab = specials | {
            t for i in fit_idx for t in PairDataset([train[i]], tok, 256).enc["input_ids"][0]
        }
        ranking = frequency_ranking(ckpt)
        curve = []
        for n in [*GRID, len(ranking)]:
            keep = fit_vocab | set(ranking[:n])
            if not keep <= available:  # a pruned checkpoint cannot grow its vocabulary back
                continue
            ds.enc["input_ids"] = [[t if t in keep else tok.unk_token_id for t in s] for s in ids]
            dl = make_loader(ds, tok, 64, shuffle=False)
            acc = float((predict_logits(model, dl, device).argmax(-1) == y).mean())
            rows = len(keep | specials)
            curve.append({"top_n": n, "total_params": NON_EMB + rows * DIM, "val_acc": acc})
            print(f"{label:20s} top_n {n:6d} {curve[-1]['total_params'] / 1e6:5.1f}M val {acc:.4f}")
        if curve[-1]["total_params"] < FULL_PARAMS:
            # already-pruned checkpoint: its full-vocabulary val accuracy is the training log's
            full = json.loads((Path(run) / "seeds" / f"seed_{seed}.json").read_text())[
                "best_val_acc"
            ]
            curve.append({"top_n": len(ranking), "total_params": FULL_PARAMS, "val_acc": full})
            print(f"{label:20s} full vocabulary (training log) 70.8M val {full:.4f}")
        results.append({"label": label, "checkpoint": str(ckpt), "seed": seed, "curve": curve})
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
