"""Accuracy of the submitted models on a labelled file (the mentor's val.jsonl), with a chart.

Run predictions first, then plot:
python scripts/predict_test.py --input val.jsonl --out report/mentor_val --device cpu
python scripts/plot_val_results.py --gold val.jsonl --preds report/mentor_val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = [  # file stem, short label, parameters (M)
    ("A_masked_kd_full_70.8M", "A: masked KD", 70.8),
    ("B_masked_kd_pruned_45.5M", "B: A, pruned vocab", 45.5),
    ("C_plain_full_70.8M", "C: plain", 70.8),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--preds", default="report/mentor_val")
    p.add_argument("--fig", default="report/figures/mentor_val")
    args = p.parse_args()

    gold = {str(r["idx"]): r["label"] for r in map(json.loads, open(args.gold, encoding="utf-8"))}
    majority = max(sum(v == lab for v in gold.values()) for lab in set(gold.values())) / len(gold)
    rows = []
    for stem, label, params in MODELS:
        preds = {
            r["idx"]: r["prediction"]
            for r in csv.DictReader(open(Path(args.preds) / f"{stem}_predictions.csv"))
        }
        y = np.array([gold[k] for k in preds])
        ok = np.array([preds[k] == gold[k] for k in preds])
        lo, hi = binomtest(int(ok.sum()), len(ok)).proportion_ci(0.95, "wilson")
        rows.append(
            {
                "model": stem,
                "label": label,
                "params_M": params,
                "n": len(ok),
                "correct": int(ok.sum()),
                "accuracy": float(ok.mean()),
                "ci95": [float(lo), float(hi)],
                "acc_entailment": float(ok[y == "entailment"].mean()),
                "acc_not_entailment": float(ok[y == "not_entailment"].mean()),
            }
        )
        r = rows[-1]
        print(
            f"{label:20s} {params:5.1f}M  acc {r['accuracy']:.4f} ({r['correct']}/{r['n']}) "
            f"95% CI [{lo:.3f}, {hi:.3f}]  ent {r['acc_entailment']:.3f} "
            f"not_ent {r['acc_not_entailment']:.3f}"
        )
    Path(args.preds, "summary.json").write_text(
        json.dumps({"majority": majority, "models": rows}, indent=2)
    )

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.3, 2.15))  # sized to sit beside text in the report
    x = np.arange(len(rows))
    w = 0.26
    acc = [r["accuracy"] * 100 for r in rows]
    err = [
        [a - r["ci95"][0] * 100 for a, r in zip(acc, rows, strict=True)],
        [r["ci95"][1] * 100 - a for a, r in zip(acc, rows, strict=True)],
    ]
    ax.bar(x - w, acc, w, yerr=err, capsize=3, color="#3a47c9", label="overall (95% CI)")
    ax.bar(x, [r["acc_entailment"] * 100 for r in rows], w, color="#8e96e0", label="gold entail.")
    ax.bar(
        x + w,
        [r["acc_not_entailment"] * 100 for r in rows],
        w,
        color="#c9cdf0",
        label="gold not-entail.",
    )
    for xi, a, r in zip(x, acc, rows, strict=True):  # label sits above the CI whisker
        ax.text(
            xi - w, r["ci95"][1] * 100 + 1.2, f"{a:.1f}", ha="center", fontsize=7, weight="bold"
        )
    ax.axhline(majority * 100, ls=":", color="grey", lw=0.8)
    ax.text(
        -0.5,
        majority * 100 - 4.5,
        f"majority {majority * 100:.1f}",
        fontsize=6,
        color="grey",
        ha="left",
    )
    ax.set_xticks(x, [f"{r['label'].split(':')[0]}\n{r['params_M']}M" for r in rows])
    ax.set_ylabel("accuracy % (val.jsonl)")
    ax.set_ylim(40, 100)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.fig}.pdf")
    fig.savefig(f"{args.fig}.png", dpi=200)
    print(f"-> {args.fig}.pdf/.png, {args.preds}/summary.json")


if __name__ == "__main__":
    main()
