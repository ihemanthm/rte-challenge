"""Error analysis on dev predictions: error rate per phenomenon tag + a sample of mistakes.

python -m rte.analysis --predictions outputs/<run>/dev_predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from rte.baselines import overlap
from rte.data import load_jsonl

NEG = re.compile(r"\b(not|no|never|none|nobody|nothing|neither|nor|without)\b|n't", re.I)
NUM = re.compile(r"\d")


def tag(rows: list[dict]) -> dict[str, np.ndarray]:
    ov = overlap(rows)
    plen = np.array([len(r["premise"].split()) for r in rows])
    gold_ent = np.array([r["label"] == "entailment" for r in rows])
    return {
        "negation_in_hypothesis": np.array([bool(NEG.search(r["hypothesis"])) for r in rows]),
        "numbers_or_dates": np.array([bool(NUM.search(r["hypothesis"])) for r in rows]),
        "long_premise_p90": plen >= np.percentile(plen, 90),
        "high_overlap_not_entailed": (ov >= 0.8) & ~gold_ent,  # lexical-overlap trap
        "low_overlap_entailed": (ov < 0.5) & gold_ent,  # needs paraphrase / world knowledge
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--n-samples", type=int, default=25)
    args = p.parse_args()

    rows = load_jsonl(args.predictions)
    wrong = np.array([r["pred"] != r["label"] for r in rows])
    tags = tag(rows)
    print(f"overall: n={len(rows)} acc={1 - wrong.mean():.4f}\n")
    print(f"{'tag':28s} {'n':>4s} {'err_rate':>9s}")
    for name, mask in tags.items():
        n = int(mask.sum())
        err = wrong[mask].mean() if n else float("nan")
        print(f"{name:28s} {n:4d} {err:9.3f}")
    print(f"{'(untagged)':28s} {int((~np.any(list(tags.values()), axis=0)).sum()):4d}")

    out = Path(args.predictions).with_name("dev_errors.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pred", "p_entailment", *tags, "premise", "hypothesis"])
        for i in np.flatnonzero(wrong):
            r = rows[i]
            w.writerow(
                [
                    r.get("idx"),
                    r["label"],
                    r["pred"],
                    f"{r['p_entailment']:.3f}",
                    *[int(m[i]) for m in tags.values()],
                    r["premise"],
                    r["hypothesis"],
                ]
            )
    print(f"\n{int(wrong.sum())} errors -> {out} (annotate a sample of {args.n_samples} by hand)")


if __name__ == "__main__":
    main()
