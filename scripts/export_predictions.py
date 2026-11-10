"""Copy a run's dev predictions into submission/ as JSONL and CSV (idx, prediction, p_entailment).

python scripts/export_predictions.py --run outputs/student_deberta_v3_xsmall_pruned
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="submission")
    p.add_argument("--name", default="dev_predictions")
    args = p.parse_args()
    rows = [json.loads(line) for line in open(Path(args.run) / "dev_predictions.jsonl")]
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    keep = ["idx", "premise", "hypothesis", "pred", "p_entailment"]
    with open(out / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
    with open(out / f"{args.name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "prediction", "p_entailment"])
        for r in rows:
            w.writerow([r["idx"], r["pred"], f"{r['p_entailment']:.6f}"])
    acc = sum(r["pred"] == r["label"] for r in rows if "label" in r) / len(rows)
    print(f"wrote {len(rows)} predictions to {out}/{args.name}.{{jsonl,csv}} (dev acc {acc:.4f})")


if __name__ == "__main__":
    main()
