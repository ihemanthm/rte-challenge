"""Materialise data/train.jsonl (from the challenge zip) and data/dev.jsonl (GLUE RTE validation).

Both files share one schema: {"premise", "hypothesis", "label", "idx"} with string labels
"entailment" / "not_entailment". The dev split is used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

GLUE_LABELS = {0: "entailment", 1: "not_entailment"}  # GLUE RTE label ids


def extract_train(zip_path: Path, out: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("train.jsonl"))
        lines = zf.read(member).decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    write_jsonl(rows, out / "train.jsonl")
    return rows


def fetch_dev(out: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "rte", split="validation")
    rows = [
        {
            "premise": r["sentence1"],
            "hypothesis": r["sentence2"],
            "label": GLUE_LABELS[r["label"]],
            "idx": r["idx"],
        }
        for r in ds
    ]
    write_jsonl(rows, out / "dev.jsonl")
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True, help="challenge dataset zip")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    train = extract_train(args.zip, args.out)
    dev = fetch_dev(args.out)

    pair = lambda r: (r["premise"], r["hypothesis"])  # noqa: E731
    leak = {pair(r) for r in train} & {pair(r) for r in dev}
    for name, rows in (("train", train), ("dev", dev)):
        print(f"{name}: {len(rows)} rows, labels={dict(Counter(r['label'] for r in rows))}")
    print(f"train/dev pair overlap: {len(leak)}")
    if leak:
        raise SystemExit("train/dev overlap detected")


if __name__ == "__main__":
    main()
