"""Materialise data/train.jsonl and data/dev.jsonl (GLUE RTE validation).

train.jsonl comes from the challenge zip (`--zip`) or, with no zip, is rebuilt from the public
GLUE RTE train split: the challenge file is exactly GLUE train with each text field read through
a TSV/CSV parser (which strips surrounding quotes and un-doubles inner ones). Either way the result
is checked against the fingerprint of the official challenge file, so both paths give identical
data and row order (the teacher logits are row-aligned to it).

Both files share one schema: {"premise", "hypothesis", "label", "idx"} with string labels
"entailment" / "not_entailment". The dev split is used for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

GLUE_LABELS = {0: "entailment", 1: "not_entailment"}  # GLUE RTE label ids
# sha256 of json.dumps(rows, sort_keys=True, ensure_ascii=False) for the official train.jsonl
TRAIN_FINGERPRINT = "dac4a2e0e5134aa8442701a2332d72112eb40489e68ce41ce48231754a9ea6d6"


def fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _tsv_field(s: str) -> str:
    return next(csv.reader(io.StringIO(s), delimiter="\t"))[0] if s else s


def fetch_train(out: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "rte", split="train")
    rows = [
        {
            "premise": _tsv_field(r["sentence1"]),
            "hypothesis": _tsv_field(r["sentence2"]),
            "label": GLUE_LABELS[r["label"]],
            "idx": r["idx"],
        }
        for r in ds
    ]
    write_jsonl(rows, out / "train.jsonl")
    return rows


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
    p.add_argument("--zip", type=Path, help="challenge dataset zip; omit to rebuild from GLUE")
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    train = extract_train(args.zip, args.out) if args.zip else fetch_train(args.out)
    if fingerprint(train) != TRAIN_FINGERPRINT:
        raise SystemExit("train.jsonl does not match the official challenge file fingerprint")
    print("train fingerprint matches the official challenge file")
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
