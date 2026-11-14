"""Test-day inference: run every model in models/ on one input file, fully offline.

python scripts/predict_test.py --input test.jsonl
python scripts/predict_test.py --input test.csv --models models --out predictions --device cpu

Input: .jsonl, .json, .csv or .tsv with a premise and a hypothesis column (also accepted:
sentence1/sentence2, text_a/text_b). An id column (idx / id / index) is kept if present,
otherwise the row number is used. A label column is optional; if present, accuracy is printed.

Output, per model: <out>/<model>_predictions.csv (idx, prediction, label_id, p_entailment)
and .jsonl (input fields + prediction). label_id follows GLUE: 0 = entailment, 1 = not_entailment.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never wait on the network: models are local
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from rte import ID2LABEL, LABEL2ID  # noqa: E402
from rte.data import PairDataset, make_loader  # noqa: E402
from rte.device import pick_device  # noqa: E402
from rte.modeling import load_checkpoint  # noqa: E402
from rte.params import count_params  # noqa: E402
from rte.train import predict_logits  # noqa: E402

ALIASES = {
    "premise": ("premise", "sentence1", "text_a", "Premise", "Sentence1"),
    "hypothesis": ("hypothesis", "sentence2", "text_b", "Hypothesis", "Sentence2"),
    "idx": ("idx", "id", "index", "ID", "Id", "pair_id"),
    "label": ("label", "gold_label", "Label", "labels"),
}


def read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):  # {"data": [...]} or column-oriented {"premise": [...], ...}
            lists = [v for v in data.values() if isinstance(v, list)]
            if len(lists) == 1 and isinstance(lists[0][0], dict):
                data = lists[0]
            else:
                keys = [k for k, v in data.items() if isinstance(v, list)]
                data = [
                    dict(zip(keys, vals, strict=True))
                    for vals in zip(*(data[k] for k in keys), strict=True)
                ]
        raw = data
    elif suffix in (".csv", ".tsv", ".txt"):
        delim = "\t" if suffix == ".tsv" or text.count("\t") > text.count(",") else ","
        raw = list(csv.DictReader(text.splitlines(), delimiter=delim))
    else:
        sys.exit(f"unsupported input type {suffix}: use .jsonl, .json, .csv or .tsv")
    if not raw:
        sys.exit("input file has no rows")

    def col(name: str, required: bool) -> str | None:
        for key in ALIASES[name]:
            if key in raw[0]:
                return key
        if required:
            sys.exit(f"no {name} column; found columns {list(raw[0])}")
        return None

    p, h, i, y = (
        col("premise", True),
        col("hypothesis", True),
        col("idx", False),
        col("label", False),
    )
    rows = []
    for n, r in enumerate(raw):
        row = {"idx": r[i] if i else n, "premise": str(r[p]), "hypothesis": str(r[h])}
        if y and r.get(y) not in (None, "", -1, "-1"):
            lab = r[y]
            row["label"] = (
                ID2LABEL[int(lab)] if str(lab).strip().lstrip("-").isdigit() else str(lab).strip()
            )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="test file (.jsonl / .json / .csv / .tsv)")
    ap.add_argument("--models", default="models", help="folder with one sub-folder per model")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    rows = read_rows(Path(args.input))
    labelled = all("label" in r and r["label"] in LABEL2ID for r in rows)
    print(f"{len(rows)} pairs from {args.input}" + (" (labelled)" if labelled else ""))
    model_dirs = sorted(d for d in Path(args.models).iterdir() if (d / "config.json").exists())
    if not model_dirs:
        sys.exit(f"no models found in {args.models}/ (each needs config.json + model.safetensors)")
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for d in model_dirs:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(d)
        model = load_checkpoint(d).to(device).eval()
        dl = make_loader(PairDataset(rows, tok, 256), tok, args.batch_size, shuffle=False)
        probs = torch.softmax(torch.tensor(predict_logits(model, dl, device)), -1).numpy()
        preds = probs.argmax(-1)
        with open(out / f"{d.name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "prediction", "label_id", "p_entailment"])
            for r, k, pr in zip(rows, preds, probs, strict=True):
                w.writerow([r["idx"], ID2LABEL[int(k)], int(k), f"{pr[0]:.6f}"])
        with open(out / f"{d.name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for r, k, pr in zip(rows, preds, probs, strict=True):
                rec = {
                    **r,
                    "prediction": ID2LABEL[int(k)],
                    "label_id": int(k),
                    "p_entailment": float(pr[0]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        msg = f"{d.name}: {count_params(model)['total'] / 1e6:.1f}M params, {time.time() - t0:.0f}s"
        if labelled:
            acc = sum(
                ID2LABEL[int(k)] == r["label"] for k, r in zip(preds, rows, strict=True)
            ) / len(rows)
            msg += f", accuracy {acc:.4f}"
        print(msg + f" -> {out}/{d.name}_predictions.csv")
        del model


if __name__ == "__main__":
    main()
