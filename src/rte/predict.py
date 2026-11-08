"""Score a saved checkpoint on any jsonl file (dev now, hidden test later).

python -m rte.predict --checkpoint outputs/<run>/model --input data/dev.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rte import ID2LABEL
from rte.data import PairDataset, labels_of, load_jsonl, make_loader, write_jsonl
from rte.device import pick_device
from rte.train import predict_logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="run dir or its model/ subdir")
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="defaults to <run>/<input-stem>_predict_output.jsonl")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if (ckpt / "model").is_dir():
        ckpt = ckpt / "model"
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32).to(device)

    rows = load_jsonl(args.input)
    dl = make_loader(PairDataset(rows, tok, args.max_length), tok, args.batch_size, shuffle=False)
    probs = torch.softmax(torch.tensor(predict_logits(model, dl, device)), -1).numpy()

    # keep the input fields so the file also feeds rte.analysis
    out = [
        {**r, "pred": ID2LABEL[int(pr.argmax())], "p_entailment": float(pr[0])}
        for r, pr in zip(rows, probs, strict=True)
    ]
    # distinct name so it never clobbers the training run's dev_predictions.jsonl
    dest = (
        Path(args.output)
        if args.output
        else ckpt.parent / f"{Path(args.input).stem}_predict_output.jsonl"
    )
    write_jsonl(out, dest)
    print(f"wrote {len(out)} predictions -> {dest}")
    if "label" in rows[0]:
        print(f"accuracy {(probs.argmax(-1) == labels_of(rows)).mean():.4f}")


if __name__ == "__main__":
    main()
