"""Assemble the three submitted models into models/<name>/ (weights + tokenizer + model_info.json).

All three are DeBERTa-v3-xsmall; they differ in training method and vocabulary size.
A model whose source checkpoint is missing is skipped with a message.

python scripts/package_models.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

MODELS = [
    {
        "name": "A_masked_kd_full_70.8M",
        "source": "outputs/distill_masked_deberta_v3_xsmall/model",
        "role": "accuracy pick: highest validation accuracy of all 20 xsmall runs",
        "method": "masked knowledge distillation (alpha 0.5, T 2), seed 21",
        "params": 70_830_722,
        "val_acc": 0.8266,
        "dev_acc": 0.7798,
    },
    {
        "name": "B_masked_kd_pruned_45.5M",
        "source": "outputs/distill_masked_deberta_v3_xsmall_pruned/model",
        "role": "efficiency pick: model A with a pruned vocabulary (-36% parameters)",
        "method": "model A + post-hoc vocabulary pruning (train tokens + top-60k frequent)",
        "params": 45_542_402,
        "val_acc": 0.7984,  # honest validation: vocabulary of the fit split only
        "dev_acc": 0.7834,
    },
    {
        "name": "C_plain_full_70.8M",
        "source": "outputs/student_deberta_v3_xsmall/model",
        "role": "hedge: trained without the teacher, so its errors differ from A's",
        "method": "plain fine-tuning on gold labels, seed 21",
        "params": 70_830_722,
        "val_acc": 0.8105,
        "dev_acc": 0.8087,
    },
]
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def main() -> None:
    out = Path("models")
    for m in MODELS:
        src = Path(m["source"])
        if not (src / "model.safetensors").exists():
            print(f"skip {m['name']}: {src} not found")
            continue
        seed = (src / "selected_seed.txt").read_text().strip()
        assert seed == "21", f"{src} holds seed {seed}, expected 21"
        dst = out / m["name"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            shutil.copy2(src / f, dst / f)
        info = {k: v for k, v in m.items() if k != "source"}
        info["architecture"] = "microsoft/deberta-v3-xsmall + 2-way classification head"
        (dst / "model_info.json").write_text(json.dumps(info, indent=2))
        print(f"{m['name']}: {sum((dst / f).stat().st_size for f in FILES) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
