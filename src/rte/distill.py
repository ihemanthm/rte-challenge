"""Knowledge distillation entry point: same training loop as rte.train, with a `distill:` block.

python -m rte.distill --config configs/distill_deberta_v3_xsmall.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rte.config import Config
from rte.train import run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--teacher-logits", dest="teacher_logits", help="override distill.teacher_logits"
    )
    p.add_argument("--epochs", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--run-name", dest="run_name", help="write to outputs/<run-name>")
    args = p.parse_args()

    cfg = Config.load(args.config, epochs=args.epochs, seeds=args.seeds, run_name=args.run_name)
    if cfg.distill is None:
        raise SystemExit("config has no `distill:` block")
    if args.teacher_logits:
        cfg.distill.teacher_logits = args.teacher_logits
    if not Path(cfg.distill.teacher_logits).exists():
        raise SystemExit(
            f"teacher logits not found: {cfg.distill.teacher_logits}\n"
            "Produce them on Colab with notebooks/colab_teacher.ipynb (rte.train --oof)."
        )
    run(cfg)


if __name__ == "__main__":
    main()
