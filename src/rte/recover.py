"""Rebuild a run's metrics.json from its printed training log (e.g. a lost Colab runtime).

Copy the output of the `rte.train` cell into a text file, then:

    python -m rte.recover --config configs/baseline_deberta_v3_base.yaml --log teacher_log.txt

Per-seed val/dev accuracy and per-epoch history come from the log. Parameter counts are
computed from the model config alone (no weights downloaded). Dev probabilities and the
checkpoint cannot be recovered, so the result is flagged `recovered_from_log`.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from rte.config import Config

_EPOCH = re.compile(r"seed (\d+) epoch\s+(\d+) loss ([\d.na]+) val_acc ([\d.]+)")
_SEED = re.compile(r"^seed (\d+): val ([\d.]+) dev ([\d.]+)", re.M)


def parse_log(text: str) -> list[dict]:
    """Seed records in log order; history keeps the last attempt if a seed was restarted."""
    history: dict[int, list[dict]] = {}
    for s, e, loss, val in _EPOCH.findall(text):
        h = history.setdefault(int(s), [])
        if int(e) == 1:
            h.clear()
        h.append({"epoch": int(e), "loss": float(loss), "val_acc": float(val)})
    seeds = {}
    for s, val, dev in _SEED.findall(text):
        h = history.get(int(s), [])
        best = max(h, key=lambda r: r["val_acc"]) if h else None
        seeds[int(s)] = {
            "seed": int(s),
            "best_val_acc": float(val),
            "best_epoch": best["epoch"] if best else None,
            "history": h,
            "dev_acc": float(dev),
        }
    return list(seeds.values())


def model_params(model_name: str) -> dict[str, int]:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from rte.params import count_params

    conf = AutoConfig.from_pretrained(model_name, num_labels=2)
    return count_params(AutoModelForSequenceClassification.from_config(conf))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log", required=True, help="text file with the rte.train cell output")
    p.add_argument("--run-name", dest="run_name")
    args = p.parse_args()
    cfg = Config.load(args.config, run_name=args.run_name)

    seeds = parse_log(open(args.log, encoding="utf-8").read())
    missing = sorted(set(cfg.seeds) - {s["seed"] for s in seeds})
    if not seeds:
        raise SystemExit("no 'seed N: val X dev Y' lines found in the log")
    if missing:
        print(f"warning: seeds {missing} are not in the log; metrics cover {len(seeds)} seeds")

    accs = np.array([s["dev_acc"] for s in seeds])
    best = max(seeds, key=lambda s: s["best_val_acc"])  # first max, as in rte.train
    params = model_params(cfg.model_name)
    metrics = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "distilled": cfg.distill is not None,
        "dev_acc_mean": float(accs.mean()),
        "dev_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "selected_seed": best["seed"],
        "selected_seed_dev_acc": best["dev_acc"],
        "params": params,
        "seeds": seeds,
        "recovered_from_log": True,
    }
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    print(
        f"{cfg.run_name}: dev {accs.mean():.4f} ± {metrics['dev_acc_std']:.4f} over "
        f"{len(accs)} seeds, params {params['total'] / 1e6:.1f}M -> {run_dir / 'metrics.json'}"
    )


if __name__ == "__main__":
    main()
