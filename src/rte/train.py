"""Config-driven fine-tuning, shared by the baseline, the plain student and the distilled student.

    python -m rte.train --config configs/student_deberta_v3_xsmall.yaml
    python -m rte.train --config configs/baseline_deberta_v3_base.yaml --oof   # teacher logits

Model selection never touches dev: each seed early-stops on a premise-grouped hold-out of train.
Dev is scored once per seed, and results are reported as mean ± std over seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from rte import ID2LABEL, LABEL2ID
from rte.config import Config, snapshot_environment
from rte.data import (
    PairDataset,
    grouped_holdout,
    grouped_kfold,
    labels_of,
    load_jsonl,
    make_loader,
    write_jsonl,
)
from rte.device import autocast_dtype, pick_device
from rte.params import count_params, fmt
from rte.seed import set_seed

_LAYER_RE = re.compile(r"\.layer\.(\d+)\.")
_HEAD_KEYS = ("classifier", "pooler")


def load_model(name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # transformers>=5 defaults to the checkpoint dtype; DeBERTa-v3 ships fp16 weights, and
        # AdamW on fp16 params underflows (eps, v) and diverges to NaN after one step.
        # Master weights stay fp32; mixed precision is applied via autocast on CUDA only.
        dtype=torch.float32,
    )


def param_groups(model, cfg: Config) -> list[dict]:
    """AdamW groups with optional layer-wise lr decay; no weight decay on biases/LayerNorm."""
    n_layers = model.config.num_hidden_layers
    head_lr = cfg.head_lr or cfg.lr
    groups: dict[tuple[float, float], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in _HEAD_KEYS):
            lr = head_lr
        else:
            m = _LAYER_RE.search(name)
            depth = (
                0
                if "embeddings." in name and m is None
                else (int(m.group(1)) + 1 if m else n_layers)
            )
            scale = cfg.layer_decay ** (n_layers + 1 - depth) if cfg.layer_decay else 1.0
            lr = cfg.lr * scale
        no_decay = p.ndim == 1 or "LayerNorm" in name or name.endswith(".bias")
        wd = 0.0 if no_decay else cfg.weight_decay
        groups.setdefault((lr, wd), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "row_id")}
        out.append(model(**inputs).logits.float().cpu())
    return torch.cat(out).numpy()


def kd_loss(student, teacher, labels, alpha: float, t: float) -> torch.Tensor:
    """Hinton et al. (2015): alpha * CE(hard) + (1 - alpha) * T^2 * KL(teacher_T || student_T)."""
    hard = F.cross_entropy(student, labels)
    soft = F.kl_div(
        F.log_softmax(student / t, dim=-1), F.softmax(teacher / t, dim=-1), reduction="batchmean"
    )
    return alpha * hard + (1 - alpha) * (t * t) * soft


def fit(cfg: Config, rows: list[dict], tokenizer, device, seed: int, teacher=None, log=print):
    """Train one model on `rows`, early-stopping on a grouped hold-out. Returns (model, info)."""
    set_seed(seed)
    fit_idx, val_idx = grouped_holdout(rows, cfg.val_fraction, seed)
    train_ds = PairDataset([rows[i] for i in fit_idx], tokenizer, cfg.max_length, row_ids=fit_idx)
    val_ds = PairDataset([rows[i] for i in val_idx], tokenizer, cfg.max_length, row_ids=val_idx)
    train_dl = make_loader(train_ds, tokenizer, cfg.batch_size, shuffle=True, seed=seed)
    val_dl = make_loader(val_ds, tokenizer, cfg.eval_batch_size, shuffle=False)
    val_y = labels_of([rows[i] for i in val_idx])

    model = load_model(cfg.model_name).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg))
    total_steps = cfg.epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)
    amp = autocast_dtype(device)
    teacher_t = (
        torch.tensor(teacher, dtype=torch.float32, device=device) if teacher is not None else None
    )

    best_acc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for batch in train_dl:
            row_ids = batch.pop("row_id")
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast(device.type, dtype=amp) if amp else nullcontext()
            with ctx:
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits.float()
            if teacher_t is not None:
                d = cfg.distill
                loss = kd_loss(
                    logits, teacher_t[row_ids.to(device)], batch["labels"], d.alpha, d.temperature
                )
            else:
                loss = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()

        val_acc = float((predict_logits(model, val_dl, device).argmax(-1) == val_y).mean())
        history.append({"epoch": epoch, "loss": running / len(train_dl), "val_acc": val_acc})
        log(
            f"  seed {seed} epoch {epoch:2d} loss {running / len(train_dl):.4f} "
            f"val_acc {val_acc:.4f} ({time.time() - t0:.0f}s)"
        )
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


def run(cfg: Config) -> dict:
    device = pick_device(cfg.device)
    run_dir = cfg.run_dir
    snapshot_environment(run_dir, cfg)
    print(f"run {cfg.run_name} | model {cfg.model_name} | device {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_rows, dev_rows = load_jsonl(cfg.train_file), load_jsonl(cfg.dev_file)
    dev_y = labels_of(dev_rows)
    dev_dl = make_loader(
        PairDataset(dev_rows, tokenizer, cfg.max_length),
        tokenizer,
        cfg.eval_batch_size,
        shuffle=False,
    )

    teacher = None
    if cfg.distill:
        teacher = np.load(cfg.distill.teacher_logits)
        assert teacher.shape == (len(train_rows), 2), f"teacher logits {teacher.shape} misaligned"

    seeds_info, dev_probs, best = [], [], None
    for seed in cfg.seeds:
        model, info = fit(cfg, train_rows, tokenizer, device, seed, teacher=teacher)
        logits = predict_logits(model, dev_dl, device)
        info["dev_acc"] = float((logits.argmax(-1) == dev_y).mean())
        print(f"seed {seed}: val {info['best_val_acc']:.4f} dev {info['dev_acc']:.4f}")
        seeds_info.append(info)
        dev_probs.append(torch.softmax(torch.tensor(logits), -1).numpy())
        # keep the seed with the best *validation* accuracy (dev is not used for selection)
        if best is None or info["best_val_acc"] > best[0]["best_val_acc"]:
            best = (info, model.cpu(), dev_probs[-1])  # previous best is released here
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    params = count_params(best[1])
    accs = np.array([s["dev_acc"] for s in seeds_info])
    metrics = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "distilled": cfg.distill is not None,
        "dev_acc_mean": float(accs.mean()),
        "dev_acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "selected_seed": best[0]["seed"],
        "selected_seed_dev_acc": best[0]["dev_acc"],
        "params": params,
        "seeds": seeds_info,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "params.json").write_text(json.dumps(params, indent=2))
    np.save(run_dir / "dev_probs_all_seeds.npy", np.stack(dev_probs))
    write_dev_predictions(dev_rows, best[2], run_dir / "dev_predictions.jsonl")
    if cfg.save_model:
        best[1].save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "model")

    print(
        f"\n{cfg.run_name}: dev acc {metrics['dev_acc_mean']:.4f} ± {metrics['dev_acc_std']:.4f} "
        f"over {len(accs)} seeds | params total {fmt(params['total'])}, "
        f"non-embedding {fmt(params['non_embedding'])}"
    )
    return metrics


def write_dev_predictions(rows: list[dict], probs: np.ndarray, path) -> None:
    out = []
    for r, p in zip(rows, probs, strict=True):
        out.append({**r, "pred": ID2LABEL[int(p.argmax())], "p_entailment": float(p[0])})
    write_jsonl(out, path)


def run_oof(cfg: Config) -> np.ndarray:
    """Out-of-fold teacher logits: each train row is scored by a model that never saw it."""
    device = pick_device(cfg.device)
    run_dir = cfg.run_dir.with_name(cfg.run_name + "_oof")
    snapshot_environment(run_dir, cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    rows = load_jsonl(cfg.train_file)
    oof = np.zeros((len(rows), 2), dtype=np.float32)
    seed = cfg.seeds[0]
    for k, (tr_idx, ho_idx) in enumerate(grouped_kfold(rows, cfg.oof_folds)):
        print(f"fold {k + 1}/{cfg.oof_folds}: train {len(tr_idx)} held-out {len(ho_idx)}")
        model, _ = fit(cfg, [rows[i] for i in tr_idx], tokenizer, device, seed + k)
        ho_dl = make_loader(
            PairDataset([rows[i] for i in ho_idx], tokenizer, cfg.max_length),
            tokenizer,
            cfg.eval_batch_size,
            shuffle=False,
        )
        oof[ho_idx] = predict_logits(model, ho_dl, device)
        del model
    acc = float((oof.argmax(-1) == labels_of(rows)).mean())
    np.save(run_dir / "teacher_logits_train.npy", oof)
    (run_dir / "oof_metrics.json").write_text(json.dumps({"oof_acc": acc, "seed": seed}, indent=2))
    print(f"OOF accuracy {acc:.4f} -> {run_dir / 'teacher_logits_train.npy'}")
    return oof


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--oof", action="store_true", help="export out-of-fold teacher logits instead")
    p.add_argument("--epochs", type=int, help="override (e.g. smoke tests)")
    p.add_argument("--seeds", type=int, nargs="+", help="override seed list")
    p.add_argument("--run-name", dest="run_name")
    args = p.parse_args()
    cfg = Config.load(args.config, epochs=args.epochs, seeds=args.seeds, run_name=args.run_name)
    if args.oof:
        run_oof(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
