"""Typed run configuration loaded from YAML, plus run-directory bookkeeping."""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

import yaml


@dataclass
class DistillConfig:
    teacher_logits: str  # .npy, shape (n_train, 2), row-aligned with train_file
    alpha: float = 0.5  # weight on hard-label CE; (1 - alpha) on KL to teacher
    temperature: float = 2.0
    mask_teacher_errors: bool = False  # CE only where the teacher is wrong


@dataclass
class Config:
    run_name: str
    model_name: str
    train_file: str = "data/train.jsonl"
    dev_file: str = "data/dev.jsonl"
    output_dir: str = "outputs"
    max_length: int = 256
    seeds: list[int] = field(default_factory=lambda: [13, 21, 42, 87, 100])
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 2e-5
    head_lr: float | None = None  # classifier head lr; defaults to lr
    layer_decay: float | None = None  # layer-wise lr decay, e.g. 0.9
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.1  # grouped hold-out from train for early stopping
    patience: int = 3  # early-stopping patience in epochs
    device: str = "auto"
    save_model: bool = True  # save the best-validation seed's checkpoint
    oof_folds: int = 5  # used by `rte.train --oof` to export teacher logits
    embedding_rank: int | None = None  # low-rank (SVD) word embedding, see rte.modeling
    distill: DistillConfig | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @classmethod
    def load(cls, path: str | Path, **overrides) -> Config:
        raw = yaml.safe_load(Path(path).read_text())
        raw.update({k: v for k, v in overrides.items() if v is not None})
        if raw.get("distill"):
            raw["distill"] = DistillConfig(**raw["distill"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def mirror(run_dir: Path) -> None:
    """Copy new/changed files of run_dir to $RTE_MIRROR_DIR/<run> (e.g. Google Drive).

    Training writes to fast local disk; the mirror keeps finished seeds/folds safe if the
    runtime is lost. Mirror failures (flaky network mounts) warn but never stop training.
    """
    root = os.environ.get("RTE_MIRROR_DIR")
    if not root:
        return
    dest_root = Path(root) / run_dir.name
    for attempt in range(1, 4):
        try:
            for src in sorted(run_dir.rglob("*")):
                if not src.is_file():
                    continue
                dest = dest_root / src.relative_to(run_dir)
                st = src.stat()
                if dest.exists():
                    dt = dest.stat()
                    if dt.st_size == st.st_size and int(dt.st_mtime) == int(st.st_mtime):
                        continue
                os.makedirs(dest.parent, exist_ok=True)
                shutil.copy2(src, dest)
            return
        except OSError as e:
            print(f"warning: mirror to {dest_root} failed (attempt {attempt}/3): {e}")
            time.sleep(5 * attempt)
    print(f"warning: {dest_root} may be incomplete; local {run_dir} is intact")


def snapshot_environment(run_dir: Path, cfg: Config) -> None:
    """Persist resolved config + exact package versions so a run can be reproduced."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    env = {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs}
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))
