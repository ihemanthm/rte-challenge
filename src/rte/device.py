from __future__ import annotations

import torch


def pick_device(preferred: str | None = None) -> torch.device:
    """cuda -> mps -> cpu, unless a device is forced via config/CLI."""
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 autocast only on CUDA GPUs that support it; MPS/CPU stay fp32 for stability."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None
