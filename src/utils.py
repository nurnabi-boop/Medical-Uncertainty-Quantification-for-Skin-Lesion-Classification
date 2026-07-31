"""Reproducibility, device detection, and small IO helpers.

Heavy imports (torch) are done at module top-level because every caller of this
module already needs torch. CLIs that must print --help without torch should
import :mod:`src.config` instead, which is dependency-light.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch; optionally force deterministic kernels.

    Determinism on CUDA is best-effort: a few ops have no deterministic kernel,
    so we set the env flag and ``use_deterministic_algorithms(warn_only=True)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding so augmentation/shuffle order is reproducible."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(prefer: str = "auto"):
    """Resolve a torch.device from a preference string ("auto"|"cpu"|"cuda")."""
    import torch

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_report(prefer: str = "auto") -> str:
    """Human-readable one-liner about the selected compute device."""
    import torch

    dev = get_device(prefer)
    if dev.type == "cuda":
        return f"device=cuda ({torch.cuda.get_device_name(0)})"
    return "device=cpu"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
