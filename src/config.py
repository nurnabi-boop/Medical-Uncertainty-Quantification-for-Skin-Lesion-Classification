"""Central configuration for medical-uq-selective-prediction.

Design goals:
  * One dataclass holds every knob so runs are fully described by a config.
  * Torch-free on import (only stdlib) so `python -m src.dataset --help` and other
    CLIs print help without importing heavy deps.
  * A single ``--smoke`` profile shrinks the whole pipeline (~50 images, 1 epoch,
    ensemble of 2, 5 MC samples) so the end-to-end run finishes in a few minutes.
  * YAML load/save for reproducible, shareable run specs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional

# HAM10000 diagnostic categories (the `dx` column). Sorted for a deterministic
# label mapping that does not depend on row order in the CSV.
HAM10000_CLASSES: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Human-readable names, used only for reporting / the Gradio app.
HAM10000_CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


@dataclass
class Config:
    """All hyperparameters and paths for a run.

    Anything that affects results lives here so a run is reproducible from the
    saved YAML alone.
    """

    # ---- reproducibility ----
    seed: int = 42
    deterministic: bool = True

    # ---- data ----
    data_root: str = "data"
    metadata_csv: str = "HAM10000_metadata.csv"
    # Image folders shipped by the Kaggle dataset; joined by image_id.
    image_dirs: List[str] = field(
        default_factory=lambda: ["HAM10000_images_part_1", "HAM10000_images_part_2"]
    )
    classes: List[str] = field(default_factory=lambda: list(HAM10000_CLASSES))
    num_classes: int = 7
    img_size: int = 224
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Lesion grouping column: the same physical lesion must never cross splits.
    group_col: str = "lesion_id"
    id_col: str = "image_id"
    label_col: str = "dx"

    # ---- dataloaders ----
    batch_size: int = 32
    num_workers: int = 2  # Windows-safe default; smoke sets this to 0
    pin_memory: bool = True
    # "weighted_loss" (class-weighted CE) or "weighted_sampler" (balanced sampling)
    imbalance: str = "weighted_loss"

    # ---- model ----
    arch: str = "resnet34"
    pretrained: bool = True
    dropout_p: float = 0.3

    # ---- optimization ----
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True  # mixed precision (ignored on CPU)
    label_smoothing: float = 0.0
    early_stop_patience: int = 8  # epochs without val macro-F1 improvement

    # ---- uncertainty ----
    mc_samples: int = 30  # T stochastic forward passes for MC Dropout
    ensemble_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # M=5

    # ---- conformal ----
    conformal_alpha: float = 0.10  # target miscoverage (=> 90% coverage)

    # ---- io ----
    models_dir: str = "models"
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # ---- smoke ----
    smoke: bool = False
    smoke_n_images: int = 50
    smoke_epochs: int = 1
    smoke_ensemble: int = 2
    smoke_mc: int = 5

    # ------------------------------------------------------------------ helpers
    def apply_smoke(self) -> "Config":
        """Shrink every expensive knob for a fast end-to-end pass."""
        self.smoke = True
        self.epochs = self.smoke_epochs
        self.mc_samples = self.smoke_mc
        self.ensemble_seeds = list(range(self.smoke_ensemble))
        self.num_workers = 0
        self.batch_size = min(self.batch_size, 16)
        self.early_stop_patience = max(self.early_stop_patience, self.epochs)
        return self

    def to_yaml(self, path: str) -> None:
        import yaml  # local import keeps module torch/3p-free at import time

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Register the common config-override flags shared by all CLIs."""
    g = parser.add_argument_group("config")
    g.add_argument("--config", type=str, default=None, help="Path to a YAML config to load as defaults.")
    g.add_argument("--root", "--data-root", dest="data_root", type=str, default=None,
                   help="Dataset root folder (contains metadata CSV + image dirs).")
    g.add_argument("--seed", type=int, default=None, help="Global random seed.")
    g.add_argument("--img-size", type=int, default=None, help="Square image size.")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--imbalance", choices=["weighted_loss", "weighted_sampler"], default=None,
                   help="Class-imbalance strategy.")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    g.add_argument("--smoke", action="store_true",
                   help="Fast end-to-end profile (~50 images, 1 epoch, ensemble of 2, 5 MC samples).")


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config from CLI args: YAML defaults < explicit flags < --smoke."""
    cfg = Config.from_yaml(args.config) if getattr(args, "config", None) else Config()

    # Apply any explicitly-provided override flags (None means "not provided").
    for name in ("data_root", "seed", "img_size", "batch_size", "num_workers",
                 "epochs", "lr", "imbalance", "device"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)

    if getattr(args, "smoke", False):
        cfg.apply_smoke()
    return cfg
