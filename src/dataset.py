"""Dataset: lesion-grouped split, leakage assertion, and dataloaders (HAM10000).

Key guarantees:
  * The split is grouped by ``lesion_id`` — the same physical lesion (which may
    have several dermatoscopic images) never appears in two splits. This is
    asserted in :func:`assert_no_leakage` and fails loudly.
  * The split is seeded and stratified by per-lesion class so it is reproducible
    and keeps the (very imbalanced) class distribution across splits.
  * Class imbalance is handled either by class-weighted loss (weights returned
    for the trainer) or by a balanced ``WeightedRandomSampler`` on the train set.
  * ``--make-synthetic`` writes a tiny HAM10000-shaped dataset (metadata CSV +
    image folders, with multi-image lesions) so the whole pipeline and every
    acceptance check run offline, without the 2.6 GB Kaggle download.

CLI:
    python -m src.dataset --root data                 # split + leakage check + report
    python -m src.dataset --make-synthetic --root data/synthetic
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import Config, config_from_args, add_config_args

# ImageNet statistics (ResNet34 pretrained backbone expects these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ===========================================================================
# Synthetic data generator (offline pipeline / acceptance-check fallback)
# ===========================================================================
# Approximate HAM10000 class frequencies — heavily nv-dominated.
_SYNTH_CLASS_PROB = {
    "akiec": 0.03, "bcc": 0.05, "bkl": 0.11, "df": 0.012,
    "mel": 0.11, "nv": 0.67, "vasc": 0.018,
}
# Distinct base colors per class so a model can learn *something* on synthetic data.
_SYNTH_BASE_COLORS = [
    (180, 120, 110), (150, 90, 90), (200, 170, 130), (120, 80, 70),
    (90, 60, 70), (170, 140, 130), (200, 90, 100),
]
_SYNTH_LOCALIZATIONS = ["back", "lower extremity", "trunk", "upper extremity",
                        "abdomen", "face", "chest", "foot", "neck", "scalp"]


def _gen_synth_image(rng: np.random.Generator, cls_idx: int, px: int) -> np.ndarray:
    """Class-conditional RGB image: tinted background + off-color lesion blob + noise."""
    base = np.array(_SYNTH_BASE_COLORS[cls_idx % len(_SYNTH_BASE_COLORS)], dtype=np.float64)
    img = np.ones((px, px, 3), dtype=np.float64) * base
    yy, xx = np.mgrid[0:px, 0:px]
    cx = px / 2 + rng.normal(0, px * 0.06)
    cy = px / 2 + rng.normal(0, px * 0.06)
    radius = px * (0.18 + 0.12 * rng.random())
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
    lesion_color = np.clip(base * 0.55 + rng.normal(0, 18, 3), 0, 255)
    img[mask] = lesion_color
    img += rng.normal(0, 9, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_synthetic_dataset(cfg: Config, n_images: int, px: int = 96,
                           min_per_class: int = 6) -> str:
    """Write a tiny HAM10000-shaped dataset under ``cfg.data_root``.

    Produces ``HAM10000_metadata.csv`` plus the two image folders, with some
    lesions having 2–3 images (so grouping/leakage logic is genuinely exercised).
    Returns the metadata CSV path.
    """
    from PIL import Image

    root = cfg.data_root
    os.makedirs(root, exist_ok=True)
    img_dirs = [os.path.join(root, d) for d in cfg.image_dirs]
    for d in img_dirs:
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    classes = list(cfg.classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Per-class image counts: a floor for representation, remainder by frequency.
    counts = {c: min_per_class for c in classes}
    remaining = max(0, n_images - sum(counts.values()))
    if remaining > 0:
        probs = np.array([_SYNTH_CLASS_PROB[c] for c in classes], dtype=float)
        probs /= probs.sum()
        for c in rng.choice(classes, size=remaining, p=probs):
            counts[c] += 1

    rows: List[dict] = []
    lesion_n = 0
    image_n = 0
    for c in classes:
        n_c = counts[c]
        idx = 0
        while idx < n_c:
            size = min(int(rng.integers(1, 4)), n_c - idx)  # 1..3 images / lesion
            lesion_n += 1
            lesion_id = f"SYN_LES_{lesion_n:05d}"
            for _ in range(size):
                image_id = f"SYN_IMG_{image_n:06d}"
                arr = _gen_synth_image(rng, class_to_idx[c], px)
                dst_dir = img_dirs[image_n % len(img_dirs)]
                Image.fromarray(arr).save(os.path.join(dst_dir, image_id + ".jpg"), quality=90)
                rows.append({
                    "lesion_id": lesion_id,
                    "image_id": image_id,
                    "dx": c,
                    "dx_type": "synthetic",
                    "age": float(rng.integers(25, 85)),
                    "sex": rng.choice(["male", "female"]),
                    "localization": rng.choice(_SYNTH_LOCALIZATIONS),
                })
                image_n += 1
            idx += size

    df = pd.DataFrame(rows)
    csv_path = os.path.join(root, cfg.metadata_csv)
    df.to_csv(csv_path, index=False)
    print(f"[synthetic] wrote {len(df)} images across {lesion_n} lesions -> {csv_path}")
    return csv_path


# ===========================================================================
# Metadata loading
# ===========================================================================
def load_metadata(cfg: Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Read the metadata CSV, resolve each image_id to a file path, map dx->label."""
    csv_path = os.path.join(cfg.data_root, cfg.metadata_csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}\n"
            f"  - Download HAM10000:  kaggle datasets download -d "
            f"kmader/skin-cancer-mnist-ham10000 -p {cfg.data_root} --unzip\n"
            f"  - Or generate a synthetic dataset:  python -m src.dataset "
            f"--make-synthetic --root {cfg.data_root}"
        )
    df = pd.read_csv(csv_path)
    for col in (cfg.id_col, cfg.group_col, cfg.label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from {csv_path}. Found: {list(df.columns)}")

    # Build image_id -> path by scanning the configured image folders.
    exts = (".jpg", ".jpeg", ".png")
    id_to_path: Dict[str, str] = {}
    for d in cfg.image_dirs:
        folder = os.path.join(cfg.data_root, d)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                id_to_path[stem] = os.path.join(folder, fn)
    if not id_to_path:
        raise FileNotFoundError(
            f"No image files found under {cfg.data_root} in {cfg.image_dirs}."
        )

    df["path"] = df[cfg.id_col].map(id_to_path)
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        warnings.warn(f"{n_missing} image_ids in the CSV have no matching file; dropping them.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    unknown = sorted(set(df[cfg.label_col]) - set(cfg.classes))
    if unknown:
        raise ValueError(f"Unexpected dx labels not in config.classes: {unknown}")
    class_to_idx = {c: i for i, c in enumerate(cfg.classes)}
    df["label"] = df[cfg.label_col].map(class_to_idx).astype(int)
    return df, class_to_idx


# ===========================================================================
# Lesion-grouped seeded split + leakage assertion
# ===========================================================================
def lesion_grouped_split(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign each row a split in {train,val,test}, grouping by lesion_id.

    Lesions (not images) are split, stratified by the lesion's class where the
    per-class lesion counts allow it.
    """
    from sklearn.model_selection import train_test_split

    gc, lc = cfg.group_col, cfg.label_col
    # One label per lesion (HAM10000 lesions are single-dx; warn if not).
    nunique = df.groupby(gc)[lc].nunique()
    if (nunique > 1).any():
        warnings.warn(f"{int((nunique > 1).sum())} lesion(s) have >1 dx; using the majority label.")
    lesion_label = df.groupby(gc)[lc].agg(lambda s: s.mode().iat[0])
    lesions = lesion_label.index.to_numpy()
    labels = lesion_label.to_numpy()

    def _strat(y: np.ndarray, frac: float):
        """Return ``y`` if a stratified split is feasible, else None (random split).

        sklearn requires every class to have >=2 members AND each side of the
        split (sklearn rounds the test side up) to hold at least one per class,
        i.e. ``min(n_test, n_train) >= n_classes``.
        """
        klasses, counts = np.unique(y, return_counts=True)
        n = len(y)
        n_test = int(np.ceil(n * frac))
        n_train = n - n_test
        if counts.min() < 2 or min(n_test, n_train) < len(klasses):
            return None
        return y

    # Split off test, then carve val out of the remainder.
    les_tv, les_test, y_tv, _ = train_test_split(
        lesions, labels, test_size=cfg.test_frac, random_state=cfg.seed,
        stratify=_strat(labels, cfg.test_frac))

    val_rel = cfg.val_frac / (1.0 - cfg.test_frac)
    les_train, les_val = train_test_split(
        les_tv, test_size=val_rel, random_state=cfg.seed, stratify=_strat(y_tv, val_rel))

    split_of = {l: "train" for l in les_train}
    split_of.update({l: "val" for l in les_val})
    split_of.update({l: "test" for l in les_test})
    df = df.copy()
    df["split"] = df[gc].map(split_of)
    if df["split"].isna().any():
        raise RuntimeError("Some rows were not assigned a split (lesion mapping gap).")
    return df


def assert_no_leakage(df: pd.DataFrame, group_col: str = "lesion_id") -> None:
    """Fail loudly if any lesion_id appears in more than one split."""
    groups = {s: set(df.loc[df["split"] == s, group_col]) for s in ("train", "val", "test")}
    problems = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = groups[a] & groups[b]
        if inter:
            sample = list(sorted(inter))[:5]
            problems.append(f"{len(inter)} lesion(s) shared by {a}/{b} (e.g. {sample})")
    if problems:
        raise AssertionError("LESION LEAKAGE DETECTED:\n  " + "\n  ".join(problems))
    print(f"LEAKAGE CHECK PASSED - no {group_col} appears in more than one split "
          f"({len(groups['train'])}/{len(groups['val'])}/{len(groups['test'])} lesions "
          f"train/val/test, all disjoint).")


# ===========================================================================
# Imbalance handling
# ===========================================================================
def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> "np.ndarray":
    """Inverse-frequency class weights (normalized to mean 1) for weighted CE."""
    counts = np.zeros(num_classes, dtype=np.float64)
    vc = train_df["label"].value_counts()
    for k, v in vc.items():
        counts[int(k)] = v
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)
    return (w / w.mean()).astype(np.float32)


def make_weighted_sampler(train_df: pd.DataFrame, num_classes: int):
    """A WeightedRandomSampler that draws classes ~uniformly each epoch."""
    import torch

    counts = np.zeros(num_classes, dtype=np.float64)
    for k, v in train_df["label"].value_counts().items():
        counts[int(k)] = v
    counts = np.clip(counts, 1.0, None)
    per_class_w = 1.0 / counts
    sample_w = per_class_w[train_df["label"].to_numpy()]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(train_df), replacement=True)


# ===========================================================================
# Transforms + Dataset + DataLoaders
# ===========================================================================
def build_transforms(cfg: Config):
    """Albumentations train/eval transforms (resize 224, ImageNet norm, light aug)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_ = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return {"train": train, "val": eval_, "test": eval_}


class LesionDataset:
    """Torch Dataset over a split DataFrame; returns (CHW float tensor, int label)."""

    def __init__(self, df: pd.DataFrame, transform, label_col: str = "label"):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        img = np.array(Image.open(self.paths[i]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, self.labels[i]


def subsample_smoke(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep ~cfg.smoke_n_images images while preserving lesion grouping + class coverage."""
    gc, lc = cfg.group_col, cfg.label_col
    rng = np.random.default_rng(cfg.seed)
    lesion_label = df.groupby(gc)[lc].first()
    sizes = df.groupby(gc).size()
    lesions = list(lesion_label.index)
    rng.shuffle(lesions)

    chosen: set = set()
    # Guarantee >=2 lesions per present class so all three splits can see it.
    for c in df[lc].unique():
        cls_lesions = [l for l in lesions if lesion_label[l] == c]
        chosen.update(cls_lesions[:2])
    count = int(sum(sizes[l] for l in chosen))
    for l in lesions:
        if count >= cfg.smoke_n_images:
            break
        if l not in chosen:
            chosen.add(l)
            count += int(sizes[l])
    return df[df[gc].isin(chosen)].reset_index(drop=True)


def build_dataloaders(cfg: Config) -> Dict:
    """End-to-end: load -> (smoke subsample) -> split -> assert -> datasets/loaders."""
    import torch
    from .utils import set_seed, seed_worker

    set_seed(cfg.seed, cfg.deterministic)
    df, class_to_idx = load_metadata(cfg)
    if cfg.smoke:
        df = subsample_smoke(df, cfg)
    df = lesion_grouped_split(df, cfg)
    assert_no_leakage(df, cfg.group_col)

    tfms = build_transforms(cfg)
    datasets = {s: LesionDataset(df[df["split"] == s], tfms[s]) for s in ("train", "val", "test")}

    train_df = df[df["split"] == "train"]
    class_weights = compute_class_weights(train_df, cfg.num_classes)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=(cfg.pin_memory and torch.cuda.is_available()),
                  worker_init_fn=seed_worker, generator=g)

    if cfg.imbalance == "weighted_sampler":
        sampler = make_weighted_sampler(train_df, cfg.num_classes)
        train_loader = torch.utils.data.DataLoader(datasets["train"], sampler=sampler, **common)
    else:
        train_loader = torch.utils.data.DataLoader(datasets["train"], shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(datasets["val"], shuffle=False, **common)
    test_loader = torch.utils.data.DataLoader(datasets["test"], shuffle=False, **common)

    return {
        "df": df,
        "datasets": datasets,
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "class_weights": class_weights,
        "class_to_idx": class_to_idx,
        "classes": list(cfg.classes),
    }


# ===========================================================================
# Reporting / CLI
# ===========================================================================
def _print_report(df: pd.DataFrame, cfg: Config) -> None:
    gc = cfg.group_col
    splits = ("train", "val", "test")
    n_img = {s: int((df["split"] == s).sum()) for s in splits}
    n_les = {s: df.loc[df["split"] == s, gc].nunique() for s in splits}

    print(f"\n=== HAM10000 lesion-grouped split (seed={cfg.seed}) ===")
    print(f"Total: {len(df)} images / {df[gc].nunique()} lesions")
    print(f"Images  -> train={n_img['train']}  val={n_img['val']}  test={n_img['test']}")
    print(f"Lesions -> train={n_les['train']}  val={n_les['val']}  test={n_les['test']}")

    ct = pd.crosstab(df[cfg.label_col], df["split"]).reindex(index=cfg.classes, columns=splits, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    print("\nPer-class counts (images):")
    header = f"{'class':<7}" + "".join(f"{s:>7}" for s in splits) + f"{'total':>7}"
    print(header)
    print("-" * len(header))
    for c in cfg.classes:
        row = ct.loc[c]
        print(f"{c:<7}" + "".join(f"{int(row[s]):>7}" for s in splits) + f"{int(row['total']):>7}")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset",
        description="HAM10000 lesion-grouped split + leakage check + dataloaders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(parser)
    parser.add_argument("--make-synthetic", action="store_true",
                        help="Generate a tiny synthetic HAM10000-like dataset under <root> "
                             "(for testing the pipeline without the 2.6GB download).")
    parser.add_argument("--synthetic-n", type=int, default=300,
                        help="Number of synthetic images to generate with --make-synthetic.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.make_synthetic:
        make_synthetic_dataset(cfg, n_images=args.synthetic_n)

    df, _ = load_metadata(cfg)
    if cfg.smoke:
        df = subsample_smoke(df, cfg)
        print(f"[smoke] subsampled to {len(df)} images / {df[cfg.group_col].nunique()} lesions")
    df = lesion_grouped_split(df, cfg)
    _print_report(df, cfg)
    assert_no_leakage(df, cfg.group_col)

    # Sanity: build one real batch through the albumentations pipeline (workers=0).
    try:
        import torch
        tfms = build_transforms(cfg)
        ds = LesionDataset(df[df["split"] == "train"], tfms["train"])
        loader = torch.utils.data.DataLoader(ds, batch_size=min(8, len(ds)), num_workers=0)
        xb, yb = next(iter(loader))
        print(f"Sanity batch: images={tuple(xb.shape)} dtype={xb.dtype} "
              f"range=[{xb.min():.2f},{xb.max():.2f}] | labels={tuple(yb.shape)}")
    except Exception as e:  # pragma: no cover - sanity only
        print(f"[warn] sanity batch failed: {e}")

    # Persist the split for full reproducibility / later phases.
    from .utils import ensure_dir
    ensure_dir(cfg.results_dir)
    out = os.path.join(cfg.results_dir, "splits.csv")
    df[[cfg.id_col, cfg.group_col, cfg.label_col, "label", "split"]].to_csv(out, index=False)
    print(f"Saved split assignments -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
