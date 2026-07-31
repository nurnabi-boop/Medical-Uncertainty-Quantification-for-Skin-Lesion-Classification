# medical-uq-selective-prediction

Train a skin-lesion classifier, then make it **know when it doesn't know.** This
repo compares three ways to quantify predictive uncertainty — **MC Dropout**,
**Deep Ensembles**, and **Conformal Prediction** — and shows that *deferring the
most uncertain cases to a clinician* raises accuracy on the cases that are kept.

The contribution is the **uncertainty evaluation** (calibration + risk–coverage +
conformal coverage), not raw classification accuracy.

> Dataset: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000),
> 7-class dermatoscopic lesions. Swappable behind `src/dataset.py`.

## Principles

- **Reproducible** — global seed, deterministic kernels where possible, config-driven.
- **Honest splits** — grouped by `lesion_id` so the same lesion never appears in two
  splits; leakage is asserted and fails loudly.
- **Phased** — every phase ends in a runnable command + an acceptance check.
- **`--smoke` everywhere** — ~50 images, 1 epoch, ensemble of 2, 5 MC samples, so the
  whole pipeline runs end-to-end in a few minutes.

## Install

```bash
pip install -r requirements.txt
```

Python 3.14 is supported (uses `cp314` wheels for `torch==2.12.0` /
`torchvision==0.27.0`). CPU/CUDA is auto-detected.

## Data

```bash
# Kaggle CLI (needs ~/.kaggle/kaggle.json credentials)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data --unzip
```

This yields `data/HAM10000_metadata.csv` and two image folders
(`HAM10000_images_part_1`, `HAM10000_images_part_2`), joined by `image_id`.

No Kaggle access? Phase 1 ships a synthetic-data generator that mimics the real
layout so the entire pipeline and every acceptance check can run offline:

```bash
python -m src.dataset --make-synthetic --root data/synthetic   # [Phase 1]
```

## Repo layout

```
medical-uq-selective-prediction/
├── data/                 # dataset (gitignored)
├── src/
│   ├── config.py         # dataclass config + YAML/CLI overrides + --smoke profile
│   ├── utils.py          # seeding, device detection, IO helpers
│   ├── dataset.py        # lesion-grouped split + leakage assert + loaders
│   ├── model.py          # ResNet34 with a dropout head (for MC dropout)
│   ├── train.py          # trains ONE model; takes a seed (used for ensembles)
│   ├── calibration.py    # ECE, reliability diagram, temperature scaling
│   ├── uncertainty.py    # MC Dropout + Deep Ensemble -> probs + uncertainty scores
│   ├── conformal.py      # split-conformal APS from scratch
│   ├── selective.py      # risk-coverage curves + AURC
│   └── visualize.py      # reliability diagrams, risk-coverage plots, set-size hist
├── results/              # metrics.csv, metrics.json, plots/
├── models/               # checkpoints (one per ensemble seed)
├── app.py                # Gradio: image -> prediction + uncertainty + conformal set
├── README.md
└── requirements.txt
```

## Build phases & acceptance checks

| Phase | What | Acceptance check |
|------:|------|------------------|
| 0 | Scaffold | `pip install` succeeds; `python -m src.dataset --help` runs |
| 1 | Data | `python -m src.dataset --root data` prints split sizes, per-class counts, `LEAKAGE CHECK PASSED` |
| 2 | Base classifier | `--smoke` trains cleanly; real run reports test macro-F1, per-class recall, multiclass AUROC |
| 3 | Calibration | prints ECE before/after temperature scaling; saves reliability diagram |
| 4 | Uncertainty | per-image entropy + mutual information for MC Dropout & Deep Ensemble |
| 5 | Conformal | split-conformal APS; empirical coverage ≈ 1−α |
| 6 | Selective prediction | risk–coverage curves + AURC; deferring raises kept-set accuracy |
| 7 | Gradio app | upload image → prediction + uncertainty + conformal set |

**Status: Phase 0 complete.**
