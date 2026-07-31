"""medical-uq-selective-prediction: uncertainty quantification + selective prediction
for skin-lesion classification (HAM10000).

Package modules are built phase by phase:
  config       - dataclass config + YAML/CLI overrides + --smoke profile
  utils        - seeding, device detection, small IO helpers
  dataset      - lesion-grouped split, leakage assertion, dataloaders
  model        - ResNet34 with MC-dropout head
  train        - single-model training (seeded; reused for deep ensembles)
  calibration  - ECE, reliability diagram, temperature scaling
  uncertainty  - MC Dropout + Deep Ensemble (entropy / mutual information)
  conformal    - split-conformal APS from scratch
  selective    - risk-coverage curves + AURC
  visualize    - reliability diagrams, risk-coverage plots, set-size histograms
"""

__version__ = "0.1.0"
