"""
Configuration dataclass for LHRM baseline.
Every value is tagged: PAPER / REPO / ASSUMED
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LHRMConfig:
    # === Dataset ===
    dataset: str = "aapd"               # REPO (choices: aapd, reuters)
    data_dir: str = "data/processed"    # REPO

    # === BERT ===
    bert_name: str = "bert-base-uncased"  # ASSUMED (paper says "BERT", hidden 768)
    max_len: int = 256                    # ASSUMED via A10 rule (P90 word-length > 128)
    hidden_size: int = 768                # PAPER

    # === CSAM Variant ===
    variant: str = "L"   # ASSUMED — selected by 1-epoch pilot per A6
    # Choices: L (literal Eq.4-8), L2 (scaled dot-product in Eq.4), R (repo-style fallback)
    separate_label_encoder: bool = False  # ASSUMED (shared weights per A2)

    # === Training ===
    epochs_aapd: int = 15              # PAPER
    epochs_reuters: int = 10           # PAPER
    batch_size: int = 8                # PAPER
    lr: float = 2e-5                   # PAPER
    weight_decay: float = 0.01         # PAPER
    seed: int = 42                     # REPO (paper doesn't specify)
    use_amp: bool = True               # ASSUMED (speed; record it)
    num_workers: int = 0               # REPO (Windows compatibility)

    # === Focal Loss ===
    loss: str = "focal"                # PAPER (assumed from text)
    focal_gamma: float = 2.0           # ASSUMED per A9 (paper doesn't report)
    focal_alpha: float = 0.25          # ASSUMED per A9 (paper doesn't report)
    # focal_alpha_t: positive=0.25, negative=0.75

    # === HRRM (Gradient Boosting) ===
    gb_n_estimators: int = 100         # PAPER
    gb_learning_rate: float = 0.1      # PAPER
    gb_max_depth: int = 3              # PAPER
    gb_subsample: float = 1.0          # PAPER
    gb_criterion: str = "friedman_mse" # PAPER
    gb_loss: str = "squared_error"     # PAPER
    gb_validation_fraction: float = 0.1  # PAPER (INERT: n_iter_no_change=None)
    gb_tol: float = 1e-4               # PAPER (INERT: n_iter_no_change=None)
    # Note: n_iter_no_change=None -> no early stopping (per A13)

    # === Evaluation ===
    eval_k_values: list = field(default_factory=lambda: [1, 3, 5])
    bootstrap_n: int = 1000            # REPO (95% CI)

    # === Paths ===
    experiment_id: str = "lhrm_baseline_aapd_L_focal_seed42_v1"
    out_dir: str = "experiments"       # REPO
    checkpoint_dir: str = "checkpoints"  # REPO

    def epochs(self) -> int:
        return self.epochs_aapd if self.dataset == "aapd" else self.epochs_reuters

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
