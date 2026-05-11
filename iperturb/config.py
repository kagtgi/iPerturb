"""Configuration dataclasses for iPerturb pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GRNConfig:
    """Configuration for GRN construction."""

    gene_list_file: str = "gene_list.txt"
    cache_dir: str = "grn_cache"
    out_file: str = "grn_edges.tsv"

    # Edge selection
    greedy_reward: float = 0.15
    level_conf: dict = field(default_factory=lambda: {1: 1.0, 2: 0.60, 3: 0.35, 4: 0.20})

    # Parameter budget (Rule-of-10: N_genes × 3 gene-params + edge params ≤ N_obs/10)
    n_genes: int = 2_000
    n_observations: int = 280_000        # total cell-perturbation observations
    params_gene: int = 3                 # V_j, α_j, b_j
    params_edge_signed: int = 2          # K_d, n  (sign fixed from DB)
    params_edge_unsigned: int = 3        # K_d, n, s_ij (sign learned)

    @property
    def param_budget(self) -> int:
        return self.n_observations // 10

    @property
    def gene_param_total(self) -> int:
        return self.params_gene * self.n_genes

    @property
    def edge_param_budget(self) -> int:
        return self.param_budget - self.gene_param_total

    # STRING
    string_min_score: int = 700          # 400=permissive, 700=high, 900=very high

    # COXPRESdb
    coex_topn: int = 5                   # top-N co-expressed partners per gene
    coxpresdb_zip: str = ""              # path to Hsa_union_coex.zip
    skip_coxpresdb: bool = False

    # GeneHancer
    gh_gff_path: str = ""
    gh_tfbs_path: str = ""
    gh_tissue_path: str = ""
    tissue_filter: str = ""              # e.g. "K562" or "RPE1"


@dataclass
class TrainConfig:
    """Configuration for GRNN training."""

    n_epochs: int = 50
    lr: float = 1e-3
    train_frac: float = 0.70
    val_frac: float = 0.10
    seed: int = 42

    # Loss weights
    lam_wmse: float = 1.0
    lam_afda: float = 1.0
    lam_delta: float = 1.0
    lam_deg: float = 2.0
    lam_balance: float = 0.5
    gamma_focus: float = 2.0
    top_k_deg: int = 20

    # GRNN forward
    max_iter: int = 100
    eps: float = 1e-5

    device: str = "cpu"                  # overridden at runtime if CUDA available


@dataclass
class EvalConfig:
    """Configuration for GRNN evaluation."""

    n_eval_runs: int = 10
    subsample_frac: float = 0.30
    top_k_pearson: int = 20
    lfc_threshold: float = 0.1


@dataclass
class PipelineConfig:
    """Top-level configuration combining all sub-configs."""

    cell_line: str = "K562"             # "K562" or "RPE1"
    data_path: str = ""                 # path to .h5ad file
    n_top_hvgs: int = 2_000
    perturbation_col: str = "gene"
    control_label: str = "non-targeting"
    min_cells: int = 5
    plot_dir: str = "grn_plots"

    grn: GRNConfig = field(default_factory=GRNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
