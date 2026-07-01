"""
Centralized configuration for the perturbation benchmarking toolkit.

All configurable parameters live here. Users should modify this file
(or override at runtime) to adapt the evaluation to their dataset.
"""

import torch

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATA_PATH: str = "K562.h5ad"  # Path to the .h5ad Perturb-seq file

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SUBSAMPLE_FRAC: float = 0.20       # Fraction of cells kept per perturbation
MIN_CELLS_PER_PERT: int = 5        # Minimum cells after subsampling to evaluate
MAX_T3_CELLS: int = 512            # Max cells per pert for Tier-3 distance matrices

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CTRL_LABEL: str = "non-targeting"   # obs value identifying control cells
PERT_COL: str = "gene"              # obs column holding perturbation labels
TOP_K_DE: int = 50                  # Number of top DE genes for Tier-2 metrics
DIR_ACC_THRESHOLD: float = 0.1      # Minimum |delta| to count for directional accuracy

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "results"         # Directory for saving evaluation results

# ---------------------------------------------------------------------------
# Train/test split (eval/splits.py) — Level 6 generalizability regimes
# ---------------------------------------------------------------------------
SPLIT_MODE: str = "none"            # none | random | simulation | unseen_pert
                                    #      | unseen_context | unseen_both
                                    # "none" reproduces the legacy all-perts behavior.
TEST_FRAC: float = 0.20             # fraction of perts (or combos/contexts) held out as test
COMBO_SEP: str = "+"                # combinatorial perturbation separator (e.g. Norman "GENE1+GENE2")
CONTEXT_COL: str = "cell_type"      # obs column for unseen-context / unseen-both regimes

# ---------------------------------------------------------------------------
# Central scoring (eval/metrics.py) + canonical evaluation gene panel
# ---------------------------------------------------------------------------
CENTRAL_SCORING: bool = True        # runner recomputes metrics from each model's "scoring" payload
N_EVAL_GENES: int = 2000            # size of the canonical HVG panel
EVAL_PANEL: str = "hvg+de"          # "hvg" | "hvg+de"
PANEL_INCLUDE_DE: bool = True       # union per-pert top-DE genes into the panel
TOP_K_DE_SMALL: int = 20            # k for PearsonΔ20 / top-DEG metrics (Systema)

# ---------------------------------------------------------------------------
# Extended metric suite (7-level taxonomy) — heavy metrics are opt-in
# ---------------------------------------------------------------------------
ENABLE_KL_JS: bool = True           # L2 KL + Jensen-Shannon divergence
WASSERSTEIN_BACKEND: str = "sliced" # L2 Wasserstein: "sliced" | "none"
N_WASSERSTEIN_PROJ: int = 50        # random projections for sliced Wasserstein
ENABLE_T3_HEAVY: bool = True        # master switch for per-cell distributional metrics
ENABLE_ETEST: bool = True           # L2/L4 permutation E-test significance
N_ETEST_PERM: int = 100             # E-test permutations

# Level 5 — biological validity (eval/enrichment.py); offline GMTs under eval/resources/
ENABLE_ENRICHMENT: bool = False     # GSEA/GO/KEGG/Hallmark/Reactome (needs gseapy + GMTs)
ENRICHMENT_MAX_PERTS: int = 50      # cap perts for enrichment (cost control)

# Level 7 — GRN inference track (eval/grn.py)
ENABLE_GRN_TRACK: bool = False      # AUROC/AUPRC/EPR/recovery/SHD + signed/motif/hub
GRN_REFERENCE: str = "dorothea"     # bundled reference network name
GRN_TOPK_EDGES: int = 1000          # top edges considered for EPR/hub recovery

# Advanced biology & mechanism (eval/bio_advanced.py) — each opt-in
ENABLE_ESSENTIALITY: bool = False   # effect-magnitude vs DepMap essentiality
DEPMAP_TABLE: str = ""              # path to bundled essentiality CSV
ENABLE_TF_ACTIVITY: bool = False    # decoupler ULM/VIPER over DoRothEA/CollecTRI
TF_REGULONS: str = "collectri"      # collectri | dorothea
ENABLE_CELLTYPE_ID: bool = False    # cell-type identity preservation (needs CONTEXT_COL)
ENABLE_GI: bool = True              # genetic-interaction track (auto-active iff combos present)
ENABLE_VARIANCE_FIDELITY: bool = True  # per-gene cell-to-cell variance correlation
TF_REGULONS_FILE: str = ""          # CSV source,target (CollecTRI/DoRothEA) for TF activity

# Bundled offline resources (eval/resources/; populated by fetch_resources.py).
# The "full" biology profile (eval.profiles.apply_full_profile) flips the L5/L7
# ENABLE_* flags on and points these at the fetched files per cell line.
RESOURCE_DIR: str = "eval/resources"
GRN_REFERENCE_FILE: str = ""        # TSV regulator<TAB>target[<TAB>sign] (DoRothEA)
DEPMAP_K562_FILE: str = ""          # gene,score CSV (DepMap CRISPR gene effect, K562)
DEPMAP_RPE1_FILE: str = ""          # gene,score CSV (DepMap CRISPR gene effect, RPE1)
ENRICHMENT_GMT_DIR: str = ""        # dir of MSigDB .gmt files (license-restricted; manual fetch)

# ---------------------------------------------------------------------------
# Dataset registry (eval/datasets_registry.py)
# ---------------------------------------------------------------------------
DATASET: str = "custom"             # custom | norman | adamson | dixit
                                    #   | replogle_k562 | replogle_rpe1 | sciplex3
NORMAN_DIR: str = "./norman_data"   # download/cache dir for the Norman dataset
TRAIN_DATASET = None                # Level 6 cross-dataset: fit on this dataset ...
EVAL_DATASET = None                 # ... and score on this one (None = same as DATASET)

# Curated multi-dataset matrix (genetic + chemical, multiple cell lines).
# adamson/dixit load via GEARS PertData (same path as norman); replogle_* and
# sciplex3 download a standardized h5ad (scPerturb / Zenodo) — URLs overridable.
MODALITY: str = "genetic"           # genetic | chemical (set per-dataset by the registry)
DATASET_CACHE_DIR: str = "./data_cache"
DATASET_MAX_LOAD_CELLS: int = 200000   # cap cells read before subsampling (0 = off; bounds RAM on genome-wide Replogle)
REPLOGLE_K562_URL: str = "https://zenodo.org/record/7041849/files/ReplogleWeissman2022_K562_essential.h5ad?download=1"
REPLOGLE_RPE1_URL: str = "https://zenodo.org/record/7041849/files/ReplogleWeissman2022_rpe1.h5ad?download=1"
SCIPLEX3_URL: str = "https://zenodo.org/record/7041849/files/SrivatsanTrapnell2020_sciplex3.h5ad?download=1"

# ---------------------------------------------------------------------------
# Resource budget — single A100 40GB VRAM VM (models run sequentially)
# ---------------------------------------------------------------------------
MAX_CELLS_PER_PERT: int = 500       # absolute per-pert cell cap (sampling.py)
GENE_EMBEDDING_FILE: str = ""       # GenePert: gene->vec file (pickle/csv); "" → PCA fallback
GENEPERT_ALPHA: float = 1.0         # GenePert ridge regularization
SCGEN_EPOCHS: int = 100
BIOLORD_EPOCHS: int = 100
CHEMCPA_EPOCHS: int = 100            # chemCPA (chemical track) training epochs
IPERTURB_EPOCHS: int = 50           # iPerturb GRNN training epochs
IPERTURB_USE_FULL_GRN: bool = True  # build iPerturb's real L1–L4 GeneHancer GRN (else regulon)
IPERTURB_SUPPORT_DIR: str = ""      # GeneHancer v5.26 + support files; "" → eval/resources/iperturb
IPERTURB_TISSUE_FILTER: str = "K562"  # GeneHancer tissue filter (set per dataset cell line by the runner)
# iPerturb 2.0 engine (network propagation + parameter inference)
IPERTURB_ENGINE: str = "propagate"     # propagate (Tier-1, default) | hill (legacy GRNN) | deep (Tier-2)
IPERTURB_PROP_OP: str = "rwr"          # rwr (personalized PageRank) | heat (heat kernel)
IPERTURB_PROP_NORM: str = "sym"        # sym | rw  (adjacency normalization)
IPERTURB_PROP_SIGNED: bool = False     # use DB edge signs in diffusion (else sign-free magnitude)
IPERTURB_ALPHA: float = 0.2            # RWR restart / heat time (grid-searched if tuned)
IPERTURB_TUNE_ALPHA: bool = True       # grid-search alpha on the headline metric (inner train/val)
IPERTURB_READOUT: str = "lowrank"      # lowrank | diag  (influence -> delta readout)
IPERTURB_READOUT_RANK: int = 64        # low-rank readout rank
IPERTURB_RIDGE_LAMBDA: float = 1.0     # ridge regularization for readout + epistasis
IPERTURB_USE_EPISTASIS: bool = True    # analytic co-propagation epistasis for combos
RIDGE_ALPHA: float = 1.0
SCGEN_BATCH: int = 128
BIOLORD_BATCH: int = 128
CPA_BATCH: int = 128
SCGPT_EVAL_BATCH: int = 64
STATE_INFER_BATCH: int = 64
C2S_LOAD_4BIT: bool = True
C2S_MAX_GENES_PER_CELL: int = 200
C2S_BATCH: int = 8

# ---------------------------------------------------------------------------
# Leaderboard (eval/collect_results.py)
# ---------------------------------------------------------------------------
BASELINE_MODELS: tuple = ("nonctrl_mean", "matching_mean")
LEADERBOARD_REF: str = "nonctrl_mean"
