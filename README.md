# iPerturb

**Gene Regulatory Neural Network (GRNN) for CRISPRi Perturb-seq perturbation prediction.**

iPerturb builds a multi-evidence gene regulatory network (GRN), fits a biologically-constrained
neural ODE to CRISPRi knockdown data, and predicts genome-wide transcriptional responses to
unseen perturbations.

---

## Overview

```
Perturb-seq .h5ad
      │
      ▼ HVG selection (2 000 genes)
      │
      ▼ GRN construction  ──── TRRUST · OmniPath · CollecTRI  (L1, curated)
      │                   ──── GeneHancer TFBS               (L2, enhancer)
      │                   ──── STRING PPI + transitive close  (L3, PPI)
      │                   ──── COXPRESdb co-expression        (L4, coex)
      │
      ▼ Parameter-budget greedy selection  (Rule-of-10: ≤ N_obs / 10 params)
      │
      ▼ GRNN training  (dual-channel Hill kinetics + Picard iteration)
      │   Loss: WMSE + AFDA + Balance + ΔMse + DEG-MSE
      │
      ▼ Evaluation (Pearson Δ, centroid accuracy, directional accuracy, MSE)
      │
      ▼ Publication-quality figures + Cytoscape.js interactive GRN
```

---

## Installation

```bash
pip install -e .
# or: pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

---

## Quick Start

### 1. Build the GRN

```bash
python scripts/build_grn.py \
    --gene-list gene_list.txt \
    --cache-dir grn_cache \
    --out grn_edges.tsv \
    --cell-line K562 \
    --gh-gff GeneHancer_v5.26.gff \
    --gh-tfbs GeneHancer_TFBSs_v5.26.txt \
    --gh-tissue GeneHancer_Tissues_v5.26.txt \
    --coxpresdb-zip Hsa_union_coex.zip
```

### 2. Train the GRNN

```bash
python scripts/train_grnn.py \
    --h5ad K562.h5ad \
    --grn grn_edges.tsv \
    --cell-line K562 \
    --n-epochs 50 \
    --device cuda \
    --viz-genes GATA1 KLF1 MYC
```

### 3. Full pipeline (GRN + training in one step)

```bash
python scripts/run_pipeline.py \
    --h5ad K562.h5ad \
    --cell-line K562 \
    --gh-gff GeneHancer_v5.26.gff \
    --gh-tfbs GeneHancer_TFBSs_v5.26.txt \
    --gh-tissue GeneHancer_Tissues_v5.26.txt \
    --n-epochs 50 \
    --device cuda
```

### 4. Python API

```python
import scanpy as sc
from iperturb.data import load_dataset, split_dataset
from iperturb.grn  import build_grn
from iperturb.model import grn_tsv_to_grnn
from iperturb.train import train_grnn, evaluate_all

# Load data
dataset, gene_names = load_dataset("K562.h5ad", gene_names=[])
train_ds, val_ds, test_ds = split_dataset(dataset)

# Build GRN
selected, _ = build_grn(set(gene_names), cache_dir="grn_cache")
selected.to_csv("grn_edges.tsv", sep="\t", index=False)

# Build + train model
model, grn_df = grn_tsv_to_grnn("grn_edges.tsv", gene_names, dataset.x0.numpy())
train_grnn(model, train_ds, val_ds, n_epochs=50)

# Evaluate
metrics = evaluate_all(model, test_ds)
print(metrics)
```

---

## Data

| Dataset | Cell line | Source |
|---------|-----------|--------|
| K562.h5ad | K562 (CML) | Replogle et al. 2022 (figshare) |
| RPE1.h5ad | hTERT-RPE1 | Replogle et al. 2022 (figshare) |

Optional external files (improve GRN quality):

| File | Source |
|------|--------|
| `GeneHancer_v5.26.gff` | GeneHancer v5.26 (requires licence) |
| `GeneHancer_TFBSs_v5.26.txt` | GeneHancer v5.26 |
| `GeneHancer_Tissues_v5.26.txt` | GeneHancer v5.26 |
| `Hsa_union_coex.zip` | COXPRESdb v8 (free, coxpresdb.jp) |

---

## Package Structure

```
iPerturb/
├── iperturb/
│   ├── __init__.py
│   ├── config.py         # Configuration dataclasses
│   ├── utils.py          # logmean, inv_softplus, HTTP session
│   ├── grn/
│   │   ├── __init__.py   # build_grn() entry point
│   │   ├── sources.py    # TRRUST · OmniPath · GeneHancer · STRING · COXPRESdb
│   │   ├── merge.py      # merge_edges()
│   │   └── select.py     # greedy_select_connected()
│   ├── model.py          # GRNN (dual-channel Hill + Picard iteration)
│   ├── data.py           # PerturbseqDataset, split_dataset, load_dataset
│   ├── train.py          # grnn_loss, train_grnn, evaluate_all, subsample_evaluate
│   └── visualize.py      # Heatmaps, loss curves, Cytoscape.js, GraphML export
├── scripts/
│   ├── build_grn.py      # CLI: GRN construction
│   ├── train_grnn.py     # CLI: GRNN training + evaluation
│   └── run_pipeline.py   # CLI: full end-to-end pipeline
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## GRNN Model

The GRNN models steady-state gene expression under CRISPRi knockdown using a
dual-channel thermodynamic update:

```
drive_ij = ŵ_ij · φ(x_i)           — signed regulatory drive
h_act_j  = Σ max(0,  drive_ij)      — total activation input
h_rep_j  = Σ max(0, −drive_ij)      — total repression input

x_ss_j = (V_j · (1 + h_act_j) + b_j) / (1 + α_j + h_rep_j)
```

Weight parameterisation enforces biological priors from the GRN database:
- **sign ∈ {+1, −1}**: `ŵ = sign · sigmoid(w_raw)` — direction fixed, magnitude learned.
- **sign = 0**: `ŵ = tanh(w_raw)` — direction and magnitude both learned from data.

---

## Loss Function

| Component | Description |
|-----------|-------------|
| **WMSE** | Importance-weighted MSE; `w_i ∝ \|Δobs_i\| + ε` |
| **AFDA** | Autofocus Direction-Aware focal loss on genes with `\|Δobs\| ≥ 0.1` |
| **Balance** | Soft up-fraction alignment; prevents systematic activation bias |
| **Delta** | Plain MSE on `Δ = x − x0`; prevents the Δ=0 shortcut |
| **DEG** | Top-k importance-weighted MSE on the most-perturbed genes |

---

## Parameter Budget (Rule-of-10)

```
N_genes  = 2 000
Params per gene: V_j, α_j, b_j             → 3 × 2 000  =   6 000
Params per edge:
  sign ≠ 0 (DB-constrained): K_d, n        → 2 params
  sign = 0  (learned):        K_d, n, s_ij → 3 params
Edge param budget = N_obs / 10 − gene_params = 22 000
```

The greedy selection stops when the cumulative edge-parameter cost hits 22 000,
ensuring the model is not over-parameterised relative to the number of
perturbation observations.

---

## Citation

If you use iPerturb, please cite:

> [Manuscript in preparation]

---

## License

MIT
