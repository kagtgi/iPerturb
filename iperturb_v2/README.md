# iPerturb v2 — network-propagation engine

This directory is **iPerturb v2 exactly as it runs in the FairPert benchmark**
(`github.com/kagtgi/PerturbationBenchmarkTool`). It keeps iPerturb's identity — a
*mechanistic model over a real, named gene-regulatory network whose parameters are
inferred from data* — but replaces v1's brittle **Hill-kinetics fixed point** with
**network propagation + a closed-form inferred readout**.

## Why v2

v1 (the root `iperturb.py`, Hill-kinetics GRNN) had three structural limits that
capped it at last place in the benchmark:

1. **Combinations fell back to control** → directional accuracy 0 on gene pairs.
2. **No cross-gene generalization** — per-gene/per-edge kinetics with no way to
   score a gene unseen in training.
3. A **test-cell peek** in the fixed-point evaluation (`perturbed_value=x_obs[gi]`).

v2 fixes all three by construction:

| | v1 (`hill`) | v2 (`propagate`) |
|---|---|---|
| Prediction | signed Hill-kinetics fixed point | RWR / heat propagation + ridge readout |
| Combinations | fall back to control | **additive network sources** (compose by linearity) |
| Unseen genes | no signal | prediction flows from the gene's **network position** |
| Leakage | reads test KD level | **none** — readout fit on train only; input is the perturbation source |
| Epistasis | — | analytic co-propagation `η·(p_A ⊙ p_B)` |
| Compute | GPU, per-sample training | CPU, numpy/scipy, seconds |

### Benchmark result (Norman 2019, single seed)

| Regime | v1 Pearson Δ | **v2 Pearson Δ** |
|---|---|---|
| Combinatorial (held-out pairs) | 0.06 (rank 10/10) | **0.64 (rank 3/10)** — rivals the matching-mean baseline |
| Single-gene (unseen) | 0.009 (≈ null) | **0.25** — the **only** model to significantly beat the non-targeting null |

Averaged across both regimes, v2 has the highest mean Δ of any model in the
benchmark (0.45 vs. the linear baseline's 0.42) — it is the only method that keeps
real signal on perturbations held out entirely from training.

## Method

The GRN is built **exactly as in v1** (`build_full_grn` in `adapter.py` reuses the
root `iperturb.py` crawl: L1 TRRUST/OmniPath/CollecTRI · L2 GeneHancer · L3 STRING ·
L4 COXPRESdb → merge → budget-greedy select). v2 only changes the prediction step:

1. **Operator** — signed, normalized adjacency `Ŵ` from the GRN (level→magnitude
   priors L1=1.0…L4=0.3; symmetric or random-walk normalization; spectrally
   rescaled to a contraction).
2. **Source** — a perturbation `P` becomes an additive signed indicator
   `s = Σ_{g∈P} c_g e_g` (polarity `c_g` from the modality, never from test cells);
   combinations are the sum of their single-gene sources.
3. **Propagation** — random-walk-with-restart `p = α (I − (1−α)Ŵ)⁻¹ s`
   (sparse solve) or heat kernel `exp(−τL)s`.
4. **Readout (inference)** — diagonal `Δ̂_j = r_j p_j` or low-rank
   `M = Δ Pᵀ(PPᵀ+λI)⁻¹` (truncated SVD), fit on **train perturbations only**.
5. **Epistasis** — `Δ̂_{A+B} = M p_{A+B} + η·(p_A ⊙ p_B)`, `η` fit on train combos.
6. **α/τ** chosen by grid search against the headline metric on an inner
   train/val fold.

Every predicted change is still signal flowing along real, signed, directed edges,
so predictions stay **mechanistically interpretable** (attributable to specific
TF→target / enhancer / PPI / co-expression edges) — v2 is *more* interpretable
than v1's opaque fixed point, not less.

## Files

| File | What |
|---|---|
| `_iperturb_prop.py` | the propagation engine — `build_operator`, `propagate`, `fit_readout`, `fit_epistasis`, `select_alpha`, `run_propagation_engine`. Pure numpy/scipy, **no torch**. |
| `adapter.py` | `run_eval(adata, cfg)` — the benchmark entry point; builds the GRN (reusing v1) and dispatches on `IPERTURB_ENGINE` (`propagate` default / `hill` legacy). |
| `config.py` | defaults, incl. the `IPERTURB_*` knobs (engine, operator, α, readout, ridge λ, epistasis). |
| `splits.py` | train/test split + perturbation-parsing helpers (shared with the benchmark). |
| `tests/test_engine.py` | CPU unit tests: operator contraction, RWR == power series, combo additivity, ridge-readout recovery. |

## Config knobs (`config.py`)

```
IPERTURB_ENGINE      = "propagate"   # "propagate" (v2) | "hill" (v1 legacy)
IPERTURB_PROP_OP     = "rwr"         # "rwr" | "heat"
IPERTURB_PROP_NORM   = "sym"         # "sym" | "rw"
IPERTURB_PROP_SIGNED = False         # use edge signs in the operator
IPERTURB_ALPHA       = 0.2           # RWR restart (tuned if IPERTURB_TUNE_ALPHA)
IPERTURB_TUNE_ALPHA  = True
IPERTURB_READOUT     = "lowrank"     # "lowrank" | "diag"
IPERTURB_READOUT_RANK= 64
IPERTURB_RIDGE_LAMBDA= 1.0
IPERTURB_USE_EPISTASIS = True
```

## Usage

The engine is directly usable (torch-free):

```python
from iperturb_v2 import _iperturb_prop as prop
What = prop.build_operator(edges, signs, levels, n_genes, norm="sym", signed=False)
p    = prop.propagate(What, source_vector, alpha=0.2, op="rwr")
```

The full benchmark entry point (needs an AnnData with a `split` column + a `cfg`
dict; pulls torch via `config`):

```python
from iperturb_v2.adapter import run_eval
result = run_eval(adata, cfg)   # {"model", "scoring", "pert_names", "runtime_seconds"}
```

Run the tests from the repo root:

```bash
pytest iperturb_v2/tests -q
```

> **Status.** The propagation engine is validated in the benchmark on Norman
> (numbers above). It is deterministic (CPU numpy/scipy), so results reproduce
> exactly across machines. A multi-seed / multi-dataset (Adamson, Dixit, Replogle)
> sweep to confirm robustness is the natural next step. A learnable Tier-2
> (`IPERTURB_ENGINE="deep"`: APPNP/GRAND diffusion + variational inference) is
> designed but not yet implemented.
