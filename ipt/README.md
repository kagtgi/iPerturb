# iPerturb (`ipt/`) — LNBI rebuild

Context-specific GRN + Hill-form learnable propagation for unseen CRISPRi perturbations (K562, RPE1).
Protocol frozen in [`PREREG.md`](PREREG.md) (hash it before fitting; addenda A–H are dated and were written before the run they govern).

## Layout
- `ipt/data/` h5ad -> per-perturbation mean log1p(CP10K) deltas (batch-matched controls), target-held-out folds
- `ipt/graph/l1.py` curated L1 (CollecTRI / OmniPath / TRRUST); sign 0 never promoted; conflicts -> unsigned
- `ipt/models/` `hill.py` (TF4), `graphnet.py` (baseline-centred fixed point, persistent clamp), `basis_hill.py` (Chebyshev Hill + sparse/dense matmul), `peredge.py` (per-edge shapes), `scc.py` (SCC-aware solver), `baselines.py`
- `ipt/metrics.py` centred / residualised Pearson, top-k, DEG AUPRC, centroid rank, paired bootstrap
- `scripts/` experiments (`g1_check` ... `g8_fallback`, `bench_*`, `gears_run.py`), `build_numbers.py`, `make_tables.py`, `fig*.py`
- `paper/` LNCS manuscript; **every number in the text comes from `paper/numbers.tex`** (generated), tables from `paper/tables/`

## Reproduce (order)
```
py scripts/g1_check.py              # coverage + sign signal + folds   (needs artifacts/data/*.npz, see ipt/data/prepare.py on the h5ad files)
py scripts/g2_run.py --line K562_essential --folds 0 1 2 ; py scripts/g2_run.py --line rpe1 --folds 0 1 2
py scripts/g4_controls.py --line K562_essential ; py scripts/g4_controls.py --line rpe1   # association smoothing + controls
py scripts/g5_gnn.py --line K562_essential ; py scripts/g5_gnn.py --line rpe1
py scripts/g6_explain.py ; py scripts/g7_context.py ; py scripts/g8_fallback.py
py scripts/scc_bench.py ; py scripts/make_bench_graphs.py ; py scripts/bench_matmul.py ; py scripts/bench_peredge.py   # GPU on the VM
# GEARS on the VM (gears-env): scripts/gears_run.py, then convert predictions with scripts/gears_to_scores.py
py scripts/build_numbers.py ; py scripts/make_tables.py ; py scripts/make_bench_table.py ; py scripts/fig*.py
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
pytest -q
```
Data are not redistributed (GeneHancer is licence-gated; Replogle h5ad from scPerturb).
