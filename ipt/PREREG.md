# Pre-registration (frozen before any graph-model fit; 2026-10-10)

Data: Replogle K562-essential and RPE1 (h5ad -> `ipt/data/prepare.py`). Target = batch-matched mean log1p(CP10K) delta, perturbed gene excluded from every score. Folds: 3 target-held-out (70/15/15) per line, strata = (TF status, control-expression tertile, L1 out-degree bin); all train-independent.

## Models
- B0 no-change; B1 train-mean; B2 ridge on top-k PCs of training Delta (k in {5,10,20,40} chosen on validation).
- B3 = graph residual model with linear transfer; H2 = same with shared-Hill4 transfer (`hill_fixk`: global k=0.3 fixed; `hill_regk`: per-regulator k, ridge to 0.3). Identical graph, decoder, optimiser, grid (lr {0.003,0.01} x u-ridge {1e-3,1e-1}), epochs <=120, patience 20, selection by validation masked MSE on reachable validation targets.
- Graph = signed L1 core only (CollecTRI/OmniPath/TRRUST; sign 0 and conflicting-sign edges excluded; conf = min(n_db,3)/3). L5 OFF. No L2-L4 in G2.
- Graph models predict the cross-fitted residual of B2 for *reachable* targets (>=1 signed out-edge); unreachable targets use B2 unchanged (explicit fallback).
- Intervention: target clamped to (1-zeta)*x0 at every update; zeta from training-target knockdown ratio only.
- Controls: degree-preserving rewired graph (same fit budget), 1 rewire per seed.

## Metrics (macro over targets)
Primary: centered Pearson on Delta (train-mean removed from pred and obs). Secondary: raw Pearson, top-20 centered Pearson, RMSE, DEG AUPRC (labels |Welch z|>4, cell-level, no replicates; score = |pred - train_mean|), nearest-centroid normalised rank. Reported for all test targets AND for reachable vs unreachable separately. Paired bootstrap over targets (10,000), 95% CI.

## Gate G2 decision rule (K562, all three folds pooled, reachable targets)
PASS iff for H2 vs B3 AND H2 vs B2, mean primary-metric difference > 0 with 95% CI excluding 0, AND real graph beats rewired graph (paired, same model). Otherwise Step 5B (negative/conditional paper). Because ~6% of targets are reachable, the pooled reachable sample is small; this is stated as a power limitation, not tuned away.

## Not allowed after seeing results
Changing metric, gate rule, fallback, or graph definition. Any added analysis is labelled exploratory.
