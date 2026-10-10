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

---
# Addendum A (written 2026-10-10 AFTER seeing G2 fold-0 result, before running anything in it; labelled EXPLORATORY-PRE-REGISTERED)

Observed on K562 fold 0: graph residual models collapsed to B2 (validation MSE identical to 5 digits); an oracle one-hop readout with its scale fitted on the test targets also adds nothing (kappa=0.014); only ~1% of the strongest residual genes are L1 one-hop targets. Gate G2 therefore likely fails; the remaining G2 folds are still run as registered.

Association-channel ablation (plan Go/no-go #1 allows this, labelled non-causal): graph-smoothed target embedding in the B2 model. e_p = (1-alpha) G[p] + alpha * mean_{j in N(p)} G[j]; alpha in {0,0.25,0.5,0.75} and k in {5,10,20,40} selected on validation (alpha=0 is B2, so the family nests B2). Layers (undirected, unsigned): L1 curated, L3 STRING>=700, L4 COXPRESdb top-20 partners per gene (cached). Variants: none, L1, L1+L3, L1+L3+L4, each with a degree-preserving rewired control selected with the same grid. All 3 folds x 2 lines, ALL test targets. Primary metric unchanged (centered Pearson); decision: graph helps only if real-minus-rewired paired bootstrap CI excludes 0 at the same budget. L5 stays off. Claim wording if positive: "association prior improves low-rank embedding", never "regulatory mechanism".
