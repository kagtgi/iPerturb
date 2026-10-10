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

---
# Addendum B (written 2026-10-10 after seeing Addendum-A results; before running these controls)

Addendum A: STRING (L3) smoothing lifted centered Pearson by ~+0.10 (K562, RPE1) over no-graph and over degree-preserving rewired. Before any claim, these controls decide whether the gain is perturbation-specific:
1. **Magnitude-residualised metric** `pcc_resid`: project the train-mean direction out of both pred and obs rows (target gene excluded), then Pearson. If the gain disappears here, it only predicted response strength.
2. **Self-target zeroing**: set each training row's own-target entry of Delta to the training column mean before the SVD (and before the ridge fit) for no-graph, L3 and L1+L3 (real and rewired). If the gain collapses, it is imported neighbour knockdown profiles; if it survives, it is not.
3. **Neighbour-mean baseline** (no SVD): pred = (1-a) B2 + a * mean training Delta of the test gene's STRING neighbours that are training targets (a in {0,.25,.5,.75,1}, selected on validation; fallback B2 if none).
4. **Data-only kNN graph**: cosine kNN among gene loadings (k=20 PCs, training data only) with per-gene degree matched to the STRING graph, same smoothing grid. If it matches STRING, the external prior adds nothing.
5. **Unit of analysis**: per-unique-target means across folds, then paired bootstrap over unique targets.
6. Report selected (k, alpha) per fold; the grid is NOT extended post hoc. Save test prediction arrays (float16) so metrics can be recomputed without refitting.
7. DEG-AUPRC caveat: labels are dominated by the shared response (no-change has the best DEG AUPRC on RPE1); report with that note.
Decision rule: the "association prior" claim is made only if the gain survives (1), (2) real-vs-rewired, and beats (4) or is reported as equal to it.

---
# Addendum C (written 2026-10-11, before running B5)

B5 = learned message-passing control on the SAME undirected graph (L1+L3) as the best smoothing variant: 2-layer GCN (row-normalised adjacency, hidden 64, residual self-weights) over the 40 training-derived gene loadings; node embedding -> low-rank coefficients; prediction = train mean + E[t] G_k^T; MSE excluding the perturbed gene; AdamW, lr in {1e-3, 3e-3}, <=200 epochs, patience 20, selected on validation MSE; 3 seeds; real vs degree-preserving rewired graph. Question: does learned message passing beat fixed smoothing (smooth_L1+L3) and B2? Same primary metric, same folds, unit = unique target means.

---
# Addendum D (2026-10-11, before running explanations)

Neighbour-ablation faithfulness for smooth_L1+L3 (selected k, alpha per fold from g4cfg): attribution a_j = <pred(N)-pred(N\j), pred(N)-train_mean> (prediction-only; no observed data). For test targets with >=6 association neighbours, remove the top-K (K=1,2,3,5) attributed neighbours vs K random neighbours (50 draws) and compare the drop in centered Pearson against the OBSERVED profile; unit = unique target; paired bootstrap. Positive = model-emphasised neighbours are the useful ones. Association explanation only; no causal path claim.

---
# Addendum E (2026-10-11, user-directed: accelerate learnable-Hill training via matrix multiplication). Computational study, pre-registered before any timing.

Related work checked: FastKAN/KAN (learnable univariate edge functions via basis expansions, RBF evaluation as matmul), scKAN/KAN-GRN (KAN for GRN *inference*). We therefore do NOT claim first learnable-basis nonlinearity; the claim is limited to: (i) a monotone/bounded Hill-shaped transfer expressed in a Chebyshev basis, (ii) propagation as one SpMM/dense matmul instead of gather/scatter, (iii) measured speed, memory and numerical equivalence against the exact Hill4 edge-wise reference.

Implementations compared (same graph, same parameters, same K=24 damped steps, same clamp): E0 exact Hill4 + edge gather/index_add (current GraphNet; reference); E1 exact Hill4 + sparse matmul; E2 exact Hill4 + dense matmul; C1 Chebyshev(r) Hill + sparse matmul; C2 Chebyshev(r) Hill + dense matmul; P1 per-edge Hill4 (k per edge) edge-wise (cost reference for H1). Chebyshev degree r in {4,8,16,32}; coefficients fitted by least squares to Hill4 on a k grid (log-spaced 0.05..1), differentiable in per-regulator k via table interpolation.
Graphs: signed L1 core (K562: G=8563, E=10,489) and L1+L3 association graph (E ~ 4.7e5, both directions). Batch B in {32,128,512}. Devices: A100 (VM) and CPU.
Metrics: seconds per forward+backward step (median of 20 after 5 warmup), peak GPU memory, max |Hill_cheb - Hill_exact| over [0,1] x k grid, max |x*_impl - x*_E0| over the equilibrium and max |predicted delta difference| for 200 real targets with fitted-scale parameters.
Acceptance (stated now, reported either way): equivalence if max equilibrium difference < 1e-3 at some r<=32; "speedup" reported as measured ratio vs E0 with the batch/graph it applies to; no claim about accuracy of perturbation prediction (G2 already showed Hill=linear there).

---
# Addendum F (2026-10-11): does the context-specific layer carry context-specific signal? (supports pillar 1, "context-specific GRN")

User framing: iPerturb = context-specific GRN + learnable Hill-form propagation. Test on K562 target-held-out folds (same 3 folds, same smoothing grid and metrics as Addendum B/A): association smoothing over L1+L3 (base) versus base + GeneHancer L2 built for K562 (cell-line-active enhancers + TFBS), base + L2 built for HepG2 (foreign context), base + L2 for Jurkat (foreign), and base + degree-preserving rewired K562-L2. Context specificity is supported only if L2(K562) beats both foreign-context L2 variants and the rewired control (paired bootstrap over unique targets, CI excludes 0); otherwise reported as "no measurable context-specific predictive gain from L2". RPE1 cannot be tested because its L2 is empty (tissue label mismatch, documented). Also report graph-level context specificity: TF out-degree divergence of L2 across K562/HepG2/Jurkat and Jaccard of edge sets (descriptive, no predictive claim).
