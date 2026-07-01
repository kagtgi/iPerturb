"""iPerturb 2.0 — Tier 1: analytic network-propagation engine (CPU, numpy/scipy).

Replaces iPerturb's brittle Hill-kinetics fixed point with the *linearized steady
state of a signed graph diffusion* over the same L1-L4 GRN, then infers a closed-form
readout (and an analytic combo-epistasis term) on the train split. Keeps iPerturb's
identity — a mechanistic GRN model whose parameters are inferred — but:

  * **composes combos** (additive source ``s_{A+B}=s_A+s_B``) instead of falling back
    to control (fixes DirAcc 0.0 on doubles),
  * **generalizes to unseen genes** (propagation from a gene's network position yields
    a structured delta even with zero observed training delta — where Linear has a
    zero column), and
  * stays fully interpretable (edges / paths / per-gene response inspectable).

References: network propagation in biology (Cowen et al., Nat Rev Genet 2017; HotNet),
personalized PageRank / RWR, APPNP "Predict then Propagate" (fixed-propagation limit).

No torch; pure numpy/scipy so it runs on CPU/CI. See ``eval/models/iperturb.py`` for
the adapter wiring (engine selector ``IPERTURB_ENGINE='propagate'``).
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, factorized

logger = logging.getLogger(__name__)

_LEVEL_W = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.3}  # mirrors the GRNN level priors


def _logmean(X) -> np.ndarray:
    """Bias-free centroid in log1p space: log1p(mean(expm1(X)))  (matches iPerturb)."""
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    return np.log1p(np.expm1(X).mean(0)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. Operator
# --------------------------------------------------------------------------- #
def build_operator(edges, signs, levels, n, *, norm="sym", signed=False):
    """Signed/weighted, normalized diffusion operator ``What`` (N x N, sparse).

    ``edges`` = list of (src_idx, tgt_idx) with influence flowing src -> tgt, so
    ``A[tgt, src] = sign * level_magnitude`` (directed L1; L3/L4 enter both ways
    because the GRN build already emits both orientations).

    Normalized to a contraction so RWR's inverse exists and converges:
      * ``sym``: ``D^{-1/2} A D^{-1/2}`` with ``D = diag(sum_j |A_ij|)``;
      * ``rw`` : ``A D_col^{-1}`` (column-stochastic on |A|).
    Then spectral-rescaled by ``rho(|What|)`` so ``rho(|What|) <= 1``.
    """
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)
    rows, cols, vals = [], [], []
    for (s, t), sg, lv in zip(edges, signs, levels):
        mag = _LEVEL_W.get(int(lv), 0.3)
        w = (float(sg) if (signed and sg != 0) else 1.0) * mag
        if w == 0.0:
            w = mag  # sign==0 in signed mode -> use +magnitude
        rows.append(t); cols.append(s); vals.append(w)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    absA = abs(A)
    if norm == "rw":
        dc = np.asarray(absA.sum(0)).ravel()
        dc[dc == 0] = 1.0
        What = A @ sp.diags(1.0 / dc)
    else:  # symmetric
        dr = np.asarray(absA.sum(1)).ravel()
        dr[dr == 0] = 1.0
        Dis = sp.diags(1.0 / np.sqrt(dr))
        What = Dis @ A @ Dis
    # spectral rescale (power iteration on |What|) -> contraction
    aW = abs(What).tocsr()
    v = np.random.default_rng(0).random(n)
    v /= np.linalg.norm(v) + 1e-12
    rho = 1.0
    for _ in range(50):
        u = aW @ v
        rho = np.linalg.norm(u) + 1e-12
        v = u / rho
    if rho > 1.0:
        What = What / (rho * 1.001)
    return What.tocsr()


# --------------------------------------------------------------------------- #
# 2. Propagation:  influence p = P(alpha) s
# --------------------------------------------------------------------------- #
def propagate(What, S, alpha, op="rwr"):
    """Propagate source columns ``S`` (N x m, dense) -> influence ``P`` (N x m).

    * ``rwr`` : personalized PageRank / RWR, ``P = alpha (I-(1-alpha)What)^{-1} S``.
      Pre-factorized solve (one LU, reused for all columns / predict time).
    * ``heat``: heat kernel, ``P = expm_multiply(-tau L, S)``, ``L=I-What``, ``tau=alpha``.
    """
    n = What.shape[0]
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 1:
        S = S[:, None]
    if op == "heat":
        L = (sp.identity(n, format="csc") - What).tocsc()
        return np.asarray(expm_multiply(-float(alpha) * L, S))
    M = (sp.identity(n, format="csc") - (1.0 - alpha) * What).tocsc()
    solve = factorized(M)  # LU once
    P = np.column_stack([solve(S[:, j]) for j in range(S.shape[1])])
    return alpha * P


# --------------------------------------------------------------------------- #
# 3. Readout (closed-form inference):  influence -> expression delta
# --------------------------------------------------------------------------- #
def fit_readout(P_tr, D_tr, *, lam=1.0, rank=None):
    """Closed-form ridge readout mapping propagated influence -> delta.

    ``P_tr, D_tr`` are (N x n_tr). Returns a dict the apply step understands.
    * diagonal (rank=None): per-gene ``r_j = <P_j,D_j>/(<P_j,P_j>+lam)``.
    * low-rank: ``M = D P^T (P P^T + lam I)^{-1}`` truncated to ``rank`` (SVD).
    """
    if rank is None:
        num = (P_tr * D_tr).sum(1)
        den = (P_tr * P_tr).sum(1) + lam
        return {"kind": "diag", "r": (num / den).astype(np.float64)}
    P = P_tr; D = D_tr
    G = P @ P.T + lam * np.eye(P.shape[0])
    M = D @ P.T @ np.linalg.inv(G)            # (N x N)
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    k = int(min(rank, len(sv)))
    return {"kind": "lowrank", "U": U[:, :k] * sv[:k], "Vt": Vt[:k, :]}


def apply_readout(ro, P):
    """Delta_hat (N x m) from influence P (N x m)."""
    if ro["kind"] == "diag":
        return ro["r"][:, None] * P
    return ro["U"] @ (ro["Vt"] @ P)


# --------------------------------------------------------------------------- #
# 4. Combo epistasis (analytic co-propagation):  eta * (p_A (.) p_B)
# --------------------------------------------------------------------------- #
def fit_epistasis(P_pairs, resid, *, lam=1.0):
    """Per-gene ``eta_j`` fit on train-combo additive residuals.

    ``P_pairs`` = Hadamard overlap (p_A (.) p_B) for train combos (N x n_combo);
    ``resid`` = Delta_combo - additive prediction (N x n_combo). Closed-form 1-D ridge.
    """
    if P_pairs.shape[1] == 0:
        return {"eta": np.zeros(P_pairs.shape[0])}
    num = (P_pairs * resid).sum(1)
    den = (P_pairs * P_pairs).sum(1) + lam
    return {"eta": (num / den).astype(np.float64)}


# --------------------------------------------------------------------------- #
# 5. Headline-metric proxy (mean-centred, pref-referenced PearsonD)
# --------------------------------------------------------------------------- #
def _mc(x):
    return x - x.mean()


def _headline(Dhat, Dtrue, offset):
    """Mean over perts of Pearson( mc(Dhat+offset), mc(Dtrue+offset) ).

    ``offset = x0 - pref`` makes this the SV-corrected (pref-referenced) PearsonD
    the benchmark scores (pred-pref vs true-pref, mean-centred per pert)."""
    rs = []
    for j in range(Dhat.shape[1]):
        a = _mc(Dhat[:, j] + offset); b = _mc(Dtrue[:, j] + offset)
        sa, sb = a.std(), b.std()
        if sa > 1e-9 and sb > 1e-9:
            rs.append(float(np.mean(a * b) / (sa * sb)))
    return float(np.mean(rs)) if rs else -1.0


# --------------------------------------------------------------------------- #
# 6. alpha / tau selection (grid + inner val on the headline proxy)
# --------------------------------------------------------------------------- #
def select_alpha(What, S_tr, D_tr, offset, *, alphas, op, lam, rank, seed=42):
    """Inner 85/15 split over train perts; pick alpha maximizing headline proxy."""
    n_tr = S_tr.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_tr)
    n_val = max(1, int(round(0.15 * n_tr)))
    va, tr = idx[:n_val], idx[n_val:]
    best, best_a = -2.0, alphas[0]
    for a in alphas:
        P = propagate(What, S_tr, a, op)
        ro = fit_readout(P[:, tr], D_tr[:, tr], lam=lam, rank=rank)
        Dhat_va = apply_readout(ro, P[:, va])
        score = _headline(Dhat_va, D_tr[:, va], offset)
        logger.info("  iPerturb-prop alpha=%.3f  val headline=%.4f", a, score)
        if score > best:
            best, best_a = score, a
    return best_a, best


# --------------------------------------------------------------------------- #
# 7. Orchestrator called by the adapter
# --------------------------------------------------------------------------- #
def run_propagation_engine(*, adata, train_mask, test_mask, obs_all, gene_names,
                           gene2idx, x0, hvg_pos, grn_df, pert_ref, ctrl_label,
                           combo_sep, min_cells, cfg, parse_gene_set):
    """Full Tier-1 flow. Returns (pred_c, true_c, names, info).

    ``grn_df`` = DataFrame with columns source,target,sign,level (the same GRN TSV
    the Hill engine consumes). Predicts EVERY test pert (singles + combos); only
    perts whose genes are all off-GRN fall back to ``pert_ref`` (never ``ctrl``).
    """
    N = len(gene_names)
    op = str(cfg.get("IPERTURB_PROP_OP", "rwr"))
    norm = str(cfg.get("IPERTURB_PROP_NORM", "sym"))
    signed = bool(cfg.get("IPERTURB_PROP_SIGNED", False))
    lam = float(cfg.get("IPERTURB_RIDGE_LAMBDA", 1.0))
    readout = str(cfg.get("IPERTURB_READOUT", "lowrank"))
    rank = int(cfg.get("IPERTURB_READOUT_RANK", 64)) if readout == "lowrank" else None
    use_epi = bool(cfg.get("IPERTURB_USE_EPISTASIS", True))
    tune = bool(cfg.get("IPERTURB_TUNE_ALPHA", True))
    alpha0 = float(cfg.get("IPERTURB_ALPHA", 0.2))
    seed = int(cfg.get("RANDOM_SEED", 42))

    # --- operator ---
    edges, signs, levels = [], [], []
    for _, row in grn_df.iterrows():
        s, t = str(row["source"]), str(row["target"])
        if s in gene2idx and t in gene2idx:
            edges.append((gene2idx[s], gene2idx[t]))
            signs.append(int(row.get("sign", 0)) if str(row.get("sign", 0)) not in ("", "nan") else 0)
            levels.append(int(row.get("level", 1)))
    What = build_operator(edges, signs, levels, N, norm=norm, signed=signed)
    rank = min(rank, N) if rank else None

    def source_of(pert):
        gs = [g for g in parse_gene_set(pert, ctrl_label, combo_sep) if g in gene2idx]
        s = np.zeros(N)
        for g in gs:
            s[gene2idx[g]] = 1.0           # CRISPRa over-expression; readout learns sign
        return s, gs

    # --- training targets (centroid deltas over gene_names, train split only) ---
    train_perts = [p for p in np.unique(obs_all[train_mask]) if p != str(ctrl_label)]
    S_cols, D_cols, kept = [], [], []
    for p in train_perts:
        m = (obs_all == p) & train_mask
        if m.sum() < min_cells:
            continue
        s, gs = source_of(p)
        if s.sum() == 0:
            continue
        S_cols.append(s)
        D_cols.append(_logmean(adata.X[m])[hvg_pos].astype(np.float64) - x0.astype(np.float64))
        kept.append(p)
    if not S_cols:
        raise RuntimeError("iPerturb-prop: no trainable perturbations with GRN coverage")
    S_tr = np.column_stack(S_cols)
    D_tr = np.column_stack(D_cols)
    offset = x0.astype(np.float64) - np.asarray(pert_ref, dtype=np.float64)

    # --- infer alpha + readout ---
    alphas = ([0.05, 0.1, 0.2, 0.3, 0.5] if tune else [alpha0])
    if tune and len(alphas) > 1:
        alpha, _ = select_alpha(What, S_tr, D_tr, offset, alphas=alphas, op=op,
                                lam=lam, rank=rank, seed=seed)
    else:
        alpha = alpha0
    P_tr = propagate(What, S_tr, alpha, op)
    ro = fit_readout(P_tr, D_tr, lam=lam, rank=rank)

    # --- epistasis on train combos (additive residuals) ---
    epi = {"eta": np.zeros(N)}
    if use_epi:
        cP, cR = [], []
        for j, p in enumerate(kept):
            gs = [g for g in parse_gene_set(p, ctrl_label, combo_sep) if g in gene2idx]
            if len(gs) < 2:
                continue
            sA = np.zeros(N); sA[gene2idx[gs[0]]] = 1.0
            sB = np.zeros(N); sB[gene2idx[gs[1]]] = 1.0
            pA = propagate(What, sA, alpha, op)[:, 0]
            pB = propagate(What, sB, alpha, op)[:, 0]
            cP.append(pA * pB)
            cR.append(D_tr[:, j] - apply_readout(ro, P_tr[:, j:j + 1])[:, 0])
        if cP:
            epi = fit_epistasis(np.column_stack(cP), np.column_stack(cR), lam=lam)

    logger.info("iPerturb-prop: op=%s norm=%s signed=%s alpha=%.3f readout=%s rank=%s "
                "epistasis=%s | %d edges, %d train perts",
                op, norm, signed, alpha, readout, rank, use_epi, len(edges), len(kept))

    # --- predict every test pert ---
    pred_c, true_c, names, n_modeled = [], [], [], 0
    test_perts = [p for p in np.unique(obs_all[test_mask]) if p != str(ctrl_label)]
    for p in test_perts:
        m = (obs_all == p) & test_mask
        if m.sum() < min_cells:
            continue
        x_obs = _logmean(adata.X[m])[hvg_pos].astype(np.float32)
        s, gs = source_of(p)
        if s.sum() == 0:
            pred = np.asarray(pert_ref, dtype=np.float32)   # fair non-trivial fallback
        else:
            P = propagate(What, s, alpha, op)
            Dhat = apply_readout(ro, P)[:, 0]
            if use_epi and len(gs) >= 2:
                sA = np.zeros(N); sA[gene2idx[gs[0]]] = 1.0
                sB = np.zeros(N); sB[gene2idx[gs[1]]] = 1.0
                pA = propagate(What, sA, alpha, op)[:, 0]
                pB = propagate(What, sB, alpha, op)[:, 0]
                Dhat = Dhat + epi["eta"] * (pA * pB)
            pred = (x0.astype(np.float64) + Dhat).astype(np.float32)
            n_modeled += 1
        pred_c.append(pred); true_c.append(x_obs); names.append(p)
    info = {"alpha": alpha, "n_edges": len(edges), "n_modeled": n_modeled,
            "op": op, "readout": readout}
    return pred_c, true_c, names, info
