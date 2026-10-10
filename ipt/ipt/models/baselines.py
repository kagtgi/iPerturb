"""B0 no-change, B1 train-mean, B2 ridge on a low-rank gene embedding (Ahlmann-Eltze-style).
All fit from (D_train, tgt_train) only; predict(tgt) uses the perturbed gene's embedding row.
tgt = column index of the perturbed gene on the gene axis."""
from __future__ import annotations
import numpy as np

KS = (5, 10, 20, 40)


class NoChange:
    def fit(self, D, tgt): self.G = D.shape[1]; return self
    def predict(self, tgt): return np.zeros((len(tgt), self.G), np.float32)


class MeanShift:
    def fit(self, D, tgt): self.b = D.mean(0); return self
    def predict(self, tgt): return np.tile(self.b, (len(tgt), 1)).astype(np.float32)


class Linear:
    def __init__(self, k=10, ridge=1e-2): self.k, self.ridge = k, ridge

    def fit(self, D, tgt, V=None):
        D = D.astype(np.float64); self.b = D.mean(0); Dc = D - self.b
        if V is None:
            _, _, Vt = np.linalg.svd(Dc, full_matrices=False); V = Vt.T
        k = min(self.k, V.shape[1]); self.G = V[:, :k]
        P = self.G[tgt]
        self.W = np.linalg.solve(P.T @ P + self.ridge * np.eye(k), P.T @ (Dc @ self.G))
        return self

    def predict(self, tgt):
        return (self.b + self.G[tgt] @ self.W @ self.G.T).astype(np.float32)


def fit_linear_select(D, tgt, Dv, tv, ks=KS):
    """Choose k on validation by MSE excluding the perturbed gene."""
    Dc = D.astype(np.float64) - D.mean(0)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False); V = Vt.T
    best, bm = None, np.inf
    mask = np.ones_like(Dv, bool); mask[np.arange(len(tv)), tv] = False
    for k in ks:
        m = Linear(k).fit(D, tgt, V)
        e = float((((m.predict(tv) - Dv) ** 2) * mask).sum() / mask.sum())
        if e < bm: best, bm = m, e
    return best


def cross_fit_linear(D, tgt, k, n_splits=5, seed=0):
    """Out-of-fold B2 predictions on the training set (residual target for graph models)."""
    rng = np.random.default_rng(seed); idx = rng.permutation(len(D)); out = np.zeros_like(D, dtype=np.float32)
    for f in range(n_splits):
        te = idx[f::n_splits]; tr = np.setdiff1d(idx, te)
        out[te] = Linear(k).fit(D[tr], tgt[tr]).predict(tgt[te])
    return out
