"""Matmul-friendly Hill propagation (PREREG Addendum E).

* Chebyshev-basis Hill: H(x;k) ~= sum_b a_b(k) T_b(2x-1), coefficients least-squares fitted on a log-k grid and
  linearly interpolated in log k, so per-regulator k stays learnable and differentiable.
* Propagation as one sparse (spmm) or dense matmul instead of edge gather + index_add.
Fixed-point equations are identical to GraphNet; only the arithmetic changes.
"""
from __future__ import annotations
import numpy as np
import torch
from numpy.polynomial import chebyshev as C
from ipt.models.hill import hill4
from ipt.models.graphnet import GraphNet

K_GRID = np.exp(np.linspace(np.log(0.05), np.log(1.0), 48))


def fit_table(r: int, n: float = 2.0) -> np.ndarray:
    """(len(K_GRID), r+1) Chebyshev coefficients of Hill4(.;k,n) on [0,1]."""
    x = 0.5 * (np.cos(np.pi * (np.arange(4096) + 0.5) / 4096) + 1.0)
    tab = []
    for k in K_GRID:
        y = hill4(torch.as_tensor(x), torch.tensor(float(k), dtype=torch.float64), n).numpy()
        tab.append(C.chebfit(2 * x - 1, y, r))
    return np.stack(tab)


def coef_from_k(k, table):
    """Differentiable interpolation of coefficient table in log k. k: (G,) or (1,) -> (G or 1, r+1)."""
    lo, hi = float(np.log(K_GRID[0])), float(np.log(K_GRID[-1]))
    pos = ((torch.log(k.clamp(K_GRID[0], K_GRID[-1])) - lo) / (hi - lo) * (len(K_GRID) - 1))
    i0 = pos.floor().clamp(0, len(K_GRID) - 2).long()
    f = (pos - i0.float())[:, None]
    return (1 - f) * table[i0] + f * table[i0 + 1]


def cheb_clenshaw(x, coef):
    """x: (B,G) in [0,1]; coef: (G or 1, r+1). Elementwise Clenshaw recurrence."""
    t = 2 * x - 1
    r = coef.shape[1] - 1
    b1 = b2 = torch.zeros_like(x)
    for j in range(r, 0, -1):
        b1, b2 = coef[:, j][None, :] + 2 * t * b1 - b2, b1
    return coef[:, 0][None, :] + t * b1 - b2


def cheb_stack(x, coef):
    """Basis-matrix form: Phi (B,G,r+1) contracted with coefficients (matmul-style)."""
    t = 2 * x - 1
    T = [torch.ones_like(t), t]
    for _ in range(2, coef.shape[1]):
        T.append(2 * t * T[-1] - T[-2])
    Phi = torch.stack(T[: coef.shape[1]], dim=-1)
    return torch.einsum("bgr,gr->bg", Phi, coef.expand(x.shape[1], -1))


class FastGraphNet(GraphNet):
    def __init__(self, *a, hill_impl="exact", prop_impl="edge", r=16, cheb_eval="clenshaw", **kw):
        super().__init__(*a, **kw)
        self.hill_impl, self.prop_impl, self.r, self.cheb_eval = hill_impl, prop_impl, r, cheb_eval
        if hill_impl == "cheb":
            self.register_buffer("table", torch.as_tensor(fit_table(r, self.n), dtype=torch.float32))
        order = torch.argsort(self.dst * self.G + self.src)  # row-major (dst,src) so the sparse matrix is coalesced
        self.register_buffer("order", order)

    def T(self, x):
        if self.transfer == "linear":
            return x
        if self.hill_impl == "exact":
            return super().T(x)
        k = self.k()
        k = k if k.numel() > 1 else k.expand(self.G)
        coef = coef_from_k(k, self.table)
        return (cheb_clenshaw if self.cheb_eval == "clenshaw" else cheb_stack)(x, coef)

    def _prep(self, w):
        if self.prop_impl == "edge":
            return w
        if self.prop_impl == "spmm":
            idx = torch.stack([self.dst[self.order], self.src[self.order]])
            return torch.sparse_coo_tensor(idx, w[self.order], (self.G, self.G), is_coalesced=True)
        W = torch.zeros(self.G, self.G, device=w.device, dtype=w.dtype)
        return W.index_put((self.src, self.dst), w)

    def _inflow(self, hd, wp):
        if self.prop_impl == "edge":
            return super()._inflow(hd, wp)
        if self.prop_impl == "spmm":
            return torch.sparse.mm(wp, hd.T).T
        return hd @ wp
