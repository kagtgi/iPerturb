"""Per-edge learnable Hill-form shapes: edge-wise evaluation vs one sparse matmul (PREREG Addendum H).

Edge e=(j->i) has its own shape H_e(x)=sum_b c_eb T_b(2x-1) and strength folded into c_eb.
msg_i = sum_{e: j->i} sum_b c_eb [Phi_b(x_j) - Phi_b(x0_j)].
'edge': gather Phi(x)[:, src] -> (B,E,R) then scatter-add (O(B E R) activations).
'spmm': Phi(x)-Phi(x0) -> (B,G*R) times one sparse (G*R x G) matrix with E*R non-zeros (O(B G R)).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from ipt.models.basis_hill import fit_table, coef_from_k, K_GRID


def cheb_stack_basis(x, R):
    t = 2 * x - 1
    T = [torch.ones_like(t), t]
    for _ in range(2, R):
        T.append(2 * t * T[-1] - T[-2])
    return torch.stack(T[:R], dim=-1)


class PerEdgeCheb(nn.Module):
    def __init__(self, src, dst, x0, r=8, impl="spmm", steps=24, damping=0.5, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.R, self.impl, self.steps, self.damping = r + 1, impl, steps, damping
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.G = len(x0)
        x0 = torch.as_tensor(x0, dtype=torch.float32).clamp(0.02, 0.98)
        self.register_buffer("x0", x0)
        table = torch.as_tensor(fit_table(r), dtype=torch.float32)
        k = torch.as_tensor(rng.uniform(0.1, 0.9, len(src)), dtype=torch.float32)
        w = torch.as_tensor(rng.normal(0, 0.15, len(src)), dtype=torch.float32)
        self.coef = nn.Parameter(w[:, None] * coef_from_k(k, table))          # (E, R)
        rows = self.dst.repeat_interleave(self.R)
        cols = (self.src[:, None] * self.R + torch.arange(self.R)[None, :]).reshape(-1)
        perm = torch.argsort(rows * (self.G * self.R) + cols)
        self.register_buffer("perm", perm)
        self.register_buffer("sidx", torch.stack([rows[perm], cols[perm]]))

    def _inflow(self, dphi):                                              # dphi (B,G,R)
        B = dphi.shape[0]
        if self.impl == "edge":
            g = dphi[:, self.src]                                           # (B,E,R)
            msg = (g * self.coef[None]).sum(-1)                             # (B,E)
            return torch.zeros(B, self.G, device=dphi.device).index_add_(1, self.dst, msg)
        Wt = torch.sparse_coo_tensor(self.sidx, self.coef.reshape(-1)[self.perm], (self.G, self.G * self.R), is_coalesced=True)
        return torch.sparse.mm(Wt, dphi.reshape(B, self.G * self.R).T).T

    def forward(self, tgt, zeta=0.8):
        B = len(tgt)
        x0 = self.x0[None, :].expand(B, -1)
        M = torch.zeros(B, self.G, dtype=torch.bool, device=x0.device)
        M[torch.arange(B, device=x0.device), tgt] = True
        cv = ((1 - zeta) * self.x0[tgt]).clamp(0.02, 0.98)[:, None].expand(B, self.G)
        phi0 = cheb_stack_basis(x0, self.R)
        base = torch.log(x0) - torch.log1p(-x0)
        x = torch.where(M, cv, x0)
        for _ in range(self.steps):
            cand = torch.where(M, cv, torch.sigmoid(base + self._inflow(cheb_stack_basis(x, self.R) - phi0)))
            x = torch.where(M, cv, (1 - self.damping) * x + self.damping * cand)
        return x
