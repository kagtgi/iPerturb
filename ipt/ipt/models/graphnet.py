"""Baseline-centered fixed-point regulatory propagation with persistent CRISPRi clamp.

F_i(x) = sigmoid(logit(x0_i) + sum_j w_ji (T(x_j) - T(x0_j))),  T = hill4(., k_j) ('hill4') or identity ('linear').
Target coordinate is clamped to (1-zeta)*x0 at every update. Delta_hat_i = d_i (x*_i - x0_i), i != target.
w_ji = rho * s_ji * c_ji * softplus(a) / max(1, sum_j' |.|)  (incoming normalisation, plan 5.3).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ipt.models.hill import hill4

EPS = 0.02


def logit(x): return torch.log(x) - torch.log1p(-x)


class GraphNet(nn.Module):
    def __init__(self, src, dst, sign, conf, x0, transfer="hill4", n=2.0, k_mode="shared",
                 k_init=0.3, k_bounds=(0.05, 1.0), steps=24, damping=0.5, edge_k=False):
        super().__init__()
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("sign", torch.as_tensor(sign, dtype=torch.float32))
        self.register_buffer("conf", torch.as_tensor(conf, dtype=torch.float32))
        self.register_buffer("x0", torch.as_tensor(x0, dtype=torch.float32).clamp(EPS, 1 - EPS))
        G = len(x0); self.G, self.transfer, self.n = G, transfer, n
        self.k_mode, self.kb, self.steps, self.damping = k_mode, k_bounds, steps, damping
        self.a = nn.Parameter(torch.tensor(0.0)); self.rho = nn.Parameter(torch.tensor(0.0))
        self.u = nn.Parameter(torch.zeros(G)); self.d0 = nn.Parameter(torch.tensor(0.0))
        z = (k_init - k_bounds[0]) / (k_bounds[1] - k_bounds[0])
        self.kz = nn.Parameter(torch.full((G if k_mode == "regulator" else 1,), float(np.log(z / (1 - z)))))
        self.register_buffer("k_bar", torch.tensor(float(k_init)))

    def k(self):
        return self.kb[0] + (self.kb[1] - self.kb[0]) * torch.sigmoid(self.kz)

    def weights(self):
        wt = self.sign * self.conf * Fn.softplus(self.a + 0.5413)  # softplus(0.5413)=1 at init
        inc = torch.zeros(self.G, device=wt.device).index_add_(0, self.dst, wt.abs())
        wt = wt / torch.clamp(inc[self.dst], min=1.0)
        return torch.sigmoid(self.rho + 1.0) * 2.0 * wt     # rho_eff in (0,2), 1.46 at init

    def decoder(self):
        return Fn.softplus(self.d0 + self.u)

    def T(self, x):
        if self.transfer == "linear": return x
        k = self.k(); k = k if k.numel() == 1 else k[None, :]
        return hill4(x, k, self.n)

    def _prep(self, w):
        return w

    def _inflow(self, hd, w):
        """Edge-wise gather/scatter reference: inflow_i = sum_j w_ji * hd_j."""
        return torch.zeros(hd.shape[0], self.G, device=hd.device, dtype=hd.dtype).index_add_(1, self.dst, hd[:, self.src] * w[None, :])

    def equilibrium(self, tgt, zeta):
        B = len(tgt); x0 = self.x0[None, :].expand(B, -1)
        M = torch.zeros(B, self.G, dtype=torch.bool, device=self.x0.device); M[torch.arange(B, device=self.x0.device), tgt] = True
        cv = ((1 - zeta) * self.x0[tgt]).clamp(EPS, 1 - EPS)[:, None].expand(B, self.G)
        w = self.weights(); base = logit(x0); h0 = self.T(x0)
        wp = self._prep(w)
        step = lambda x: torch.where(M, cv, torch.sigmoid(base + self._inflow(self.T(x) - h0, wp)))
        x = torch.where(M, cv, x0)
        for _ in range(self.steps):
            x = torch.where(M, cv, (1 - self.damping) * x + self.damping * step(x))
        return x, (step(x) - x).abs().amax(1), M

    def forward(self, tgt, zeta):
        x, resid, M = self.equilibrium(tgt, zeta)
        out = torch.where(M, torch.zeros(()), self.decoder()[None, :] * (x - self.x0[None, :]))
        return out, resid
