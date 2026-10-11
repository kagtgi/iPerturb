"""Certified anti-vanishing propagation: log-odds state, leaky Hill transfer, contraction guaranteed for ALL parameter values.

z_i <- (1-eta) z_i + eta [ z0_i + sum_j w_ji (T(sigmoid z_j) - T(x0_j)) ],  T(x) = (1-leak) H(x;k) + leak * x
Certificate (appendix, Theorem 1): with L_T = sup |T'| and ||W||_inf the maximal total incoming weight, the damped map is a contraction in the
sup norm with constant q_eta = 1 - eta (1 - q),  q = (L_T / 4) ||W||_inf.  We parametrise ||W||_inf = gamma_max * sigmoid(rho) * 4 / L_T(k)
with gamma_max <= 1/2 so that q <= gamma_max for every value the optimiser can reach (k in [0.05, 1], any a, rho).
The clamp is imposed in log-odds at every update; output = d_i (z*_i - z0_i) is not squashed, the leak keeps dT/dx > 0 where the Hill curve is flat.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from ipt.models.hill import hill4

EPS = 0.02


def logit(x):
    return torch.log(x) - torch.log1p(-x)


def hill_lipschitz(k, n=2.0):
    """Closed form for n = 2 and k <= sqrt(3):  max_x |H'(x)| = 9 (1 + k^2) / (8 sqrt(3) k)  (attained at 1 - x = k / sqrt(3))."""
    if n != 2.0:
        raise ValueError("closed form is for n = 2")
    return 9.0 * (1.0 + k ** 2) / (8.0 * np.sqrt(3.0) * k)


class GraphNet2(nn.Module):
    def __init__(self, src, dst, sign, conf, x0, transfer="hill4", leak=0.1, n=2.0, k_init=0.3, steps=24, damping=0.5,
                 gamma_max=0.5, d_init=0.3, tol=None):
        super().__init__()
        self.register_buffer("src", torch.as_tensor(np.array(src), dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(np.array(dst), dtype=torch.long))
        self.register_buffer("sign", torch.as_tensor(np.array(sign), dtype=torch.float32))
        self.register_buffer("conf", torch.as_tensor(np.array(conf), dtype=torch.float32))
        x0 = torch.as_tensor(np.array(x0), dtype=torch.float32).clamp(EPS, 1 - EPS)
        self.register_buffer("x0", x0)
        self.G, self.transfer, self.leak, self.n, self.steps, self.damping = len(x0), transfer, leak, n, steps, damping
        self.gamma_max, self.tol, self.last_iters = gamma_max, tol, 0
        self.a = nn.Parameter(torch.tensor(0.0))
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.kz = nn.Parameter(torch.tensor(float(np.log((k_init - 0.05) / (1.0 - k_init)))))
        self.u = nn.Parameter(torch.zeros(self.G))
        self.d0 = nn.Parameter(torch.tensor(float(np.log(np.expm1(d_init)))))

    def k(self):
        return 0.05 + 0.95 * torch.sigmoid(self.kz)

    def lipschitz_T(self):
        if self.transfer == "linear":
            return torch.tensor(1.0)
        return (1 - self.leak) * hill_lipschitz(self.k(), self.n) + self.leak

    def weights(self):
        """Incoming-normalised signed weights scaled so that ||W||_inf = gamma_max sigmoid(rho) 4 / L_T  (so q <= gamma_max)."""
        wt = self.sign * self.conf * Fn.softplus(self.a + 0.5413)
        inc = torch.zeros(self.G, device=wt.device).index_add_(0, self.dst, wt.abs())
        wn = wt / torch.clamp(inc[self.dst], min=1e-6)            # every gene with parents has incoming |weight| sum 1
        return self.gamma_max * torch.sigmoid(self.rho) * 4.0 / self.lipschitz_T() * wn

    def certificate(self):
        w = self.weights().detach()
        inc = torch.zeros(self.G).index_add_(0, self.dst, w.abs())
        L, winf = float(self.lipschitz_T()), float(inc.max())
        q = L / 4.0 * winf
        return dict(L_T=L, W_inf=winf, q=q, q_eta=1 - self.damping * (1 - q))

    def T(self, x):
        if self.transfer == "linear":
            return x
        return (1 - self.leak) * hill4(x, self.k(), self.n) + self.leak * x

    def decoder(self):
        return Fn.softplus(self.d0 + self.u)

    def equilibrium(self, tgt, zeta, z_init=None, tol=None, max_iter=500):
        B = len(tgt)
        dev = self.x0.device
        z0 = logit(self.x0)[None, :].expand(B, -1)
        M = torch.zeros(B, self.G, dtype=torch.bool, device=dev)
        M[torch.arange(B, device=dev), tgt] = True
        zc = logit(((1 - zeta) * self.x0[tgt]).clamp(EPS, 1 - EPS))[:, None].expand(B, self.G)
        w = self.weights()
        h0 = self.T(self.x0)[None, :]
        z = torch.where(M, zc, z0 if z_init is None else z_init)
        tol = self.tol if tol is None else tol
        n_it = max_iter if tol is not None else self.steps
        self.last_iters = n_it
        for it in range(n_it):
            d = self.T(torch.sigmoid(z)) - h0
            infl = torch.zeros(B, self.G, device=dev).index_add_(1, self.dst, d[:, self.src] * w[None, :])
            z_new = torch.where(M, zc, (1 - self.damping) * z + self.damping * (z0 + infl))
            if tol is not None and float((z_new - z).abs().max()) < tol:
                z = z_new
                self.last_iters = it + 1
                break
            z = z_new
        return z, z0, M

    def forward(self, tgt, zeta):
        z, z0, M = self.equilibrium(tgt, zeta)
        return torch.where(M, torch.zeros(()), self.decoder()[None, :] * (z - z0))
