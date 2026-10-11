"""Anti-vanishing propagation: state in log-odds (no output saturation), leaky Hill transfer, skip/damped update.

z_i <- (1-eta) z_i + eta [ z0_i + sum_j w_ji (T(sigmoid z_j) - T(x0_j)) ],  T(x) = (1-leak) H(x;k) + leak * x
Output = d_i (z*_i - z0_i): the response is not squashed by a sigmoid, the leak keeps dT/dx > 0 where the Hill curve
is flat, weights are initialised to a target spectral gain, and the clamp is imposed in log-odds at every update.
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


class GraphNet2(nn.Module):
    def __init__(self, src, dst, sign, conf, x0, transfer="hill4", leak=0.1, n=2.0, k_init=0.3, steps=24, damping=0.5,
                 gain=0.9, d_init=0.3):
        super().__init__()
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long))
        self.register_buffer("sign", torch.as_tensor(sign, dtype=torch.float32))
        self.register_buffer("conf", torch.as_tensor(conf, dtype=torch.float32))
        x0 = torch.as_tensor(x0, dtype=torch.float32).clamp(EPS, 1 - EPS)
        self.register_buffer("x0", x0)
        self.G, self.transfer, self.leak, self.n, self.steps, self.damping = len(x0), transfer, leak, n, steps, damping
        self.a = nn.Parameter(torch.tensor(0.0))
        self.rho = nn.Parameter(torch.tensor(float(np.log(gain / (2 - gain)))))   # gain*sigmoid-> init spectral gain
        self.kz = nn.Parameter(torch.tensor(float(np.log((k_init - 0.05) / (1.0 - k_init)))))
        self.u = nn.Parameter(torch.zeros(self.G))
        self.d0 = nn.Parameter(torch.tensor(float(np.log(np.expm1(d_init)))))

    def k(self):
        return 0.05 + 0.95 * torch.sigmoid(self.kz)

    def weights(self):
        wt = self.sign * self.conf * Fn.softplus(self.a + 0.5413)
        inc = torch.zeros(self.G, device=wt.device).index_add_(0, self.dst, wt.abs())
        return 2.0 * torch.sigmoid(self.rho) * wt / torch.clamp(inc[self.dst], min=1.0)

    def T(self, x):
        if self.transfer == "linear":
            return x
        return (1 - self.leak) * hill4(x, self.k(), self.n) + self.leak * x

    def decoder(self):
        return Fn.softplus(self.d0 + self.u)

    def equilibrium(self, tgt, zeta):
        B = len(tgt)
        dev = self.x0.device
        z0 = logit(self.x0)[None, :].expand(B, -1)
        M = torch.zeros(B, self.G, dtype=torch.bool, device=dev)
        M[torch.arange(B, device=dev), tgt] = True
        zc = logit(((1 - zeta) * self.x0[tgt]).clamp(EPS, 1 - EPS))[:, None].expand(B, self.G)
        w = self.weights()
        h0 = self.T(self.x0)[None, :]
        z = torch.where(M, zc, z0)
        for _ in range(self.steps):
            d = self.T(torch.sigmoid(z)) - h0
            infl = torch.zeros(B, self.G, device=dev).index_add_(1, self.dst, d[:, self.src] * w[None, :])
            z = torch.where(M, zc, (1 - self.damping) * z + self.damping * (z0 + infl))
        return z, z0, M

    def forward(self, tgt, zeta):
        z, z0, M = self.equilibrium(tgt, zeta)
        out = torch.where(M, torch.zeros(()), self.decoder()[None, :] * (z - z0))
        return out
