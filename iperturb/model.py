"""
GRNN — Gene Regulatory Neural Network.

Architecture
────────────
Each gene j has three kinetic parameters (always learned):
  V_j    — maximum expression (log-parameterised)
  α_j    — degradation rate   (log-parameterised)
  b_j    — basal transcription (softplus-parameterised)

Each directed edge i→j has:
  K_d_ij — Hill half-saturation constant (softplus-parameterised)
  n_ij   — Hill coefficient ∈ [1, 4]     (sigmoid-parameterised)
  ŵ_ij   — effective weight (sign-dependent parameterisation below)

Weight parameterisation
───────────────────────
  sign ∈ {+1, −1}  →  ŵ = sign · sigmoid(w_raw)
    • sigmoid maps w_raw → (0,1); multiplying by sign fixes direction.
    • Only the magnitude is a free parameter.
    • sign=+1: ŵ ∈ (0,+1)   (activation constrained by DB)
    • sign=−1: ŵ ∈ (−1, 0)  (repression constrained by DB)

  sign = 0          →  ŵ = tanh(w_raw) ∈ (−1,+1)
    • Both sign and magnitude are free parameters learned from data.

Kinetic-parameter initialisation
─────────────────────────────────
  log_V   = log(x0_j)             max expression ≈ control baseline
  log_α   = −2.3                  slow degradation prior
  b_raw   = −4.0                  near-zero basal transcription
  Kd_raw  = inv_softplus(x0_src)  φ(x0_src) ≈ 0.5 (midpoint of Hill curve)
  n_raw   = 0                     Hill coeff starts at ≈ 2.5

Forward pass
────────────
Dual-channel Picard iteration to steady state:

    drive  = ŵ · φ(x_src)           (signed regulatory drive)
    h_act  = Σ max(0,  drive)        (activation drive,    always ≥ 0)
    h_rep  = Σ max(0, −drive)        (repression magnitude, always ≥ 0)
    x_ss_j = (V_j · (1 + h_act_j) + b_j) / (1 + α_j + h_rep_j)

Both activation and repression channels have guaranteed non-zero gradients.
Convergence is monitored empirically (n_iters == max_iter signals failure).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import inv_softplus

log = logging.getLogger("iperturb")


class GRNN(nn.Module):
    """Gene Regulatory Neural Network."""

    def __init__(
        self,
        gene_names: list[str],
        src_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        signs: torch.Tensor,
        levels: torch.Tensor,
        x0_init: torch.Tensor,
        max_iter: int = 100,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.gene_names = gene_names
        self.gene2idx   = {g: i for i, g in enumerate(gene_names)}
        self.N          = len(gene_names)
        self.E          = src_idx.numel()
        self.max_iter   = max_iter
        self.eps        = eps

        self.register_buffer("src_idx", src_idx.long())
        self.register_buffer("tgt_idx", tgt_idx.long())
        self.register_buffer("signs",   signs.float())    # {−1, 0, +1}
        self.register_buffer("levels",  levels.float())   # {1, 2, 3, 4}

        log_x0         = torch.log(x0_init.clamp(min=1e-6))
        self.log_V     = nn.Parameter(log_x0.clone())
        self.log_alpha = nn.Parameter(torch.full((self.N,), -2.3))
        self.b_raw     = nn.Parameter(torch.full((self.N,), -4.0))

        self.Kd_raw = nn.Parameter(inv_softplus(x0_init[src_idx].clamp(min=1e-3)))
        self.n_raw  = nn.Parameter(torch.zeros(self.E))

        # Biologically-informed w_raw initialisation
        def _inv_sig(p: float) -> float:
            return float(np.log(p / (1.0 - p)))

        w_init = torch.zeros(self.E)
        act = signs ==  1;  rep = signs == -1;  unk = signs == 0
        l1  = levels == 1;  l2  = levels == 2;  l3  = levels == 3;  l4  = levels == 4

        w_init[act & l1] = _inv_sig(0.85);  w_init[act & l2] = _inv_sig(0.65)
        w_init[act & l3] = _inv_sig(0.45);  w_init[act & l4] = _inv_sig(0.25)
        w_init[rep & l1] = _inv_sig(0.90);  w_init[rep & l2] = _inv_sig(0.70)
        w_init[rep & l3] = _inv_sig(0.50);  w_init[rep & l4] = _inv_sig(0.30)
        w_init[unk]      = torch.randn(unk.sum()) * 0.1

        self.w_raw = nn.Parameter(w_init)

    # ── Derived quantities ────────────────────────────────────────────────────
    @property
    def V(self):
        return torch.exp(self.log_V)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def b(self):
        return F.softplus(self.b_raw)

    @property
    def Kd(self):
        return F.softplus(self.Kd_raw) + 1e-6

    @property
    def n(self):
        return 1.0 + 3.0 * torch.sigmoid(self.n_raw)

    @property
    def w_hat(self) -> torch.Tensor:
        """Effective weight (sign-constrained or fully free)."""
        is_signed = self.signs != 0
        return torch.where(
            is_signed,
            self.signs * torch.sigmoid(self.w_raw),
            torch.tanh(self.w_raw),
        )

    @property
    def eff_weight(self) -> np.ndarray:
        """eff_ij = ŵ_ij · φ_ij(x0_src) — regulatory drive at baseline."""
        with torch.no_grad():
            x0_src = self.V[self.src_idx] * 0.8
            xn     = x0_src.pow(self.n)
            Kdn    = self.Kd.pow(self.n)
            phi    = xn / (Kdn + xn + 1e-12)
            return (self.w_hat * phi).cpu().numpy()

    def eff_map(self) -> dict[tuple[str, str], float]:
        """Return a {(source, target): eff_weight} dict."""
        arr = self.eff_weight
        return {
            (self.gene_names[int(s)], self.gene_names[int(t)]): float(w)
            for s, t, w in zip(self.src_idx.cpu(), self.tgt_idx.cpu(), arr)
        }

    # ── Forward (Picard iteration) ────────────────────────────────────────────
    def _step(self, x: torch.Tensor) -> torch.Tensor:
        """One Picard iteration step (dual-channel)."""
        x_src  = x[self.src_idx]
        xn     = x_src.pow(self.n)
        Kdn    = self.Kd.pow(self.n)
        phi    = xn / (Kdn + xn + 1e-12)
        drive  = self.w_hat * phi

        h_act = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_rep = torch.zeros(self.N, device=x.device, dtype=x.dtype)
        h_act.scatter_add_(0, self.tgt_idx, drive.clamp(min=0))
        h_rep.scatter_add_(0, self.tgt_idx, (-drive).clamp(min=0))

        return (self.V * (1.0 + h_act) + self.b) / (1.0 + self.alpha + h_rep)

    def forward(
        self,
        x0: torch.Tensor,
        perturbed_idx: int | None = None,
        perturbed_value: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Run Picard iteration to steady state.

        Returns
        -------
        (x_ss, n_iters)
          n_iters == max_iter indicates non-convergence; callers should log this.
        """
        x = x0.clone()
        if perturbed_idx is not None and perturbed_value is not None:
            x[perturbed_idx] = perturbed_value
        for t in range(self.max_iter):
            x_new = self._step(x)
            if perturbed_idx is not None and perturbed_value is not None:
                x_new[perturbed_idx] = perturbed_value
            if (x_new - x).norm() < self.eps:
                return x_new, t + 1
            x = x_new
        return x, self.max_iter


# ── Factory from TSV ──────────────────────────────────────────────────────────

def grn_tsv_to_grnn(
    tsv_path: str,
    gene_names: list[str],
    x0: np.ndarray,
    max_iter: int = 100,
    eps: float = 1e-5,
) -> tuple[GRNN, "pd.DataFrame"]:
    """
    Build a GRNN from a GRN TSV file.

    TSV columns expected: source  target  level  sign  db

    Returns
    -------
    (model, grn_df)
    """
    import pandas as pd

    df       = pd.read_csv(tsv_path, sep="\t")
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    df       = df[df["source"].isin(gene2idx) & df["target"].isin(gene2idx)].copy()

    src_idx = torch.tensor([gene2idx[g] for g in df["source"]], dtype=torch.long)
    tgt_idx = torch.tensor([gene2idx[g] for g in df["target"]], dtype=torch.long)
    signs   = torch.tensor(df["sign"].fillna(0).astype(int).values,  dtype=torch.float32)
    levels  = torch.tensor(df["level"].fillna(4).astype(int).values, dtype=torch.float32)
    x0_t    = torch.tensor(x0, dtype=torch.float32)

    model   = GRNN(gene_names, src_idx, tgt_idx, signs, levels, x0_t, max_iter, eps)
    n_unk   = int((signs == 0).sum())
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "GRNN: %d genes | %d edges (%d sign=0 / %d DB-constrained) | %d params",
        len(gene_names), len(df), n_unk, len(df) - n_unk, n_params,
    )
    return model, df
