"""
Training, evaluation, and metrics for GRNN.

Loss function components
────────────────────────
  WMSE     — Weighted MSE: w_i ∝ |Δobs_i|+ε, normalised to sum=1.
             Strongly DE genes contribute more; reduces mode collapse.
  AFDA     — Autofocus Direction-Aware Loss (focal-style).
             Applied to genes with |Δobs| ≥ 0.1.
             L_afda = mean[(1−agreement)^γ · (Δpred − Δobs)²]
  Balance  — Sign-fraction alignment: prevents systematic activation bias.
             L_balance = (frac_pred_up − frac_obs_up)²
  Delta MSE — prevents Δ=0 shortcut
  DEG MSE  — top-k importance-weighted MSE on most-changed genes
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from .model import GRNN
    from .data import PerturbseqDataset

log = logging.getLogger("iperturb")


# ── Loss function ─────────────────────────────────────────────────────────────

def grnn_loss(
    x_pred: torch.Tensor,
    x_obs: torch.Tensor,
    x0: torch.Tensor,
    top_k: int = 20,
    lam_wmse: float = 1.0,
    lam_afda: float = 1.0,
    lam_delta: float = 1.0,
    lam_deg: float = 2.0,
    lam_balance: float = 0.5,
    gamma_focus: float = 2.0,
) -> torch.Tensor:
    """
    Multi-component GRNN loss.

    Components
    ----------
    WMSE   : importance-weighted MSE on absolute expression.
    AFDA   : autofocus direction-aware focal loss on Δ genes (|Δ| ≥ 0.1).
    Balance: soft up-fraction alignment to prevent sign bias.
    Delta  : plain MSE on Δ = x − x0 (prevents Δ=0 shortcut).
    DEG    : top-k weighted MSE on most strongly perturbed genes.
    """
    d_obs  = x_obs  - x0
    d_pred = x_pred - x0

    # WMSE
    w_obs     = d_obs.abs() + 1e-6
    w_obs     = w_obs / w_obs.sum()
    loss_wmse = (w_obs * (x_pred - x_obs).pow(2)).sum()

    # AFDA
    sig_mask = d_obs.abs() >= 0.1
    if sig_mask.any():
        dp        = d_pred[sig_mask]
        do        = d_obs[sig_mask]
        cos_raw   = (dp * do) / ((dp.abs() + 1e-8) * (do.abs() + 1e-8))
        agreement = (1.0 + cos_raw) / 2.0
        focus_w   = (1.0 - agreement).clamp(0.0, 1.0).pow(gamma_focus)
        loss_afda = (focus_w * (dp - do).pow(2)).mean()
    else:
        loss_afda = x_pred.new_tensor(0.0)

    # Balance
    tau          = 0.05
    loss_balance = (torch.sigmoid(d_pred / tau).mean() - torch.sigmoid(d_obs / tau).mean()).pow(2)

    # Delta MSE
    loss_delta = F.mse_loss(d_pred, d_obs)

    # Weighted DEG MSE
    delta_obs = d_obs.abs()
    topk_idx  = delta_obs.topk(min(top_k, len(delta_obs))).indices
    weights   = delta_obs[topk_idx] / (delta_obs[topk_idx].sum() + 1e-12)
    loss_deg  = (weights * (x_pred[topk_idx] - x_obs[topk_idx]).pow(2)).sum()

    return (lam_wmse    * loss_wmse
            + lam_afda    * loss_afda
            + lam_balance * loss_balance
            + lam_delta   * loss_delta
            + lam_deg     * loss_deg)


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def _eval_loss(
    model: "GRNN",
    subset,
    x0: torch.Tensor,
    device: str,
    top_k: int,
    lam_wmse: float,
    lam_afda: float,
    lam_delta: float,
    lam_deg: float,
    lam_balance: float,
    gamma_focus: float,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
            x_obs  = batch["x_obs"][0].to(device)
            x_pred, _ = model(
                x0,
                perturbed_idx=int(batch["perturbed_idx"][0]),
                perturbed_value=float(batch["perturbed_value"][0]),
            )
            total += grnn_loss(x_pred, x_obs, x0,
                               top_k, lam_wmse, lam_afda,
                               lam_delta, lam_deg, lam_balance, gamma_focus).item()
    return total / max(len(subset), 1)


def sign_balance_report(
    model: "GRNN", subset, x0: torch.Tensor, device: str
) -> tuple[float, float]:
    """Returns (frac_pred_up, frac_obs_up) across the subset."""
    model.eval()
    pred_ups, obs_ups = [], []
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
            x_obs  = batch["x_obs"][0].to(device)
            x_pred, _ = model(
                x0,
                perturbed_idx=int(batch["perturbed_idx"][0]),
                perturbed_value=float(batch["perturbed_value"][0]),
            )
            d_pred = x_pred - x0;  d_obs = x_obs - x0
            pred_ups.append((d_pred > 0).float().mean().item())
            obs_ups.append( (d_obs  > 0).float().mean().item())
    return float(np.mean(pred_ups)), float(np.mean(obs_ups))


def convergence_report(
    model: "GRNN", subset, x0: torch.Tensor, device: str
) -> tuple[float, float]:
    """
    Returns (median_iters, frac_non_converged).
    frac_non_converged > 0.05 warrants investigation.
    """
    model.eval()
    iters = []
    with torch.no_grad():
        for batch in DataLoader(subset, batch_size=1, shuffle=False, num_workers=0):
            _, n = model(
                x0,
                perturbed_idx=int(batch["perturbed_idx"][0]),
                perturbed_value=float(batch["perturbed_value"][0]),
            )
            iters.append(n)
    arr = np.array(iters)
    return float(np.median(arr)), float((arr == model.max_iter).mean())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_grnn(
    model: "GRNN",
    train_ds,
    val_ds,
    n_epochs: int = 50,
    lr: float = 1e-3,
    top_k_deg: int = 20,
    lam_wmse: float = 1.0,
    lam_afda: float = 1.0,
    lam_delta: float = 1.0,
    lam_deg: float = 2.0,
    lam_balance: float = 0.5,
    gamma_focus: float = 2.0,
    device: str | None = None,
) -> tuple[list[float], list[float]]:
    """
    Train the GRNN with Adam + cosine annealing LR schedule.

    Returns
    -------
    (train_history, val_history) — per-epoch loss lists.
    """
    from tqdm import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model  = model.to(device)
    base   = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    x0     = base.x0.to(device)
    loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    train_h, val_h = [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", leave=False):
            x_obs = batch["x_obs"][0].to(device)
            opt.zero_grad()
            x_pred, _ = model(
                x0,
                perturbed_idx=int(batch["perturbed_idx"][0]),
                perturbed_value=float(batch["perturbed_value"][0]),
            )
            loss = grnn_loss(
                x_pred, x_obs, x0,
                top_k_deg, lam_wmse, lam_afda, lam_delta, lam_deg, lam_balance, gamma_focus,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        train_h.append(epoch_loss / len(loader))
        val_h.append(_eval_loss(
            model, val_ds, x0, device,
            top_k_deg, lam_wmse, lam_afda, lam_delta, lam_deg, lam_balance, gamma_focus,
        ))

        if epoch % 5 == 0 or epoch == 1:
            p_up, o_up   = sign_balance_report(model, val_ds, x0, device)
            med_it, f_nc = convergence_report(model, val_ds, x0, device)
            log.info(
                "Epoch %3d  train=%.4f  val=%.4f  "
                "pred_up=%.2f  obs_up=%.2f  bias=%.2f  "
                "med_iters=%.0f  non_conv=%.1f%%",
                epoch, train_h[-1], val_h[-1],
                p_up, o_up, p_up - o_up,
                med_it, 100 * f_nc,
            )

    return train_h, val_h


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_all(
    model: "GRNN",
    dataset,
    top_k_pearson: int = 20,
    lfc_threshold: float = 0.1,
    device: str = "cpu",
    batch_size: int = 1,
) -> dict:
    """
    Evaluate GRNN and return all iPerturb metrics.

    Metrics
    -------
    (a) pearson_deltaK_mean    — mean Pearson r on top-K DEGs per perturbation
    (a) pearson_delta_all_mean — mean Pearson r on all genes
    (b) centroid_accuracy      — fraction of samples with correct mean-Δ sign
    (c) mse_delta              — MSE on Δ = x − x0
    (d) directional_accuracy   — per-gene sign accuracy for |LFC| ≥ threshold
    """
    model.eval()
    model.to(device)

    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    x0   = base.x0.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    pearson_topk_list = []
    pearson_all_list  = []
    centroid_correct  = 0
    mse_delta_list    = []
    dir_correct = dir_total = 0

    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"][0].to(device)
            pidx  = int(batch["perturbed_idx"][0])
            pval  = float(batch["perturbed_value"][0])

            x_pred, _ = model(x0, perturbed_idx=pidx, perturbed_value=pval)

            d_obs  = (x_obs  - x0).cpu().numpy()
            d_pred = (x_pred - x0).cpu().numpy()

            # Pearson — all genes
            if d_obs.std() > 1e-8 and d_pred.std() > 1e-8:
                pearson_all_list.append(float(np.corrcoef(d_pred, d_obs)[0, 1]))
            else:
                pearson_all_list.append(float("nan"))

            # Pearson — top-k by |d_obs|
            k        = min(top_k_pearson, len(d_obs))
            topk_idx = np.argpartition(np.abs(d_obs), -k)[-k:]
            d_obs_k, d_pred_k = d_obs[topk_idx], d_pred[topk_idx]
            if d_obs_k.std() > 1e-8 and d_pred_k.std() > 1e-8:
                pearson_topk_list.append(float(np.corrcoef(d_pred_k, d_obs_k)[0, 1]))
            else:
                pearson_topk_list.append(float("nan"))

            # Centroid accuracy
            if np.sign(d_pred.mean()) == np.sign(d_obs.mean()) and d_obs.mean() != 0:
                centroid_correct += 1

            # MSE on Δ
            mse_delta_list.append(float(np.mean((d_pred - d_obs) ** 2)))

            # Directional accuracy
            mask = np.abs(d_obs) >= lfc_threshold
            if mask.sum() > 0:
                dir_correct += int((np.sign(d_pred[mask]) == np.sign(d_obs[mask])).sum())
                dir_total   += int(mask.sum())

    n = len(pearson_topk_list)
    if n == 0:
        nan = float("nan")
        return {
            f"pearson_delta{top_k_pearson}_mean": nan,
            "pearson_delta_all_mean":             nan,
            "centroid_accuracy":                  nan,
            "mse_delta":                          nan,
            "directional_accuracy":               nan,
            "directional_lfc_threshold":          lfc_threshold,
            "directional_n_gene_pert_pairs":      0,
        }

    def _nanmean(lst):
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")

    return {
        f"pearson_delta{top_k_pearson}_mean": _nanmean(pearson_topk_list),
        "pearson_delta_all_mean":             _nanmean(pearson_all_list),
        "centroid_accuracy":                  centroid_correct / n,
        "mse_delta":                          float(np.mean(mse_delta_list)),
        "directional_accuracy":               (dir_correct / dir_total
                                               if dir_total > 0 else float("nan")),
        "directional_lfc_threshold":          lfc_threshold,
        "directional_n_gene_pert_pairs":      dir_total,
    }


def subsample_evaluate(
    model: "GRNN",
    test_ds,
    n_runs: int = 10,
    subsample_frac: float = 0.30,
    top_k_pearson: int = 20,
    lfc_threshold: float = 0.1,
    device: str = "cpu",
    seeds: list[int] | None = None,
) -> list[dict]:
    """
    Evaluate GRNN on N random subsamples of test_ds.

    Returns a list of metric dicts (one per run), each with 'seed' and
    'n_eval_samples' added.
    """
    import torch.utils.data

    if seeds is None:
        seeds = list(range(n_runs))

    all_metrics = []
    P = len(test_ds)

    for seed in seeds:
        rng = np.random.default_rng(seed)
        m   = max(1, int(subsample_frac * P))
        idx = rng.choice(P, size=m, replace=False)
        sub = torch.utils.data.Subset(test_ds, idx)

        metrics = evaluate_all(
            model, sub,
            top_k_pearson=top_k_pearson,
            lfc_threshold=lfc_threshold,
            device=device,
        )
        metrics["seed"]           = seed
        metrics["n_eval_samples"] = m
        all_metrics.append(metrics)

        log.info(
            "Eval run %2d/%2d (seed=%d, n=%d)  Pearson Δ%d=%.4f  "
            "centroid=%.4f  MSE=%.5f  dir=%.4f",
            seeds.index(seed) + 1, len(seeds), seed, m,
            top_k_pearson,
            metrics[f"pearson_delta{top_k_pearson}_mean"],
            metrics["centroid_accuracy"],
            metrics["mse_delta"],
            metrics["directional_accuracy"],
        )

    return all_metrics
