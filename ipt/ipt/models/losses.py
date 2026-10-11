"""Losses aligned with the evaluation metric (centred, per-perturbation, perturbed gene excluded)."""
from __future__ import annotations
import torch

EPS = 1e-8


def _mask(y, tgt):
    m = torch.ones_like(y, dtype=torch.bool)
    m[torch.arange(len(tgt)), tgt] = False
    return m


def centred_corr_loss(pred, y, tgt):
    """1 - Pearson(pred_row, y_row) over genes without the perturbed gene, averaged over rows.
    Scale-invariant, so small responses keep a usable gradient (no vanishing with tiny effects)."""
    m = _mask(y, tgt).float()
    n = m.sum(1, keepdim=True)
    pm = (pred * m).sum(1, keepdim=True) / n
    ym = (y * m).sum(1, keepdim=True) / n
    pc, yc = (pred - pm) * m, (y - ym) * m
    cov = (pc * yc).sum(1)
    den = torch.sqrt((pc ** 2).sum(1) + EPS) * torch.sqrt((yc ** 2).sum(1) + EPS)
    return (1 - cov / den).mean()


def normalised_mse(pred, y, tgt):
    """MSE after dividing each row by the observed row scale (every perturbation contributes comparably)."""
    m = _mask(y, tgt).float()
    n = m.sum(1)
    scale = torch.sqrt(((y ** 2) * m).sum(1) / n + EPS)[:, None]
    return ((((pred - y) / scale) ** 2) * m).sum() / m.sum()


def huber(pred, y, tgt, delta=1.0):
    m = _mask(y, tgt).float()
    scale = torch.sqrt(((y ** 2) * m).sum(1) / m.sum(1) + EPS)[:, None]
    r = (pred - y) / scale
    q = torch.where(r.abs() <= delta, 0.5 * r ** 2, delta * (r.abs() - 0.5 * delta))
    return (q * m).sum() / m.sum()


def combined(pred, y, tgt, w_corr=1.0, w_huber=0.1):
    return w_corr * centred_corr_loss(pred, y, tgt) + w_huber * huber(pred, y, tgt)


def plain_mse(pred, y, tgt):
    m = _mask(y, tgt).float()
    return (((pred - y) ** 2) * m).sum() / m.sum()


LOSSES = {"mse": plain_mse, "corr": centred_corr_loss, "nmse": normalised_mse, "huber": huber, "corr+huber": combined}
