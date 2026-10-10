"""Honest evaluation metrics for CRISPR perturbation prediction."""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def pearson_rows(
    a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Pearson correlation per row.

    Args:
        a, b: (P, G) arrays
        mask: (P, G) bool, compute correlation only where True per row

    Returns:
        (P,) Pearson r per row. Zero-variance rows return 0.0, never NaN.
    """
    P = a.shape[0]
    result = np.zeros(P)

    for p in range(P):
        if mask is not None:
            idx = mask[p]
        else:
            idx = np.ones(a.shape[1], dtype=bool)

        a_p = a[p, idx]
        b_p = b[p, idx]

        a_std = np.std(a_p)
        b_std = np.std(b_p)

        if a_std > 0 and b_std > 0:
            result[p] = np.corrcoef(a_p, b_p)[0, 1]
        else:
            result[p] = 0.0

    return result


def exclude_target_mask(target_idx: np.ndarray, G: int) -> np.ndarray:
    """Boolean mask excluding target columns.

    Args:
        target_idx: (P,) int, column index of target (-1 if off-axis)
        G: number of genes

    Returns:
        (P, G) bool, False at each row's target column.
    """
    P = len(target_idx)
    mask = np.ones((P, G), dtype=bool)

    for p in range(P):
        if target_idx[p] >= 0:
            mask[p, target_idx[p]] = False

    return mask


def pcc_delta(
    pred: np.ndarray, obs: np.ndarray, target_idx: np.ndarray
) -> np.ndarray:
    """Pearson correlation excluding perturbed gene.

    Args:
        pred, obs: (P, G) predictions and observations
        target_idx: (P,) column index of perturbed gene

    Returns:
        (P,) Pearson r per perturbation, excluding target.
    """
    mask = exclude_target_mask(target_idx, pred.shape[1])
    return pearson_rows(pred, obs, mask)


def pcc_delta_centered(
    pred: np.ndarray, obs: np.ndarray, train_mean: np.ndarray, target_idx: np.ndarray
) -> np.ndarray:
    """Pearson after subtracting training mean (Systema-style centering).

    Args:
        pred, obs: (P, G) predictions and observations
        train_mean: (G,) mean of training-set observations
        target_idx: (P,) column index of perturbed gene

    Returns:
        (P,) centered Pearson r per perturbation, excluding target.
    """
    pred_c = pred - train_mean[np.newaxis, :]
    obs_c = obs - train_mean[np.newaxis, :]
    return pcc_delta(pred_c, obs_c, target_idx)


def top_k_pcc(
    pred: np.ndarray,
    obs: np.ndarray,
    target_idx: np.ndarray,
    k: int = 20,
    centered_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pearson restricted to top k genes by |obs|.

    Args:
        pred, obs: (P, G) predictions and observations
        target_idx: (P,) column index of perturbed gene
        k: number of top genes to use
        centered_mean: if given, subtract from both pred and obs first

    Returns:
        (P,) Pearson r on top-k genes per perturbation, excluding target.
    """
    P, G = pred.shape

    if centered_mean is not None:
        pred = pred - centered_mean[np.newaxis, :]
        obs = obs - centered_mean[np.newaxis, :]

    mask = np.zeros((P, G), dtype=bool)

    for p in range(P):
        # Get absolute values, excluding target
        abs_obs = np.abs(obs[p, :])
        if target_idx[p] >= 0:
            abs_obs = abs_obs.copy()
            abs_obs[target_idx[p]] = -np.inf  # Exclude target

        # Find top k indices
        top_idx = np.argsort(-abs_obs)[:k]
        mask[p, top_idx] = True

    return pearson_rows(pred, obs, mask)


def rmse_delta(
    pred: np.ndarray, obs: np.ndarray, target_idx: np.ndarray
) -> np.ndarray:
    """RMSE excluding perturbed gene.

    Args:
        pred, obs: (P, G) predictions and observations
        target_idx: (P,) column index of perturbed gene

    Returns:
        (P,) RMSE per perturbation, excluding target.
    """
    mask = exclude_target_mask(target_idx, pred.shape[1])
    P = pred.shape[0]
    rmse = np.zeros(P)

    for p in range(P):
        idx = mask[p]
        diff = pred[p, idx] - obs[p, idx]
        rmse[p] = np.sqrt(np.mean(diff ** 2))

    return rmse


def deg_auprc(
    pred: np.ndarray, deg_labels: np.ndarray, target_idx: np.ndarray
) -> Dict:
    """Average precision using score=|pred|, excluding target gene.

    Args:
        pred: (P, G) predictions (use absolute value as score)
        deg_labels: (P, G) binary labels (1=DE, 0=non-DE)
        target_idx: (P,) column index of perturbed gene

    Returns:
        dict with:
            - ap: (P,) average precision per perturbation (NaN if 0 or all positives)
            - prevalence: (P,) fraction of positives per row
            - n_skipped: int, number of rows where AP could not be computed
            - precision_at_k: dict with k->array of precision@k
    """
    mask = exclude_target_mask(target_idx, pred.shape[1])
    P = pred.shape[0]

    ap = np.full(P, np.nan)
    prevalence = np.zeros(P)
    n_skipped = 0

    for p in range(P):
        idx = mask[p]
        y_true = deg_labels[p, idx].astype(int)
        y_score = np.abs(pred[p, idx])

        n_pos = np.sum(y_true)
        n_neg = np.sum(1 - y_true)

        prevalence[p] = n_pos / len(y_true) if len(y_true) > 0 else 0

        if n_pos == 0 or n_neg == 0:
            n_skipped += 1
        else:
            ap[p] = average_precision_score(y_true, y_score)

    return {
        "ap": ap,
        "prevalence": prevalence,
        "n_skipped": n_skipped,
    }


def precision_at_k(
    pred: np.ndarray, deg_labels: np.ndarray, target_idx: np.ndarray, k: int = 50
) -> np.ndarray:
    """Precision@k using score=|pred|, excluding target gene.

    Args:
        pred: (P, G) predictions (use absolute value as score)
        deg_labels: (P, G) binary labels
        target_idx: (P,) column index of perturbed gene
        k: number of top predictions to consider

    Returns:
        (P,) precision@k per perturbation (NaN if no positives in row).
    """
    mask = exclude_target_mask(target_idx, pred.shape[1])
    P = pred.shape[0]
    prec_at_k = np.full(P, np.nan)

    for p in range(P):
        idx = mask[p]
        y_true = deg_labels[p, idx].astype(int)
        y_score = np.abs(pred[p, idx])

        if np.sum(y_true) == 0:
            continue

        sorted_idx = np.argsort(-y_score)[:k]
        prec_at_k[p] = np.mean(y_true[sorted_idx])

    return prec_at_k


def centroid_accuracy(
    pred: np.ndarray, obs: np.ndarray, train_mean: np.ndarray, target_idx: np.ndarray
) -> Dict:
    """Cross-perturbation accuracy (centroid matching).

    For each perturbation p, compute centered Pearson (subtract train_mean from pred and obs,
    exclude ALL target genes from columns used) between pred[p] and obs[q] for all q.

    Args:
        pred, obs: (P, G) predictions and observations
        train_mean: (G,) training mean
        target_idx: (P,) column index of perturbed gene per perturbation

    Returns:
        dict with:
            - top1: fraction of perturbations where obs[p] is top correlation
            - norm_rank: mean normalized rank (1.0 perfect, 0.5 chance)
            - norm_rank_each: (P,) normalized rank per perturbation
    """
    P, G = pred.shape

    # Subtract training mean
    pred_c = pred - train_mean[np.newaxis, :]
    obs_c = obs - train_mean[np.newaxis, :]

    # Create mask excluding all target genes
    exclude_cols = np.zeros(G, dtype=bool)
    for p in range(P):
        if target_idx[p] >= 0:
            exclude_cols[target_idx[p]] = True

    keep_cols = ~exclude_cols

    pred_c = pred_c[:, keep_cols]
    obs_c = obs_c[:, keep_cols]

    # Compute correlation between all pairs
    corr_matrix = np.zeros((P, P))
    for p in range(P):
        a_std = np.std(pred_c[p, :])
        for q in range(P):
            b_std = np.std(obs_c[q, :])
            if a_std > 0 and b_std > 0:
                corr_matrix[p, q] = np.corrcoef(pred_c[p, :], obs_c[q, :])[0, 1]
            else:
                corr_matrix[p, q] = 0.0

    # For each p, rank its correlation with obs
    top1_count = 0
    norm_ranks = np.zeros(P)

    for p in range(P):
        correlations = corr_matrix[p, :]
        # Rank in descending order; ties broken pessimistically (higher rank)
        ranks = rankdata(-correlations, method="max")
        rank_of_p = ranks[p]

        if rank_of_p == 1:
            top1_count += 1

        # Normalized rank: 1 - (rank - 1) / (P - 1)
        # rank=1 -> norm_rank=1, rank=P -> norm_rank=0
        norm_ranks[p] = 1.0 - (rank_of_p - 1) / (P - 1) if P > 1 else 1.0

    return {
        "top1": top1_count / P,
        "norm_rank": np.mean(norm_ranks),
        "norm_rank_each": norm_ranks,
    }


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict:
    """Bootstrap CI and p-value for paired differences.

    Resamples targets (rows) with replacement, ignoring NaN pairs.

    Args:
        a, b: (P,) or (P, G) arrays of measurements
        n_boot: number of bootstrap samples
        seed: random seed

    Returns:
        dict with:
            - mean_diff: mean of a - b
            - ci_low, ci_high: 2.5/97.5 percentiles of bootstrap mean diff
            - p_two_sided: two-sided p-value (capped at 1.0)
            - n: number of valid (non-NaN) pairs used
            - win_frac: fraction where a > b
    """
    rng = np.random.RandomState(seed)

    # Flatten and remove NaN pairs
    a_flat = a.ravel()
    b_flat = b.ravel()
    valid = ~(np.isnan(a_flat) | np.isnan(b_flat))

    a_valid = a_flat[valid]
    b_valid = b_flat[valid]
    n = len(a_valid)

    diff_obs = a_valid - b_valid
    mean_diff_obs = np.mean(diff_obs)

    # Bootstrap
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(diff_obs[idx])

    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    # Two-sided bootstrap p-value for H0: mean difference == 0
    p_two_sided = min(2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0)), 1.0)

    win_frac = np.mean(a_valid > b_valid)

    return {
        "mean_diff": mean_diff_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_two_sided": p_two_sided,
        "n": n,
        "win_frac": win_frac,
    }
