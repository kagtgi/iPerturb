"""Network-propagation kernels (Cowen et al. 2017) used to propagate TRAINING RESPONSES to unseen perturbed genes.

S_pj = amount of 'fluid' from seed p that reaches j.  kernels: rwr (random walk with restart / personalised PageRank),
heat (diffusion kernel exp(-tL)), trunc (K-step random walk), onehop (neighbour mean).
Normalisation: 'rw'  W = A D^-1  (column-stochastic, fluid conserving)  or  'sym'  W = D^-1/2 A D^-1/2.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, expm_multiply


def normalise(A: sp.csr_matrix, kind: str = "rw") -> sp.csr_matrix:
    d = np.asarray(A.sum(0)).ravel()
    if kind == "rw":
        return (A @ sp.diags(1.0 / np.maximum(d, 1e-12))).tocsr()
    dm = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (sp.diags(dm) @ A @ sp.diags(dm)).tocsr()


def kernel_scores(A, seeds, kernel="rwr", alpha=0.5, norm="rw", t=1.0, steps=2):
    """Return dense (len(seeds), G) matrix; row i = propagation scores from seed gene seeds[i]."""
    G = A.shape[0]
    P0 = np.zeros((G, len(seeds)))
    P0[np.asarray(seeds), np.arange(len(seeds))] = 1.0
    if kernel == "onehop":
        W = normalise(A, "rw")
        return (W @ P0).T
    if kernel == "trunc":
        W = normalise(A, norm)
        P, acc = P0, np.zeros_like(P0)
        for _ in range(steps):
            P = W @ P
            acc += P
        return acc.T
    if kernel == "heat":
        d = np.asarray(A.sum(0)).ravel()
        L = sp.diags(d) - A
        return expm_multiply(-t * L.tocsc(), P0).T
    W = normalise(A, norm)
    M = (sp.eye(G, format="csc") - (1 - alpha) * W).tocsc()
    return (alpha * splu(M).solve(P0)).T


def propagate_responses(S, seed_index, train_rows, Ytrain, gamma=0.0, deg=None, exclude_self=True):
    """Delta_hat_p = sum_j S_pj / deg_j^gamma * Delta_j / sum_j (...) over training targets j (train_rows = gene index of each training row)."""
    w = S[:, train_rows].copy()
    if gamma and deg is not None:
        w = w / np.maximum(deg[train_rows], 1.0)[None, :] ** gamma
    if exclude_self:
        for i, p in enumerate(seed_index):
            w[i, np.where(train_rows == p)[0]] = 0.0
    tot = w.sum(1, keepdims=True)
    ok = (tot[:, 0] > 1e-12)
    out = np.zeros((len(S), Ytrain.shape[1]), np.float32)
    out[ok] = (w[ok] @ Ytrain) / tot[ok]
    return out, ok
