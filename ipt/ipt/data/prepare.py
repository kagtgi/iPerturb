"""Per-perturbation mean-delta targets from a Replogle-style h5ad (dense integer counts).

Estimand (plan 3.3): Delta_p = mean_cells log1p(CP10K)_p - batch-matched mean log1p(CP10K)_control.
Size factor = library size on the stored gene axis. Streams cells in chunks; outputs one npz.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
import numpy as np
import scipy.sparse as sp

CTRL = "control"


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def normalize(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1
    return np.log1p(X / tot * scale).astype(np.float32)


def aggregate(chunks, pert_codes, batch_codes, n_pert, n_batch, n_genes, ctrl_code):
    """chunks: iterable of (start, X_dense). pert_codes/batch_codes: int arrays over all cells.
    Returns sums, sumsq (n_pert,G), counts (n_pert,n_batch), control per-batch sums (n_batch,G)."""
    S = np.zeros((n_pert, n_genes)); Q = np.zeros((n_pert, n_genes))
    N = np.zeros((n_pert, n_batch), dtype=np.int64)
    CB = np.zeros((n_batch, n_genes))
    for start, X in chunks:
        L = normalize(X)
        pc = pert_codes[start:start + len(L)]; bc = batch_codes[start:start + len(L)]
        M = sp.csr_matrix((np.ones(len(L)), (pc, np.arange(len(L)))), shape=(n_pert, len(L)))
        S += M @ L; Q += M @ (L.astype(np.float64) ** 2)
        np.add.at(N, (pc, bc), 1)
        c = pc == ctrl_code
        if c.any():
            B = sp.csr_matrix((np.ones(c.sum()), (bc[c], np.arange(c.sum()))), shape=(n_batch, c.sum()))
            CB += B @ L[c]
    return S, Q, N, CB


def finalize(S, Q, N, CB, ctrl_idx):
    """Return n, mean, unbiased var per perturbation, and batch-matched control mean per perturbation."""
    n = N.sum(1)
    mean = S / np.maximum(n, 1)[:, None]
    var = np.maximum(Q / np.maximum(n, 1)[:, None] - mean ** 2, 0) * (n / np.maximum(n - 1, 1))[:, None]
    cn = N[ctrl_idx].astype(np.float64); has = cn > 0
    cm_b = np.zeros_like(CB); cm_b[has] = CB[has] / cn[has][:, None]
    W = N.astype(np.float64) * has[None, :]
    wsum = W.sum(1)
    matched = np.where(wsum[:, None] > 0, (W @ cm_b) / np.maximum(wsum, 1e-12)[:, None], mean[ctrl_idx][None, :])
    return n, mean, var, matched, int(has.sum())


def run(h5ad: str, out: str, min_cells: int = 20, chunk: int = 20000):
    import h5py
    h = h5py.File(h5ad, "r")
    o = h["obs"]

    def col(k):
        c = o[k]
        return _dec(np.array(c["categories"])[np.array(c["codes"])]) if isinstance(c, h5py.Group) else _dec(np.array(c))

    genes = _dec(np.array(h["var"]["gene_name"]))
    pert = col("perturbation"); batch = col("batch")
    perts, pc = np.unique(pert, return_inverse=True)
    batches, bc = np.unique(batch, return_inverse=True)
    X = h["X"]; n_cells, G = X.shape
    gen = ((s, np.asarray(X[s:s + chunk])) for s in range(0, n_cells, chunk))
    ci = int(np.where(perts == CTRL)[0][0])
    gen = ((s, np.asarray(X[s:s + chunk])) for s in range(0, n_cells, chunk))
    S, Q, N, CB = aggregate(gen, pc, bc, len(perts), len(batches), G, ci)
    n, mean, var, matched, n_cb = finalize(S, Q, N, CB, ci)
    delta = (mean - matched).astype(np.float32)
    keep = np.array([(p != CTRL) and n[i] >= min_cells for i, p in enumerate(perts)])
    gidx = {g: i for i, g in enumerate(genes)}
    tgt = np.array([gidx.get(p, -1) for p in perts])
    np.savez_compressed(
        out, genes=genes.astype(str), perts=perts[keep].astype(str), n_cells=n[keep], delta=delta[keep],
        var_p=var[keep].astype(np.float32), mean_lp=mean[keep].astype(np.float32),
        matched_ctrl=matched[keep].astype(np.float32), target_idx=tgt[keep],
        ctrl_mean=mean[ci].astype(np.float32), ctrl_var=var[ci].astype(np.float32), ctrl_n=int(n[ci]),
        n_ctrl_batches=n_cb, min_cells=min_cells,
    )
    meta = dict(src=h5ad, n_cells=int(n_cells), n_genes=int(G), n_perts_kept=int(keep.sum()),
                on_axis=int((tgt[keep] >= 0).sum()), ctrl_cells=int(n[ci]), ctrl_batches=n_cb)
    print(json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("h5ad"); ap.add_argument("out")
    ap.add_argument("--min-cells", type=int, default=20)
    a = ap.parse_args(); run(a.h5ad, a.out, a.min_cells)
