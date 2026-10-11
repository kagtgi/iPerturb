"""Careful preprocessing of a Replogle-style Perturb-seq h5ad (dense integer counts) -> perturbation effects with uncertainty.

Decisions (see paper, Methods 'Preprocessing'):
* cell QC per gem group with 5-MAD outlier rules on log UMI and log genes (both sides) and on mitochondrial fraction (upper side only);
  no universal mito cutoff (knockdowns can change metabolism); single-guide-target cells only (nperts == 1);
* NO batch correction, NO imputation, NO cell-cycle regression; control cells matched by gem group;
* effect of perturbation p: Delta (difference of mean log1p CP10K vs batch-matched controls) with a Welch standard error, and a
  pseudobulk log2 fold change computed from summed raw UMIs (library-size normalised, pseudocount); cells are not treated as replicates
  for inference -- the SE is descriptive and used only for precision weighting;
* knockdown efficiency  k_p = 1 - mean_lin(target | p) / mean_lin(target | matched control)  (linear CP10K scale), reported only when the target is
  detected in controls (>= 0.05 CP10K = 5 CPM), otherwise flagged and left unknown;
* everything fitted later (gene sets, edges, hyper-parameters) uses training perturbations only.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import scipy.sparse as sp

CTRL = "control"
MAD_K = 5.0


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def mad_mask(values, groups, k=MAD_K, two_sided=True):
    """True = keep. Outliers beyond k MADs of the group median (upper side only if two_sided is False)."""
    keep = np.ones(len(values), bool)
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        v = values[i]
        med = np.median(v)
        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-12
        z = (v - med) / mad
        keep[i] = (np.abs(z) <= k) if two_sided else (z <= k)
    return keep


def cell_qc(umi, genes, mito, batch, nperts=None):
    """Returns the keep mask and a dict of removal counts."""
    k1 = mad_mask(np.log1p(umi), batch)
    k2 = mad_mask(np.log1p(genes), batch)
    k3 = mad_mask(mito, batch, two_sided=False)
    k4 = np.ones(len(umi), bool) if nperts is None else (np.asarray(nperts) <= 1)
    keep = k1 & k2 & k3 & k4
    return keep, dict(n=len(keep), kept=int(keep.sum()), fail_umi=int((~k1).sum()), fail_genes=int((~k2).sum()), fail_mito=int((~k3).sum()),
                      fail_multiguide=int((~k4).sum()))


def normalise(X, scale=1e4):
    tot = X.sum(1, keepdims=True)
    tot[tot == 0] = 1
    lin = X / tot * scale
    return np.log1p(lin).astype(np.float32), lin.astype(np.float32), tot[:, 0]


def aggregate2(chunks, pert_codes, batch_codes, keep, n_pert, n_batch, n_genes, ctrl_code):
    """Accumulate log1p sums/sums of squares, linear CP10K sums, raw count sums and library sizes per perturbation,
    per-(pert,batch) cell counts, and per-batch control sums (log, linear, raw, library). Masked cells are skipped."""
    S = np.zeros((n_pert, n_genes)); Q = np.zeros((n_pert, n_genes)); SL = np.zeros((n_pert, n_genes)); SR = np.zeros((n_pert, n_genes))
    LIB = np.zeros(n_pert)
    N = np.zeros((n_pert, n_batch), dtype=np.int64)
    CB = np.zeros((n_batch, n_genes)); CBL = np.zeros((n_batch, n_genes)); CBR = np.zeros((n_batch, n_genes)); CBLIB = np.zeros(n_batch)
    for start, X in chunks:
        m = keep[start:start + len(X)]
        if not m.any():
            continue
        X = np.asarray(X)[m]
        L, lin, tot = normalise(X.astype(np.float32))
        pc = pert_codes[start:start + len(m)][m]
        bc = batch_codes[start:start + len(m)][m]
        M = sp.csr_matrix((np.ones(len(L)), (pc, np.arange(len(L)))), shape=(n_pert, len(L)))
        S += M @ L; Q += M @ (L.astype(np.float64) ** 2); SL += M @ lin; SR += M @ X.astype(np.float64)
        LIB += M @ tot
        np.add.at(N, (pc, bc), 1)
        c = pc == ctrl_code
        if c.any():
            B = sp.csr_matrix((np.ones(c.sum()), (bc[c], np.arange(c.sum()))), shape=(n_batch, c.sum()))
            CB += B @ L[c]; CBL += B @ lin[c]; CBR += B @ X[c].astype(np.float64); CBLIB += B @ tot[c]
    return dict(S=S, Q=Q, SL=SL, SR=SR, LIB=LIB, N=N, CB=CB, CBL=CBL, CBR=CBR, CBLIB=CBLIB)


def finalize2(A, ctrl_idx, tgt_idx, min_cells=20, det_thresh=0.05):
    N, S, Q = A["N"], A["S"], A["Q"]
    n = N.sum(1)
    nn = np.maximum(n, 1)
    mean = S / nn[:, None]
    var = np.maximum(Q / nn[:, None] - mean ** 2, 0) * (n / np.maximum(n - 1, 1))[:, None]
    cn = N[ctrl_idx].astype(np.float64)
    has = cn > 0
    W = N.astype(np.float64) * has[None, :]
    wsum = W.sum(1)

    def matched(CB, scale=None):
        cm_b = np.zeros_like(CB)
        cm_b[has] = CB[has] / cn[has][:, None]
        return np.where(wsum[:, None] > 0, (W @ cm_b) / np.maximum(wsum, 1e-12)[:, None], 0.0)

    m_log = matched(A["CB"])
    m_lin = matched(A["CBL"])
    delta = mean - m_log
    nc = float(n[ctrl_idx])
    ctrl_var = var[ctrl_idx]
    se = np.sqrt(var / nn[:, None] + ctrl_var[None, :] / max(nc, 1.0))
    # pseudobulk log2FC from summed raw UMIs (CPM, pseudocount 1) against the batch-matched control CPM
    cpm = A["SR"] / np.maximum(A["LIB"], 1)[:, None] * 1e6
    ctrl_cpm_b = np.zeros((len(has), A["CBR"].shape[1]))
    ctrl_cpm_b[has] = A["CBR"][has] / np.maximum(A["CBLIB"][has], 1)[:, None] * 1e6
    m_cpm = np.where(wsum[:, None] > 0, (W @ ctrl_cpm_b) / np.maximum(wsum, 1e-12)[:, None], 0.0)
    lfc = np.log2((cpm + 1.0) / (m_cpm + 1.0))
    # knockdown efficiency of the target gene (linear CP10K scale), only where detected in controls
    mean_lin = A["SL"] / nn[:, None]
    kd = np.full(len(n), np.nan)
    det = np.zeros(len(n), bool)
    for i, t in enumerate(tgt_idx):
        if t >= 0:
            c = m_lin[i, t]
            det[i] = c >= det_thresh
            if det[i]:
                kd[i] = 1.0 - mean_lin[i, t] / c
    return dict(n_cells=n, delta=delta.astype(np.float32), se=se.astype(np.float32), lfc_pb=lfc.astype(np.float32), kd=kd,
                kd_detected=det, ctrl_mean=mean[ctrl_idx].astype(np.float32), ctrl_n=int(nc), n_ctrl_batches=int(has.sum()),
                matched_ctrl=m_log.astype(np.float32))


def run(h5ad, out, min_cells=20, chunk=20000):
    import h5py
    h = h5py.File(h5ad, "r")
    o = h["obs"]

    def col(k, numeric=False):
        if k not in o:
            return None
        c = o[k]
        if isinstance(c, h5py.Group):
            return _dec(np.array(c["categories"])[np.array(c["codes"])])
        a = np.array(c)
        return a.astype(float) if numeric else _dec(a)

    genes = _dec(np.array(h["var"]["gene_name"]))
    pert, batch = col("perturbation"), col("batch")
    umi, ng, mito = col("UMI_count", True), col("ngenes", True), col("percent_mito", True)
    nperts = col("nperts", True)
    if umi is None:           # fall back to the matrix itself
        umi = np.asarray(h["X"][:, :].sum(1))
    X = h["X"]
    n_cells, G = X.shape
    if ng is None:
        ng = np.zeros(n_cells)
        for s in range(0, n_cells, chunk):
            ng[s:s + chunk] = (np.asarray(X[s:s + chunk]) > 0).sum(1)
    if mito is None:
        mito = np.zeros(n_cells)
    keep, qc = cell_qc(umi, ng, mito, batch, nperts)
    perts, pc = np.unique(pert, return_inverse=True)
    batches, bc = np.unique(batch, return_inverse=True)
    ci = int(np.where(perts == CTRL)[0][0])
    gen = ((s, np.asarray(X[s:s + chunk])) for s in range(0, n_cells, chunk))
    A = aggregate2(gen, pc, bc, keep, len(perts), len(batches), G, ci)
    gidx = {g: i for i, g in enumerate(genes)}
    tgt = np.array([gidx.get(p, -1) for p in perts])
    R = finalize2(A, ci, tgt, min_cells)
    n_raw = np.bincount(pc, minlength=len(perts))
    keepp = np.array([(p != CTRL) and R["n_cells"][i] >= min_cells for i, p in enumerate(perts)])
    np.savez_compressed(
        out, genes=genes.astype(str), perts=perts[keepp].astype(str), n_cells=R["n_cells"][keepp], n_cells_raw=n_raw[keepp],
        delta=R["delta"][keepp], se=R["se"][keepp], lfc_pb=R["lfc_pb"][keepp], kd=R["kd"][keepp], kd_detected=R["kd_detected"][keepp],
        target_idx=tgt[keepp], ctrl_mean=R["ctrl_mean"], ctrl_n=R["ctrl_n"], matched_ctrl=R["matched_ctrl"][keepp])
    meta = dict(src=str(h5ad), qc=qc, n_perts_kept=int(keepp.sum()), kd_detected=int(R["kd_detected"][keepp].sum()),
                median_kd=float(np.nanmedian(R["kd"][keepp])), ctrl_cells=R["ctrl_n"], ctrl_batches=R["n_ctrl_batches"])
    json.dump(meta, open(str(out) + ".json", "w"), indent=1)
    print(json.dumps(meta))
    return meta


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
