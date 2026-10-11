"""Per-condition pseudobulk of the GEARS-processed data (log1p CP10K, <=100 cells/perturbation, 3000 controls), cached once per line.

Runs on the VM. Output ~/gdata/pseudobulk_<line>.npz: conds (e.g. 'AARS2+ctrl', 'ctrl'), mean [n_cond x G], n_cells, genes.
Chunked read of the backed sparse matrix so RAM stays small.
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True)
a = ap.parse_args()
H = Path.home() / "gdata"
x = ad.read_h5ad(H / "pertdata" / a.line.lower() / "perturb_processed.h5ad", backed="r")
cond = x.obs["condition"].astype(str).to_numpy()
conds, inv = np.unique(cond, return_inverse=True)
n = np.bincount(inv)
S = np.zeros((len(conds), x.n_vars), np.float64)
step = 20000
for s in range(0, x.n_obs, step):
    blk = x.X[s:s + step]
    blk = blk if sp.issparse(blk) else sp.csr_matrix(blk)
    ind = sp.csr_matrix((np.ones(blk.shape[0]), (inv[s:s + step], np.arange(blk.shape[0]))), shape=(len(conds), blk.shape[0]))
    S += np.asarray((ind @ blk).todense())
M = (S / n[:, None]).astype(np.float32)
genes = np.asarray(x.var["gene_name"].astype(str)) if "gene_name" in x.var else np.asarray(x.var_names.astype(str))
np.savez_compressed(H / f"pseudobulk_{a.line}.npz", conds=conds, mean=M, n_cells=n, genes=genes)
print(a.line, "conditions", len(conds), "genes", len(genes), "ctrl cells", n[list(conds).index("ctrl")])
