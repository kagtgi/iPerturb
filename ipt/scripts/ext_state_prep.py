"""STATE (arc-state) input prep, runs on the VM in ~/state-env.

Builds, per line, a control-cell h5ad (3000 non-targeting cells, log1p CP10K, gene symbols as var_names) with the
obs columns the released Replogle ST-SE checkpoints expect (gene, cell_line, gem_group), and the TSV of perturbations
to simulate (union of our fold test targets and the per-fold 300-target training samples that exist in the checkpoint
one-hot map).  Also writes targets_<line>.json with what was dropped (target not in the checkpoint perturbation map).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True)
ap.add_argument("--map", required=True, help="pert_onehot_map.pt of the checkpoint")
ap.add_argument("--cells", type=int, default=100)
ap.add_argument("--embed", default="hvg", help="hvg: obsm[X_hvg] = log1p CP10K of the checkpoint HVG list (var_dims.pkl)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
H = Path.home() / "gdata"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
h5 = H / ("ReplogleWeissman2022_K562_essential.h5ad" if a.line.startswith("K562") else "ReplogleWeissman2022_rpe1.h5ad")
folds = json.load(open(H / f"folds_{a.line}.json"))
pm = {str(k) for k in torch.load(a.map, weights_only=False)}
batches = {str(k) for k in pickle.load(open(Path(a.map).parent / "batch_onehot_map.pkl", "rb"))}

adata = sc.read_h5ad(h5)
var_names = np.asarray(adata.var.index.astype(str))
ctrl = np.where(adata.obs["perturbation"].astype(str).to_numpy() == "control")[0]
rng = np.random.default_rng(1)
ci = np.sort(rng.choice(ctrl, min(3000, len(ctrl)), replace=False))
c = adata[ci].copy()
c.var_names = var_names
c.var_names_make_unique()
sc.pp.normalize_total(c, target_sum=1e4)
sc.pp.log1p(c)
c.X = sp.csr_matrix(c.X)
gem = c.obs["batch"].astype(str).to_numpy()
print("gem groups in ctrl:", len(set(gem)), "in checkpoint batch map:", len(set(gem) & batches))
vd = pickle.load(open(Path(a.map).parent / "var_dims.pkl", "rb"))
hv = [str(g) for g in vd["gene_names"]]
pos = {g: i for i, g in enumerate(c.var_names)}
Xd = c.X.toarray()
Xh = np.zeros((c.n_obs, len(hv)), np.float32)
miss = 0
for j, g in enumerate(hv):
    if g in pos:
        Xh[:, j] = Xd[:, pos[g]]
    else:
        miss += 1
print("HVG genes missing from our axis:", miss, "/", len(hv))
c.obsm["X_hvg"] = Xh
c.obs = pd.DataFrame({"gene": "non-targeting", "cell_line": a.line.split("_")[0].lower(), "gem_group": gem}, index=c.obs_names)
c.write_h5ad(out / f"ctrl_{a.line}.h5ad")

want_test = sorted({g for f in folds for g in f["test"]})
rows, report = [], {"test_in_map": 0, "test_total": len(want_test), "dropped_test": []}
keep = [g for g in want_test if g in pm]
report["test_in_map"] = len(keep)
report["dropped_test"] = [g for g in want_test if g not in pm]
tr_all = set()
for fo, f in enumerate(folds):
    tr = [g for g in f["train"] if g in pm]
    r2 = np.random.default_rng(1 + fo)
    s = [str(g) for g in r2.choice(tr, min(300, len(tr)), replace=False)]
    tr_all |= set(s)
    report[f"train_sample_fold{fo}"] = s
    report[f"n_train_in_map_fold{fo}"] = len(tr)
perts = sorted(set(keep) | tr_all)
pd.DataFrame({"perturbation": perts, "num_cells": a.cells}).to_csv(out / f"perts_{a.line}.tsv", sep="\t", index=False)
json.dump(report, open(out / f"targets_{a.line}.json", "w"))
print(a.line, "perturbations to simulate:", len(perts), "test in map", len(keep), "/", len(want_test))
