"""Turn `state tx infer` output (obsm[X_hvg] predictions for virtual cells) into pred_fold{k}.npz for ext_to_scores.py.

delta = mean predicted X_hvg of the simulated perturbation cells - mean predicted X_hvg of the non-targeting cells in the same output.
genes = the checkpoint's 2000-HVG list (all other genes get zero delta in the scorer, a documented limitation).
targets per fold = fold test targets that exist in the checkpoint's one-hot perturbation map.
"""
import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True)
ap.add_argument("--h5ad", required=True)
ap.add_argument("--var-dims", required=True)
ap.add_argument("--targets-json", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
H = Path.home() / "gdata"
folds = json.load(open(H / f"folds_{a.line}.json"))
rep = json.load(open(a.targets_json))
genes = [str(g) for g in pickle.load(open(a.var_dims, "rb"))["gene_names"]]
x = ad.read_h5ad(a.h5ad)
P = np.asarray(x.obsm["X_hvg"], np.float32)
lab = x.obs["gene"].astype(str).to_numpy()
cm = P[lab == "non-targeting"].mean(0)
mean = {}
for g in np.unique(lab):
    if g != "non-targeting":
        mean[g] = P[lab == g].mean(0) - cm
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for fo, f in enumerate(folds):
    tests = [g for g in f["test"] if g in mean]
    tr = [g for g in rep[f"train_sample_fold{fo}"] if g in mean]
    np.savez_compressed(out / f"pred_fold{fo}.npz", targets=np.array(tests), delta=np.stack([mean[g] for g in tests]),
                        genes=np.array(genes), train_targets=np.array(tr), train_delta=np.stack([mean[g] for g in tr]),
                        n_test_total=len(f["test"]), n_test_excluded=len(f["test"]) - len(tests))
    print(a.line, "fold", fo, "test targets predicted", len(tests), "of", len(f["test"]), "train sample", len(tr))
