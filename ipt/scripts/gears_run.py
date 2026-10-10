"""B4: official GEARS (cell-gears) trained on our target-held-out folds. Runs on the VM (gears-env).

Pipeline per line: subsample cells, log1p(CP10K), keep conditions whose gene is in GEARS' GO gene set,
process once, then per fold: custom split (our train/valid/test target lists), train, predict test targets.
Outputs delta predictions (pred - control mean) aligned to our gene axis, plus the supported-target list.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from gears import GEARS, PertData

ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--cells", type=int, default=100)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

H = Path.home() / "gdata"
OUT = H / f"gears_{a.line}"
OUT.mkdir(exist_ok=True)
h5 = H / ("ReplogleWeissman2022_K562_essential.h5ad" if a.line.startswith("K562") else "ReplogleWeissman2022_rpe1.h5ad")
folds = json.load(open(H / f"folds_{a.line}.json"))
want = sorted({g for f in folds for k in ("train", "valid", "test") for g in f[k]})

pdata = PertData(str(H / "pertdata"))          # downloads gene2go_all.pkl (Harvard Dataverse) on first use
go_genes = set(pdata.gene2go.keys())
adata = sc.read_h5ad(h5)
rng = np.random.default_rng(a.seed)
pert = adata.obs["perturbation"].astype(str).to_numpy()
var_names = np.asarray(adata.var["gene_name"].astype(str) if "gene_name" in adata.var else adata.var_names.astype(str))
vset = set(var_names)
keep_pert = {g for g in want if g.upper() in {x.upper() for x in go_genes} or g in go_genes}
keep_pert = {g for g in keep_pert if g in vset}
idx = []
for g in sorted(keep_pert):
    ii = np.where(pert == g)[0]
    idx += rng.choice(ii, min(len(ii), a.cells), replace=False).tolist()
ci = np.where(pert == "control")[0]
idx += rng.choice(ci, min(len(ci), 3000), replace=False).tolist()
adata = adata[sorted(idx)].copy()
adata.var_names = var_names
adata.var_names_make_unique()
adata.var["gene_name"] = adata.var_names.astype(str)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X = sp.csr_matrix(adata.X)
p2 = adata.obs["perturbation"].astype(str)
adata.obs["condition"] = np.where(p2 == "control", "ctrl", p2 + "+ctrl")
adata.obs["cell_type"] = a.line
adata.obs = adata.obs[["condition", "cell_type"]]
print(f"{a.line}: kept {len(keep_pert)}/{len(want)} targets in GO set; cells {adata.n_obs}", flush=True)
pdata.new_data_process(dataset_name=a.line, adata=adata)
gene_list = list(pdata.adata.var["gene_name"])
ctrl_mean = np.asarray(pdata.adata[pdata.adata.obs["condition"] == "ctrl"].X.mean(0)).ravel()
json.dump(sorted(keep_pert), open(OUT / "supported.json", "w"))

for fo in a.folds:
    f = folds[fo]
    sd = {k: [g + "+ctrl" for g in f[kk] if g in keep_pert] for k, kk in (("train", "train"), ("val", "valid"), ("test", "test"))}
    pickle.dump(sd, open(OUT / f"split_{fo}.pkl", "wb"))
    pdata.prepare_split(split="custom", seed=a.seed, split_dict_path=str(OUT / f"split_{fo}.pkl"))
    pdata.get_dataloader(batch_size=32, test_batch_size=128)
    model = GEARS(pdata, device="cuda", weight_bias_track=False, proj_name="gears", exp_name=f"{a.line}_f{fo}")
    model.model_initialize(hidden_size=64)
    model.train(epochs=a.epochs)
    tests = [g[:-5] for g in sd["test"]]
    pred = model.predict([[g] for g in tests])
    P = np.stack([np.asarray(pred[g]) - ctrl_mean for g in tests]).astype(np.float32) if tests else np.zeros((0, len(gene_list)), np.float32)
    np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(tests), delta=P, genes=np.array(gene_list))
    print(f"fold {fo}: predicted {len(tests)} test targets", flush=True)
