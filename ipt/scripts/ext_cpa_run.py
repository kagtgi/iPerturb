"""B7: official CPA (pip cpa-tools 0.8.8) with fixed pre-computed perturbation embeddings so that unseen targets can be predicted. VM, cpa-env. One fold per process.

CPA natively learns one embedding per perturbation (no representation for an unseen gene). The package's own unseen-perturbation route is the `drug_embeddings`
argument of CPAModule (used by the RDKit-fingerprint path of CPA(..., use_rdkit_embeddings=True) for unseen drugs): a frozen nn.Embedding whose row i is the
description of perturbation i, passed through CPA's learned perturbation network. Here row i = the GEARS GO-similarity neighbour vector of the perturbed gene
(top-20 GO-Jaccard neighbours as importance weights over the essential-perturbation gene list, identical to the biolord attribute); control and <PAD> rows are zero.
Hyper-parameters = the cpa-tools Norman tutorial (nb loss on raw counts, n_latent 32, batch 2048, early stopping patience 5, check every 5 epochs), except
doser_type 'linear' retained and max_epochs/max_time reduced (recorded in the output file).
Training data: <=50 cells per training target (+ 3000 controls), <=20 cells per validation target (early stopping). Unseen targets are predicted by feeding 30 control
cells labelled with the target ('ood' split, never trained on) and reading CPA_pred; counts are normalised to CP10K + log1p per cell (as in the tutorial) and the
delta = mean(pred) - mean(log1p CP10K of the same control cells).
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn

import sys, types
_tk = types.ModuleType('tkinter'); _tk.N = None; sys.modules['tkinter'] = _tk  # cpa-tools 0.8.8 has a stray `from tkinter import N` (unused); tkinter is not installed on the VM
import cpa

ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--train-cells", type=int, default=15)
ap.add_argument("--valid-cells", type=int, default=10)
ap.add_argument("--ood-cells", type=int, default=30)
ap.add_argument("--max-epochs", type=int, default=2000)
ap.add_argument("--max-minutes", type=int, default=25)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()
H = Path.home() / "gdata"
OUT = Path(a.out or Path.home() / "ext" / "cpa") / a.line
OUT.mkdir(parents=True, exist_ok=True)
fo = a.fold
f = json.load(open(H / f"folds_{a.line}.json"))[fo]
h5 = H / ("ReplogleWeissman2022_K562_essential.h5ad" if a.line.startswith("K562") else "ReplogleWeissman2022_rpe1.h5ad")

# GO-neighbour vectors (same construction as ext_biolord_run.py)
gene_list = pickle.load(open(H / "pertdata" / "essential_all_data_pert_genes.pkl", "rb"))
go = pd.read_csv(H / "pertdata" / "go_essential_all" / "go_essential_all.csv", header=None, names=["source", "target", "importance"], low_memory=False)
go = go.sort_values(["target", "importance"], ascending=[True, False], kind="stable").groupby("target").head(21)
go = go[go["source"].isin(set(gene_list))]
gpos = {g: i for i, g in enumerate(gene_list)}
nb = {}
for t, d in go.groupby("target"):
    v = np.zeros(len(gene_list), np.float32)
    v[[gpos[s] for s in d["source"]]] = d["importance"].to_numpy(np.float32)
    nb[t] = v

raw = ad.read_h5ad(h5)
var_names = np.asarray(raw.var["gene_name"].astype(str)) if "gene_name" in raw.var else np.asarray(raw.var_names.astype(str))
pert = raw.obs["perturbation"].astype(str).to_numpy()
have = {g for g in set(pert) if g in nb and nb[g].sum() > 0}
tr = [g for g in f["train"] if g in have and (pert == g).any()]
va = [g for g in f["valid"] if g in have and (pert == g).any()]
te = [g for g in f["test"] if g in have]
excl = len(f["test"]) - len(te)
rng = np.random.default_rng(a.seed + fo)


def pick(g, n):
    ii = np.where(pert == g)[0]
    return rng.choice(ii, min(len(ii), n), replace=False)


idx, lab, spl = [], [], []
for g in tr:
    i = pick(g, a.train_cells); idx += i.tolist(); lab += [g] * len(i); spl += ["train"] * len(i)
for g in va:
    i = pick(g, a.valid_cells); idx += i.tolist(); lab += [g] * len(i); spl += ["valid"] * len(i)
ci = np.where(pert == "control")[0]
ci = rng.choice(ci, 3000, replace=False)
idx += ci.tolist(); lab += ["ctrl"] * len(ci); spl += ["train"] * len(ci)
X = raw.X[np.array(idx)]
X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
print(f"{a.line} fold {fo}: counts max {X.max():.0f}, integer-valued {bool(np.allclose(X.data, np.round(X.data)))}; train {len(tr)} valid {len(va)} test {len(te)} (excluded {excl})", flush=True)

# virtual unseen-target cells = control cells relabelled (never trained on)
ctrl_rows = np.where(np.array(lab) == "ctrl")[0]
vx, vl = [], []
for g in te:
    sel = rng.choice(ctrl_rows, a.ood_cells, replace=False)
    vx.append(sel); vl += [g] * a.ood_cells
vsel = np.concatenate(vx)
Xall = sp.vstack([X, X[vsel]]).tocsr()
labs = np.array(lab + vl)
splits = np.array(spl + ["ood"] * len(vl))
adata = ad.AnnData(X=Xall.astype(np.float32))
adata.var_names = var_names
adata.var_names_make_unique()
adata.obs["condition"] = labs
adata.obs["split"] = splits
adata.obs["cell_type"] = a.line
adata.obs["dose_value"] = 1.0
adata.layers["counts"] = adata.X.copy()
del raw

cpa.CPA.pert_encoder = None
cpa.CPA.covars_encoder = None
cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl", dosage_key="dose_value", categorical_covariate_keys=["cell_type"],
                      is_count_data=True, max_comb_len=1)
penc = cpa.CPA.pert_encoder
D = len(next(iter(nb.values())))
keep = np.stack([nb[g] for g in have]).sum(0) > 0
W = np.zeros((len(penc), int(keep.sum())), np.float32)
for k, i in penc.items():
    if k in nb:
        W[i] = nb[k][keep]
emb = nn.Embedding(len(penc), W.shape[1], padding_idx=0)
emb.weight.data.copy_(torch.tensor(W))
emb.weight.requires_grad = False
print("perturbation embedding", tuple(W.shape), "perts with vector", int((np.abs(W).sum(1) > 0).sum()), flush=True)

model_params = {"n_latent": 32, "recon_loss": "nb", "doser_type": "linear", "n_hidden_encoder": 256, "n_layers_encoder": 4, "n_hidden_decoder": 256,
                "n_layers_decoder": 2, "use_batch_norm_encoder": True, "use_layer_norm_encoder": False, "use_batch_norm_decoder": False,
                "use_layer_norm_decoder": False, "dropout_rate_encoder": 0.2, "dropout_rate_decoder": 0.0, "variational": False, "seed": 8206}
trainer_params = {"n_epochs_kl_warmup": None, "n_epochs_adv_warmup": 50, "n_epochs_mixup_warmup": 10, "n_epochs_pretrain_ae": 10, "mixup_alpha": 0.1,
                  "lr": 0.0001, "wd": 3.2170178270865573e-06, "adv_steps": 3, "reg_adv": 10.0, "pen_adv": 20.0, "adv_lr": 0.0001, "adv_wd": 7.051355554517135e-06,
                  "n_layers_adv": 2, "n_hidden_adv": 128, "use_batch_norm_adv": True, "use_layer_norm_adv": False, "dropout_rate_adv": 0.3, "step_size_lr": 25,
                  "do_clip_grad": False, "adv_loss": "cce", "gradient_clip_value": 5.0}
model = cpa.CPA(adata=adata, split_key="split", train_split="train", valid_split="valid", test_split="ood", drug_embeddings=emb, **model_params)
t0 = time.time()
h, m = divmod(a.max_minutes, 60)
model.train(max_epochs=a.max_epochs, use_gpu=True, batch_size=2048, plan_kwargs=trainer_params, early_stopping_patience=5, check_val_every_n_epoch=5,
            max_time=f"00:{h:02d}:{m:02d}:00")
print(f"trained in {time.time() - t0:.0f}s", flush=True)

# counterfactual prediction for every unseen target and for a 300-target training sample
trs = [str(g) for g in np.random.default_rng(a.seed + fo).choice(tr, min(300, len(tr)), replace=False)]
adata.layers["X_true"] = adata.X.copy()
ov = np.where(splits == "ood")[0]
ctrl_adata = adata[adata.obs["condition"] == "ctrl"]


def lognorm(M):
    M = np.asarray(M, np.float64)
    M = M / np.maximum(M.sum(1, keepdims=True), 1e-9) * 1e4
    return np.log1p(M)


def predict_targets(names, cells=a.ood_cells):
    out = {}
    for s in range(0, len(names), 100):
        chunk = names[s:s + 100]
        rows = []
        for g in chunk:
            sel = rng.choice(ctrl_rows, cells, replace=False)
            rows.append(sel)
        rr = np.concatenate(rows)
        sub = adata[rr].copy()
        xc = np.asarray(sub.X.todense()) if sp.issparse(sub.X) else np.asarray(sub.X)
        sub.obs["condition"] = np.repeat(chunk, cells)
        sub.obs["split"] = "ood"
        model.predict(sub, batch_size=2048)
        pr = lognorm(sub.obsm["CPA_pred"])
        cx = lognorm(xc)
        for i, g in enumerate(chunk):
            out[g] = (pr[i * cells:(i + 1) * cells].mean(0) - cx[i * cells:(i + 1) * cells].mean(0)).astype(np.float32)
    return out


pt = predict_targets(te)
ptr = predict_targets(trs)
np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(te), delta=np.stack([pt[g] for g in te]), genes=np.array(adata.var_names),
                    train_targets=np.array(trs), train_delta=np.stack([ptr[g] for g in trs]), n_test_total=len(f["test"]), n_test_excluded=excl,
                    train_cells=a.train_cells, max_minutes=a.max_minutes, train_seconds=time.time() - t0)
print(f"fold {fo}: predicted {len(te)} test targets", flush=True)
