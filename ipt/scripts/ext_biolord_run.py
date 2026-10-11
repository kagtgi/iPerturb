"""B8: official biolord (pip biolord 0.0.3) with the published unseen-single-perturbation setup, on our target-held-out folds. Runs on the VM (biolord-env).

Setup copied from nitzanlab/biolord_reproducibility (scripts/biolord/adamson + notebooks/perturbations/adamson preprocessing, optimal params):
  - one row per perturbation = mean log1p-CP10K expression (pseudobulk of the GEARS-processed cells) + one 'ctrl' row;
  - the perturbation is described by ONE ordered attribute = GEARS GO-similarity neighbour vector: for each perturbed gene, the 20 top GO-Jaccard
    neighbours (nlargest(21) over go_essential_all.csv, as in the published notebook) as importance weights over the essential-perturbation gene list,
    restricted to columns that are non-zero for at least one perturbation;
  - biolord train(max_epochs=1000, early stopping patience 10 on the 'test'(=our validation) split, batch 32), optimal hyper-parameters from adamson_config_optimal.py;
  - prediction for an unseen target: control row with its ordered attribute replaced by the target's neighbour vector -> module.get_expression.
Targets without a GO-neighbour vector, or absent from the pseudobulk, are excluded and counted.
Output per fold: pred_fold{k}.npz (targets, delta = pred - control mean, genes, train_targets, train_delta).
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

import biolord

ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--max-epochs", type=int, default=1000)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=None)
a = ap.parse_args()

H = Path.home() / "gdata"
OUT = Path(a.out or Path.home() / "ext" / "biolord") / a.line
OUT.mkdir(parents=True, exist_ok=True)
folds = json.load(open(H / f"folds_{a.line}.json"))
pb = np.load(H / f"pseudobulk_{a.line}.npz", allow_pickle=True)
conds = [str(c) for c in pb["conds"]]
cidx = {c: i for i, c in enumerate(conds)}
M, genes = pb["mean"], [str(g) for g in pb["genes"]]
ctrl_mean = M[cidx["ctrl"]]

# GEARS GO-neighbour vectors (published biolord preprocessing)
gene_list = pickle.load(open(H / "pertdata" / "essential_all_data_pert_genes.pkl", "rb"))
go = pd.read_csv(H / "pertdata" / "go_essential_all" / "go_essential_all.csv", header=None, names=["source", "target", "importance"])
go = go.sort_values(["target", "importance"], ascending=[True, False], kind="stable").groupby("target").head(21)
go = go[go["source"].isin(set(gene_list))]
gpos = {g: i for i, g in enumerate(gene_list)}
nb = {}
for t, d in go.groupby("target"):
    v = np.zeros(len(gene_list), np.float32)
    v[[gpos[s] for s in d["source"]]] = d["importance"].to_numpy(np.float32)
    nb[t] = v
perts = sorted({c[:-5] for c in conds if c.endswith("+ctrl")})
has = [p for p in perts if p in nb and nb[p].sum() > 0]
keep_idx = np.stack([nb[p] for p in has]).sum(0) > 0
print(f"{a.line}: perturbations in pseudobulk {len(perts)}, with GO-neighbour vector {len(has)}, attribute dim {int(keep_idx.sum())}", flush=True)
hset = set(has)

# optimal params of adamson_config_optimal.py (autoencoder_lr -> latent_lr / latent_wd as named in that file; unknown_attribute_noise_param = biolord default 0.1)
vp = dict(seed=42, use_batch_norm=False, use_layer_norm=False, step_size_lr=45, attribute_dropout_rate=0.1, cosine_scheduler=True,
          scheduler_final_lr=1e-5, n_latent=32, n_latent_attribute_ordered=512, reconstruction_penalty=1000.0, attribute_nn_width=64,
          attribute_nn_depth=6, attribute_nn_lr=0.001, attribute_nn_wd=4e-8, latent_lr=0.0001, latent_wd=0.001, unknown_attribute_penalty=10000.0,
          decoder_width=64, decoder_depth=1, decoder_activation=False, attribute_nn_activation=False, unknown_attributes=False,
          decoder_lr=0.001, decoder_wd=0.01, early_stopping_patience=10, n_latent_attribute_categorical=16)
module_params = {k: vp[k] for k in ("attribute_nn_width", "attribute_nn_depth", "use_batch_norm", "use_layer_norm", "attribute_dropout_rate", "seed",
                                    "n_latent_attribute_ordered", "n_latent_attribute_categorical", "reconstruction_penalty", "unknown_attribute_penalty",
                                    "decoder_width", "decoder_depth", "decoder_activation", "attribute_nn_activation", "unknown_attributes")}
trainer_params = dict(n_epochs_warmup=0, latent_lr=vp["latent_lr"], latent_wd=vp["latent_wd"], attribute_nn_lr=vp["attribute_nn_lr"],
                      attribute_nn_wd=vp["attribute_nn_wd"], step_size_lr=vp["step_size_lr"], cosine_scheduler=vp["cosine_scheduler"],
                      scheduler_final_lr=vp["scheduler_final_lr"], decoder_lr=vp["decoder_lr"], decoder_wd=vp["decoder_wd"])

rows = has + ["ctrl"]
X = np.stack([M[cidx[p + "+ctrl"]] for p in has] + [ctrl_mean]).astype(np.float32)
E = np.stack([nb[p][keep_idx] for p in has] + [np.zeros(int(keep_idx.sum()), np.float32)])
ad = anndata.AnnData(X=X)
ad.obs_names = rows
ad.obs["condition"] = rows
ad.obsm["perturbation_neighbors"] = E
key = "perturbation_neighbors"
t0 = time.time()
for fo in a.folds:
    f = folds[fo]
    tr = [g for g in f["train"] if g in hset]
    va = [g for g in f["valid"] if g in hset]
    te = [g for g in f["test"] if g in hset]
    excl = len(f["test"]) - len(te)
    lab = {g: "train" for g in tr}
    lab.update({g: "test" for g in va})      # biolord's 'test' split = validation / early stopping
    lab.update({g: "ood" for g in te})       # evaluated unseen targets
    ad.obs["split"] = [lab.get(r, "unused") for r in rows]
    sub = ad[[l != "unused" or r == "ctrl" for r, l in zip(rows, ad.obs["split"])]].copy()
    sub.obs.loc["ctrl", "split"] = "train"
    biolord.Biolord.setup_anndata(sub, ordered_attributes_keys=[key], categorical_attributes_keys=None, retrieval_attribute_key=None)
    model = biolord.Biolord(adata=sub, n_latent=vp["n_latent"], model_name="norman", module_params=module_params, train_classifiers=False, split_key="split")
    # torch BatchNorm1d in biolord's MLPs raises on a size-1 batch; keep the published batch size 32 unless it leaves a size-1 remainder
    ntr, nva = len(tr) + 1, len(va)
    bs = next(b for b in range(32, 20, -1) if ntr % b != 1 and nva % b != 1)
    t1 = time.time()
    model.train(max_epochs=a.max_epochs, batch_size=bs, plan_kwargs=trainer_params, early_stopping=True,
                early_stopping_patience=vp["early_stopping_patience"], check_val_every_n_epoch=5, num_workers=1, enable_checkpointing=False)
    print(f"fold {fo}: batch_size {bs}; trained in {time.time() - t1:.0f}s, train {len(tr)} valid {len(va)} test {len(te)} (excluded no-GO/absent {excl})", flush=True)
    model.module.eval()
    ds_ctrl = model.get_dataset(sub[sub.obs["condition"] == "ctrl"].copy())
    emb = {p: E[i] for i, p in enumerate(has)}

    def predict(names):
        out = []
        for p in names:
            d = {k: v.clone() for k, v in ds_ctrl.items()}
            d[key] = torch.tensor(emb[p], dtype=torch.float32)[None, :].repeat(d[key].shape[0], 1).to(d[key].device)
            pr, _ = model.module.get_expression(d)
            out.append(pr.detach().cpu().numpy().mean(0) - ctrl_mean)
        return np.stack(out).astype(np.float32)

    rng2 = np.random.default_rng(a.seed + fo)
    trs = [str(g) for g in rng2.choice(tr, min(300, len(tr)), replace=False)]
    np.savez_compressed(OUT / f"pred_fold{fo}.npz", targets=np.array(te), delta=predict(te), genes=np.array(genes), train_targets=np.array(trs),
                        train_delta=predict(trs), n_test_total=len(f["test"]), n_test_excluded=excl, max_epochs=a.max_epochs)
    print(f"fold {fo}: predicted {len(te)} test targets; elapsed {time.time() - t0:.0f}s", flush=True)
