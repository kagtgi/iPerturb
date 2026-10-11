"""Addendum J(b): metric-aligned losses for the smoothing head (gradient-trained W), extended alpha grid."""
import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
from ipt.models.losses import LOSSES

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--ext", default="", help="optional extra edge file(s) (tsv with columns a,b) to add to the graph, comma separated")
ap.add_argument("--tag", default="")
ap.add_argument("--losses", default="mse_closed,mse,corr,nmse,huber,corr+huber")
ap.add_argument("--base", default="l1l3")
ap.add_argument("--alphas", default="0.5,0.75,0.9,1.0")
a = ap.parse_args()
torch.set_num_threads(3)
KS = (20, 40)
ALPHAS = tuple(float(x) for x in a.alphas.split(","))

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
folds = json.load(open(ART / "folds" / f"{a.line}.json"))


def pairs(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)


l1 = load_l1(CACHE)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
parts = []
if a.base in ("l1l3",):
    parts.append(pairs(l1.source, l1.target))
if a.base in ("l1l3", "l3"):
    parts.append(pairs(st.a.str.upper(), st.b.str.upper()))
for fpath in [p for p in a.ext.split(",") if p]:
    d = pd.read_csv(fpath, sep="\t")
    ca, cb = ("a", "b") if "a" in d.columns else ("source", "target")
    parts.append(pairs(d[ca].astype(str).str.upper(), d[cb].astype(str).str.upper()))
E = np.unique(np.sort(np.concatenate(parts), 1), axis=0) if parts else np.zeros((0, 2), int)
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(G, G))
A = (A + A.T).tocsr()
A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()
Ahat = sp.diags(1 / np.maximum(deg, 1)) @ A
print("edges", len(E), flush=True)


def train_W(emb_tr, Gk, b, Ytr, ttr, emb_va, Yva, tva, loss_name, steps=150, lr=0.02, ridge=1e-2):
    k = Gk.shape[1]
    P = emb_tr
    W0 = np.linalg.solve(P.T @ P + ridge * np.eye(k), P.T @ ((Ytr - b) @ Gk))
    if loss_name == "mse_closed":
        return W0
    W = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float32))
    Gt, bt = torch.tensor(Gk, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    Pt, Yt, tt = torch.tensor(P, dtype=torch.float32), torch.tensor(Ytr, dtype=torch.float32), torch.as_tensor(ttr)
    Pv, Yv, tv = torch.tensor(emb_va, dtype=torch.float32), torch.tensor(Yva, dtype=torch.float32), torch.as_tensor(tva)
    lossf = LOSSES[loss_name]
    opt = torch.optim.Adam([W], lr=lr)
    best, bw = np.inf, W.detach().clone()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(bt + Pt @ W @ Gt.T, Yt, tt) + ridge * (W ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v = float(lossf(bt + Pv @ W @ Gt.T, Yv, tv))
        if v < best:
            best, bw = v, W.detach().clone()
    return bw.numpy().astype(np.float64)


rows, cfgs = [], []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float64) for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    _, _, Vt = np.linalg.svd(Y["train"] - tm, full_matrices=False)
    V = Vt.T
    obs, tt = Y["test"], T["test"]
    deg_lab = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    for loss_name in a.losses.split(","):
        best = None
        for k in KS:
            Gk = V[:, :k]
            for al in ALPHAS:
                emb = (1 - al) * Gk + al * (Ahat @ Gk)
                W = train_W(emb[T["train"]], Gk, tm, Y["train"], T["train"], emb[T["valid"]], Y["valid"], T["valid"], loss_name)
                pv = tm + emb[T["valid"]] @ W @ Gk.T
                lv = float(LOSSES["mse" if loss_name == "mse_closed" else loss_name](torch.tensor(pv, dtype=torch.float32), torch.tensor(Y["valid"], dtype=torch.float32), torch.as_tensor(T["valid"])))
                if best is None or lv < best[0]:
                    best = (lv, k, al, W, emb, Gk)
        lv, k, al, W, emb, Gk = best
        p = (tm + emb[T["test"]] @ W @ Gk.T).astype(np.float32)
        cfgs.append(dict(fold=fo, loss=loss_name, k=k, alpha=al, val_loss=lv))
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pres = M.pcc_delta_resid(p, obs, tm, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        ap_ = M.deg_auprc(p - tm, deg_lab, tt)["ap"]
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, loss=loss_name, target=f["test"][i], pcc_c=pc[i], pcc_resid=pres[i], top20_c=t20[i], deg_ap=ap_[i], cent_rank=cn[i]))
        print(f"fold {fo} {loss_name}: k={k} alpha={al} val {lv:.4f} test rho_c {np.mean(pc):.4f}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g11_{a.line}{a.tag}.tsv", sep="\t", index=False)
pd.DataFrame(cfgs).to_csv(ART / "scores" / f"g11cfg_{a.line}{a.tag}.tsv", sep="\t", index=False)
u = df.groupby(["loss", "target"], as_index=False)[["pcc_c", "pcc_resid", "top20_c", "deg_ap", "cent_rank"]].mean()
print(u.groupby("loss").mean(numeric_only=True).round(4).to_string())
