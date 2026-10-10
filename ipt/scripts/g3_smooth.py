"""Addendum A: association-prior smoothing of the perturbed gene's embedding in the B2 low-rank model."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
from ipt.models import baselines as B

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
KS, ALPHAS = (5, 10, 20, 40), (0.0, 0.25, 0.5, 0.75)

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
folds = json.load(open(ART / "folds" / f"{a.line}.json"))


def pairs_to_idx(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)


l1 = load_l1(CACHE)
L1 = pairs_to_idx(l1.source, l1.target)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
L3 = pairs_to_idx(st.a.str.upper(), st.b.str.upper())
cx = pickle.load(open(CACHE / "l4_coex_parsed.pkl", "rb"))
cx = cx.assign(source=cx.source.str.upper(), target=cx.target.str.upper())
cx = cx.sort_values("mr", ascending=False).groupby("source").head(20)
L4 = pairs_to_idx(cx.source, cx.target)
layers = {"none": [], "L1": [L1], "L1+L3": [L1, L3], "L1+L3+L4": [L1, L3, L4]}


def edges_of(names):
    if not names:
        return np.zeros((0, 2), int)
    e = np.concatenate(names)
    e = np.sort(e, 1)
    return np.unique(e, axis=0)


def adjacency(e):
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    d = np.asarray(A.sum(1)).ravel()
    return sp.diags(1 / np.maximum(d, 1)) @ A


def rewired(e, seed):
    g = nx.Graph()
    g.add_edges_from(map(tuple, e))
    nx.double_edge_swap(g, nswap=5 * len(e), max_tries=50 * len(e), seed=seed)
    return np.array(g.edges())


def fit_smooth(D, tgt, V, Ahat, k, alpha, ridge=1e-2):
    Gk = V[:, :k]
    emb = (1 - alpha) * Gk + alpha * (Ahat @ Gk)
    b = D.mean(0)
    P = emb[tgt]
    W = np.linalg.solve(P.T @ P + ridge * np.eye(k), P.T @ ((D - b) @ Gk))
    return lambda t: (b + emb[t] @ W @ Gk.T).astype(np.float32)


rows = []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float64) for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    Dc = Y["train"] - tm
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    V = Vt.T
    mask = np.ones_like(Y["valid"], bool)
    mask[np.arange(len(T["valid"])), T["valid"]] = False
    preds = {"B0_nochange": np.zeros_like(Y["test"]), "B1_trainmean": np.tile(tm, (len(T["test"]), 1))}
    variants = []
    for name, ls in layers.items():
        e = edges_of(ls)
        variants.append((name, "real", adjacency(e) if len(e) else sp.csr_matrix((G, G))))
        if name != "none":
            variants.append((name, "rewired", adjacency(rewired(e, a.seed + fo))))
    for name, gname, Ahat in variants:
        best, bm = None, np.inf
        for k in KS:
            for al in ALPHAS:
                pr = fit_smooth(Y["train"], T["train"], V, Ahat, k, al)
                ev = float((((pr(T["valid"]) - Y["valid"]) ** 2) * mask).sum() / mask.sum())
                if ev < bm:
                    best, bm, cfg = pr, ev, (k, al)
        preds[f"smooth_{name}|{gname}"] = best(T["test"])
        print(f"fold {fo} {name}|{gname} cfg k={cfg[0]} alpha={cfg[1]} val_mse {bm:.6f}", flush=True)
    obs, tt = Y["test"], T["test"]
    deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    # reachable = target has any L1 edge in either direction (association neighbourhood non-empty)
    nb1 = np.asarray(adjacency(edges_of([L1])).sum(1)).ravel() > 0
    for m, p in preds.items():
        parts = m.split("|")
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pr_ = M.pcc_delta(p, obs, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        rm = M.rmse_delta(p, obs, tt)
        ap_ = M.deg_auprc(p - tm, deg, tt)["ap"]
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, model=parts[0], graph=parts[1] if len(parts) > 1 else "-",
                             target=f["test"][i], has_l1=bool(nb1[tt[i]]), pcc_c=pc[i], pcc=pr_[i], top20_c=t20[i],
                             rmse=rm[i], deg_ap=ap_[i], cent_rank=cn[i]))
df = pd.DataFrame(rows)
(ART / "scores").mkdir(exist_ok=True)
df.to_csv(ART / "scores" / f"g3_{a.line}.tsv", sep="\t", index=False)
print(df.groupby(["model", "graph"])[["pcc_c", "pcc", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4).to_string())
