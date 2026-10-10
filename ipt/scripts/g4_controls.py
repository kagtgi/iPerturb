"""Addendum B controls for the association-smoothing gain (see PREREG.md)."""
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
KS, ALPHAS, BLEND = (5, 10, 20, 40), (0.0, 0.25, 0.5, 0.75), (0.0, 0.25, 0.5, 0.75, 1.0)

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


def edges_of(parts):
    e = np.sort(np.concatenate(parts), 1)
    return np.unique(e, axis=0)


def adj01(e):
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


def rownorm(A):
    d = np.asarray(A.sum(1)).ravel()
    return sp.diags(1 / np.maximum(d, 1)) @ A


def rewired(e, seed):
    g = nx.Graph()
    g.add_edges_from(map(tuple, e))
    nx.double_edge_swap(g, nswap=5 * len(e), max_tries=50 * len(e), seed=seed)
    return np.array(g.edges())


def knn_matched(V, deg, k_pc=20):
    """Cosine kNN among gene loadings (training data only); per-gene degree matched to `deg`."""
    X = V[:, :k_pc]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rows, cols = [], []
    for s in range(0, G, 1000):
        S = X[s:s + 1000] @ X.T
        for r in range(S.shape[0]):
            d = int(deg[s + r])
            if d == 0:
                continue
            S[r, s + r] = -2
            nb = np.argpartition(-S[r], d)[:d]
            rows += [s + r] * d
            cols += nb.tolist()
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(G, G)).tocsr()
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


def fit_smooth(D, tgt, V, Ahat, k, alpha, ridge=1e-2):
    Gk = V[:, :k]
    emb = (1 - alpha) * Gk + alpha * (Ahat @ Gk)
    b = D.mean(0)
    P = emb[tgt]
    W = np.linalg.solve(P.T @ P + ridge * np.eye(k), P.T @ ((D - b) @ Gk))
    return lambda t: (b + emb[t] @ W @ Gk.T).astype(np.float32)


def select(D, tgt, V, Ahat, Dv, tv, mask):
    best, bm, cfg = None, np.inf, None
    for k in KS:
        for al in ALPHAS:
            pr = fit_smooth(D, tgt, V, Ahat, k, al)
            ev = float((((pr(tv) - Dv) ** 2) * mask).sum() / mask.sum())
            if ev < bm:
                best, bm, cfg = pr, ev, (k, al)
    return best, bm, cfg


rows, cfgs, preds_out = [], [], {}
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float64) for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    mask = np.ones_like(Y["valid"], bool)
    mask[np.arange(len(T["valid"])), T["valid"]] = False
    # SVD with and without the self-target entry (zeroed to the training column mean)
    Dz = Y["train"].copy()
    Dz[np.arange(len(T["train"])), T["train"]] = tm[T["train"]]
    V = {}
    for tag, D in (("self", Y["train"]), ("noself", Dz)):
        _, _, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)
        V[tag] = Vt.T
    A3 = adj01(edges_of([L3]))
    A13 = adj01(edges_of([L1, L3]))
    A3r = adj01(rewired(edges_of([L3]), a.seed + fo))
    A13r = adj01(rewired(edges_of([L1, L3]), a.seed + fo))
    Ak = knn_matched(V["self"], np.asarray(A3.sum(1)).ravel())
    nbm = {t: i for i, t in enumerate(T["train"])}  # train target gene -> train row (first occurrence)
    variants = [("none", "real", "self", None), ("L3", "real", "self", A3), ("L3", "rewired", "self", A3r),
                ("L1+L3", "real", "self", A13), ("L1+L3", "rewired", "self", A13r),
                ("none", "real", "noself", None), ("L3", "real", "noself", A3), ("L3", "rewired", "noself", A3r),
                ("L1+L3", "real", "noself", A13), ("L1+L3", "rewired", "noself", A13r),
                ("dataknn", "real", "self", Ak)]
    preds = {"B0_nochange": np.zeros_like(Y["test"]), "B1_trainmean": np.tile(tm, (len(T["test"]), 1))}
    for name, gname, svd, A in variants:
        Ahat = rownorm(A) if A is not None else sp.csr_matrix((G, G))
        D = Dz if svd == "noself" else Y["train"]
        pr, bm, cfg = select(D, T["train"], V[svd], Ahat, Y["valid"], T["valid"], mask)
        key = f"smooth_{name}|{gname}|{svd}"
        preds[key] = pr(T["test"])
        cfgs.append(dict(line=a.line, fold=fo, variant=key, k=cfg[0], alpha=cfg[1], val_mse=bm))
        print(f"fold {fo} {key} k={cfg[0]} alpha={cfg[1]} val_mse {bm:.6f}", flush=True)
    # neighbour-mean baseline (no SVD), blended with B2 on validation
    b2 = B.fit_linear_select(Y["train"], T["train"], Y["valid"], T["valid"])
    A3n = A3.tocsr()

    def nbr_pred(tg, blend):
        base = b2.predict(tg).astype(np.float64)
        out = base.copy()
        for r, t in enumerate(tg):
            nb = [nbm[j] for j in A3n.indices[A3n.indptr[t]:A3n.indptr[t + 1]] if j in nbm]
            if nb:
                out[r] = (1 - blend) * base[r] + blend * Y["train"][nb].mean(0)
        return out.astype(np.float32)

    bv = min(BLEND, key=lambda al: float((((nbr_pred(T["valid"], al) - Y["valid"]) ** 2) * mask).sum()))
    preds["nbrmean_L3|real|self"] = nbr_pred(T["test"], bv)
    cfgs.append(dict(line=a.line, fold=fo, variant="nbrmean_L3|real|self", k=-1, alpha=bv, val_mse=np.nan))
    preds["B2_linear"] = b2.predict(T["test"])
    obs, tt = Y["test"], T["test"]
    deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    for m, p in preds.items():
        preds_out[f"f{fo}|{m}"] = p.astype(np.float16)
        parts = m.split("|")
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pres = M.pcc_delta_resid(p, obs, tm, tt)
        pr_ = M.pcc_delta(p, obs, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        rm = M.rmse_delta(p, obs, tt)
        ap_ = M.deg_auprc(p - tm, deg, tt)["ap"]
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, model=parts[0], graph=parts[1] if len(parts) > 1 else "-",
                             svd=parts[2] if len(parts) > 2 else "-", target=f["test"][i], pcc_c=pc[i], pcc_resid=pres[i],
                             pcc=pr_[i], top20_c=t20[i], rmse=rm[i], deg_ap=ap_[i], cent_rank=cn[i]))
df = pd.DataFrame(rows)
(ART / "scores").mkdir(exist_ok=True)
df.to_csv(ART / "scores" / f"g4_{a.line}.tsv", sep="\t", index=False)
pd.DataFrame(cfgs).to_csv(ART / "scores" / f"g4cfg_{a.line}.tsv", sep="\t", index=False)
(ART / "preds").mkdir(exist_ok=True)
np.savez_compressed(ART / "preds" / f"g4_{a.line}.npz", **preds_out)
df["mid"] = df.model + "|" + df.graph + "|" + df.svd
print(df.groupby("mid")[["pcc_c", "pcc_resid", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4).to_string())
