"""B5: learned 2-layer GCN over the L1+L3 association graph (real vs degree-preserving rewired). See PREREG Addendum C."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
torch.set_num_threads(3)
K = 40

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
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
E13 = np.unique(np.sort(np.concatenate([pairs_to_idx(l1.source, l1.target), pairs_to_idx(st.a.str.upper(), st.b.str.upper())]), 1), axis=0)


def rewired(e, seed):
    g = nx.Graph()
    g.add_edges_from(map(tuple, e))
    nx.double_edge_swap(g, nswap=5 * len(e), max_tries=50 * len(e), seed=seed)
    return np.array(g.edges())


def norm_adj(e):
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    d = np.asarray(A.sum(1)).ravel()
    A = (sp.diags(1 / np.maximum(d, 1)) @ A).tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), (G, G)).coalesce()


class GCN(nn.Module):
    def __init__(self, hidden=64, k=K):
        super().__init__()
        self.n1, self.s1, self.n2, self.s2 = nn.Linear(k, hidden), nn.Linear(k, hidden), nn.Linear(hidden, k), nn.Linear(hidden, k)

    def forward(self, X, A):
        h = torch.relu(torch.sparse.mm(A, self.n1(X)) + self.s1(X))
        return torch.sparse.mm(A, self.n2(h)) + self.s2(h)


def masked_mse(p, y, t):
    m = torch.ones_like(y, dtype=torch.bool)
    m[torch.arange(len(t)), t] = False
    return (((p - y) ** 2) * m).sum() / m.sum()


rows = []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float64) for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    _, _, Vt = np.linalg.svd(Y["train"] - tm, full_matrices=False)
    Gk = torch.tensor(Vt[:K].T, dtype=torch.float32)
    X = Gk / Gk.std(0, keepdim=True)
    b = torch.tensor(tm, dtype=torch.float32)
    t_tr, t_va, t_te = (torch.as_tensor(T[k]) for k in ("train", "valid", "test"))
    Ytr, Yva = (torch.tensor(Y[k], dtype=torch.float32) for k in ("train", "valid"))
    obs, tt = Y["test"], T["test"]
    deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    for gname in ("real", "rewired"):
        A = norm_adj(E13 if gname == "real" else rewired(E13, 42 + fo))
        for seed in a.seeds:
            best = None
            for lr in (1e-3, 3e-3):
                torch.manual_seed(seed)
                net = GCN()
                opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
                bv, bs, bad = np.inf, None, 0
                for ep in range(200):
                    opt.zero_grad()
                    E = net(X, A)
                    loss = masked_mse(b + E[t_tr] @ Gk.T, Ytr, t_tr)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        v = float(masked_mse(b + net(X, A)[t_va] @ Gk.T, Yva, t_va))
                    if v < bv - 1e-9:
                        bv, bs, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
                    else:
                        bad += 1
                        if bad >= 20:
                            break
                if best is None or bv < best[0]:
                    best = (bv, bs, lr)
            net = GCN()
            net.load_state_dict(best[1])
            with torch.no_grad():
                p = (b + net(X, A)[t_te] @ Gk.T).numpy()
            print(f"fold {fo} B5_gcn|{gname}|s{seed} lr={best[2]} val_mse {best[0]:.6f}", flush=True)
            pc = M.pcc_delta_centered(p, obs, tm, tt)
            pres = M.pcc_delta_resid(p, obs, tm, tt)
            pr_ = M.pcc_delta(p, obs, tt)
            t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
            rm = M.rmse_delta(p, obs, tt)
            ap_ = M.deg_auprc(p - tm, deg, tt)["ap"]
            cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
            for i in range(len(tt)):
                rows.append(dict(line=a.line, fold=fo, model="B5_gcn", graph=gname, seed=seed, target=f["test"][i], pcc_c=pc[i],
                                 pcc_resid=pres[i], pcc=pr_[i], top20_c=t20[i], rmse=rm[i], deg_ap=ap_[i], cent_rank=cn[i]))
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g5_{a.line}.tsv", sep="\t", index=False)
print(df.groupby(["model", "graph"])[["pcc_c", "pcc_resid", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4).to_string())
