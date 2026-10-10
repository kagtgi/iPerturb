"""Addendum F: context-specific L2 (GeneHancer) vs foreign-context L2 in association smoothing, K562."""
import json, pickle, sys
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1

ART = Path(__file__).parents[1] / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
line = "K562_essential"
KS, ALPHAS = (5, 10, 20, 40), (0.0, 0.25, 0.5, 0.75)
z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]; gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
perts = [p.upper() for p in z["perts"]]; prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
folds = json.load(open(ART / "folds" / f"{line}.json"))


def pairs(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)


l1 = load_l1(CACHE); L1 = pairs(l1.source, l1.target)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb")); L3 = pairs(st.a.str.upper(), st.b.str.upper())
L2 = {}
for nm, f in (("K562", "l2_fullaxis_K562_8002c0173f81.pkl"), ("HepG2", "l2_fullaxis_HepG2_d69da0c42ad5.pkl"), ("Jurkat", "l2_fullaxis_Jurkat_f1898bae0459.pkl")):
    d = pickle.load(open(CACHE / f, "rb"))[0]
    L2[nm] = pairs(d.source.str.upper(), d.target.str.upper())
uniq = lambda e: np.unique(np.sort(e, 1), axis=0)
base = uniq(np.concatenate([L1, L3]))

# descriptive graph-level context specificity
desc = {}
for a_ in L2:
    for b_ in L2:
        if a_ < b_:
            A, B = set(map(tuple, uniq(L2[a_]))), set(map(tuple, uniq(L2[b_])))
            desc[f"jaccard_{a_}_{b_}"] = len(A & B) / len(A | B)
for nm in L2:
    desc[f"n_edges_{nm}"] = int(len(uniq(L2[nm])))
od = {nm: np.bincount(L2[nm][:, 0], minlength=G) for nm in L2}
for a_ in L2:
    for b_ in L2:
        if a_ < b_:
            desc[f"tf_outdeg_spearman_{a_}_{b_}"] = float(pd.Series(od[a_]).corr(pd.Series(od[b_]), method="spearman"))
json.dump(desc, open(ART / "g7_graph_context.json", "w"), indent=1); print(desc)


def adj(e):
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G)); A = (A + A.T).tocsr(); A.data[:] = 1
    d = np.asarray(A.sum(1)).ravel(); return sp.diags(1 / np.maximum(d, 1)) @ A


def rewired(e, seed):
    g = nx.Graph(); g.add_edges_from(map(tuple, e)); nx.double_edge_swap(g, nswap=3 * len(e), max_tries=30 * len(e), seed=seed); return np.array(g.edges())


def fit_smooth(D, tgt, V, Ah, k, al, ridge=1e-2):
    Gk = V[:, :k]; emb = (1 - al) * Gk + al * (Ah @ Gk); b = D.mean(0); P = emb[tgt]
    W = np.linalg.solve(P.T @ P + ridge * np.eye(k), P.T @ ((D - b) @ Gk)); return lambda t: (b + emb[t] @ W @ Gk.T).astype(np.float32)


rows, cfgs = [], []
for fo in (0, 1, 2):
    f = folds[fo]; idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float64) for k in idx}; T = {k: tix[idx[k]] for k in idx}; tm = Y["train"].mean(0)
    mask = np.ones_like(Y["valid"], bool); mask[np.arange(len(T["valid"])), T["valid"]] = False
    _, _, Vt = np.linalg.svd(Y["train"] - tm, full_matrices=False); V = Vt.T
    cand = {"base": base, "base+L2_K562": uniq(np.concatenate([base, L2["K562"]])), "base+L2_HepG2": uniq(np.concatenate([base, L2["HepG2"]])),
            "base+L2_Jurkat": uniq(np.concatenate([base, L2["Jurkat"]]))}
    cand["base+L2_K562_rewired"] = rewired(cand["base+L2_K562"], 42 + fo)
    obs, tt = Y["test"], T["test"]; deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    for name, e in cand.items():
        Ah = adj(e); best, bm, cfg = None, np.inf, None
        for k in KS:
            for al in ALPHAS:
                pr = fit_smooth(Y["train"], T["train"], V, Ah, k, al)
                ev = float((((pr(T["valid"]) - Y["valid"]) ** 2) * mask).sum() / mask.sum())
                if ev < bm: best, bm, cfg = pr, ev, (k, al)
        p = best(T["test"]); cfgs.append(dict(fold=fo, variant=name, k=cfg[0], alpha=cfg[1], val_mse=bm, n_edges=len(e)))
        print(f"fold {fo} {name} edges={len(e)} k={cfg[0]} alpha={cfg[1]} val_mse {bm:.6f}", flush=True)
        pc = M.pcc_delta_centered(p, obs, tm, tt); pres = M.pcc_delta_resid(p, obs, tm, tt); t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        ap_ = M.deg_auprc(p - tm, deg, tt)["ap"]; cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append(dict(fold=fo, variant=name, target=f["test"][i], pcc_c=pc[i], pcc_resid=pres[i], top20_c=t20[i], deg_ap=ap_[i], cent_rank=cn[i]))
df = pd.DataFrame(rows); df.to_csv(ART / "scores" / "g7_K562.tsv", sep="\t", index=False)
pd.DataFrame(cfgs).to_csv(ART / "scores" / "g7cfg_K562.tsv", sep="\t", index=False)
u = df.groupby(["variant", "target"], as_index=False)[["pcc_c", "pcc_resid", "top20_c", "deg_ap", "cent_rank"]].mean()
print(u.groupby("variant").mean(numeric_only=True).round(4).to_string())
for m in ("pcc_c", "pcc_resid"):
    for a_, b_ in (("base+L2_K562", "base"), ("base+L2_K562", "base+L2_HepG2"), ("base+L2_K562", "base+L2_Jurkat"), ("base+L2_K562", "base+L2_K562_rewired")):
        x = u[u.variant == a_].set_index("target")[m]; y = u[u.variant == b_].set_index("target")[m]; j = x.index.intersection(y.index)
        r = M.paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)
        print(f"{m}: {a_} - {b_}: {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['p_two_sided']:.3f} n={r['n']}")
