"""Addendum K: network-propagation kernels for propagating training responses to unseen perturbed genes."""
import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
from ipt.models import baselines as B
from ipt.models.diffusion import kernel_scores, propagate_responses

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--tag", default="")
ap.add_argument("--extra", default="", help="comma list of name=path (tsv with a,b or source,target) extra tier graphs")
a = ap.parse_args()

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n_c = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n_c[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
folds = json.load(open(ART / "folds" / f"{a.line}.json"))


def pairs(a_, b_):
    m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
    return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)


def adj(e):
    e = np.unique(np.sort(e, 1), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


l1 = load_l1(CACHE)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
cx = pickle.load(open(CACHE / "l4_coex_parsed.pkl", "rb"))
cx = cx.assign(source=cx.source.str.upper(), target=cx.target.str.upper()).sort_values("mr", ascending=False).groupby("source").head(20)
tiers = {"L1": adj(pairs(l1.source, l1.target)), "L3": adj(pairs(st.a.str.upper(), st.b.str.upper())), "L4": adj(pairs(cx.source, cx.target))}
for item in [x for x in a.extra.split(",") if x]:
    nm, path = item.split("=")
    d = pd.read_csv(path, sep="\t")
    ca, cb = ("a", "b") if "a" in d.columns else ("source", "target")
    tiers[nm] = adj(pairs(d[ca].astype(str).str.upper(), d[cb].astype(str).str.upper()))
tiers["L1+L3"] = adj(np.concatenate([pairs(l1.source, l1.target), pairs(st.a.str.upper(), st.b.str.upper())]))
print({k: int(v.nnz // 2) for k, v in tiers.items()}, flush=True)

KERNELS = ([("onehop", {})] + [("rwr", dict(alpha=al, norm=nm)) for al in (0.1, 0.3, 0.5, 0.7, 0.9) for nm in ("rw", "sym")]
           + [("heat", dict(t=t)) for t in (0.5, 1.0, 2.0)] + [("trunc", dict(steps=k, norm="rw")) for k in (1, 2, 3)])
GAMMAS, LAMS = (0.0, 0.5, 1.0), (0.25, 0.5, 0.75, 1.0)
rows = []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]].astype(np.float32) for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    b2 = B.fit_linear_select(Y["train"], T["train"], Y["valid"], T["valid"])
    base = {"valid": b2.predict(T["valid"]), "test": b2.predict(T["test"])}
    mval = np.ones_like(Y["valid"], bool)
    mval[np.arange(len(T["valid"])), T["valid"]] = False
    obs, tt = Y["test"].astype(np.float64), T["test"]
    deg_lab = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    results = {"B2_linear": base["test"]}
    prop_cache = {}

    def mse(p):
        return float((((p - Y["valid"]) ** 2) * mval).sum() / mval.sum())

    def run_graph(gname, A, kernels=KERNELS):
        deg = np.asarray(A.sum(0)).ravel()
        best_by_family = {}
        for kname, kw in kernels:
            S = {s: kernel_scores(A, T[s], kname, **kw) for s in ("valid", "test")}
            for gamma in (GAMMAS if kname != "onehop" else (0.0,)):
                pr = {}
                ok = {}
                for s in ("valid", "test"):
                    pr[s], ok[s] = propagate_responses(S[s], T[s], T["train"], Y["train"], gamma, deg)
                prop_cache[(gname, kname, tuple(sorted(kw.items())), gamma)] = (pr, ok)
                for lam in LAMS:
                    pv = base["valid"].copy()
                    pv[ok["valid"]] = (1 - lam) * base["valid"][ok["valid"]] + lam * pr["valid"][ok["valid"]]
                    v = mse(pv)
                    fam = kname
                    if fam not in best_by_family or v < best_by_family[fam][0]:
                        best_by_family[fam] = (v, kw, gamma, lam, pr["test"], ok["test"])
        return best_by_family

    chosen = {}
    for gname in ("L1+L3",) + tuple(k for k in tiers if k not in ("L1+L3",)):
        bf = run_graph(gname, tiers[gname])
        for fam, (v, kw, gamma, lam, prt, okt) in bf.items():
            p = base["test"].copy()
            p[okt] = (1 - lam) * base["test"][okt] + lam * prt[okt]
            name = f"prop[{gname}]|{fam}"
            results[name] = p
            chosen[name] = dict(val=v, kw=str(kw), gamma=gamma, lam=lam)
        print(f"fold {fo} {gname}: best families {[(k, round(v[0], 5)) for k, v in bf.items()]}", flush=True)
    # multi-network integration: best kernel per single-tier graph (L1, L3, L4 + extras), NNLS weights on validation
    singles = [k for k in tiers if k not in ("L1+L3",)]
    comp_v, comp_t = [base["valid"]], [base["test"]]
    for gname in singles:
        fams = {k.split("|")[1]: k for k in results if k.startswith(f"prop[{gname}]")}
        bestname = min((chosen[n]["val"], n) for n in fams.values())[1]
        key = bestname
        # rebuild validation/test predictions of that choice
        v_, kw_, gamma_, lam_ = chosen[key]["val"], eval(chosen[key]["kw"]), chosen[key]["gamma"], chosen[key]["lam"]
        fam = key.split("|")[1]
        pr, ok = prop_cache[(gname, fam, tuple(sorted(kw_.items())), gamma_)]
        for s, store in (("valid", comp_v), ("test", comp_t)):
            p = base[s].copy()
            p[ok[s]] = pr[s][ok[s]]
            store.append(p)
    X = np.stack([c[mval].ravel() if i < 0 else c.ravel() for i, c in enumerate(comp_v)], 1)
    mflat = mval.ravel()
    Xm = np.stack([c.ravel()[mflat] for c in comp_v], 1)
    ym = Y["valid"].ravel()[mflat]
    sub = np.random.default_rng(0).choice(len(ym), min(len(ym), 600000), replace=False)
    w, _ = nnls(Xm[sub].astype(np.float64), ym[sub].astype(np.float64))
    w = w / max(w.sum(), 1e-9)
    results["prop[integrated]|nnls"] = sum(wi * c for wi, c in zip(w, comp_t)).astype(np.float32)
    chosen["prop[integrated]|nnls"] = dict(w=",".join(f"{x:.2f}" for x in w), comps="B2," + ",".join(singles))
    print(f"fold {fo} integration weights: B2,{singles} = {np.round(w, 2)}", flush=True)
    for m_, p in results.items():
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pres = M.pcc_delta_resid(p, obs, tm, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        ap_ = M.deg_auprc(p - tm, deg_lab, tt)["ap"]
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, model=m_, target=f["test"][i], pcc_c=pc[i], pcc_resid=pres[i], top20_c=t20[i], deg_ap=ap_[i], cent_rank=cn[i]))
    pd.DataFrame([dict(fold=fo, model=k, **v) for k, v in chosen.items()]).to_csv(ART / "scores" / f"g12cfg_{a.line}{a.tag}_f{fo}.tsv", sep="\t", index=False)
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g12_{a.line}{a.tag}.tsv", sep="\t", index=False)
u = df.groupby(["model", "target"], as_index=False)[["pcc_c", "pcc_resid", "top20_c", "deg_ap", "cent_rank"]].mean()
print(u.groupby("model").mean(numeric_only=True).sort_values("pcc_c").round(4).to_string())
