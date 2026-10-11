"""Addendum L(d): what does careful preprocessing change? (a) agreement with the earlier targets, (b) knockdown efficiency vs response size,
(c) precision-weighted smoothing head vs unweighted (same folds, same configuration)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
res, rows = {}, []
for line in ("K562_essential", "rpe1"):
    old = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    new = np.load(ART / "data2" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in old["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    po = {p.upper(): i for i, p in enumerate(old["perts"])}
    pn = {p.upper(): i for i, p in enumerate(new["perts"])}
    common = sorted(set(po) & set(pn))
    Do = old["delta"][[po[p] for p in common]]
    Dn = new["delta"][[pn[p] for p in common]]
    tgt = old["target_idx"][[po[p] for p in common]]
    r = []
    for i in range(len(common)):
        m = np.ones(G, bool)
        if tgt[i] >= 0:
            m[tgt[i]] = False
        r.append(np.corrcoef(Do[i][m], Dn[i][m])[0, 1])
    kd = new["kd"][[pn[p] for p in common]]
    det = new["kd_detected"][[pn[p] for p in common]]
    ncell = new["n_cells"][[pn[p] for p in common]]
    mag = np.sqrt(np.mean(np.where(np.arange(G)[None, :] == tgt[:, None], 0, Dn) ** 2, axis=1))
    ok = det & np.isfinite(kd)
    from scipy.stats import spearmanr
    res[line] = dict(n_common=len(common), agree_median=float(np.median(r)), agree_p05=float(np.quantile(r, 0.05)),
                     kd_median=float(np.nanmedian(kd[ok])), kd_p10=float(np.nanquantile(kd[ok], 0.10)), kd_lt50=float(np.mean(kd[ok] < 0.5)),
                     kd_detected=int(ok.sum()), kd_resp_spearman=float(spearmanr(kd[ok], mag[ok])[0]),
                     ncells_median=float(np.median(ncell)))
    # ---- precision-weighted ridge for the smoothing head (k=40, alpha=0.9, same graph as g4)
    prow = {p: i for i, p in enumerate(po)}
    Y = old["delta"]
    tix = old["target_idx"]
    folds = json.load(open(ART / "folds" / f"{line}.json"))
    l1 = load_l1(CACHE)
    st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))

    def pairs(a_, b_):
        m = a_.isin(gi) & b_.isin(gi) & (a_ != b_)
        return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)

    e = np.unique(np.sort(np.concatenate([pairs(l1.source, l1.target), pairs(st.a.str.upper(), st.b.str.upper())]), 1), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G))
    A = ((A + A.T) > 0).astype(float).tocsr()
    Ah = sp.diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
    n_new = {p: new["n_cells"][pn[p]] if p in pn else np.nan for p in po}
    se_all = np.sqrt(old["var_p"] / old["n_cells"][:, None] + old["ctrl_var"][None, :] / int(old["ctrl_n"]))
    for fo, f in enumerate(folds):
        idx = {k: np.array([prow[x.upper()] for x in f[k]]) for k in ("train", "valid", "test")}
        Yt, Tt = Y[idx["train"]].astype(np.float64), tix[idx["train"]]
        tm = Yt.mean(0)
        _, _, Vt = np.linalg.svd(Yt - tm, full_matrices=False)
        V = Vt.T
        k, al = 40, 0.9
        Gk = V[:, :k]
        emb = (1 - al) * Gk + al * (Ah @ Gk)
        P = emb[Tt]
        ncell = np.array([n_new.get(x.upper(), np.nan) for x in f["train"]], float)
        ncell = np.where(np.isfinite(ncell), ncell, np.nanmedian(ncell))
        variants = {"unweighted": np.ones(len(Tt)), "weight_ncells": np.minimum(ncell, 200) / np.minimum(ncell, 200).mean(),
                    "weight_precision": 1.0 / np.mean(se_all[idx["train"]] ** 2, axis=1)}
        variants["weight_precision"] = variants["weight_precision"] / variants["weight_precision"].mean()
        obs = Y[idx["test"]].astype(np.float64)
        tt = tix[idx["test"]]
        for name, w in variants.items():
            Wd = np.diag(w)
            Wm = np.linalg.solve(P.T @ Wd @ P + 1e-2 * np.eye(k), P.T @ Wd @ ((Yt - tm) @ Gk))
            pred = (tm + emb[tt] @ Wm @ Gk.T).astype(np.float32)
            pc = M.pcc_delta_centered(pred, obs, tm, tt)
            for i in range(len(tt)):
                rows.append(dict(line=line, fold=fo, variant=name, target=f["test"][i], pcc_c=pc[i]))
    print(line, res[line], flush=True)
json.dump(res, open(ART / "prep2_stats.json", "w"), indent=1)
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / "g14_prep2.tsv", sep="\t", index=False)
for line, d in df.groupby("line"):
    u = d.groupby(["variant", "target"], as_index=False).pcc_c.mean()
    base = u[u.variant == "unweighted"].set_index("target").pcc_c
    for v in ("weight_ncells", "weight_precision"):
        x = u[u.variant == v].set_index("target").pcc_c
        j = x.index.intersection(base.index)
        r_ = M.paired_bootstrap(x.loc[j].to_numpy(), base.loc[j].to_numpy(), 10000, 0)
        print(line, v, f"{r_['mean_diff']:+.4f} [{r_['ci_low']:+.4f}, {r_['ci_high']:+.4f}] n={r_['n']}; mean {x.mean():.4f} vs {base.mean():.4f}")
