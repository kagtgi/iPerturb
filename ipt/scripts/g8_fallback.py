"""Addendum G (exploratory): fallback to the no-graph prediction for isolated targets, from saved predictions."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
ART = Path(__file__).parents[1] / "artifacts"; CACHE = Path("D:/iPerturb/lnbi/cache")
rows = []
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in z["genes"]]; gi = {g: i for i, g in enumerate(genes)}; G = len(genes)
    perts = [p.upper() for p in z["perts"]]; prow = {p: i for i, p in enumerate(perts)}
    Yall, tix = z["delta"], z["target_idx"]; folds = json.load(open(ART / "folds" / f"{line}.json"))
    P = np.load(ART / "preds" / f"g4_{line}.npz")
    l1 = load_l1(CACHE); st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
    def pairs(a_, b_):
        m = a_.isin(gi) & b_.isin(gi) & (a_ != b_); return np.stack([a_[m].map(gi).to_numpy(), b_[m].map(gi).to_numpy()], 1)
    e = np.concatenate([pairs(l1.source, l1.target), pairs(st.a.str.upper(), st.b.str.upper())])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(G, G)); deg = np.asarray(((A + A.T) > 0).sum(1)).ravel()
    for fo in range(3):
        f = folds[fo]; ix = np.array([prow[p.upper()] for p in f["test"]]); obs = Yall[ix].astype(np.float64); tt = tix[ix]
        tr = np.array([prow[p.upper()] for p in f["train"]]); tm = Yall[tr].mean(0)
        sm = P[f"f{fo}|smooth_L1+L3|real|self"].astype(np.float32); no = P[f"f{fo}|smooth_none|real|self"].astype(np.float32)
        iso = deg[tt] == 0; fb = sm.copy(); fb[iso] = no[iso]
        for name, p in (("smooth_L1+L3", sm), ("smooth_L1+L3+fallback", fb)):
            pc = M.pcc_delta_centered(p, obs, tm, tt); pr = M.pcc_delta_resid(p, obs, tm, tt)
            for i in range(len(tt)): rows.append(dict(line=line, fold=fo, target=f["test"][i], model=name, iso=bool(iso[i]), pcc_c=pc[i], pcc_resid=pr[i]))
df = pd.DataFrame(rows); df.to_csv(ART / "scores" / "g8_fallback.tsv", sep="\t", index=False)
for line, d in df.groupby("line"):
    u = d.groupby(["model", "target"], as_index=False)[["pcc_c", "pcc_resid"]].mean()
    iso_n = d[d.model == "smooth_L1+L3"].groupby("target").iso.first()
    print(line, "isolated test targets:", int(iso_n.sum()), "of", len(iso_n))
    a = u[u.model == "smooth_L1+L3+fallback"].set_index("target"); b = u[u.model == "smooth_L1+L3"].set_index("target")
    for m in ("pcc_c", "pcc_resid"):
        r = M.paired_bootstrap(a[m].to_numpy(), b.loc[a.index, m].to_numpy(), 10000, 0)
        print(f"  {m}: mean {a[m].mean():.4f} vs {b[m].mean():.4f}; diff {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
