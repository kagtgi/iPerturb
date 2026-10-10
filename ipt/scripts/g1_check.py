"""Gate G1: L1 coverage and training-only sign-signal test, plus fold creation (K562, RPE1)."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.graph.l1 import load_l1
from ipt.data.splits import make_folds

ROOT = Path(__file__).parents[1]; ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
e = load_l1(CACHE)
signed = e[e.sign != 0]
print(f"L1 edges {len(e)}, signed {len(signed)}, conflicts {int(e.sign_conflict.sum())}")
report = {}
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = list(z["genes"]); gi = {g.upper(): i for i, g in enumerate(genes)}
    perts = list(z["perts"]); tix = z["target_idx"]; n = z["n_cells"]
    delta = z["delta"]; varp = z["var_p"]; vc = z["ctrl_var"]; nc = int(z["ctrl_n"]); cm = z["ctrl_mean"]
    on = [i for i in range(len(perts)) if tix[i] >= 0]
    sg = signed[signed.source.isin(gi) & signed.target.isin(gi)]
    outdeg = sg.groupby("source").size(); tf = set(e.source)
    # reach: genes with >=1 signed out-edge to an on-axis target; >=100 transitive reach via BFS
    adj = {s: list(g.target) for s, g in sg.groupby("source")}
    def reach(s):
        seen = {s}; fr = [s]
        while fr:
            nx = []
            for u in fr:
                for v in adj.get(u, []):
                    if v not in seen: seen.add(v); nx.append(v)
            fr = nx
        return len(seen) - 1
    pn = [perts[i].upper() for i in on]
    has = [p in adj for p in pn]
    r = [reach(p) if p in adj else 0 for p in pn]
    # strata: TF status x baseline-expression tertile x outdeg bin (all train-independent)
    base = cm[[tix[i] for i in on]]; q = np.quantile(base, [1/3, 2/3])
    strata = [(p in tf, int(np.searchsorted(q, b)), min(int(outdeg.get(p, 0)).bit_length(), 4)) for p, b in zip(pn, base)]
    folds = make_folds([perts[i] for i in on], strata)
    (ART / "folds").mkdir(exist_ok=True)
    json.dump(folds, open(ART / "folds" / f"{line}.json", "w"))
    rep = dict(n_perts=len(perts), on_axis=len(on), frac_with_signed_out=float(np.mean(has)),
               frac_reach100=float(np.mean(np.array(r) >= 100)), median_reach=float(np.median(r)),
               n_signed_edges_on_axis=len(sg))
    # sign-signal on fold-0 TRAIN targets only. CRISPRi: expected sign(delta_target) = -sign(edge)
    tr = set(folds[0]["train"]); se = np.sqrt(varp / n[:, None] + vc / nc)  # Welch SE per gene
    rows = []
    for p, g in sg.groupby("source"):
        if p not in {x.upper() for x in tr}: continue
        i = [k for k in on if perts[k].upper() == p][0]
        for t, s in zip(g.target, g.sign):
            j = gi[t]; zz = delta[i, j] / max(se[i, j], 1e-6)
            rows.append((p, t, s, zz))
    d = pd.DataFrame(rows, columns=["tf", "tgt", "sign", "z"]); resp = d[d.z.abs() > 3]
    conc = (np.sign(resp.z) == -resp.sign)
    per_tf = resp.assign(c=conc).groupby("tf").c.agg(["mean", "size"]); per_tf = per_tf[per_tf["size"] >= 5]
    rep.update(train_edges=len(d), responsive_edges=len(resp), frac_responsive=float(len(resp) / max(len(d), 1)),
               edge_concordance=float(conc.mean()) if len(resp) else None,
               edge_binom_p=float(stats.binomtest(int(conc.sum()), len(resp), 0.5).pvalue) if len(resp) else None,
               n_tf_ge5=len(per_tf), tf_mean_concordance=float(per_tf["mean"].mean()) if len(per_tf) else None,
               tf_sign_test_p=float(stats.wilcoxon(per_tf["mean"] - 0.5).pvalue) if len(per_tf) > 5 else None)
    report[line] = rep; print(line, json.dumps(rep, indent=1))
json.dump(report, open(ART / "g1_report.json", "w"), indent=1)
