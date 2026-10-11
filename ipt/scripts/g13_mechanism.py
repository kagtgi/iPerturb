"""Addendum L: validate the clamped steady-state propagation as a biological mechanism on held-out reachable CRISPRi targets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.explain.mechanism import build_W, clamped_response, reachable, top_paths
from ipt.graph.l1 import load_l1

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--draws", type=int, default=20)
a = ap.parse_args()
rng = np.random.default_rng(0)

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n_c = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n_c[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
cm = z["ctrl_mean"]
folds = json.load(open(ART / "folds" / f"{a.line}.json"))

# ---- signed edges with provenance (curated union + SIGNOR)
e = load_l1(CACHE)
e = e[(e.sign != 0) & e.source.isin(gi) & e.target.isin(gi)][["source", "target", "sign", "dbs", "n_db"]].copy()
sg = ART / "ext" / "L1" / "signor.tsv"
if sg.exists():
    s = pd.read_csv(sg, sep="\t")
    s = s[(s.sign != 0) & s.source.isin(gi) & s.target.isin(gi)]
    s = pd.DataFrame({"source": s.source, "target": s.target, "sign": s.sign, "dbs": "SIGNOR", "n_db": 1})
    both = pd.concat([e, s])
    agg = both.groupby(["source", "target"]).agg(pos=("sign", lambda x: (x > 0).any()), neg=("sign", lambda x: (x < 0).any()),
                                                 dbs=("dbs", lambda x: ",".join(sorted(set(",".join(x).split(","))))), n_db=("n_db", "max")).reset_index()
    agg["n_db"] = agg.dbs.str.count(",") + 1
    agg = agg[agg.pos != agg.neg]            # conflicting signs excluded
    agg["sign"] = np.where(agg.pos, 1, -1)
    e = agg[["source", "target", "sign", "dbs", "n_db"]]
src, dst = e.source.map(gi).to_numpy(), e.target.map(gi).to_numpy()
sign = e.sign.to_numpy(float)
conf = (np.minimum(e.n_db.to_numpy(), 3) / 3.0)
info = {(int(s_), int(d_)): dict(dbs=db, sign=int(sg_)) for s_, d_, db, sg_ in zip(src, dst, e.dbs, e.sign)}
print(f"{a.line}: {len(e)} signed edges", flush=True)
has_out = set(src.tolist())


def world(kind):
    """kind: real | signperm | rewired"""
    s, d, sg_ = src.copy(), dst.copy(), sign.copy()
    if kind == "signperm":
        sg_ = rng.permutation(sg_)
    elif kind == "rewired":
        d = d.copy()
        pairs = set(zip(s.tolist(), d.tolist()))
        for _ in range(5 * len(d)):
            i, j = rng.integers(0, len(d), 2)
            if i == j:
                continue
            A_, B_ = (s[i], d[j]), (s[j], d[i])
            if A_[0] == A_[1] or B_[0] == B_[1] or A_ in pairs or B_ in pairs:
                continue
            pairs.discard((s[i], d[i])); pairs.discard((s[j], d[j])); pairs.add(A_); pairs.add(B_)
            d[i], d[j] = d[j], d[i]
    return build_W(s, d, sg_, conf, G, rho=0.9)[0]


def dxg_for(g, zeta):
    c = cm[g]
    return float(np.log1p((1 - zeta) * np.expm1(max(c, 0.0))) - c)


rows, reports = [], []
for fo, f in enumerate(folds):
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]] for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    ar = np.arange(len(T["train"]))
    ratio = np.expm1(np.clip(cm[T["train"]] + Y["train"][ar, T["train"]], 0, None)) / np.maximum(np.expm1(cm[T["train"]]), 1e-9)
    zeta = float(1 - np.clip(np.median(ratio), 0.05, 0.95))
    for kind in ["real"] + [f"{k}{r}" for k in ("signperm", "rewired") for r in range(a.draws if fo == 0 else a.draws)]:
        W = world(kind.rstrip("0123456789")) if kind != "real" else world("real")
        # gain kappa from training reachable perturbations (least squares through the origin)
        num = den = 0.0
        for i, g in enumerate(T["train"]):
            if g not in has_out:
                continue
            dist = reachable(W, g)
            nodes, steady, _ = clamped_response(W, g, dxg_for(g, zeta), dist)
            if len(nodes) == 0:
                continue
            obs = Y["train"][i][nodes]
            num += float(steady @ obs); den += float(steady @ steady)
        kappa = max(num / den, 0.0) if den > 0 else 0.0
        for i, g in enumerate(T["test"]):
            if g not in has_out:
                continue
            dist = reachable(W, g)
            nodes, steady, hops = clamped_response(W, g, dxg_for(g, zeta), dist)
            if len(nodes) == 0:
                continue
            obs = Y["test"][i][nodes]
            zz = obs / np.maximum(se[idx["test"][i]][nodes], 1e-6)
            resp = np.abs(zz) > 3
            pred = {"steady": steady, "K1": hops[0], "K2": hops[0] + hops[1], "K3": hops[0] + hops[1] + hops[2], "K4": sum(hops)}
            dd = np.array([dist[n] for n in nodes])
            for name, p in pred.items():
                for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 99)):
                    m = resp & (dd >= lo) & (dd <= hi) & (np.abs(p) > 1e-12)
                    if m.sum() == 0:
                        continue
                    rows.append(dict(line=a.line, fold=fo, world=kind, model=name, hop=f"{lo}" if lo < 4 else "4+", target=f["test"][i],
                                     n=int(m.sum()), hit=int((np.sign(p[m]) == np.sign(obs[m])).sum()), kappa=kappa))
            if kind == "real" and name == "steady":
                pass
            if kind == "real":
                # reached-gene correlation
                if len(nodes) >= 5 and np.std(steady) > 0 and np.std(obs) > 0:
                    rows.append(dict(line=a.line, fold=fo, world=kind, model="steady", hop="corr", target=f["test"][i], n=len(nodes),
                                     hit=float(np.corrcoef(steady, obs)[0, 1]), kappa=kappa))
        if kind == "real":
            # mechanism reports for the 3 test targets with the most responding reached genes
            cand = []
            for i, g in enumerate(T["test"]):
                if g not in has_out:
                    continue
                dist = reachable(W, g)
                nodes, steady, hops = clamped_response(W, g, dxg_for(g, zeta), dist)
                if len(nodes) == 0:
                    continue
                zz = Y["test"][i][nodes] / np.maximum(se[idx["test"][i]][nodes], 1e-6)
                cand.append((int((np.abs(zz) > 3).sum()), i, g, nodes, steady, zz, dist))
            for cnt, i, g, nodes, steady, zz, dist in sorted(cand, key=lambda x: -x[0])[:3] if fo == 0 else []:
                top = np.argsort(-np.abs(zz))[:5]
                lines = [f"## {genes[g]} ({a.line}, fold {fo}) -- {cnt} responding genes among {len(nodes)} reached", f"knockdown assumed (training median): {zeta:.2f}; gain kappa {kappa:.3f}", ""]
                for k in top:
                    t = nodes[k]
                    ps, summ = top_paths(W, g, t, 3, 3, info)
                    lines.append(f"- {genes[t]}: observed z={zz[k]:+.1f}, predicted sign {'+' if steady[k] > 0 else '-'} (hop {dist[t]}); reinforcing {summ['reinforcing']:+.3f} vs opposing {summ['opposing']:+.3f} over {summ['n_paths']} paths")
                    for p_ in ps:
                        chain = " -> ".join(genes[x] for x in p_["path"])
                        ev = "; ".join(f"{genes[ed['src']]}{'->' if ed['sign'] > 0 else '-|'}{genes[ed['dst']]} [{ed.get('dbs', '?')}]" for ed in p_["edges"])
                        lines.append(f"    path {chain} (net {p_['effect']:+.3f}): {ev}")
                reports.append("\n".join(lines))
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g13_{a.line}.tsv", sep="\t", index=False)
(ART / "reports").mkdir(exist_ok=True)
(ART / "reports" / f"mechanism_{a.line}.md").write_text("\n\n".join(reports), encoding="utf8")
acc = df[df.hop != "corr"].copy()
acc["kind"] = acc.world.str.replace(r"\d+$", "", regex=True)
g = acc.groupby(["kind", "model", "hop"]).agg(n=("n", "sum"), hit=("hit", "sum"))
g["acc"] = g.hit / g.n
print(g.round(3).to_string())
c = df[df.hop == "corr"]
print("reached-gene Pearson (real, steady): mean", round(c.hit.mean(), 3), "n targets", c.target.nunique())
