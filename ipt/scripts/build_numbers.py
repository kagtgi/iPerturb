"""Writes paper/numbers.tex: every number quoted in the manuscript text, computed from the artifacts."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.graph.l1 import load_l1
from ipt.metrics import paired_bootstrap
from ipt.viz.load import LINES, mean_ci, unique_target_scores

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
S = ART / "scores"
N = {}
SHORT = {"K562_essential": "k562", "rpe1": "rpe1"}


def put(k, v, fmt=None):
    N[k] = v if fmt is None else fmt.format(v)


def ci(r, d=3):
    return f"{r['mean_diff']:+.{d}f} [{r['ci_low']:+.{d}f}, {r['ci_high']:+.{d}f}]"


# ---- data / graph counts
l1 = load_l1("D:/iPerturb/lnbi/cache")
put("l1_unique", len(l1), "{:,}")
put("l1_signed", int((l1.sign != 0).sum()), "{:,}")
for line in LINES:
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    s = SHORT[line]
    put(f"{s}_genes", len(z["genes"]), "{:,}")
    put(f"{s}_targets", int((z["target_idx"] >= 0).sum()), "{:,}")
    put(f"{s}_perts", len(z["perts"]), "{:,}")
g1 = json.load(open(ART / "g1_report.json"))
for line in LINES:
    s, r = SHORT[line], g1[line]
    put(f"{s}_signed_out_pct", 100 * r["frac_with_signed_out"], "{:.1f}")
    put(f"{s}_reach100_pct", 100 * r["frac_reach100"], "{:.1f}")
    put(f"{s}_concord", 100 * r["edge_concordance"], "{:.0f}")
    put(f"{s}_resp_edges", r["responsive_edges"], "{:,}")
    put(f"{s}_train_edges", r["train_edges"], "{:,}")

# ---- main prediction results
for line in LINES:
    s = SHORT[line]
    U = unique_target_scores(line)
    put(f"{s}_nuniq", U.target.nunique(), "{:,}")

    def m(mid, met):
        return mean_ci(U[U.mid == mid][met])[0]

    def pb(a, b, met="pcc_c"):
        x = U[U.mid == a].set_index("target")[met]
        y = U[U.mid == b].set_index("target")[met]
        j = x.index.intersection(y.index)
        return paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)

    for key, mid in (("ip", "smooth_L1+L3|real|self"), ("b2", "B2_linear|-|-"), ("b0", "B0_nochange|-|-"), ("rew", "smooth_L1+L3|rewired|self"),
                     ("knn", "smooth_dataknn|real|self"), ("gcn", "B5_gcn|real|self"), ("nbr", "nbrmean_L3|real|self"), ("str", "smooth_L3|real|self"),
                     ("gcnrew", "B5_gcn|rewired|self")):
        for met, mk in (("pcc_c", "rc"), ("pcc_resid", "rr"), ("top20_c", "r20"), ("cent_rank", "cent"), ("deg_ap", "ap")):
            put(f"{s}_{key}_{mk}", m(mid, met), "{:.3f}")
    for tag, a, b in (("ip_b2", "smooth_L1+L3|real|self", "B2_linear|-|-"), ("ip_rew", "smooth_L1+L3|real|self", "smooth_L1+L3|rewired|self"),
                      ("ip_knn", "smooth_L1+L3|real|self", "smooth_dataknn|real|self"), ("ip_noself", "smooth_L1+L3|real|noself", "smooth_L1+L3|rewired|noself"),
                      ("ip_str_l1", "smooth_L1+L3|real|self", "smooth_L3|real|self"), ("nbr_ip", "nbrmean_L3|real|self", "smooth_L1+L3|real|self"),
                      ("gcn_ip", "B5_gcn|real|self", "smooth_L1+L3|real|self"), ("gcn_rew", "B5_gcn|real|self", "B5_gcn|rewired|self")):
        for met, mk in (("pcc_c", "rc"), ("pcc_resid", "rr")):
            put(f"{s}_d_{tag}_{mk}", ci(pb(a, b, met)))
    if (U.mid == "B4_gears|-|-").any():
        for met, mk in (("pcc_c", "rc"), ("pcc_resid", "rr"), ("top20_c", "r20"), ("cent_rank", "cent"), ("deg_ap", "ap")):
            put(f"{s}_gears_{mk}", m("B4_gears|-|-", met), "{:.3f}")
        put(f"{s}_d_ip_gears_rc", ci(pb("smooth_L1+L3|real|self", "B4_gears|-|-")))
        put(f"{s}_d_ip_gears_rr", ci(pb("smooth_L1+L3|real|self", "B4_gears|-|-", "pcc_resid")))
        put(f"{s}_gears_n", int(U[U.mid == "B4_gears|-|-"].target.nunique()), "{:,}")

# chosen smoothing hyper-parameters
for line in LINES:
    c = pd.read_csv(S / f"g4cfg_{line}.tsv", sep="\t")
    c = c[c.variant == "smooth_L1+L3|real|self"]
    put(f"{SHORT[line]}_cfg", "; ".join(f"fold {int(r.fold)}: $k$={int(r.k)}, $\\alpha$={r.alpha:g}" for r in c.itertuples()))

# ---- G2 gate (Hill vs linear vs B2, reachable targets)
for line, f in (("K562_essential", "g2all_K562_essential"), ("rpe1", "g2_rpe1")):
    d = pd.read_csv(S / f"{f}.tsv", sep="\t")
    R = d[d.reachable]
    s = SHORT[line]
    put(f"{s}_g2_reach_n", R.target.nunique(), "{:,}")
    put(f"{s}_g2_reach_pct", 100 * R.target.nunique() / d.target.nunique(), "{:.1f}")

    def pt(model, graph):
        return R[(R.model == model) & (R.graph == graph)].groupby(["fold", "target"]).pcc_c.mean()

    for tag, (ma, ga, mb, gb) in {"h2_b3": ("H2_hill_regk", "real", "B3_linear", "real"), "h2_b2": ("H2_hill_regk", "real", "B2_linear", "-"),
                                  "h2_rew": ("H2_hill_regk", "real", "H2_hill_regk", "rewired"), "b3_b2": ("B3_linear", "real", "B2_linear", "-")}.items():
        a_, b_ = pt(ma, ga), pt(mb, gb)
        j = a_.index.intersection(b_.index)
        r = paired_bootstrap(a_.loc[j].to_numpy(), b_.loc[j].to_numpy(), 10000, 0)
        put(f"{s}_g2_{tag}", ci(r, 4))

# ---- SCC + oracle
scc = json.load(open(ART / "scc_bench.json"))
for line in LINES:
    s, r = SHORT[line], scc[line]
    put(f"{s}_scc_largest", r["largest_scc"])
    put(f"{s}_scc_speed", r["median_speedup"], "{:.1f}")
    put(f"{s}_scc_diff", r["max_diff"], "{:.0e}")
    put(f"{s}_scc_nodes", r["n_nodes"], "{:,}")

# ---- explanation faithfulness
e6 = pd.read_csv(S / "g6_explain.tsv", sep="\t")
for line in LINES:
    s = SHORT[line]
    for K in (1, 5):
        d = e6[(e6.line == line) & (e6.K == K)].groupby("target")[["drop_top", "drop_rand"]].mean()
        put(f"{s}_faith{K}", ci(paired_bootstrap(d.drop_top.to_numpy(), d.drop_rand.to_numpy(), 10000, 0), 3))
        put(f"{s}_faith_n", len(d), "{:,}")

# ---- efficiency benchmarks (cpu always, gpu if present)
for dev in ("cpu", "gpu"):
    p = ART / f"bench_{dev}.json"
    if not p.exists():
        continue
    b = json.load(open(p))
    t = pd.DataFrame(b["timing"])
    for graph in ("l1", "assoc"):
        for B in sorted(t.B.unique()):
            sub = t[(t.graph == graph) & (t.B == B)].set_index("impl")
            for impl in sub.index:
                put(f"bench_{dev}_{graph}_{B}_{impl.split('_')[0]}_ms", sub.loc[impl, "sec"] * 1e3, "{:.0f}")
                if dev == "gpu":
                    put(f"bench_{dev}_{graph}_{B}_{impl.split('_')[0]}_mb", sub.loc[impl, "mem_mb"], "{:.0f}")
    for k, v in b["equivalence"].items():
        put(f"eq_{k}_dx", v["max_dx"], "{:.1e}")
        put(f"eq_{k}_dd", v["max_ddelta"], "{:.1e}")
    put("eq_scale", b["equivalence"]["cheb16_spmm"]["delta_scale"], "{:.3f}")

# ---- context-specific L2 (Addendum F)
if (S / "g7_K562.tsv").exists():
    g7 = pd.read_csv(S / "g7_K562.tsv", sep="\t")
    u = g7.groupby(["variant", "target"], as_index=False)[["pcc_c", "pcc_resid"]].mean()
    for v in u.variant.unique():
        for met, mk in (("pcc_c", "rc"), ("pcc_resid", "rr")):
            put(f"g7_{v.replace('+', '_')}_{mk}", u[u.variant == v][met].mean(), "{:.3f}")
    for tag, a, b in (("k_base", "base+L2_K562", "base"), ("k_hep", "base+L2_K562", "base+L2_HepG2"), ("k_jur", "base+L2_K562", "base+L2_Jurkat"),
                      ("k_rew", "base+L2_K562", "base+L2_K562_rewired")):
        x = u[u.variant == a].set_index("target").pcc_c
        y = u[u.variant == b].set_index("target").pcc_c
        j = x.index.intersection(y.index)
        put(f"g7_d_{tag}", ci(paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)))
if (ART / "g7_graph_context.json").exists():
    for k, v in json.load(open(ART / "g7_graph_context.json")).items():
        put("g7c_" + k.replace("-", "_"), v, "{:,.0f}" if k.startswith("n_edges") else "{:.2f}")

out = ["% generated by scripts/build_numbers.py -- do not edit"]
for k, v in sorted(N.items()):
    kk = k.replace("_", "")  # LaTeX-safe macro name suffix handled by \datanum
    out.append("\\expandafter\\def\\csname data@%s\\endcsname{%s}" % (k, str(v)))
(ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
json.dump({k: str(v) for k, v in N.items()}, open(ART / "numbers.json", "w"), indent=1)
print(len(N), "numbers written")
