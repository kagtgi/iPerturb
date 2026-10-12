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


def _f(v, d):
    t = f"{v:+.{d}f}"
    return f"{abs(v):.{d}f}" if float(t) == 0 else t


def ci(r, d=3):
    return f"{_f(r['mean_diff'], d)} [{_f(r['ci_low'], d)}, {_f(r['ci_high'], d)}]"


def sci(v, d=1):
    m, e = f"{v:.{d}e}".split("e")
    return "$" + m + chr(92) + "times10^{" + str(int(e)) + "}$"


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
    put(f"{s}_nuniq", U[U.mid == "smooth_L1+L3|real|self"].target.nunique(), "{:,}")
    put(f"{s}_b1_raw", U[U.mid == "B1_trainmean|-|-"].pcc.mean(), "{:.2f}")

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
        return R[(R.model == model) & (R.graph == graph)].groupby("target").pcc_c.mean()

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
    put(f"{s}_scc_diff", sci(r["max_diff"], 0))
    put(f"{s}_scc_nodes", r["n_nodes"], "{:,}")
    put(f"{s}_scc_reach_only", r["median_reach_vs_global"], "{:.1f}")
    put(f"{s}_scc_vs_reach", r["median_speedup_vs_reach"], "{:.1f}")

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
        put(f"eq_{k}_dx", sci(v["max_dx"]))
        put(f"eq_{k}_dd", sci(v["max_ddelta"]))
    put("eq_scale", b["equivalence"]["cheb16_spmm"]["delta_scale"], "{:.3f}")

# ---- per-edge shapes (Addendum H)
pp = ART / "bench_peredge_gpu.json"
if pp.exists():
    pe = pd.DataFrame(json.load(open(pp))["timing"])
    pe = pe[pe.graph == "assoc"]
    for r in (8, 16):
        for B in (32, 128, 512):
            for impl in ("edge", "spmm"):
                v = pe[(pe.r == r) & (pe.B == B) & (pe.impl == impl)].mem_mb.iloc[0]
                put(f"pe_{impl}_r{r}_{B}_gb", "OOM" if np.isnan(v) else f"{v / 1024:.1f}")

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


# ---- approximation error of the Chebyshev form (computed here, not typed)
import torch
from ipt.models.basis_hill import coef_from_k, cheb_clenshaw, fit_table
from ipt.models.hill import hill4

xs = torch.linspace(0, 1, 2001)
for r in (4, 8, 16, 32):
    tab = torch.as_tensor(fit_table(r), dtype=torch.float32)
    err = 0.0
    for k in np.exp(np.linspace(np.log(0.05), 0, 25)):
        c = coef_from_k(torch.full((1,), float(k)), tab)
        err = max(err, float((cheb_clenshaw(xs[None, :], c) - hill4(xs, torch.tensor(float(k)), 2.0)[None, :]).abs().max()))
    put(f"cheb_err_r{r}", sci(err))

# ---- sign-concordance tests
for line in LINES:
    s_, r = SHORT[line], g1[line]
    put(f"{s_}_binom_p", sci(r["edge_binom_p"], 0))
    put(f"{s_}_tf_wilcoxon_p", f"{r['tf_sign_test_p']:.3f}")
    put(f"{s_}_n_tf", r["n_tf_ge5"])
put("binom_p_max", sci(max(g1[l]["edge_binom_p"] for l in LINES), 0))

# ---- exploratory neighbour-free fallback (Addendum G)
if (S / "g8_fallback.tsv").exists():
    g8 = pd.read_csv(S / "g8_fallback.tsv", sep="\t")
    diffs = []
    for line in LINES:
        d = g8[g8.line == line]
        a_ = d[d.model == "smooth_L1+L3+fallback"].groupby("target").pcc_c.mean()
        b_ = d[d.model == "smooth_L1+L3"].groupby("target").pcc_c.mean()
        diffs.append(float((a_ - b_).mean()))
        iso = d[d.model == "smooth_L1+L3"].groupby("target").iso.first()
        put(f"{SHORT[line]}_iso_pct", 100 * iso.mean(), "{:.1f}")
    put("fallback_gain_max", max(diffs), "{:.3f}")

# ---- memory / time ratios from the A100 benchmark
if (ART / "bench_gpu.json").exists():
    t = pd.DataFrame(json.load(open(ART / "bench_gpu.json"))["timing"])
    t = t[t.graph == "assoc"]
    ratios, speed = [], []
    for B in (128, 512):
        e0 = t[(t.B == B) & (t.impl == "E0_exact_edge")].iloc[0]
        e1 = t[(t.B == B) & (t.impl == "E1_exact_spmm")].iloc[0]
        ratios.append(e0.mem_mb / e1.mem_mb)
        speed.append(e0.sec / e1.sec)
    put("memratio_lo", min(ratios), "{:.1f}")
    put("memratio_hi", max(ratios), "{:.1f}")
    put("speedratio_lo", min(speed), "{:.1f}")
    put("speedratio_hi", max(speed), "{:.1f}")
if (ART / "bench_peredge_gpu.json").exists():
    pe2 = pd.DataFrame(json.load(open(ART / "bench_peredge_gpu.json"))["timing"])
    pe2 = pe2[(pe2.graph == "assoc") & (pe2.r == 8) & (pe2.B == 128)].set_index("impl")
    put("pe_memratio_r8_128", pe2.loc["edge", "mem_mb"] / pe2.loc["spmm", "mem_mb"], "{:.1f}")


if (ART / "bench_peredge_gpu.json").exists():
    eqv = json.load(open(ART / "bench_peredge_gpu.json"))["equivalence"]
    put("pe_equiv", sci(max(v for v in eqv.values() if v == v), 0))

# ---- mechanism validation (Addendum L)
def wilson(h, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    ph = h / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    r = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - r, c + r

for line in LINES:
    fp = S / f"g13_{line}.tsv"
    if not fp.exists():
        continue
    g13 = pd.read_csv(fp, sep="\t")
    g13 = g13[g13.hop != "corr"]
    g13["kind"] = g13.world.str.replace(r"\d+$", "", regex=True)
    s_ = SHORT[line]
    for kind in ("real", "signperm", "rewired"):
        for hop in ("1", "2", "3", "4+"):
            d = g13[(g13.kind == kind) & (g13.model == "steady") & (g13.hop == hop)]
            n_, h_ = int(d.n.sum()), float(d.hit.sum())
            lo, hi = wilson(h_, n_)
            put(f"mech_{s_}_{kind}_h{hop.replace('+', 'p')}", f"{h_ / max(n_, 1):.2f}")
            put(f"mech_{s_}_{kind}_h{hop.replace('+', 'p')}_ci", f"[{lo:.2f}, {hi:.2f}]")
            put(f"mech_{s_}_{kind}_h{hop.replace('+', 'p')}_n", f"{n_:,}")
    cc = pd.read_csv(fp, sep="\t")
    cc = cc[cc.hop == "corr"]
    put(f"mech_{s_}_corr", f"{cc.hit.mean():.2f}")
    put(f"mech_{s_}_ntargets", int(g13[(g13.kind == 'real')].target.nunique()))

# ---- convergence certificates (appendix)
cp = ART / "certificate.json"
if cp.exists():
    cert = json.load(open(cp))
    from ipt.models.graphnet2 import hill_lipschitz
    for kk in (0.05, 0.3, 1.0):
        put(f"lh_{str(kk).replace('.', 'p')}", hill_lipschitz(kk), "{:.2f}")
    for line in LINES:
        c, s_ = cert[line], SHORT[line]
        put(f"cert_{s_}_rho_raw", c["abs_rho_raw"], "{:.1f}")
        put(f"cert_{s_}_rho_scaled", c["abs_rho_scaled"], "{:.2f}")
        put(f"cert_{s_}_inf_raw", c["inf_norm_raw"], "{:.0f}")
        put(f"cert_{s_}_inf_scaled", c["inf_norm_scaled"], "{:.1f}")
        put(f"cert_{s_}_terms_med", c["neumann_terms_1e6_median"], "{:.0f}")
        put(f"cert_{s_}_terms_max", c["neumann_terms_1e6_max"], "{:.0f}")
        put(f"cert_{s_}_qv", c["q_v"], "{:.3f}")
        put(f"cert_{s_}_vmax", c["v_max"], "{:.0f}")
        put(f"cert_{s_}_trunc_ratio", c["truncation_bound_max_ratio"], "{:.3f}")
        put(f"cert_{s_}_q", c["q_max"], "{:.3f}")
        put(f"cert_{s_}_qeta", c["q_eta_max"], "{:.3f}")
        put(f"cert_{s_}_iters", c["iters_max"], "{:d}")
        put(f"cert_{s_}_spread", sci(c["spread_max"]))
        put(f"cert_{s_}_resid", sci(c["residual_max"]))
    put("cert_rwr_coldev", sci(max(v["col_sum_dev"] for l in LINES for v in cert[l]["rwr"].values()), 0))

# ---- careful preprocessing (prepare2) and its effects
if (ART / "prep2_stats.json").exists():
    ps = json.load(open(ART / "prep2_stats.json"))
    for line in LINES:
        s_ = SHORT[line]
        q = json.load(open(ART / "data2" / f"{line}.npz.json"))
        put(f"prep_{s_}_cells", q["qc"]["n"], "{:,}")
        put(f"prep_{s_}_kept_pct", 100 * q["qc"]["kept"] / q["qc"]["n"], "{:.1f}")
        put(f"prep_{s_}_fail_genes", q["qc"]["fail_genes"], "{:,}")
        put(f"prep_{s_}_ctrl_cells", q["ctrl_cells"], "{:,}")
        put(f"prep_{s_}_ctrl_batches", q["ctrl_batches"])
        r = ps[line]
        put(f"prep_{s_}_agree", r["agree_median"], "{:.4f}")
        put(f"prep_{s_}_agree_p05", r["agree_p05"], "{:.3f}")
        put(f"prep_{s_}_kd_det", r["kd_detected"], "{:,}")
        put(f"prep_{s_}_kd_med", 100 * r["kd_median"], "{:.0f}")
        put(f"prep_{s_}_kd_lt50", 100 * r["kd_lt50"], "{:.1f}")
        put(f"prep_{s_}_kd_spearman", r["kd_resp_spearman"], "{:.2f}")
        put(f"prep_{s_}_ncells_med", r["ncells_median"], "{:.0f}")
if (S / "g14_prep2.tsv").exists():
    g14 = pd.read_csv(S / "g14_prep2.tsv", sep="\t")
    for line in LINES:
        d = g14[g14.line == line]
        u = d.groupby(["variant", "target"], as_index=False).pcc_c.mean()
        base = u[u.variant == "unweighted"].set_index("target").pcc_c
        for v, key in (("weight_ncells", "wn"), ("weight_precision", "wp")):
            x = u[u.variant == v].set_index("target").pcc_c
            j = x.index.intersection(base.index)
            put(f"prep_{SHORT[line]}_{key}", ci(paired_bootstrap(x.loc[j].to_numpy(), base.loc[j].to_numpy(), 10000, 0)))
out = ["% generated by scripts/build_numbers.py -- do not edit"]
for k, v in sorted(N.items()):
    kk = k.replace("_", "")  # LaTeX-safe macro name suffix handled by \datanum
    out.append("\\expandafter\\def\\csname data@%s\\endcsname{%s}" % (k, str(v)))
(ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
json.dump({k: str(v) for k, v in N.items()}, open(ART / "numbers.json", "w"), indent=1)
print(len(N), "numbers written")
