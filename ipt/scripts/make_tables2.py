"""Tables/macros for the propagation-kernel (g12), loss (g11) and certified-Hill (g10) studies; writes paper/tables/*.tex and paper/numbers2.tex."""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap
from ipt.viz.load import LINES

ROOT = Path(__file__).parents[1]
S = ROOT / "artifacts" / "scores"
OUT = ROOT / "paper" / "tables"
SHORT = {"K562_essential": "k562", "rpe1": "rpe1"}
BS = chr(92)
N = {}


def put(k, v):
    N[k] = v


def f(v, d=3):
    t = f"{v:+.{d}f}"
    return f"{abs(v):.{d}f}" if float(t) == 0 else t


def ci(r, d=3):
    return f"{f(r['mean_diff'], d)} [{f(r['ci_low'], d)}, {f(r['ci_high'], d)}]"


def uniq(df, key):
    return df.groupby([key, "target"], as_index=False)[["pcc_c", "pcc_resid"]].mean()


def diff(u, key, a, b, met="pcc_c"):
    x = u[u[key] == a].set_index("target")[met]
    y = u[u[key] == b].set_index("target")[met]
    j = x.index.intersection(y.index)
    return paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)


# ---------------- g12: propagation kernels, tier extensions, integration
TIER_ROWS = [("L1", "L1 curated (CollecTRI/OmniPath/TRRUST)"), ("signor", "L1 + SIGNOR"), ("L3", "L3 STRING"), ("corum", "L3 CORUM complexes"),
             ("biogrid", "L3 BioGRID physical"), ("L4", "L4 COXPRESdb"), ("gtex_coex", "L4 GTEx co-expression"), ("chipatlas", "L2 ChIP-Atlas (cell line)")]
g12 = {}
for line in LINES:
    p = S / f"g12_{line}.tsv"
    if p.exists():
        g12[line] = pd.read_csv(p, sep="\t")
if g12:
    lines = [BS + "begin{tabular}{lcc}", BS + "toprule", "Graph (random walk with restart, $" + BS + "alpha$ selected) & K562 $" + BS + "rho_c$ & RPE1 $" + BS + "rho_c$" + BS + BS + BS + "midrule"]
    U = {l: uniq(d, "model") for l, d in g12.items()}

    def mean_for(line, name):
        u = U.get(line)
        if u is None or not (u.model == name).any():
            return None
        return float(u[u.model == name].pcc_c.mean())

    rows = [("B2_linear", "Linear low-rank (B2)")]
    for key, lab in TIER_ROWS:
        rows.append((f"prop[{key}]|rwr", lab))
    rows += [("prop[L1+L3]|onehop", "L1+L3, one hop (neighbour mean)"), ("prop[L1+L3]|rwr", "L1+L3, random walk with restart (multi-hop)"),
             ("prop[L1+L3]|trunc", "L1+L3, truncated 2--3 hops"), ("prop[integrated]|nnls", "All tiers, propagated separately and integrated")]
    for name, lab in rows:
        cells = []
        for line in LINES:
            v = mean_for(line, name)
            cells.append("--" if v is None else f"{v:.3f}")
        lines.append(f"{lab} & " + " & ".join(cells) + BS + BS)
        for line, c in zip(LINES, cells):
            if c != "--":
                put(f"g12_{SHORT[line]}_{name}".replace("[", "_").replace("]", "_").replace("|", "_").replace("+", "p"), c)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "tiers.tex").write_text("\n".join(lines))
    # key paired differences
    for line in LINES:
        if line not in U:
            continue
        s_ = SHORT[line]
        u = U[line]
        for tag, a, b in (("rwr_one", "prop[L1+L3]|rwr", "prop[L1+L3]|onehop"), ("trunc_one", "prop[L1+L3]|trunc", "prop[L1+L3]|onehop"),
                          ("int_rwr", "prop[integrated]|nnls", "prop[L1+L3]|rwr"), ("int_b2", "prop[integrated]|nnls", "B2_linear"),
                          ("rwr_b2", "prop[L1+L3]|rwr", "B2_linear"), ("corum_l3", "prop[corum]|rwr", "prop[L3]|rwr"), ("biogrid_l3", "prop[biogrid]|rwr", "prop[L3]|rwr"),
                          ("signor_l1", "prop[signor]|rwr", "prop[L1]|rwr"), ("gtex_l4", "prop[gtex_coex]|rwr", "prop[L4]|rwr")):
            if (u.model == a).any() and (u.model == b).any():
                put(f"g12d_{s_}_{tag}", ci(diff(u, "model", a, b)))
    # integration weights (mean over folds) from the config files
    W = {}
    for line in LINES:
        fs = sorted(glob.glob(str(S / f"g12cfg_{line}_p*_f*.tsv")))
        ws = []
        for fp in fs:
            d = pd.read_csv(fp, sep="\t")
            r = d[d.model == "prop[integrated]|nnls"]
            if len(r):
                comps = r.iloc[0]["comps"].split(",")
                w = [float(x) for x in str(r.iloc[0]["w"]).split(",")]
                ws.append(dict(zip(comps, w)))
        if ws:
            W[line] = pd.DataFrame(ws).mean().to_dict()
            for k, v in W[line].items():
                put(f"g12w_{SHORT[line]}_{k}", f"{v:.2f}")
    pd.DataFrame(W).to_csv(ROOT / "artifacts" / "g12_integration_weights.tsv", sep="\t")

# ---------------- g11: losses
g11 = {}
for line in LINES:
    p = S / f"g11_{line}.tsv"
    if p.exists():
        g11[line] = pd.read_csv(p, sep="\t")
if g11:
    names = {"mse_closed": "Ridge (closed-form MSE)", "mse": "MSE (gradient)", "corr": "Centred-correlation loss", "nmse": "Row-normalised MSE",
             "huber": "Huber (row-normalised)", "corr+huber": "Correlation + 0.1 Huber"}
    lines = [BS + "begin{tabular}{lcccc}", BS + "toprule", "Training loss & K562 $" + BS + "rho_c$ & $" + BS + "Delta$ vs ridge & RPE1 $" + BS + "rho_c$ & $" + BS + "Delta$ vs ridge" + BS + BS + BS + "midrule"]
    Ug = {l: uniq(d, "loss") for l, d in g11.items()}
    for key, lab in names.items():
        cells = []
        for line in LINES:
            u = Ug.get(line)
            if u is None or not (u.loss == key).any():
                cells += ["--", "--"]
                continue
            m = float(u[u.loss == key].pcc_c.mean())
            d = "ref." if key == "mse_closed" else ci(diff(u, "loss", key, "mse_closed"))
            cells += [f"{m:.3f}", d]
            put(f"g11_{SHORT[line]}_{key.replace('+', 'p')}", f"{m:.3f}")
        if any(c != "--" for c in cells):
            lines.append(f"{lab} & " + " & ".join(cells) + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "losses.tex").write_text("\n".join(lines))

# ---------------- g10: certified Hill, reachable targets
g10 = {}
for line in LINES:
    p = S / f"g10_{line}.tsv"
    if p.exists():
        d = pd.read_csv(p, sep="\t")
        g10[line] = d[d.reachable]
if g10:
    names = {"B2_linear": "No graph (B2)", "gn2_linear_mse": "Linear propagation, MSE", "gn2_linear_corr": "Linear propagation, correlation loss",
             "gn2_hill4_mse": "Certified Hill, MSE", "gn2_hill4_corr": "Certified Hill, correlation loss", "gn2_hill4_corr+huber": "Certified Hill, correlation + Huber"}
    lines = [BS + "begin{tabular}{lcccc}", BS + "toprule", "Reachable targets & K562 $" + BS + "rho_c$ & $" + BS + "Delta$ vs B2 & RPE1 $" + BS + "rho_c$ & $" + BS + "Delta$ vs B2" + BS + BS + BS + "midrule"]
    Ur = {l: uniq(d, "model") for l, d in g10.items()}
    for key, lab in names.items():
        cells = []
        for line in LINES:
            u = Ur.get(line)
            if u is None or not (u.model == key).any():
                cells += ["--", "--"]
                continue
            m = float(u[u.model == key].pcc_c.mean())
            d = "ref." if key == "B2_linear" else ci(diff(u, "model", key, "B2_linear"), 4)
            cells += [f"{m:.3f}", d]
            put(f"g10_{SHORT[line]}_{key.replace('+', 'p')}", f"{m:.3f}")
            put(f"g10d_{SHORT[line]}_{key.replace('+', 'p')}", d)
            put(f"g10n_{SHORT[line]}", str(int(u.target.nunique())))
        if any(c != "--" for c in cells):
            lines.append(f"{lab} & " + " & ".join(cells) + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "certhill.tex").write_text("\n".join(lines))

with open(ROOT / "paper" / "numbers2.tex", "w", encoding="utf8") as fh:
    fh.write("% generated by scripts/make_tables2.py\n")
    for k, v in sorted(N.items()):
        fh.write("\\expandafter\\def\\csname data@%s\\endcsname{%s}\n" % (k, v))
print(len(N), "macros;", [p.name for p in OUT.glob("*.tex")])
