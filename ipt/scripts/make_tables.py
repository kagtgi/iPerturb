"""LaTeX tables generated directly from score files (no hand-typed numbers)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap
from ipt.viz.load import LINES, LINE_NAME, mean_ci, unique_target_scores

OUT = Path(__file__).parents[1] / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
ROWS = [("B0_nochange|-|-", "No change (B0)"), ("B1_trainmean|-|-", "Train mean (B1)"), ("B2_linear|-|-", "Linear low-rank (B2)"),
        ("B4_gears|-|-", "GEARS (B4)"), ("B7_cpa|-|-", "CPA (B7)"), ("B8_biolord|-|-", "biolord (B8)"),
        ("B9_scgpt_asis|-|-", "scGPT, zero-shot knockout$^\\dagger$"), ("B10_state_zeroshot|-|-", "STATE, zero-shot$^\\dagger$"),
        ("B10_state_fewshot|-|-", "STATE, few-shot$^\\dagger$"), ("smooth_L1+L3|rewired|self", "Smoothing, rewired GRN"), ("smooth_dataknn|real|self", "Smoothing, data-kNN graph"),
        ("smooth_L3|real|self", "Smoothing, STRING only"), ("smooth_L1+L3|real|self", "iPerturb: smoothing head"),
        ("B5_gcn|real|self", "iPerturb: GCN head (B5)"), ("nbrmean_L3|real|self", "iPerturb: neighbour-mean head"),
        ("prop[L1+L3]|rwr", "iPerturb: multi-hop propagation (RWR)"), ("prop[integrated]|nnls", "iPerturb: multi-tier integrated propagation")]
MET = [("pcc_c", "$\\rho_c$"), ("pcc_resid", "$\\rho_r$"), ("top20_c", "$\\rho_{20}$"), ("cent_rank", "Cent."), ("deg_ap", "AUPRC")]
U = {l: unique_target_scores(l) for l in LINES}


def fmt(m, lo, hi):
    return f"{m:.3f} \\scriptsize[{lo:.3f},{hi:.3f}]"


# Table 1: K562 and RPE1 side by side (CIBB layout); groups separated by rules
SHORT = {"No change (B0)": "No change (B0)", "Train mean (B1)": "Train mean (B1)", "Linear low-rank (B2)": "Linear low-rank (B2)",
         "Smoothing, rewired GRN": "Rewired GRN", "Smoothing, data-kNN graph": "Data-kNN graph",
         "Smoothing, STRING only": "STRING only", "scGPT, zero-shot knockout$^\\dagger$": "scGPT, zero-shot KO$^\\dagger$",
         "iPerturb: smoothing head": "Smoothing", "iPerturb: GCN head (B5)": "GCN (B5)", "iPerturb: neighbour-mean head": "Neighbour mean",
         "iPerturb: multi-hop propagation (RWR)": "Multi-hop (RWR)", "iPerturb: multi-tier integrated propagation": "Tier-integrated"}
GROUPS = [("Baselines", ("B0_nochange|-|-", "B1_trainmean|-|-", "B2_linear|-|-")),
          ("External models", ("B4_gears|-|-", "B7_cpa|-|-", "B8_biolord|-|-", "B9_scgpt_asis|-|-", "B10_state_zeroshot|-|-", "B10_state_fewshot|-|-")),
          ("Controls (same read-out)", ("smooth_L1+L3|rewired|self", "smooth_dataknn|real|self", "smooth_L3|real|self")),
          ("iPerturb read-outs of the GRN", ("smooth_L1+L3|real|self", "B5_gcn|real|self", "nbrmean_L3|real|self", "prop[L1+L3]|rwr", "prop[integrated]|nnls"))]
LAB = dict(ROWS)
MS = [("pcc_c", "$\\rho_c$"), ("pcc_resid", "$\\rho_r$"), ("top20_c", "$\\rho_{20}$"), ("cent_rank", "Cent."), ("deg_ap", "AUPRC")]
best = {l: {m: max(mean_ci(U[l][U[l].mid == mid][m])[0] for mid, lab_ in ROWS if (U[l].mid == mid).any() and 'dagger' not in lab_) for m, _ in MS} for l in LINES}
cache = {(l, mid, m): mean_ci(U[l][U[l].mid == mid][m])[0] for l in LINES for mid, _ in ROWS if (U[l].mid == mid).any() for m, _ in MS}
nm = len(MS)
lines = ["\\begin{tabular}{l" + "c" * nm + "@{\\hspace{6pt}}" + "c" * nm + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{{nm}}}{{c}}{{\\textbf{{{LINE_NAME[l]}}} ({U[l][U[l].mid == 'smooth_L1+L3|real|self'].target.nunique()} targets)}}" for l in LINES) + "\\\\",
         f"\\cmidrule(lr){{2-{nm + 1}}}\\cmidrule(lr){{{nm + 2}-{2 * nm + 1}}}",
         "Model & " + " & ".join(h for _ in LINES for _, h in MS) + "\\\\\\midrule"]
for gi_, (gname, grp) in enumerate(GROUPS):
    lines.append(f"\\multicolumn{{{2 * nm + 1}}}{{l}}{{\\emph{{{gname}}}}}\\\\")
    for mid in grp:
        if not any((l, mid, "pcc_c") in cache for l in LINES):
            continue
        cells = []
        for l in LINES:
            for m, _ in MS:
                if (l, mid, m) not in cache:
                    cells.append("--")
                    continue
                s = f"{cache[(l, mid, m)]:.3f}".replace("-0.000", "0.000")
                cells.append(f"\\textbf{{{s}}}" if abs(cache[(l, mid, m)] - best[l][m]) < 1e-9 else s)
        lab = SHORT.get(LAB[mid], LAB[mid])
        lines.append(f"\\hspace{{0.4em}}{lab} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule" if gi_ < len(GROUPS) - 1 else "\\bottomrule")
lines.append("\\end{tabular}")
(OUT / "main_results.tex").write_text("\n".join(lines))

# paired differences table
PAIRS = [("smooth_L1+L3|real|self", "smooth_none|real|self", "GRN vs no graph"),
         ("smooth_L1+L3|real|self", "smooth_L1+L3|rewired|self", "GRN vs rewired GRN"),
         ("smooth_L1+L3|real|self", "smooth_dataknn|real|self", "GRN vs data-kNN graph"),
         ("smooth_L1+L3|real|noself", "smooth_L1+L3|rewired|noself", "GRN vs rewired (self-entries zeroed)"),
         ("smooth_L1+L3|real|self", "B2_linear|-|-", "iPerturb vs linear (B2)")]
if any((U[l].mid == "B4_gears|-|-").any() for l in LINES):
    PAIRS.append(("smooth_L1+L3|real|self", "B4_gears|-|-", "iPerturb vs GEARS"))
lines = ["\\begin{tabular}{lcc}", "\\toprule", "Comparison ($\\rho_c$, mean difference [95\\% CI]) & K562 & RPE1\\\\\\midrule"]
for a_, b_, lab in PAIRS:
    cells = []
    for l in LINES:
        x = U[l][U[l].mid == a_].set_index("target").pcc_c
        y = U[l][U[l].mid == b_].set_index("target").pcc_c
        j = x.index.intersection(y.index)
        if len(j) < 5:
            cells.append("--")
            continue
        r = paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)
        cells.append(f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
    lines.append(f"\\quad {lab} & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "paired.tex").write_text("\n".join(lines))
print((OUT / "main_results.tex").read_text()[:600])
print((OUT / "paired.tex").read_text())
