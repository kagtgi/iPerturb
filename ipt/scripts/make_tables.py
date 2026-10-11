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


lines = ["\\begin{tabular}{l" + "c" * len(MET) * 1 + "}", "\\toprule"]
for l in LINES:
    d = U[l]
    lines.append(f"\\multicolumn{{{len(MET) + 1}}}{{l}}{{\\textbf{{{LINE_NAME[l]}}} ({d.target.nunique()} held-out targets)}}\\\\")
    lines.append("Model & " + " & ".join(h for _, h in MET) + "\\\\\\midrule")
    best = {m: max(mean_ci(d[d.mid == mid][m])[0] for mid, lab_ in ROWS if (d.mid == mid).any() and 'dagger' not in lab_) for m, _ in MET}
    for mid, lab in ROWS:
        v = d[d.mid == mid]
        if v.empty:
            continue
        cells = []
        for m, _ in MET:
            mu, lo, hi = mean_ci(v[m])
            s = f"{mu:.3f}"
            cells.append(f"\\textbf{{{s}}}" if abs(mu - best[m]) < 1e-9 else s)
        lines.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
lines[-1] = "\\bottomrule"
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
    lines.append(f"{lab} & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
(OUT / "paired.tex").write_text("\n".join(lines))
print((OUT / "main_results.tex").read_text()[:600])
print((OUT / "paired.tex").read_text())
