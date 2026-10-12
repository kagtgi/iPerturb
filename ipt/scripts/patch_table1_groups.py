from pathlib import Path

p = Path(__file__).parent / "make_tables.py"
s = p.read_text(encoding="utf8")
a = s.index("SHORT = {")
b = s.index("LAB = dict(ROWS)")
new = '''SHORT = {"No change (B0)": "No change (B0)", "Train mean (B1)": "Train mean (B1)", "Linear low-rank (B2)": "Linear low-rank (B2)",
         "Smoothing, rewired GRN": "Rewired GRN", "Smoothing, data-kNN graph": "Data-kNN graph",
         "Smoothing, STRING only": "STRING only", "scGPT, zero-shot knockout$^\\\\dagger$": "scGPT, zero-shot KO$^\\\\dagger$",
         "iPerturb: smoothing head": "Smoothing", "iPerturb: GCN head (B5)": "GCN (B5)", "iPerturb: neighbour-mean head": "Neighbour mean",
         "iPerturb: multi-hop propagation (RWR)": "Multi-hop (RWR)", "iPerturb: multi-tier integrated propagation": "Tier-integrated"}
GROUPS = [("Baselines", ("B0_nochange|-|-", "B1_trainmean|-|-", "B2_linear|-|-")),
          ("External models", ("B4_gears|-|-", "B7_cpa|-|-", "B8_biolord|-|-", "B9_scgpt_asis|-|-", "B10_state_zeroshot|-|-", "B10_state_fewshot|-|-")),
          ("Controls (same read-out)", ("smooth_L1+L3|rewired|self", "smooth_dataknn|real|self", "smooth_L3|real|self")),
          ("iPerturb read-outs of the GRN", ("smooth_L1+L3|real|self", "B5_gcn|real|self", "nbrmean_L3|real|self", "prop[L1+L3]|rwr", "prop[integrated]|nnls"))]
'''
s = s[:a] + new + s[b:]
s = s.replace("for gi_, grp in enumerate(GROUPS):\n", "for gi_, (gname, grp) in enumerate(GROUPS):\n    lines.append(f\"\\\\multicolumn{{{2 * nm + 1}}}{{l}}{{\\\\emph{{{gname}}}}}\\\\\\\\\")\n")
s = s.replace('lines.append(f"{lab} & " + " & ".join(cells) + "\\\\\\\\")', 'lines.append(f"\\\\quad {lab} & " + " & ".join(cells) + "\\\\\\\\")')
p.write_text(s, encoding="utf8")
print(s[s.index("for gi_"):s.index("for gi_") + 900])
