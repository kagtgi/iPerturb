"""Efficiency table (A100): time / peak memory of the shared-Hill operator variants."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
g = pd.DataFrame(json.load(open(ROOT / "artifacts" / "bench_gpu.json"))["timing"])
impls = ["E0_exact_edge", "E1_exact_spmm", "E2_exact_dense", "C1_cheb16_spmm"]
BS = "\\"
lines = [BS + "begin{tabular}{llrrrr}", BS + "toprule", "Graph & Batch & E0 & E1 & E2 & C1" + BS + BS + BS + "midrule"]
for gname, glab in (("l1", "curated, 10k edges"), ("assoc", "association, 247k edges")):
    for B in (128, 512):
        cells = []
        for k in impls:
            r = g[(g.graph == gname) & (g.B == B) & (g.impl == k)].iloc[0]
            cells.append(f"{r.sec * 1e3:.0f} / {r.mem_mb:.0f}")
        lines.append(f"{glab} & {B} & " + " & ".join(cells) + BS + BS)
    lines.append(BS + "midrule")
lines[-1] = BS + "bottomrule"
lines.append(BS + "end{tabular}")
out = ROOT / "paper" / "tables" / "bench.tex"
out.write_text("\n".join(lines))
print(out.read_text())
