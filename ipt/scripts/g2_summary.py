"""Paired summaries and bootstrap CIs for the pre-registered G2 decision."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap

ART = Path(__file__).parents[1] / "artifacts" / "scores"
name = sys.argv[1] if len(sys.argv) > 1 else "g2_K562_essential"
df = pd.read_csv(ART / f"{name}.tsv", sep="\t")
df["mid"] = df.model + "|" + df.graph + "|" + df.seed.astype(str)
metric = "pcc_c"
print("targets (unique):", df.target.nunique(), " reachable:", df[df.reachable].target.nunique())
for sub, d in (("ALL", df), ("REACHABLE", df[df.reachable]), ("UNREACHABLE", df[~df.reachable])):
    print(f"\n== {sub}: mean per model (graph models averaged over seeds) ==")
    t = d.groupby(["model", "graph"])[["pcc_c", "pcc", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4)
    print(t.to_string())


def per_target(d, model, graph):
    x = d[(d.model == model) & (d.graph == graph)].groupby(["fold", "target"])[metric].mean()
    return x


R = df[df.reachable]
pairs = [("H2_hill_regk", "real", "B3_linear", "real"), ("H2_hill_fixk", "real", "B3_linear", "real"),
         ("H2_hill_regk", "real", "B2_linear", "-"), ("H2_hill_fixk", "real", "B2_linear", "-"),
         ("B3_linear", "real", "B2_linear", "-"),
         ("H2_hill_regk", "real", "H2_hill_regk", "rewired"), ("H2_hill_fixk", "real", "H2_hill_fixk", "rewired"),
         ("B3_linear", "real", "B3_linear", "rewired")]
print("\n== paired bootstrap on REACHABLE targets, primary metric (centered Pearson), a minus b ==")
for ma, ga, mb, gb in pairs:
    a, b = per_target(R, ma, ga), per_target(R, mb, gb)
    j = a.index.intersection(b.index)
    if len(j) < 3:
        print(ma, ga, "vs", mb, gb, "n<3")
        continue
    r = paired_bootstrap(a.loc[j].to_numpy(), b.loc[j].to_numpy(), n_boot=10000, seed=0)
    print(f"{ma}|{ga} - {mb}|{gb}: diff {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
          f"p={r['p_two_sided']:.3f} win={r['win_frac']:.2f} n={r['n']}")
