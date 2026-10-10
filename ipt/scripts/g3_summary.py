"""Summaries + paired bootstrap for the Addendum-A smoothing ablation (all test targets)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap
ART = Path(__file__).parents[1] / "artifacts" / "scores"
for line in ("K562_essential", "rpe1"):
    df = pd.read_csv(ART / f"g3_{line}.tsv", sep="\t"); df["mid"] = df.model + "|" + df.graph
    print(f"\n######## {line}: targets {df.target.nunique()} (with L1 neighbour: {df[df.has_l1].target.nunique()})")
    print(df.groupby("mid")[["pcc_c", "pcc", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4).to_string())
    pt = lambda m, sub=df: sub[sub.mid == m].groupby(["fold", "target"]).pcc_c.mean()
    for sub_name, sub in (("ALL", df), ("has_L1", df[df.has_l1])):
        print(f"-- paired bootstrap ({sub_name}), centered Pearson, a - b")
        for L in ("L1", "L1+L3", "L1+L3+L4"):
            for a_, b_ in ((f"smooth_{L}|real", "smooth_none|real"), (f"smooth_{L}|real", f"smooth_{L}|rewired")):
                x, y = pt(a_, sub), pt(b_, sub); j = x.index.intersection(y.index)
                r = paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)
                print(f"  {a_} - {b_}: {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f},{r['ci_high']:+.4f}] p={r['p_two_sided']:.3f} win={r['win_frac']:.2f} n={r['n']}")
