"""Unique-target-level summaries and paired bootstrap for Addenda A-C (+GEARS if present). Writes results.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.metrics import paired_bootstrap

ART = Path(__file__).parents[1] / "artifacts"
S = ART / "scores"
METRICS = ["pcc_c", "pcc_resid", "pcc", "top20_c", "rmse", "deg_ap", "cent_rank"]
out = {}
for line in ("K562_essential", "rpe1"):
    g4 = pd.read_csv(S / f"g4_{line}.tsv", sep="\t")
    g4["mid"] = g4.model + "|" + g4.graph + "|" + g4.svd
    g5 = pd.read_csv(S / f"g5_{line}.tsv", sep="\t")
    g5 = g5.groupby(["fold", "target", "graph"], as_index=False)[METRICS].mean()  # average over seeds
    g5["mid"] = "B5_gcn|" + g5.graph + "|self"
    df = pd.concat([g4[["fold", "target", "mid"] + METRICS], g5[["fold", "target", "mid"] + METRICS]])
    gp = ART / "scores" / f"gears_{line}.tsv"
    if gp.exists():
        df = pd.concat([df, pd.read_csv(gp, sep="\t")[["fold", "target", "mid"] + METRICS]])
    u = df.groupby(["mid", "target"], as_index=False)[METRICS].mean()  # unique-target means across folds
    nt = u.target.nunique()
    tab = u.groupby("mid")[METRICS].mean().round(4)
    print(f"\n######## {line}: unique test targets {nt}")
    print(tab.to_string())
    pairs = [("smooth_L1+L3|real|self", "smooth_none|real|self"), ("smooth_L1+L3|real|self", "smooth_L1+L3|rewired|self"),
             ("smooth_L3|real|self", "smooth_L3|rewired|self"), ("smooth_L1+L3|real|noself", "smooth_none|real|noself"),
             ("smooth_L1+L3|real|noself", "smooth_L1+L3|rewired|noself"), ("smooth_L1+L3|real|self", "smooth_dataknn|real|self"),
             ("smooth_L1+L3|real|self", "nbrmean_L3|real|self"), ("smooth_L1+L3|real|self", "B5_gcn|real|self"),
             ("B5_gcn|real|self", "B5_gcn|rewired|self"), ("B5_gcn|real|self", "smooth_none|real|self"),
             ("smooth_L1|real|self" if False else "smooth_L1+L3|real|self", "B2_linear|-|-")]
    if gp.exists():
        pairs += [("smooth_L1+L3|real|self", "B4_gears|-|-"), ("B4_gears|-|-", "B2_linear|-|-")]
    res = []
    for m in ("pcc_c", "pcc_resid", "top20_c", "deg_ap"):
        for a_, b_ in pairs:
            x = u[u.mid == a_].set_index("target")[m]
            y = u[u.mid == b_].set_index("target")[m]
            j = x.index.intersection(y.index)
            if len(j) < 5:
                continue
            r = paired_bootstrap(x.loc[j].to_numpy(), y.loc[j].to_numpy(), 10000, 0)
            res.append(dict(metric=m, a=a_, b=b_, diff=r["mean_diff"], lo=r["ci_low"], hi=r["ci_high"], p=r["p_two_sided"], n=r["n"], win=r["win_frac"]))
    pr = pd.DataFrame(res)
    print(pr[pr.metric.isin(["pcc_c", "pcc_resid"])].round(4).to_string(index=False))
    out[line] = dict(n_unique=int(nt), table=tab.to_dict(orient="index"), paired=res)
    cfg = pd.read_csv(S / f"g4cfg_{line}.tsv", sep="\t")
    print(cfg[cfg.variant.str.contains("L3|real")].groupby("variant")[["k", "alpha"]].agg(lambda s: list(s)).to_string())
json.dump(out, open(ART / "results.json", "w"), indent=1, default=float)
