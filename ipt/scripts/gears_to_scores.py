"""Score GEARS predictions (deltas on its gene axis) with the same metrics, folds and training mean as every other model."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M

ART = Path(__file__).parents[1] / "artifacts"
GD = ART / "gears"
rows = []
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in z["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)
    perts = [p.upper() for p in z["perts"]]
    prow = {p: i for i, p in enumerate(perts)}
    Yall, tix, n = z["delta"], z["target_idx"], z["n_cells"]
    se = np.sqrt(z["var_p"] / n[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
    folds = json.load(open(ART / "folds" / f"{line}.json"))
    for fo in range(3):
        fp = GD / line / f"pred_fold{fo}.npz"
        if not fp.exists():
            print("missing", fp)
            continue
        d = np.load(fp, allow_pickle=True)
        names = [str(t).upper() for t in d["targets"]]
        ggenes = [str(g).upper() for g in d["genes"]]
        col = np.array([gi.get(g, -1) for g in ggenes])
        P = np.zeros((len(names), G), np.float32)
        ok = col >= 0
        P[:, col[ok]] = d["delta"][:, ok]
        # GEARS is offset by its own control mean; centre it on the mean of its predictions for training targets
        if "train_delta" in d.files:
            gtm = np.zeros(G, np.float32)
            gtm[col[ok]] = d["train_delta"].mean(0)[ok]
            centring = "gears_train_mean"
        else:
            gtm = P.mean(0)
            centring = "test_prediction_mean"
        f = folds[fo]
        tr = np.array([prow[p.upper()] for p in f["train"]])
        tm = Yall[tr].mean(0)
        ix = np.array([prow[t] for t in names])
        obs, tt = Yall[ix].astype(np.float64), tix[ix]
        P_raw = P.copy()
        P = (P - gtm + tm).astype(np.float32)   # after subtracting tm this is P - gtm
        deg = np.abs(obs / np.maximum(se[ix], 1e-6)) > 4
        pc = M.pcc_delta_centered(P, obs, tm, tt)
        pres = M.pcc_delta_resid(P, obs, tm, tt)
        pr = M.pcc_delta(P_raw, obs, tt)
        t20 = M.top_k_pcc(P, obs, tt, 20, centered_mean=tm)
        rm = M.rmse_delta(P, obs, tt)
        ap = M.deg_auprc(P - tm, deg, tt)["ap"]
        # centroid rank among the evaluated (GEARS-supported) test targets only
        cn = M.centroid_accuracy(P, obs, tm, tt)["norm_rank_each"]
        for i, t in enumerate(names):
            rows.append(dict(line=line, fold=fo, target=t, mid="B4_gears|-|-", pcc_c=pc[i], pcc_resid=pres[i], pcc=pr[i], top20_c=t20[i],
                             rmse=rm[i], deg_ap=ap[i], cent_rank=cn[i]))
        print(line, fo, len(names), "targets scored; centring:", centring)
df = pd.DataFrame(rows)
for line, d in df.groupby("line"):
    d.drop(columns="line").to_csv(ART / "scores" / f"gears_{line}.tsv", sep="\t", index=False)
    print(line, d.groupby("fold").size().to_dict(), d.pcc_c.mean().round(4))
