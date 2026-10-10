"""Gate G2: B0/B1/B2 vs graph residual models (B3 linear, H2 Hill4) on target-held-out folds."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt import metrics as M
from ipt.graph.l1 import load_l1
from ipt.models import baselines as B
from ipt.models.fit import fit_graph, rewire

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0])
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--out", default="g2")
a = ap.parse_args()
torch.set_num_threads(4)

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix, n = z["delta"], z["target_idx"], z["n_cells"]
se = np.sqrt(z["var_p"] / n[:, None] + z["ctrl_var"][None, :] / int(z["ctrl_n"]))
cm = z["ctrl_mean"]
folds = json.load(open(ART / "folds" / f"{a.line}.json"))

e = load_l1(CACHE)
e = e[(e.sign != 0) & e.source.isin(gi) & e.target.isin(gi)]
src0, dst0 = e.source.map(gi).to_numpy(), e.target.map(gi).to_numpy()
sign0 = e.sign.to_numpy(np.float32)
conf0 = (np.minimum(e.n_db.to_numpy(), 3) / 3).astype(np.float32)
has_out = set(src0.tolist())
zs = (cm - cm.mean()) / cm.std()
x0 = 1 / (1 + np.exp(-zs))

rows = []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]] for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    # training-only knockdown efficacy: fraction of linear-scale expression remaining at the target gene
    ar = np.arange(len(T["train"]))
    ratio = np.expm1(np.clip(cm[T["train"]] + Y["train"][ar, T["train"]], 0, None)) / np.maximum(np.expm1(cm[T["train"]]), 1e-9)
    zeta = float(1 - np.clip(np.median(ratio), 0.05, 0.95))
    b2 = B.fit_linear_select(Y["train"], T["train"], Y["valid"], T["valid"])
    kbest = b2.G.shape[1]
    base_pred = b2.predict(T["test"])
    preds = {"B0_nochange": B.NoChange().fit(Y["train"], T["train"]).predict(T["test"]),
             "B1_trainmean": B.MeanShift().fit(Y["train"], T["train"]).predict(T["test"]),
             "B2_linear": base_pred}
    reach = {k: np.array([t in has_out for t in T[k]]) for k in T}
    cf = B.cross_fit_linear(Y["train"], T["train"], kbest)
    R_tr = (Y["train"] - cf)[reach["train"]]
    t_tr = T["train"][reach["train"]]
    R_va = (Y["valid"] - b2.predict(T["valid"]))[reach["valid"]]
    t_va = T["valid"][reach["valid"]]
    print(f"line {a.line} fold {fo} zeta {zeta:.2f} k {kbest} reachable train/val/test "
          f"{reach['train'].sum()}/{reach['valid'].sum()}/{reach['test'].sum()}", flush=True)
    variants = {"B3_linear": ("linear", "shared", False),
                "H2_hill_fixk": ("hill4", "shared", True),
                "H2_hill_regk": ("hill4", "regulator", False)}
    grid = [(lr, lu) for lr in (0.003, 0.01) for lu in (1e-3, 1e-1)]
    for seed in a.seeds:
        for gname, rew in (("real", False), ("rewired", True)):
            graph = rewire(src0, dst0, sign0, conf0, seed) if rew else (src0, dst0, sign0, conf0)
            for vname, (tr_, km, fk) in variants.items():
                best = None
                for lr, lu in grid:
                    net, info = fit_graph(graph, x0, tr_, km, R_tr, t_tr, R_va, t_va, zeta, lr, lu, seed, freeze_k=fk)
                    if best is None or info["val_mse"] < best[1]["val_mse"]:
                        best = (net, info)
                net, info = best
                p = base_pred.copy()
                with torch.no_grad():
                    r, res = net(torch.as_tensor(T["test"][reach["test"]]), zeta)
                p[reach["test"]] += r.numpy()
                preds[f"{vname}|{gname}|s{seed}"] = p
                print(f"  {vname}|{gname}|s{seed} val_mse {info['val_mse']:.5f} ep {info['best_epoch']} "
                      f"resid {float(res.max()):.1e}", flush=True)
    obs, tt = Y["test"], T["test"]
    deg = np.abs(obs / np.maximum(se[idx["test"]], 1e-6)) > 4
    for m, p in preds.items():
        parts = m.split("|")
        base = dict(line=a.line, fold=fo, model=parts[0], graph=parts[1] if len(parts) > 1 else "-",
                    seed=parts[2] if len(parts) > 2 else "-")
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pr = M.pcc_delta(p, obs, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        rm = M.rmse_delta(p, obs, tt)
        ap_ = M.deg_auprc(p - tm, deg, tt)["ap"]
        cn = M.centroid_accuracy(p, obs, tm, tt)["norm_rank_each"]
        for i in range(len(tt)):
            rows.append({**base, "target": f["test"][i], "reachable": bool(reach["test"][i]), "pcc_c": pc[i],
                         "pcc": pr[i], "top20_c": t20[i], "rmse": rm[i], "deg_ap": ap_[i], "cent_rank": cn[i]})
df = pd.DataFrame(rows)
(ART / "scores").mkdir(exist_ok=True)
df.to_csv(ART / "scores" / f"{a.out}_{a.line}.tsv", sep="\t", index=False)
g = df.groupby(["model", "graph", "reachable"])[["pcc_c", "top20_c", "rmse", "deg_ap", "cent_rank"]].mean().round(4)
print(g.to_string())
