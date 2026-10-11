"""Addendum J(c): anti-vanishing Hill propagation (GraphNet2) with metric-aligned losses on reachable targets."""
import argparse
import copy
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
from ipt.models.graphnet2 import GraphNet2
from ipt.models.losses import LOSSES

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="K562_essential")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.set_num_threads(3)

z = np.load(ART / "data" / f"{a.line}.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
perts = [p.upper() for p in z["perts"]]
prow = {p: i for i, p in enumerate(perts)}
Yall, tix = z["delta"], z["target_idx"]
cm = z["ctrl_mean"]
folds = json.load(open(ART / "folds" / f"{a.line}.json"))
e = load_l1(CACHE)
e = e[(e.sign != 0) & e.source.isin(gi) & e.target.isin(gi)]
src, dst = e.source.map(gi).to_numpy(), e.target.map(gi).to_numpy()
sign = e.sign.to_numpy(np.float32)
conf = (np.minimum(e.n_db.to_numpy(), 3) / 3).astype(np.float32)
has_out = set(src.tolist())
x0 = 1 / (1 + np.exp(-(cm - cm.mean()) / cm.std()))


def fit(transfer, loss_name, R_tr, t_tr, R_va, t_va, zeta, lr, seed=0, epochs=120, batch=32, patience=20):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GraphNet2(src, dst, sign, conf, x0, transfer=transfer)
    lossf = LOSSES[loss_name]
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    Rt, tt = torch.as_tensor(R_tr, dtype=torch.float32), torch.as_tensor(t_tr)
    Rv, tv = torch.as_tensor(R_va, dtype=torch.float32), torch.as_tensor(t_va)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        perm = rng.permutation(len(tt))
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(net(tt[b], zeta), Rt[b], tt[b]) + 1e-3 * (net.u ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            v = float(lossf(net(tv, zeta), Rv, tv)) if len(tv) else float(loss)
        if v < best - 1e-9:
            best, state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net, best


rows, diag = [], []
for fo in a.folds:
    f = folds[fo]
    idx = {k: np.array([prow[p.upper()] for p in f[k]]) for k in ("train", "valid", "test")}
    Y = {k: Yall[idx[k]] for k in idx}
    T = {k: tix[idx[k]] for k in idx}
    tm = Y["train"].mean(0)
    ar = np.arange(len(T["train"]))
    ratio = np.expm1(np.clip(cm[T["train"]] + Y["train"][ar, T["train"]], 0, None)) / np.maximum(np.expm1(cm[T["train"]]), 1e-9)
    zeta = float(1 - np.clip(np.median(ratio), 0.05, 0.95))
    b2 = B.fit_linear_select(Y["train"], T["train"], Y["valid"], T["valid"])
    cf = B.cross_fit_linear(Y["train"], T["train"], b2.G.shape[1])
    reach = {k: np.array([t in has_out for t in T[k]]) for k in T}
    R_tr, t_tr = (Y["train"] - cf)[reach["train"]], T["train"][reach["train"]]
    R_va, t_va = (Y["valid"] - b2.predict(T["valid"]))[reach["valid"]], T["valid"][reach["valid"]]
    base = b2.predict(T["test"])
    obs, tt = Y["test"].astype(np.float64), T["test"]
    preds = {"B2_linear": base}
    for transfer in ("linear", "hill4"):
        for loss_name in ("mse", "corr", "corr+huber"):
            best = None
            for lr in (0.003, 0.01):
                net, v = fit(transfer, loss_name, R_tr, t_tr, R_va, t_va, zeta, lr)
                if best is None or v < best[1]:
                    best = (net, v)
            net = best[0]
            p = base.copy()
            with torch.no_grad():
                r = net(torch.as_tensor(T["test"][reach["test"]]), zeta).numpy()
            p[reach["test"]] += r
            name = f"gn2_{transfer}_{loss_name}"
            preds[name] = p
            diag.append(dict(fold=fo, model=name, val_loss=best[1], pred_std=float(r.std()), resid_std=float(R_tr.std()),
                             collapsed=bool(r.std() < 1e-4)))
            print(f"fold {fo} {name} val {best[1]:.4f} pred_std {r.std():.4f} resid_std {R_tr.std():.4f}", flush=True)
    for m_, p in preds.items():
        pc = M.pcc_delta_centered(p, obs, tm, tt)
        pres = M.pcc_delta_resid(p, obs, tm, tt)
        t20 = M.top_k_pcc(p, obs, tt, 20, centered_mean=tm)
        for i in range(len(tt)):
            rows.append(dict(line=a.line, fold=fo, model=m_, target=f["test"][i], reachable=bool(reach["test"][i]), pcc_c=pc[i], pcc_resid=pres[i], top20_c=t20[i]))
df = pd.DataFrame(rows)
df.to_csv(ART / "scores" / f"g10_{a.line}{a.tag}.tsv", sep="\t", index=False)
pd.DataFrame(diag).to_csv(ART / "scores" / f"g10diag_{a.line}{a.tag}.tsv", sep="\t", index=False)
R = df[df.reachable]
u = R.groupby(["model", "target"], as_index=False)[["pcc_c", "pcc_resid", "top20_c"]].mean()
print(u.groupby("model")[["pcc_c", "pcc_resid", "top20_c"]].mean().round(4).to_string())
