"""SCC statistics and SCC-vs-global solver equivalence/time on the real signed L1 graphs."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.graph.l1 import load_l1
from ipt.models.graphnet import GraphNet
from ipt.models.hill import hill4
from ipt.models.scc import Problem, solve_global, solve_global_reach, solve_scc

ROOT = Path(__file__).parents[1]
ART = ROOT / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
report = {}
for line in ("K562_essential", "rpe1"):
    z = np.load(ART / "data" / f"{line}.npz", allow_pickle=True)
    genes = [g.upper() for g in z["genes"]]
    gi = {g: i for i, g in enumerate(genes)}
    cm = z["ctrl_mean"]
    x0 = 1 / (1 + np.exp(-(cm - cm.mean()) / cm.std()))
    e = load_l1(CACHE)
    e = e[(e.sign != 0) & e.source.isin(gi) & e.target.isin(gi)]
    src, dst = e.source.map(gi).to_numpy(), e.target.map(gi).to_numpy()
    sign = e.sign.to_numpy(np.float32)
    conf = (np.minimum(e.n_db.to_numpy(), 3) / 3).astype(np.float32)
    net = GraphNet(src, dst, sign, conf, x0, transfer="hill4")
    w = net.weights().detach().numpy().astype(float)
    T = lambda x: hill4(torch.as_tensor(x), torch.tensor(0.3, dtype=torch.float64), 2.0).numpy()
    P = Problem(src, dst, w, x0.astype(float), T)
    sizes = sorted((len(P.cond.nodes[c]["members"]) for c in P.cond.nodes), reverse=True)
    n_nontriv = sum(s > 1 for s in sizes)
    targets = [t for t in sorted(set(src.tolist())) if len(__import__("networkx").descendants(P.graph, t)) > 0][:12]
    rec = []
    for t in targets:
        clamp = 0.14 * x0[t]
        t0 = time.perf_counter(); xg, ig = solve_global(P, t, clamp)
        t1 = time.perf_counter(); xs, nu = solve_scc(P, t, clamp)
        t2 = time.perf_counter()
        xr, _ = solve_global_reach(P, t, clamp)
        t3 = time.perf_counter()
        rec.append(dict(target=genes[t], reach=len(__import__("networkx").descendants(P.graph, t)), max_diff=float(np.abs(xg - xs).max()), max_diff_reach=float(np.abs(xg - xr).max()), t_greach=t3 - t2,
                        t_global=t1 - t0, t_scc=t2 - t1, iters_global=ig))
    report[line] = dict(n_nodes=len(genes), n_edges=len(src), largest_scc=int(P.largest_scc), n_nontrivial_scc=int(n_nontriv),
                        top_sizes=sizes[:5], solves=rec,
                        median_speedup=float(np.median([r["t_global"] / max(r["t_scc"], 1e-9) for r in rec])),
                        median_speedup_vs_reach=float(np.median([r["t_greach"] / max(r["t_scc"], 1e-9) for r in rec])),
                        median_reach_vs_global=float(np.median([r["t_global"] / max(r["t_greach"], 1e-9) for r in rec])),
                        max_diff=float(max(r["max_diff"] for r in rec)))
    print(line, json.dumps({k: v for k, v in report[line].items() if k != "solves"}, indent=1), flush=True)
json.dump(report, open(ART / "scc_bench.json", "w"), indent=1)
