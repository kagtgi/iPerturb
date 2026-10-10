"""Addendum H benchmark: per-edge Chebyshev shapes, edge-wise vs one sparse matmul."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.models.peredge import PerEdgeCheb

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", default="artifacts/bench_graphs.npz"); ap.add_argument("--out", default="artifacts/bench_peredge_cpu.json")
ap.add_argument("--device", default="cpu"); ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512])
ap.add_argument("--rs", type=int, nargs="+", default=[8, 16]); ap.add_argument("--iters", type=int, default=7)
a = ap.parse_args(); dev = torch.device(a.device); Z = np.load(a.graphs); rng = np.random.default_rng(0)
sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
res, eq = [], {}
for g in ("l1", "assoc"):
    G = int(Z[f"{g}_G"]); E = len(Z[f"{g}_src"])
    for r in a.rs:
        models = {}
        for impl in ("edge", "spmm"):
            m = PerEdgeCheb(Z[f"{g}_src"], Z[f"{g}_dst"], Z[f"{g}_x0"], r=r, impl=impl).to(dev)
            models[impl] = m
        models["spmm"].load_state_dict(models["edge"].state_dict())
        for B in a.batches:
            tgt = torch.as_tensor(rng.integers(0, G, B), device=dev)
            for impl, m in models.items():
                try:
                    def step():
                        m.zero_grad(set_to_none=True); m(tgt).pow(2).mean().backward()
                    for _ in range(2): step()
                    sync()
                    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats()
                    ts = []
                    for _ in range(a.iters):
                        sync(); t = time.perf_counter(); step(); sync(); ts.append(time.perf_counter() - t)
                    mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
                    sec = float(np.median(ts))
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    sec, mem = float("nan"), float("nan"); print("FAIL", g, r, B, impl, str(e)[:60], flush=True)
                    if dev.type == "cuda": torch.cuda.empty_cache()
                res.append(dict(graph=g, E=E, r=r, B=B, impl=impl, sec=sec, mem_mb=mem))
                print(f"{g} E={E} r={r} B={B} {impl}: {sec*1e3:.1f} ms mem {mem:.0f} MB", flush=True)
        tgt = torch.as_tensor(rng.integers(0, G, 16), device=dev)
        with torch.no_grad():
            try:
                eq[f"{g}_r{r}"] = float((models["edge"](tgt) - models["spmm"](tgt)).abs().max())
            except (RuntimeError, torch.OutOfMemoryError):
                eq[f"{g}_r{r}"] = float("nan")
        print("equiv", g, r, eq[f"{g}_r{r}"], flush=True)
        del models
        if dev.type == "cuda": torch.cuda.empty_cache()
json.dump(dict(device=a.device, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, timing=res, equivalence=eq), open(a.out, "w"), indent=1)
