"""PREREG Addendum E benchmark: exact edge-wise Hill vs Chebyshev/matmul propagation (time, memory, equivalence)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.models.basis_hill import FastGraphNet
from ipt.models.hill import hill4

ap = argparse.ArgumentParser()
ap.add_argument("--graphs", default="artifacts/bench_graphs.npz")
ap.add_argument("--out", default="artifacts/bench_cpu.json")
ap.add_argument("--device", default="cpu")
ap.add_argument("--batches", type=int, nargs="+", default=[32, 128, 512])
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
dev = torch.device(a.device)
Z = np.load(a.graphs)
rng = np.random.default_rng(0)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def make(name, **kw):
    G = int(Z[f"{name}_G"])
    m = FastGraphNet(Z[f"{name}_src"], Z[f"{name}_dst"], Z[f"{name}_sign"], Z[f"{name}_conf"], Z[f"{name}_x0"],
                     transfer="hill4", k_mode="regulator", steps=a.steps, **kw).to(dev)
    with torch.no_grad():
        m.kz.copy_(torch.as_tensor(rng.normal(0, 1, m.kz.shape), dtype=torch.float32))  # heterogeneous regulator k
    return m, G


def time_fb(m, G, B):
    tgt = torch.as_tensor(rng.integers(0, G, B), device=dev)
    def step():
        m.zero_grad(set_to_none=True)
        out, _ = m(tgt, 0.8)
        out.pow(2).mean().backward()
    for _ in range(3):
        step()
    sync()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(a.iters):
        sync(); t = time.perf_counter(); step(); sync(); ts.append(time.perf_counter() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    return float(np.median(ts)), mem


def per_edge_hill(name, B):
    """Cost reference for H1: edge-specific k_ji, edge-wise Hill + scatter (same fixed-point skeleton)."""
    G = int(Z[f"{name}_G"]); src = torch.as_tensor(Z[f"{name}_src"], device=dev); dst = torch.as_tensor(Z[f"{name}_dst"], device=dev)
    E = len(src); x0 = torch.as_tensor(Z[f"{name}_x0"], dtype=torch.float32, device=dev).clamp(0.02, 0.98)[None, :].expand(B, -1).contiguous()
    ke = torch.rand(E, device=dev, requires_grad=True) * 0.9 + 0.06
    ke = ke.detach().requires_grad_(True); w = (torch.rand(E, device=dev) * 0.3).requires_grad_(True)
    base = torch.log(x0) - torch.log1p(-x0)
    def step():
        h0 = hill4(x0[:, src], ke[None, :], 2.0)
        x = x0.clone()
        for _ in range(a.steps):
            msg = (hill4(x[:, src], ke[None, :], 2.0) - h0) * w[None, :]
            x = 0.5 * x + 0.5 * torch.sigmoid(base + torch.zeros(B, G, device=dev).index_add_(1, dst, msg))
        x.pow(2).mean().backward()
    for _ in range(3):
        step()
    sync()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(a.iters):
        sync(); t = time.perf_counter(); step(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), (torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan"))


configs = [("E0_exact_edge", dict(hill_impl="exact", prop_impl="edge")),
           ("E1_exact_spmm", dict(hill_impl="exact", prop_impl="spmm")),
           ("E2_exact_dense", dict(hill_impl="exact", prop_impl="dense")),
           ("C1_cheb16_spmm", dict(hill_impl="cheb", prop_impl="spmm", r=16)),
           ("C2_cheb16_dense", dict(hill_impl="cheb", prop_impl="dense", r=16)),
           ("C2s_cheb16_dense_stack", dict(hill_impl="cheb", prop_impl="dense", r=16, cheb_eval="stack"))]
res = []
for name in ("l1", "assoc"):
    G = int(Z[f"{name}_G"]); E = len(Z[f"{name}_src"])
    for B in a.batches:
        for cname, kw in configs:
            if "dense" in cname and G > 12000:
                continue
            try:
                m, G_ = make(name, **kw)
                t, mem = time_fb(m, G_, B)
            except RuntimeError as e:  # OOM etc.
                t, mem = float("nan"), float("nan"); print("fail", name, B, cname, str(e)[:80])
            res.append(dict(graph=name, G=G, E=E, B=B, impl=cname, sec=t, mem_mb=mem))
            print(f"{name} E={E} B={B} {cname}: {t*1e3:.1f} ms  mem {mem:.0f} MB", flush=True)
            del m
        try:
            t, mem = per_edge_hill(name, B)
        except RuntimeError as e:
            t, mem = float("nan"), float("nan")
        res.append(dict(graph=name, G=G, E=E, B=B, impl="P1_peredge_hill_edge", sec=t, mem_mb=mem))
        print(f"{name} E={E} B={B} P1_peredge: {t*1e3:.1f} ms mem {mem:.0f} MB", flush=True)

# equivalence on the real L1 graph: equilibrium + decoded delta, 200 targets
name = "l1"
eq = {}
G = int(Z[f"{name}_G"])
tgt = torch.as_tensor(rng.integers(0, G, 200), device=dev)
ref, _ = make(name, hill_impl="exact", prop_impl="edge")
with torch.no_grad():
    xr, _, _ = ref.equilibrium(tgt, 0.8)
    dr, _ = ref(tgt, 0.8)
for r in (4, 8, 16, 32):
    for pi in ("spmm", "dense"):
        m, _ = make(name, hill_impl="cheb", prop_impl=pi, r=r)
        m.load_state_dict({k: v for k, v in ref.state_dict().items() if k in m.state_dict() and m.state_dict()[k].shape == v.shape}, strict=False)
        with torch.no_grad():
            x, _, _ = m.equilibrium(tgt, 0.8); d, _ = m(tgt, 0.8)
        eq[f"cheb{r}_{pi}"] = dict(max_dx=float((x - xr).abs().max()), max_ddelta=float((d - dr).abs().max()), delta_scale=float(dr.abs().max()))
        print(f"cheb r={r} {pi}: max|dx|={eq[f'cheb{r}_{pi}']['max_dx']:.2e} max|ddelta|={eq[f'cheb{r}_{pi}']['max_ddelta']:.2e} (scale {eq[f'cheb{r}_{pi}']['delta_scale']:.3f})", flush=True)
json.dump(dict(device=a.device, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               steps=a.steps, timing=res, equivalence=eq), open(a.out, "w"), indent=1)
