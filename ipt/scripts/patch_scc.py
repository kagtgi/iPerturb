"""Add a reachability-restricted global solver to scc.py and time it in scc_bench.py."""
from pathlib import Path

root = Path(__file__).parents[1]
p = root / "ipt" / "models" / "scc.py"
s = p.read_text(encoding="utf8")
if "def solve_global_reach" not in s:
    s += '''

def solve_global_reach(P, target, clamp, damping=0.5, tol=1e-13, max_iter=200000):
    """Global synchronous iteration restricted to the genes reachable from the target (no SCC ordering)."""
    nodes = [i for i in sorted(nx.descendants(P.graph, target)) if i != target]
    x = P.x0.copy()
    x[target] = clamp
    for it in range(max_iter):
        hx = P.T(x)
        new = x.copy()
        for m in nodes:
            new[m] = (1 - damping) * x[m] + damping * P.update(x, m, hx)
        new[target] = clamp
        d = np.abs(new - x).max()
        x = new
        if d < tol:
            return x, it + 1
    return x, max_iter
'''
    p.write_text(s, encoding="utf8")

q = root / "scripts" / "scc_bench.py"
b = q.read_text(encoding="utf8")
b = b.replace("from ipt.models.scc import Problem, solve_global, solve_scc", "from ipt.models.scc import Problem, solve_global, solve_global_reach, solve_scc")
old = "        t1 = time.perf_counter(); xs, nu = solve_scc(P, t, clamp)\n        t2 = time.perf_counter()\n"
new = ("        t1 = time.perf_counter(); xs, nu = solve_scc(P, t, clamp)\n        t2 = time.perf_counter()\n"
       "        xr, _ = solve_global_reach(P, t, clamp)\n        t3 = time.perf_counter()\n")
assert old in b
b = b.replace(old, new)
b = b.replace("max_diff=float(np.abs(xg - xs).max()),", "max_diff=float(np.abs(xg - xs).max()), max_diff_reach=float(np.abs(xg - xr).max()), t_greach=t3 - t2,")
b = b.replace('median_speedup=float(np.median([r["t_global"] / max(r["t_scc"], 1e-9) for r in rec])),',
              'median_speedup=float(np.median([r["t_global"] / max(r["t_scc"], 1e-9) for r in rec])),\n                        median_speedup_vs_reach=float(np.median([r["t_greach"] / max(r["t_scc"], 1e-9) for r in rec])),\n                        median_reach_vs_global=float(np.median([r["t_global"] / max(r["t_greach"], 1e-9) for r in rec])),')
q.write_text(b, encoding="utf8")
print("patched")
