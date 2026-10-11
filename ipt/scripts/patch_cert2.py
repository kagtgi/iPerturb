"""convergence_cert.py: weighted-norm certificate q_v = 1 - 1/max(v), v = (I-|W|)^-1 1  (only needs rho(|W|) < 1) and an empirical truncation check."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "convergence_cert.py"
s = p.read_text(encoding="utf8")
if "q_v" not in s:
    s = s.replace("import numpy as np\n", "import numpy as np\nimport scipy.sparse as sp\nfrom scipy.sparse.linalg import splu\n", 1)
    old = "    # (b) certified nonlinear operators"
    new = '''    # weighted sup-norm certificate: v = (I - |W|)^-1 1 >= 1 gives |W| v = v - 1 <= (1 - 1/max v) v, hence ||W_RR||_v <= q_v < 1
    A_abs = abs(Wsc).tocsc()
    v = splu((sp.identity(G, format="csc") - A_abs).tocsc()).solve(np.ones(G))
    q_v = 1.0 - 1.0 / float(v.max())
    rec["q_v"] = q_v
    rec["v_max"] = float(v.max())
    ratios = []
    for g in has_out:
        nodes, steady, per_hop = clamped_response(Wsc, g, -1.0, hops=4)
        if len(nodes) == 0:
            continue
        b = np.asarray(Wsc[nodes][:, [g]].todense()).ravel() * -1.0
        vv = v[nodes]
        nb = float(np.max(np.abs(b) / vv))
        for K in (1, 2, 3, 4):
            err = float(np.max(np.abs(steady - sum(per_hop[:K])) / vv))
            bound = q_v ** K / (1 - q_v) * nb
            ratios.append(err / max(bound, 1e-300))
    rec["truncation_bound_max_ratio"] = float(max(ratios))     # <= 1 means the bound holds on every checked perturbation
    # (b) certified nonlinear operators'''
    assert old in s
    s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf8")
print("ok")
