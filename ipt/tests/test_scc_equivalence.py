import numpy as np
from ipt.models.hill import hill4
from ipt.models.scc import Problem, solve_global, solve_scc
import torch


def make(seed, G=40, E=90, transfer="hill4"):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < E:
        a, b = rng.integers(0, G, 2)
        if a != b:
            pairs.add((int(a), int(b)))
    src, dst = map(np.array, zip(*sorted(pairs)))
    w = rng.normal(0, 0.35, E)
    x0 = rng.uniform(0.2, 0.8, G)
    if transfer == "hill4":
        T = lambda x: hill4(torch.as_tensor(x), torch.tensor(0.3, dtype=torch.float64), 2.0).numpy()
    else:
        T = lambda x: x
    return Problem(src, dst, w, x0, T)


def test_scc_matches_global_with_feedback():
    for seed in range(4):
        for tr in ("hill4", "linear"):
            P = make(seed, transfer=tr)
            assert P.largest_scc > 1  # random graph has feedback
            for t in (0, 5, 17):
                xg, _ = solve_global(P, t, 0.1)
                xs, _ = solve_scc(P, t, 0.1)
                assert np.abs(xg - xs).max() < 1e-9


def test_scc_dag_is_single_pass():
    src, dst = np.array([0, 1, 2]), np.array([1, 2, 3])
    P = Problem(src, dst, [0.5, -0.5, 0.5], np.full(5, 0.5), lambda x: x)
    xg, _ = solve_global(P, 0, 0.1)
    xs, nu = solve_scc(P, 0, 0.1)
    assert np.abs(xg - xs).max() < 1e-10 and nu == 3 and xs[4] == 0.5  # unreachable node untouched
