import numpy as np
from ipt.data.prepare import aggregate, finalize, normalize


def test_matches_naive():
    rng = np.random.default_rng(0)
    n, G = 300, 12
    X = rng.poisson(3, (n, G)).astype(np.float32)
    pert = rng.integers(0, 4, n); batch = rng.integers(0, 3, n)  # pert 0 = control
    chunks = [(s, X[s:s + 70]) for s in range(0, n, 70)]
    S, Q, N, CB = aggregate(chunks, pert, batch, 4, 3, G, 0)
    nn, mean, var, matched, nb = finalize(S, Q, N, CB, 0)
    L = normalize(X)
    for p in (1, 2, 3):
        m = pert == p
        assert np.allclose(mean[p], L[m].mean(0), atol=1e-5)
        assert np.allclose(var[p], L[m].var(0, ddof=1), atol=1e-4)
        w = np.array([(m & (batch == b)).sum() for b in range(3)], float)
        cm = np.stack([L[(pert == 0) & (batch == b)].mean(0) for b in range(3)])
        assert np.allclose(matched[p], (w[:, None] * cm).sum(0) / w.sum(), atol=1e-5)
    assert nb == 3
