import numpy as np
from ipt.data.prepare2 import aggregate2, cell_qc, finalize2, mad_mask, normalise


def test_mad_and_qc():
    rng = np.random.default_rng(0)
    umi = rng.normal(10000, 500, 400)
    umi[:5] = 90000
    batch = np.repeat([0, 1], 200)
    k = mad_mask(np.log1p(umi), batch)
    assert not k[:5].any() and k[5:].mean() > 0.97
    keep, qc = cell_qc(umi, umi / 3, np.full(400, 5.0), batch, nperts=np.r_[np.zeros(100), np.ones(290), 2 * np.ones(10)])
    assert keep.sum() == qc["kept"] and qc["fail_umi"] >= 5 and qc["fail_multiguide"] == 10 and keep[5:100].all()


def test_matches_naive_matched_control_and_knockdown():
    rng = np.random.default_rng(1)
    n, G = 600, 15
    X = rng.poisson(4, (n, G)).astype(np.float32)
    pert = rng.integers(0, 4, n)      # 0 = control
    batch = rng.integers(0, 3, n)
    t = 2                              # perturbation 1 targets gene 2 (knockdown)
    X[pert == 1, t] = rng.poisson(1, (pert == 1).sum())
    keep = np.ones(n, bool)
    keep[:30] = False
    A = aggregate2([(s, X[s:s + 90]) for s in range(0, n, 90)], pert, batch, keep, 4, 3, G, 0)
    tgt = np.array([-1, t, -1, -1])
    R = finalize2(A, 0, tgt)
    L, lin, _ = normalise(X.astype(np.float32))
    for p in (1, 2, 3):
        m = (pert == p) & keep
        mean = L[m].mean(0)
        w = np.array([(m & (batch == b)).sum() for b in range(3)], float)
        cm = np.stack([L[(pert == 0) & keep & (batch == b)].mean(0) for b in range(3)])
        matched = (w[:, None] * cm).sum(0) / w.sum()
        assert np.allclose(R["delta"][p], mean - matched, atol=1e-4)
        assert np.all(R["se"][p] > 0)
    assert R["kd_detected"][1] and 0.5 < R["kd"][1] < 1.0     # strong knockdown recovered
    assert not np.isfinite(R["kd"][2])                           # no target -> unknown, not 100%
