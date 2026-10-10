import numpy as np
from ipt.models.baselines import Linear, MeanShift, NoChange, fit_linear_select, cross_fit_linear


def toy(n=120, G=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(G, k)); tgt = rng.integers(0, G, n)
    D = emb[tgt] @ rng.normal(size=(k, k)) @ emb.T + 0.01 * rng.normal(size=(n, G)) + 0.5
    return D.astype(np.float32), tgt


def test_linear_recovers_low_rank():
    D, t = toy(); m = fit_linear_select(D[:90], t[:90], D[90:105], t[90:105])
    e_lin = ((m.predict(t[105:]) - D[105:]) ** 2).mean()
    e_mean = ((MeanShift().fit(D[:90], t[:90]).predict(t[105:]) - D[105:]) ** 2).mean()
    assert e_lin < e_mean


def test_b0_b1_shapes_and_crossfit():
    D, t = toy()
    assert NoChange().fit(D, t).predict(t[:5]).shape == (5, 40)
    assert np.allclose(MeanShift().fit(D, t).predict(t[:2])[0], D.mean(0), atol=1e-5)
    cf = cross_fit_linear(D, t, 3)
    assert cf.shape == D.shape and np.isfinite(cf).all()
