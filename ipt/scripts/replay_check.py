"""Independent replay: recompute Delta for random perturbations straight from the h5ad (no chunk/sparse machinery)
and compare with the stored per-perturbation deltas. Run on the VM (gears-env python)."""
import sys

import h5py
import numpy as np

h5, npz = sys.argv[1], sys.argv[2]
n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 12
z = np.load(npz, allow_pickle=True)
perts, delta = [str(p) for p in z["perts"]], z["delta"]
h = h5py.File(h5, "r")
o = h["obs"]


def col(k):
    c = o[k]
    a = np.array(c["categories"])[np.array(c["codes"])] if isinstance(c, h5py.Group) else np.array(c)
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


pert, batch = col("perturbation"), col("batch")
X = h["X"]
G = X.shape[1]


def lognorm(rows):
    x = np.asarray(X[rows], dtype=np.float64)
    return np.log1p(x / x.sum(1, keepdims=True) * 1e4)


ctrl = np.where(pert == "control")[0]
cm = {}
for b in np.unique(batch[ctrl]):
    ix = np.sort(ctrl[batch[ctrl] == b])
    cm[b] = lognorm(ix).mean(0)
rng = np.random.default_rng(0)
worst = 0.0
for i in rng.choice(len(perts), n_check, replace=False):
    ix = np.sort(np.where(pert == perts[i])[0])
    mean = lognorm(ix).mean(0)
    bs, cnt = np.unique(batch[ix], return_counts=True)
    w = np.array([c if b in cm else 0 for b, c in zip(bs, cnt)], float)
    matched = sum(wi * cm[b] for b, wi in zip(bs, w) if wi > 0) / w.sum()
    d = mean - matched
    diff = float(np.abs(d - delta[i]).max())
    worst = max(worst, diff)
    print(f"{perts[i]:>12s} cells={len(ix):5d} max|delta_replay - stored| = {diff:.2e}", flush=True)
print("WORST", worst)
