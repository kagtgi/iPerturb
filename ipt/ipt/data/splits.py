"""Target-held-out folds (70/15/15), stratified only by train-independent target properties."""
from __future__ import annotations
import hashlib, json
import numpy as np


def make_folds(targets, strata, n_folds=3, seed=42, frac=(0.70, 0.15, 0.15)):
    """targets: list[str] unique; strata: list of hashable keys (same length). Returns list of dicts."""
    targets = list(targets); strata = list(strata)
    assert len(set(targets)) == len(targets)
    out = []
    for f in range(n_folds):
        rng = np.random.default_rng(seed + 1000 * f)
        tr, va, te = [], [], []
        for s in sorted(set(strata), key=str):
            idx = [i for i, k in enumerate(strata) if k == s]
            rng.shuffle(idx)
            n = len(idx); ntr = int(round(frac[0] * n)); nva = int(round(frac[1] * n))
            tr += idx[:ntr]; va += idx[ntr:ntr + nva]; te += idx[ntr + nva:]
        d = {"fold": f, "seed": seed + 1000 * f,
             "train": sorted(targets[i] for i in tr), "valid": sorted(targets[i] for i in va),
             "test": sorted(targets[i] for i in te)}
        d["sha256"] = hashlib.sha256(json.dumps([d["train"], d["valid"], d["test"]]).encode()).hexdigest()
        out.append(d)
    return out


def check_disjoint(fold):
    a, b, c = set(fold["train"]), set(fold["valid"]), set(fold["test"])
    return not (a & b or a & c or b & c)
