"""Vectorised equivalent of `state tx infer --tsv`: append `num_cells` random-control copies per requested perturbation (label changed), no per-cell python loop."""
import sys

import anndata as ad
import numpy as np
import pandas as pd

ctrl_h5, tsv, out = sys.argv[1:4]
c = ad.read_h5ad(ctrl_h5)
t = pd.read_csv(tsv, sep="\t")
rng = np.random.RandomState(42)
src = np.concatenate([rng.choice(c.n_obs, int(n), replace=True) for n in t["num_cells"]])
lab = np.repeat(t["perturbation"].astype(str).to_numpy(), t["num_cells"].to_numpy())
obs = c.obs.iloc[src].copy()
obs["gene"] = lab
obs.index = [f"v{i}" for i in range(len(obs))]
v = ad.AnnData(X=c.X[src], obs=obs, var=c.var, obsm={"X_hvg": np.asarray(c.obsm["X_hvg"])[src]})
big = ad.concat([c, v], join="outer", merge="first")
big.obs_names_make_unique()
big.var = c.var
big.write_h5ad(out)
print("padded", c.n_obs, "->", big.n_obs)
