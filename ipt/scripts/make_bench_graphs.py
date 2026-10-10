"""Edge arrays for the efficiency benchmark: signed L1 core and the L1+L3 association graph (K562 axis)."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from ipt.graph.l1 import load_l1

ART = Path(__file__).parents[1] / "artifacts"
CACHE = Path("D:/iPerturb/lnbi/cache")
z = np.load(ART / "data" / "K562_essential.npz", allow_pickle=True)
genes = [g.upper() for g in z["genes"]]
gi = {g: i for i, g in enumerate(genes)}
G = len(genes)
cm = z["ctrl_mean"]
x0 = (1 / (1 + np.exp(-(cm - cm.mean()) / cm.std()))).astype(np.float32)
l1 = load_l1(CACHE)
s = l1[(l1.sign != 0) & l1.source.isin(gi) & l1.target.isin(gi)]
src, dst = s.source.map(gi).to_numpy(), s.target.map(gi).to_numpy()
out = dict(l1_G=G, l1_src=src, l1_dst=dst, l1_sign=s.sign.to_numpy(np.float32),
           l1_conf=(np.minimum(s.n_db.to_numpy(), 3) / 3).astype(np.float32), l1_x0=x0)
st = pickle.load(open(CACHE / "string_symbol_pairs_700.pkl", "rb"))
m = st.a.str.upper().isin(gi) & st.b.str.upper().isin(gi)
a_, b_ = st.a[m].str.upper().map(gi).to_numpy(), st.b[m].str.upper().map(gi).to_numpy()
e = np.unique(np.concatenate([np.stack([a_, b_], 1), np.stack([src, dst], 1)]), axis=0)
e = e[e[:, 0] != e[:, 1]]
e = np.unique(np.concatenate([e, e[:, ::-1]]), axis=0)  # undirected -> both directions
rng = np.random.default_rng(0)
out.update(assoc_G=G, assoc_src=e[:, 0], assoc_dst=e[:, 1], assoc_sign=rng.choice([-1.0, 1.0], len(e)).astype(np.float32),
           assoc_conf=np.ones(len(e), np.float32), assoc_x0=x0)
np.savez(ART / "bench_graphs.npz", **out)
print("L1 edges", len(src), "assoc edges", len(e), "G", G)
