"""Merge the per-fold g12 outputs (tags _p0.._p2) into g12_<line>.tsv."""
from pathlib import Path

import pandas as pd

S = Path(__file__).parents[1] / "artifacts" / "scores"
for line in ("K562_essential", "rpe1"):
    parts = [pd.read_csv(f, sep="\t") for f in sorted(S.glob(f"g12_{line}_p*.tsv"))]
    if parts:
        out = pd.concat(parts)
        out.to_csv(S / f"g12_{line}.tsv", sep="\t", index=False)
        print(line, len(parts), "folds,", out.fold.nunique(), "unique folds,", out.target.nunique(), "targets")
