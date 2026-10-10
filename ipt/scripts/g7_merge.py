"""Merge per-fold g7 outputs into g7_K562.tsv / g7cfg_K562.tsv."""
from pathlib import Path
import pandas as pd

S = Path(__file__).parents[1] / "artifacts" / "scores"
for stem in ("g7", "g7cfg"):
    parts = [pd.read_csv(f, sep="\t") for f in sorted(S.glob(f"{stem}_K562_f*.tsv"))]
    out = pd.concat(parts)
    out.to_csv(S / f"{stem}_K562.tsv", sep="\t", index=False)
    print(stem, len(parts), "parts,", len(out), "rows")
