"""Make g7_context.py runnable per fold (--folds, --tag) and add a merge script."""
from pathlib import Path

root = Path(__file__).parents[1]
p = root / "scripts" / "g7_context.py"
s = p.read_text(encoding="utf8")
if "--tag" not in s:
    s = s.replace("import json, pickle, sys\n", "import argparse, json, pickle, sys\n", 1)
    s = s.replace('line = "K562_essential"\n', 'line = "K562_essential"\nap = argparse.ArgumentParser()\nap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])\nap.add_argument("--tag", default="")\na = ap.parse_args()\n', 1)
    s = s.replace("for fo in (0, 1, 2):", "for fo in a.folds:")
    s = s.replace('df.to_csv(ART / "scores" / "g7_K562.tsv", sep="\\t", index=False)', 'df.to_csv(ART / "scores" / f"g7_K562{a.tag}.tsv", sep="\\t", index=False)')
    s = s.replace('pd.DataFrame(cfgs).to_csv(ART / "scores" / "g7cfg_K562.tsv", sep="\\t", index=False)', 'pd.DataFrame(cfgs).to_csv(ART / "scores" / f"g7cfg_K562{a.tag}.tsv", sep="\\t", index=False)')
    # the descriptive graph statistics are written by every run; keep (identical)
    p.write_text(s, encoding="utf8")

m = root / "scripts" / "g7_merge.py"
m.write_text('''"""Merge per-fold g7 outputs into g7_K562.tsv / g7cfg_K562.tsv."""
from pathlib import Path
import pandas as pd

S = Path(__file__).parents[1] / "artifacts" / "scores"
for stem in ("g7", "g7cfg"):
    parts = [pd.read_csv(f, sep="\\t") for f in sorted(S.glob(f"{stem}_K562_f*.tsv"))]
    out = pd.concat(parts)
    out.to_csv(S / f"{stem}_K562.tsv", sep="\\t", index=False)
    print(stem, len(parts), "parts,", len(out), "rows")
''', encoding="utf8")
print("ok")
