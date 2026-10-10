"""Fig 4a: aggregate the G2 gate to unique targets (matches the text and Methods)."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "fig4_ablation_efficiency.py"
s = p.read_text(encoding="utf8")
old = 'pt = lambda m, g: R[(R.model == m) & (R.graph == g)].groupby(["fold", "target"]).pcc_c.mean()'
new = 'pt = lambda m, g: R[(R.model == m) & (R.graph == g)].groupby("target").pcc_c.mean()'
assert old in s
p.write_text(s.replace(old, new), encoding="utf8")
print("ok")
