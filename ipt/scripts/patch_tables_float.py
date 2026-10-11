"""Appendix tables: allow placement 'here' to avoid near-empty pages."""
from pathlib import Path

root = Path(__file__).parents[1] / "paper" / "sections"
bs = chr(92)
for name, old in (("appendix_convergence.tex", bs + "begin{table}[t]"), ("appendix.tex", bs + "begin{table}[h]")):
    p = root / name
    s = p.read_text(encoding="utf8")
    p.write_text(s.replace(old, bs + "begin{table}[!ht]"), encoding="utf8")
    print(name, s.count(old))
