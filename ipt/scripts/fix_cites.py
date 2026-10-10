"""Normalise \\cite{a, b} -> \\cite{a,b} in the manuscript sources."""
import re
from pathlib import Path

for p in Path(__file__).parents[1].joinpath("paper", "sections").glob("*.tex"):
    s = p.read_text(encoding="utf8")
    s2 = re.sub(r"\\cite\{[^}]*\}", lambda m: m.group(0).replace(", ", ","), s)
    if s2 != s:
        p.write_text(s2, encoding="utf8")
        print("fixed", p.name)
