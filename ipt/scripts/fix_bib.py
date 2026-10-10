"""Repair control characters introduced into references.bib by shell escaping."""
from pathlib import Path

p = Path(__file__).parents[1] / "paper" / "references.bib"
s = p.read_text(encoding="utf8")
bs = chr(92)
vt = chr(11)
i = s.find("Solja")
print(repr(s[i:i + 40]))
j = s.find("Hou, Thomas")
k = s.find("Tegmark", j)
seg = s[s.find("Solja"):k]
new = "Solja{" + bs + "v{c}}i{" + bs + "'c}, Marin"
s = s.replace(seg, new + " and Hou, Thomas Y. and ")
p.write_text(s, encoding="utf8")
print(repr(s[s.find("Solja"):s.find("Tegmark") + 8]), vt in s)
