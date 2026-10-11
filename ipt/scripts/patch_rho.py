import re
from pathlib import Path

p = Path(__file__).parents[1] / "paper" / "sections" / "results.tex"
s = p.read_bytes().decode("utf8")
s2, n = re.subn(r"the best \$[^a-z$]*(?:\\r)?ho_c\$ of all", lambda m: "the best $" + chr(92) + "rho_c$ of all", s)
print(n, repr(s[s.index("the best $"):][:30]))
p.write_bytes(s2.encode("utf8"))
