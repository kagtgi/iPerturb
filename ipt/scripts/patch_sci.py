"""Make sci() emit a LaTeX \\times (the f-string escape turned it into a tab)."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "build_numbers.py"
s = p.read_text(encoding="utf8")
i = s.index("def sci(v, d=1):")
j = s.index("# ---- data / graph counts")
bs = chr(92)
new = (
    "def sci(v, d=1):\n"
    "    m, e = f\"{v:.{d}e}\".split(\"e\")\n"
    "    return \"$\" + m + chr(92) + \"times10^{\" + str(int(e)) + \"}$\"\n\n\n"
)
p.write_text(s[:i] + new + s[j:], encoding="utf8")
print(new)
