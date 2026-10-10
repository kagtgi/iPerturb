"""One-off patch: nicer number formatting in build_numbers.py and repair of a mangled appendix reference."""
from pathlib import Path

ROOT = Path(__file__).parents[1]
bs = chr(92)

# 1) build_numbers.py: avoid '-0.0000' and typeset scientific notation as LaTeX
p = ROOT / "scripts" / "build_numbers.py"
s = p.read_text(encoding="utf8")
old_ci = s[s.index("def ci(r, d=3):"):s.index("# ---- data / graph counts")]
new_ci = (
    "def _f(v, d):\n"
    "    t = f\"{v:+.{d}f}\"\n"
    "    return f\"{abs(v):.{d}f}\" if float(t) == 0 else t\n\n\n"
    "def ci(r, d=3):\n"
    "    return f\"{_f(r['mean_diff'], d)} [{_f(r['ci_low'], d)}, {_f(r['ci_high'], d)}]\"\n\n\n"
    "def sci(v, d=1):\n"
    "    m, e = f\"{v:.{d}e}\".split(\"e\")\n"
    "    return f\"${m}" + bs + "times10^{{{int(e)}}}$\"\n\n\n"
)
s = s.replace(old_ci, new_ci)
s = s.replace('put(f"{s}_scc_diff", r["max_diff"], "{:.0e}")', 'put(f"{s}_scc_diff", sci(r["max_diff"], 0))')
s = s.replace('put(f"eq_{k}_dx", v["max_dx"], "{:.1e}")', 'put(f"eq_{k}_dx", sci(v["max_dx"]))')
s = s.replace('put(f"eq_{k}_dd", v["max_ddelta"], "{:.1e}")', 'put(f"eq_{k}_dd", sci(v["max_ddelta"]))')
p.write_text(s, encoding="utf8")

# 2) appendix reference in results.tex
q = ROOT / "paper" / "sections" / "results.tex"
b = q.read_bytes()
for bad in (b"Appendix~\r\nefapp:bench", b"Appendix~\nefapp:bench", b"Appendix~efapp:bench", b"Appendix~\ref{app:bench}"):
    b = b.replace(bad, b"Appendix~" + bs.encode() + b"ref{app:bench}")
q.write_bytes(b)
print(b[b.index(b"Appendix~"):b.index(b"Appendix~") + 40])
