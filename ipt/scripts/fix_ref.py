"""Repair a mangled cross-reference in results.tex (a stray line break turned the backslash-r of \\ref into a newline)."""
from pathlib import Path

p = Path(__file__).parents[1] / "paper" / "sections" / "results.tex"
b = p.read_bytes()
bs = bytes([92])
for bad in (b"~\r\nef{sec:results_expl}", b"~\nef{sec:results_expl}", b"~\ref{sec:results_expl}"):
    b = b.replace(bad, b"~" + bs + b"ref{sec:results_expl}")
p.write_bytes(b)
i = b.index(b"case with the largest gain")
print(b[i:i + 90])
