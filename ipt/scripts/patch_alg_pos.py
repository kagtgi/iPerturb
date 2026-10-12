from pathlib import Path

p = Path(__file__).parents[1] / "paper" / "sections" / "methods.tex"
BS = chr(92)
s = p.read_text(encoding="utf8")
a = s.index(BS + "begin{algorithm}")
b = s.index(BS + "end{algorithm}") + len(BS + "end{algorithm}")
alg = s[a:b].replace(BS + "begin{algorithm}[t]", BS + "begin{algorithm}[!tb]")
s = s[:a].rstrip() + "\n" + s[b:]
lab = BS + "label{sec:grn}\n"
s = s.replace(lab, lab + alg + "\n", 1)
p.write_text(s, encoding="utf8")
print("ok")
