from pathlib import Path

P = Path(__file__).parents[1] / "paper"
BS = chr(92)
m = (P / "main.tex").read_text(encoding="utf8")
a = m.index(BS + "begin{figure}")
b = m.index(BS + "end{figure}", a) + len(BS + "end{figure}")
fig = m[a:b].replace(BS + "begin{figure}[t]", BS + "begin{figure}[!t]")
m = m[:a] + m[b:]
(P / "main.tex").write_text(m, encoding="utf8")
i = P / "sections" / "intro.tex"
s = i.read_text(encoding="utf8")
anchor = "Our contributions are twofold."
s = s.replace(anchor, fig + "\n\n" + anchor, 1)
i.write_text(s, encoding="utf8")
print(fig[:200])
