from pathlib import Path

S = Path(__file__).parents[1] / "paper" / "sections"
BS = chr(92)
m = (S / "methods.tex").read_text(encoding="utf8")
a = m.index(BS + "subsection{Constructing the context-specific GRN}")
b = m.index(BS + "subsection{Learnable propagation that follows the Hill function}")
new = (S / "grn_new.tex").read_text(encoding="utf8")
m = m[:a] + new + "\n" + m[b:]
(S / "methods.tex").write_text(m, encoding="utf8")
(S / "grn_new.tex").unlink()
print("ok")
