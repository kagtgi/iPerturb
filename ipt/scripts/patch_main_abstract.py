"""main.tex: abstract from sections/abstract.tex, numbers3, algorithm + placeins packages."""
import re
from pathlib import Path

p = Path(__file__).parents[1] / "paper" / "main.tex"
s = p.read_text(encoding="utf8")
BS = chr(92)
a, b = s.index(BS + "begin{abstract}"), s.index(BS + "keywords{")
s = s[:a] + BS + "begin{abstract}\n" + BS + "input{sections/abstract.tex}\n" + s[b:]
if "numbers3.tex" not in s:
    s = s.replace(BS + "InputIfFileExists{numbers2.tex}{}{}", BS + "InputIfFileExists{numbers2.tex}{}{}\n" + BS + "InputIfFileExists{numbers3.tex}{}{}")
if "algpseudocode" not in s:
    s = s.replace(BS + "usepackage{booktabs}", BS + "usepackage{booktabs}\n" + BS + "usepackage{algorithm}\n" + BS + "usepackage[noend]{algpseudocode}\n" + BS + "usepackage[section]{placeins}")
p.write_text(s, encoding="utf8")
print(s[:1400])
