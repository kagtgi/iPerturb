"""List literal numbers in the manuscript text that do not come through the generated \\datanum macros."""
import re
from pathlib import Path

root = Path(__file__).parents[1] / "paper"
files = sorted((root / "sections").glob("*.tex")) + [root / "main.tex"]
for p in files:
    s = p.read_text(encoding="utf8")
    s = re.sub(r"\\datanum\{[^}]*\}", "", s)
    s = re.sub(r"\\datanumif\{[^}]*\}", "", s)
    s = re.sub(r"\\(label|ref|cite|input|includegraphics|eqref)(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"%.*", "", s)
    hits = []
    for m in re.finditer(r"(?<![A-Za-z{\\_^])\d+(\.\d+)?(?![A-Za-z}_])", s):
        hits.append((m.group(0), s[max(0, m.start() - 35):m.end() + 25].replace("\n", " ")))
    print(p.name, len(hits))
    for h in hits:
        print("   ", h[0], "|", h[1])
