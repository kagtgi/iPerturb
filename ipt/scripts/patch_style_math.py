"""Use Google Sans for math text too (Greek letters in axis labels) so figures look like the overview figure."""
from pathlib import Path

p = Path(__file__).parents[1] / "ipt" / "viz" / "style.py"
s = p.read_text(encoding="utf8")
if "mathtext.fontset" not in s:
    s = s.replace('"font.size": 8.0,', '"mathtext.fontset": "custom", "mathtext.rm": "Google Sans", "mathtext.it": "Google Sans", "mathtext.bf": "Google Sans:bold",\n        "font.size": 8.0,', 1)
    p.write_text(s, encoding="utf8")
print("ok", "mathtext.fontset" in s)
