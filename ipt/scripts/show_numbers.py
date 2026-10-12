"""Print selected \\datanum values: py scripts/show_numbers.py key1 key2 ... (prefix match with a trailing *)."""
import re
import sys
from pathlib import Path

P = Path(__file__).parents[1] / "paper"
t = (P / "numbers.tex").read_text(encoding="utf8") + (P / "numbers2.tex").read_text(encoding="utf8")
d = dict(re.findall(r"data@(.+?)" + re.escape(chr(92)) + r"endcsname\{(.*)\}", t))
for k in sys.argv[1:]:
    if k.endswith("*"):
        for kk in sorted(d):
            if kk.startswith(k[:-1]):
                print(kk, "=", d[kk])
    else:
        print(k, "=", d.get(k))
