"""Make every figure use the same Google Sans fonts as the iPerturb overview figure (register the bundled TTFs)."""
from pathlib import Path

p = Path(__file__).parents[1] / "ipt" / "viz" / "style.py"
s = p.read_text(encoding="utf8")
if "FONT_DIR" not in s:
    s = s.replace("import matplotlib as mpl\n", '''import matplotlib as mpl
from pathlib import Path as _Path
from matplotlib import font_manager as _fm

# the same bundled Google Sans files that the overview figure (Fig. 1) uses
FONT_DIR = _Path("D:/iPerturb/lnbi/figure_tools/assets/fonts/Google_Sans/static")
_REGISTERED = []
for _f in ("GoogleSans-Regular.ttf", "GoogleSans-Medium.ttf", "GoogleSans-Bold.ttf"):
    _p = FONT_DIR / _f
    if _p.exists():
        _fm.fontManager.addfont(str(_p))
        _REGISTERED.append(_p.name)
''', 1)
    s = s.replace('"font.sans-serif": ["Google Sans Flex", "Roboto", "DejaVu Sans"],', '"font.sans-serif": ["Google Sans", "Google Sans Flex", "Roboto", "DejaVu Sans"],')
    p.write_text(s, encoding="utf8")
print("patched", "FONT_DIR" in s)
