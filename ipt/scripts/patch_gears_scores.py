"""Patch gears_to_scores.py: centre GEARS on its own training-mean prediction (fair to its control-mean offset)."""
from pathlib import Path

p = Path(__file__).parents[1] / "scripts" / "gears_to_scores.py"
s = p.read_text(encoding="utf8")
old = "        f = folds[fo]\n"
new = (
    "        # GEARS is offset by its own control mean; centre it on the mean of its predictions for training targets\n"
    "        if \"train_delta\" in d.files:\n"
    "            gtm = np.zeros(G, np.float32)\n"
    "            gtm[col[ok]] = d[\"train_delta\"].mean(0)[ok]\n"
    "            centring = \"gears_train_mean\"\n"
    "        else:\n"
    "            gtm = P.mean(0)\n"
    "            centring = \"test_prediction_mean\"\n"
    "        f = folds[fo]\n"
)
assert old in s
s = s.replace(old, new, 1)
old2 = "        obs, tt = Yall[ix].astype(np.float64), tix[ix]\n"
new2 = "        obs, tt = Yall[ix].astype(np.float64), tix[ix]\n        P_raw = P.copy()\n        P = (P - gtm + tm).astype(np.float32)   # after subtracting tm this is P - gtm\n"
assert old2 in s
s = s.replace(old2, new2, 1)
s = s.replace("pr = M.pcc_delta(P, obs, tt)", "pr = M.pcc_delta(P_raw, obs, tt)")
s = s.replace('print(line, fo, len(names), "targets scored")', 'print(line, fo, len(names), "targets scored; centring:", centring)')
p.write_text(s, encoding="utf8")
print("ok")
