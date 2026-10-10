import pandas as pd
from ipt.graph import l1


def test_sign_policy(tmp_path):
    cols = "source target source_genesymbol target_genesymbol is_directed is_stimulation is_inhibition consensus_direction consensus_stimulation consensus_inhibition sources".split()
    rows = [["a", "b", "A", "B", True, True, False, True, True, False, "x"],
            ["a", "c", "A", "C", True, False, False, True, False, False, "x"]]
    for n in ("omnipath_collectri.tsv", "omnipath_tf_target.tsv"):
        pd.DataFrame(rows, columns=cols).to_csv(tmp_path / n, sep="\t", index=False)
    (tmp_path / "trrust.tsv").write_text("A\tB\tRepression\t1\nA\tC\tUnknown\t2\n")
    e = l1.load_l1(tmp_path).set_index(["source", "target"])
    assert e.loc[("A", "B"), "sign"] == 0 and e.loc[("A", "B"), "sign_conflict"]
    assert e.loc[("A", "C"), "sign"] == 0 and not e.loc[("A", "C"), "sign_conflict"]
