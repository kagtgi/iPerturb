from ipt.data.splits import make_folds, check_disjoint


def test_folds_disjoint_and_stratified():
    t = [f"G{i}" for i in range(200)]; s = [i % 4 for i in range(200)]
    fs = make_folds(t, s)
    assert len(fs) == 3
    for f in fs:
        assert check_disjoint(f) and len(f["train"]) + len(f["valid"]) + len(f["test"]) == 200
        assert 25 <= len(f["test"]) <= 35
    assert fs[0]["sha256"] != fs[1]["sha256"]
    assert make_folds(t, s)[0]["sha256"] == fs[0]["sha256"]
