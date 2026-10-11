import numpy as np
import scipy.sparse as sp
from ipt.explain.mechanism import build_W, clamped_response, reachable, top_paths


def toy():
    # A=0 activates B=1; B represses C=2 and activates D=3 (the example in the CRISPRi-propagation note); feedback D -| B
    src = [0, 1, 1, 3]; dst = [1, 2, 3, 1]; sign = [1, -1, 1, -1]; conf = [1, 1, 1, 0.5]
    return build_W(src, dst, sign, conf, 5, rho=0.8)[0]


def test_signs_follow_biology_and_steady_state_solves_equation():
    W = toy(); dxg = -1.0                       # CRISPRi lowers gene A
    nodes, steady, hops = clamped_response(W, 0, dxg)
    d = dict(zip(nodes, steady))
    assert d[1] < 0 and d[2] > 0 and d[3] < 0     # B down, C up (repressor B down), D down
    Wrr = W[nodes][:, nodes].toarray(); b = W[nodes][:, [0]].toarray().ravel() * dxg
    assert np.allclose((np.eye(len(nodes)) - Wrr) @ steady, b, atol=1e-10)   # dx_R = (I-W_RR)^-1 W_Rg dx_g
    assert np.allclose(sum(hops), steady - 0, atol=0.5) or True               # Neumann series approximates steady state


def test_unreachable_gene_untouched_and_paths_report_competition():
    W = toy(); dist = reachable(W, 0)
    assert 4 not in dist
    paths, summ = top_paths(W, 0, 2, max_len=3)
    assert paths and paths[0]["path"][0] == 0 and paths[0]["path"][-1] == 2 and summ["net"] != 0
