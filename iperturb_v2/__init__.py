"""iPerturb v2 — network-propagation model, exactly as used in the FairPert benchmark.

The propagation engine (:mod:`iperturb_v2._iperturb_prop`) is pure numpy/scipy and
imports eagerly here. The benchmark entry point ``run_eval`` lives in
:mod:`iperturb_v2.adapter` and is imported lazily (it pulls in ``config`` → torch,
which is only needed for the legacy ``hill`` engine), so::

    from iperturb_v2 import _iperturb_prop      # torch-free propagation engine
    from iperturb_v2.adapter import run_eval     # full benchmark entry point
"""
from . import _iperturb_prop

__all__ = ["_iperturb_prop"]
