"""Shared utilities for iPerturb."""

from __future__ import annotations

import logging

import numpy as np
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


log = logging.getLogger("iperturb")


def logmean(X) -> np.ndarray:
    """
    Aggregate cells: log1p( mean( expm1(X) ) ).

    Input X is log1p-normalised counts (standard Scanpy convention).
    Taking the arithmetic mean in count space before re-logging avoids the
    Jensen's-inequality bias that arises from averaging in log space and ensures
    the baseline x0 is consistent across model init and training loss.
    """
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.asarray(X)
    return np.log1p(np.expm1(X).mean(axis=0)).ravel()


def inv_softplus(y, eps: float = 1e-6) -> torch.Tensor:
    """Inverse of F.softplus so softplus(inv_softplus(y)) ≈ y."""
    y = torch.as_tensor(y, dtype=torch.float32).clamp(min=eps)
    return torch.log(torch.expm1(y))


def make_session() -> requests.Session:
    """Create a requests.Session with automatic retries and a browser-like UA."""
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (GRN-research)"})
    return s


def robust_get(session: requests.Session, url: str, params=None, timeout: int = 120):
    """GET with raise_for_status."""
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r
