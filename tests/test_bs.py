"""Black-Scholes utilities used by CPI preprocessing (paper §2, §5)."""

from __future__ import annotations

import numpy as np

from src.bs import bs_vega, implied_vol, normalised_call


def test_normalised_call_bounds():
    """``max(1 - x, 0) <= c(x) <= 1`` for the normalised call price."""
    x = np.array([0.5, 1.0, 1.5])
    T = 1.0
    w = np.array([0.04, 0.04, 0.04])
    c = normalised_call(x, w)
    lo = np.maximum(1.0 - x, 0.0)
    assert np.all(c >= lo - 1e-10)
    assert np.all(c <= 1.0 + 1e-10)


def test_implied_vol_roundtrip():
    sig_true = np.array([0.12, 0.18, 0.25, 0.3])
    x = np.array([0.8, 0.95, 1.05, 1.2])
    T = 0.5
    w = sig_true ** 2 * T
    c = normalised_call(x, w)
    sig_hat = implied_vol(c, x, T)
    assert np.allclose(sig_hat, sig_true, atol=1e-6)


def test_vega_positive():
    x = np.array([0.6, 1.0, 1.5])
    sig = np.array([0.2, 0.2, 0.2])
    v = bs_vega(x, sig, 0.5)
    assert np.all(v > 0)
