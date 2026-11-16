"""Black-Scholes utilities in normalised forward-price variables.

Normalised call price ``c(x) = C(F x) / F`` of strike ``K = F x`` and
dividend-forward ``F`` is the Black-Scholes call divided by ``F``. The
corresponding closed form depends only on ``x`` and the total variance
``w = sigma^2 T`` (undiscounted). All formulas below use the log-moneyness
``k = log x`` and the total implied volatility ``s = sqrt(w)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf
from scipy.optimize import brentq


_SQRT_2 = np.sqrt(2.0)
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _ncdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(z / _SQRT_2))


def _npdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def normalised_call(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalised undiscounted call ``c(x) = N(d1) - x N(d2)`` at total
    variance ``w = sigma^2 T``, where ``d1 = -log(x)/s + s/2`` and
    ``d2 = d1 - s``. Both inputs are broadcast.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.sqrt(np.maximum(w, 1e-300))
    k = np.log(np.maximum(x, 1e-300))
    d1 = -k / s + 0.5 * s
    d2 = d1 - s
    # Intrinsic for w = 0: max(1 - x, 0).
    c = _ncdf(d1) - x * _ncdf(d2)
    tiny = w <= 0.0
    if np.any(tiny):
        c = np.where(tiny, np.maximum(1.0 - x, 0.0), c)
    return c


def normalised_vega(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``dc/ds = phi(d1)`` with ``s = sqrt(w)``; derivative of the normalised
    call with respect to total volatility (not implied vol)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    s = np.sqrt(np.maximum(w, 1e-300))
    k = np.log(np.maximum(x, 1e-300))
    d1 = -k / s + 0.5 * s
    return _npdf(d1)


def bs_vega(x: np.ndarray, sigma: np.ndarray, T: float) -> np.ndarray:
    """Normalised Black-Scholes vega per unit implied vol: ``dc/dsigma =
    sqrt(T) phi(d1)``. Equivalent to ``V / F`` of a call of forward ``F``."""
    sigma = np.asarray(sigma, dtype=float)
    w = sigma * sigma * T
    return np.sqrt(T) * normalised_vega(x, w)


def implied_total_variance(
    c: float | np.ndarray, x: float | np.ndarray,
    w_lo: float = 1e-10, w_hi: float = 25.0,
) -> np.ndarray:
    """Invert ``c = c_BS(x, w)`` for total variance ``w`` using Brent.
    Vectorised over broadcastable inputs.
    """
    c_arr = np.atleast_1d(np.asarray(c, dtype=float))
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    c_arr, x_arr = np.broadcast_arrays(c_arr, x_arr)
    out = np.empty(c_arr.shape, dtype=float)
    for idx in np.ndindex(c_arr.shape):
        ci, xi = float(c_arr[idx]), float(x_arr[idx])
        intrinsic = max(1.0 - xi, 0.0)
        if not np.isfinite(ci) or ci <= intrinsic + 1e-15 or ci >= 1.0 - 1e-15:
            out[idx] = np.nan
            continue
        try:
            out[idx] = brentq(
                lambda w: float(normalised_call(np.array([xi]), np.array([w]))[0] - ci),
                w_lo, w_hi, xtol=1e-12, rtol=1e-10,
            )
        except (ValueError, RuntimeError):
            out[idx] = np.nan
    return out.reshape(np.asarray(c).shape) if np.asarray(c).ndim else float(out)


def implied_vol(
    c: float | np.ndarray, x: float | np.ndarray, T: float | np.ndarray,
) -> np.ndarray:
    """Implied volatility ``sigma`` such that ``c = c_BS(x, sigma^2 T)``."""
    w = implied_total_variance(c, x)
    T_arr = np.asarray(T, dtype=float)
    return np.sqrt(w / T_arr)
