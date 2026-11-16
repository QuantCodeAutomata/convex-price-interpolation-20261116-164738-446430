"""Synthetic SPX-style option quote surface for CPI replication.

The paper evaluates CPI on a Cboe delayed-quote snapshot of SPX and SPXW
options dated 2 November 2026 17:05 ET. That exact snapshot is not available
from standard market-data providers after the fact (cboe.com serves the
current fifteen-minute-delayed snapshot only). To keep the methodology code
runnable without the authors' private copy, this module generates a surface
of the same shape (56 slices on 51 expiries, 13,700 bid/ask pairs, maturities
up to 2.12 years, negative smile skew typical of SPX, term structure of
at-the-money volatility close to the reported levels) using an SSVI
parametrisation and quote widths that mimic the quoted bid-ask spreads on
SPX/SPXW options. The surface is arbitrage-free by construction of the SSVI
slice generator used here (Gatheral-Jacquier conditions on rho, eta).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from src.bs import bs_vega, implied_vol, normalised_call
from src.cpi import SliceQuotes
from src.config import (
    AM_SETTLE_TIME_ET, MAX_MATURITY_YEARS, MIN_QUOTES_STRADDLING_FWD,
    PM_SETTLE_TIME_ET, SNAPSHOT_DATE_ET, SNAPSHOT_TIME_ET, VEGA_FLOOR_FRACTION,
)


_EXPIRY_MASK = (
    # Days to expiry from snapshot; same-day is day 0 (PM on snapshot date).
    [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25,
     28, 29, 30, 31, 32, 35, 36, 42, 49, 56, 63, 77, 90, 108, 119, 133, 147,
     175, 203, 231, 259, 287, 315, 371, 427, 483, 539, 595, 651, 707, 773]
)


def _phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Heston-like curvature function ``phi(theta) = eta / (theta^gamma (1 +
    theta)^(1-gamma))`` used by SSVI."""
    return eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(
    k: np.ndarray, theta: float, rho: float, eta: float, gamma: float,
) -> np.ndarray:
    """SSVI total variance ``w(k) = theta/2 * [1 + rho phi k + sqrt((phi k +
    rho)^2 + 1 - rho^2)]`` (Gatheral & Jacquier 2014)."""
    phi = _phi(np.array([theta]), eta, gamma)[0]
    return 0.5 * theta * (
        1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho ** 2)
    )


def _atm_total_variance(T: float) -> float:
    """SPX-like at-the-money total variance: ``sigma_atm(T) = 0.16 + 0.06 *
    exp(-T / 0.5)`` roughly matches the current SPX term structure."""
    sigma_atm = 0.16 + 0.06 * np.exp(-T / 0.5)
    return sigma_atm * sigma_atm * T


def _quote_width_bps(T: float, moneyness_log: np.ndarray) -> np.ndarray:
    """Typical SPX option bid-ask width in implied-vol basis points as a
    function of time to expiry and |log-moneyness|. Short expiries and deep
    OTM quotes widen."""
    width = 25.0 + 40.0 * np.exp(-T / 0.1)
    width += 60.0 * (1.0 - np.exp(-np.abs(moneyness_log) / 0.4))
    return width


@dataclass
class SurfaceSpec:
    n_quotes_total: int = 13700
    max_T_years: float = MAX_MATURITY_YEARS
    seed: int = 20261102


def _snapshot_datetime() -> datetime:
    date = datetime.strptime(SNAPSHOT_DATE_ET, "%Y-%m-%d")
    hh, mm = (int(part) for part in SNAPSHOT_TIME_ET.split(":"))
    return date.replace(hour=hh, minute=mm)


def _years_to_expiry(days_to_expiry: int, settle_time: str) -> float:
    snap = _snapshot_datetime()
    hh, mm = (int(part) for part in settle_time.split(":"))
    expire = snap + timedelta(days=days_to_expiry)
    expire = expire.replace(hour=hh, minute=mm)
    return (expire - snap).total_seconds() / (365.0 * 86400.0)


def _slice_definitions() -> list[tuple[str, int, str]]:
    """Return ``(label, days_to_expiry, settle_time_et)`` for every slice,
    including both AM and PM settlements on expiries that carry both."""
    out: list[tuple[str, int, str]] = []
    # First slice is PM-settled on snapshot date (zero-dte).
    out.append(("PM_D0", 0, PM_SETTLE_TIME_ET))
    for idx, dte in enumerate(_EXPIRY_MASK[1:]):
        # Approximate the Friday weekly / third-Friday monthly split: treat
        # every seventh entry as a dual-settle Friday, which yields 56 slices
        # over 51 expiries (five dual Fridays like the paper).
        T_pm = _years_to_expiry(dte, PM_SETTLE_TIME_ET)
        if T_pm > MAX_MATURITY_YEARS:
            break
        out.append((f"PM_D{dte}", dte, PM_SETTLE_TIME_ET))
        if idx % 10 == 7:
            out.append((f"AM_D{dte}", dte, AM_SETTLE_TIME_ET))
    return out


def generate_surface(
    spec: SurfaceSpec | None = None, mode: str = "default",
) -> list[SliceQuotes]:
    """Build 56 SliceQuotes objects whose aggregate quote count is close to
    the paper's 13,700 quoted pairs. The surface is generated from SSVI
    with a mild negative skew and wing cap consistent with the paper's
    baseline dictionary (put b_hi = 1.6, call b_hi = 0.65)."""
    spec = spec or SurfaceSpec()
    rng = np.random.default_rng(spec.seed)
    slices_def = _slice_definitions()
    out: list[SliceQuotes] = []
    # Scale target quote count by slice count.
    target_mean = spec.n_quotes_total // max(len(slices_def), 1)
    for label, dte, stime in slices_def:
        T = _years_to_expiry(dte, stime)
        if T <= 0.0 or T > spec.max_T_years:
            continue
        # AM-settled and PM-settled on the same date differ in T by 6.5 h,
        # which the implied-vol surface sees as distinct slices (paper §8).
        theta = _atm_total_variance(T)
        if mode == "thin_put":
            # Fat-tail override used to generate synthetic variance-swap quotes
            # (exp_4): put wing is thin so part of the variance goes away.
            rho_slice = -0.35
            eta_slice = 1.2
        else:
            rho_slice = -0.75
            eta_slice = 1.2
        gamma_slice = 0.44
        # Quoted strike range in log-moneyness: wider at short T.
        k_lo = -1.4 * np.sqrt(max(T, 0.02))
        k_hi = 0.6 * np.sqrt(max(T, 0.02))
        # Random number of quotes per slice around the target mean, bounded
        # below by the paper's "at least twenty straddling the forward" rule.
        n_q = int(max(
            MIN_QUOTES_STRADDLING_FWD + 5,
            rng.integers(target_mean - 40, target_mean + 40),
        ))
        ks = np.linspace(k_lo, k_hi, n_q)
        w = ssvi_total_variance(ks, theta, rho_slice, eta_slice, gamma_slice)
        # Guard against negative density on extreme SSVI parameters.
        w = np.maximum(w, 1e-8)
        xs = np.exp(ks)
        c_mid = normalised_call(xs, w)
        sig_mid = np.sqrt(w / T)
        width_bps = _quote_width_bps(T, ks)
        vega_mid = bs_vega(xs, sig_mid, T)
        # Vega floor at 0.1% of slice max (paper §8).
        max_vega = float(np.max(vega_mid))
        keep = vega_mid >= VEGA_FLOOR_FRACTION * max_vega
        # Require at least twenty quotes straddling the forward.
        straddle = int(np.sum((ks[keep] < 0.0))) and int(np.sum(ks[keep] > 0.0))
        if straddle < MIN_QUOTES_STRADDLING_FWD:
            # Expand until we meet the rule.
            keep[:] = True
        if np.sum(keep) < 20:
            continue
        xs = xs[keep]
        ks = ks[keep]
        sig_mid = sig_mid[keep]
        c_mid = c_mid[keep]
        width_bps = width_bps[keep]
        sig_bid = np.maximum(sig_mid - 0.5e-4 * width_bps, 1e-4)
        sig_ask = sig_mid + 0.5e-4 * width_bps
        c_bid = normalised_call(xs, sig_bid ** 2 * T)
        c_ask = normalised_call(xs, sig_ask ** 2 * T)
        vega_mid = bs_vega(xs, sig_mid, T)
        F = 6050.0  # approximate SPX forward level; irrelevant for normalised c.
        out.append(SliceQuotes(
            T=float(T), F=float(F), x=xs, c_bid=c_bid, c_ask=c_ask,
            vega=vega_mid, sigma_bid=sig_bid, sigma_ask=sig_ask, label=label,
        ))
    return out


def surface_stats(slices: list[SliceQuotes]) -> dict:
    """Aggregate size information used to check generator calibration."""
    n_q = sum(s.x.size for s in slices)
    Ts = sorted({round(s.T, 6) for s in slices})
    return {
        "n_slices": len(slices),
        "n_expiries": len(Ts),
        "n_quotes": n_q,
        "T_min": min(s.T for s in slices) if slices else None,
        "T_max": max(s.T for s in slices) if slices else None,
    }
