"""Fit metrics and calendar verification for a calibrated CPI surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bs import implied_vol, bs_vega, normalised_call
from .cpi import CPISolution, SliceVars, evaluate_body, evaluate_slice


@dataclass
class SliceFitStats:
    label: str
    T: float
    n_quotes: int
    n_in_spread: int
    frac_in_spread: float
    rmse_bps_vs_mid: float
    median_outside_bps: float
    median_dist_from_mid_bps: float


def _vols_from_prices(c: np.ndarray, x: np.ndarray, T: float) -> np.ndarray:
    return implied_vol(c, x, T)


def slice_fit_stats(sv: SliceVars) -> SliceFitStats:
    """Statistics of the fit for one slice against its own bid/ask quotes.

    Paper §8: "99.2% of 13,700 fitted body prices inside bid/ask spreads".
    "Inside the spread" is tested in price space; the outside- and from-mid
    distances are reported in implied-vol basis points.
    """
    q = sv.quotes
    c_fit = evaluate_body(sv, q.x)
    # Price-space in-spread test (paper §8).
    tol_px = 1e-14
    inside_px = (c_fit >= q.c_bid - tol_px) & (c_fit <= q.c_ask + tol_px)
    n_in = int(np.sum(inside_px))
    n_tot = int(c_fit.size)
    sig_fit = _vols_from_prices(c_fit, q.x, q.T)
    sig_mid = 0.5 * (q.sigma_bid + q.sigma_ask)
    err = sig_fit - sig_mid
    err_ok = err[np.isfinite(err)]
    rmse_bps = float(np.sqrt(np.mean(err_ok ** 2)) * 1e4) if err_ok.size else np.nan
    median_dist_from_mid_bps = (
        float(np.median(np.abs(err_ok)) * 1e4) if err_ok.size else np.nan
    )
    outside = ~inside_px & np.isfinite(sig_fit)
    if np.any(outside):
        # Convert the bid/ask edges to implied vols, then measure the distance
        # from the fitted implied vol to the nearer edge (paper §8).
        sig_bid = _vols_from_prices(q.c_bid[outside], q.x[outside], q.T)
        sig_ask = _vols_from_prices(q.c_ask[outside], q.x[outside], q.T)
        s_fit = sig_fit[outside]
        d_out = np.where(
            s_fit > sig_ask,
            s_fit - sig_ask,
            sig_bid - s_fit,
        )
        d_ok = d_out[np.isfinite(d_out)]
        median_outside_bps = float(np.median(d_ok) * 1e4) if d_ok.size else 0.0
    else:
        median_outside_bps = 0.0
    return SliceFitStats(
        label=q.label, T=q.T, n_quotes=n_tot, n_in_spread=n_in,
        frac_in_spread=(n_in / max(n_tot, 1)),
        rmse_bps_vs_mid=rmse_bps,
        median_outside_bps=median_outside_bps,
        median_dist_from_mid_bps=median_dist_from_mid_bps,
    )


def surface_fit_summary(sol: CPISolution) -> dict:
    """Aggregate fit statistics across the whole ladder (paper §8)."""
    per = [slice_fit_stats(sv) for sv in sol.slices]
    n_q = sum(p.n_quotes for p in per)
    n_in = sum(p.n_in_spread for p in per)
    out_bps = []
    dist_mid_bps = []
    for p, sv in zip(per, sol.slices):
        q = sv.quotes
        c_fit = evaluate_body(sv, q.x)
        sig_fit = _vols_from_prices(c_fit, q.x, q.T)
        sig_mid = 0.5 * (q.sigma_bid + q.sigma_ask)
        err = sig_fit - sig_mid
        ok = np.isfinite(err)
        dist_mid_bps.extend((np.abs(err[ok]) * 1e4).tolist())
        tol_px = 1e-14
        inside_px = (c_fit >= q.c_bid - tol_px) & (c_fit <= q.c_ask + tol_px)
        outside = ~inside_px & ok
        if np.any(outside):
            sig_bid = _vols_from_prices(q.c_bid[outside], q.x[outside], q.T)
            sig_ask = _vols_from_prices(q.c_ask[outside], q.x[outside], q.T)
            sf = sig_fit[outside]
            d = np.where(sf > sig_ask, sf - sig_ask, sig_bid - sf)
            d_ok = d[np.isfinite(d)]
            out_bps.extend((d_ok * 1e4).tolist())
    return {
        "n_quotes": n_q,
        "n_in_spread": n_in,
        "frac_in_spread": n_in / max(n_q, 1),
        "n_outside": n_q - n_in,
        "median_outside_bps": float(np.median(out_bps)) if out_bps else 0.0,
        "median_dist_from_mid_bps": (
            float(np.median(dist_mid_bps)) if dist_mid_bps else float("nan")
        ),
        "runtime_s": sol.runtime_s,
        "solver_status": sol.solver_status,
        "per_slice": [p.__dict__ for p in per],
    }


def verify_calendar(
    sol: CPISolution, n_points: int = 40000, pad_log: float = 1.0,
) -> dict:
    """Verify calendar ordering on a dense grid spanning one log-moneyness unit
    past every seam (paper §8)."""
    if len(sol.slices) < 2:
        return {"violations": 0, "max_gap": 0.0}
    k_lo = min(sv.k_l for sv in sol.slices) - pad_log
    k_hi = max(sv.k_h for sv in sol.slices) + pad_log
    ks = np.linspace(k_lo, k_hi, n_points)
    xs = np.exp(ks)
    prices = np.stack([evaluate_slice(sv, xs) for sv in sol.slices], axis=0)
    diffs = prices[1:] - prices[:-1]
    n_viol = int(np.sum(diffs < -1e-7))
    worst = float(diffs.min()) if diffs.size else 0.0
    return {"violations": n_viol, "max_gap": worst,
            "n_pairs": len(sol.slices) - 1, "n_points": n_points}
