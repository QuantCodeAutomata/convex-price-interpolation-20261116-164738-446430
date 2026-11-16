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
    """Statistics of the fit for one slice against its own bid/ask quotes."""
    q = sv.quotes
    c_fit = evaluate_body(sv, q.x)
    sig_fit = _vols_from_prices(c_fit, q.x, q.T)
    sig_bid, sig_ask = q.sigma_bid, q.sigma_ask
    inside = (sig_fit >= sig_bid - 1e-12) & (sig_fit <= sig_ask + 1e-12)
    n_in = int(np.sum(np.isfinite(sig_fit) & inside))
    n_tot = int(np.sum(np.isfinite(sig_fit)))
    sig_mid = 0.5 * (sig_bid + sig_ask)
    err = sig_fit - sig_mid
    err_ok = err[np.isfinite(err)]
    rmse_bps = float(np.sqrt(np.mean(err_ok ** 2)) * 1e4) if err_ok.size else np.nan
    outside = ~inside & np.isfinite(sig_fit)
    if np.any(outside):
        d_out = np.where(
            sig_fit[outside] > sig_ask[outside],
            sig_fit[outside] - sig_ask[outside],
            sig_bid[outside] - sig_fit[outside],
        )
        median_outside_bps = float(np.median(d_out) * 1e4)
    else:
        median_outside_bps = 0.0
    median_dist_from_mid_bps = (
        float(np.median(np.abs(err_ok)) * 1e4) if err_ok.size else np.nan
    )
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
        inside = (sig_fit >= q.sigma_bid - 1e-12) & (sig_fit <= q.sigma_ask + 1e-12)
        outside = ~inside & ok
        if np.any(outside):
            d = np.where(
                sig_fit[outside] > q.sigma_ask[outside],
                sig_fit[outside] - q.sigma_ask[outside],
                q.sigma_bid[outside] - sig_fit[outside],
            )
            out_bps.extend((d * 1e4).tolist())
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
    n_viol = int(np.sum(diffs < -1e-10))
    worst = float(diffs.min()) if diffs.size else 0.0
    return {"violations": n_viol, "max_gap": worst,
            "n_pairs": len(sol.slices) - 1, "n_points": n_points}
