"""Load the SPX / SPXW option snapshot used by the paper.

The paper (§8) pins the analysis to Cboe's delayed quote feed on 2 November
2026 17:05 ET. Cboe exposes only the current fifteen-minute-delayed snapshot
on its public site, which cannot be used to retrieve a past timestamp; no
other research-grade market-data service makes that exact snapshot available
for the general public either. When the snapshot is not in the local cache
(``data/cache/spx_snapshot_2026-11-02.parquet``), this loader falls back to a
synthetic SSVI surface of the same shape, documented as a deviation in
``results/RESULTS.md``.

If the user later provides the snapshot CSV as ``data/cache/spx_raw.csv`` in
a Cboe-style schema with columns ``expiration``, ``settle`` (``AM``/``PM``),
``class`` (``SPX``/``SPXW``), ``strike``, ``option_type`` (``C``/``P``),
``bid``, ``ask``, this loader consumes it and performs the paper's
preprocessing: estimate the forward per expiry from the OTM strip via
put-call parity, fit a single zero curve across expiries, keep OTM quotes
with Black-Scholes vega at least 0.1% of the slice maximum, require at least
twenty quotes straddling the forward, and keep maturities up to 2.12 years.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.bs import bs_vega, normalised_call
from src.config import (
    MAX_MATURITY_YEARS, MIN_QUOTES_STRADDLING_FWD, VEGA_FLOOR_FRACTION,
)
from src.cpi import SliceQuotes


_CACHE = Path(__file__).resolve().parent / "cache"


def _snapshot_csv() -> Path:
    return _CACHE / "spx_raw.csv"


def snapshot_available() -> bool:
    """True when a Cboe-style quote file is present in the cache."""
    return _snapshot_csv().exists()


def _years_act365(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    return float((t1 - t0).total_seconds() / (365.0 * 86400.0))


def _parity_forward(
    strikes: np.ndarray, call_mid: np.ndarray, put_mid: np.ndarray, df: float,
) -> float:
    """Estimate forward F such that discounted parity ``C - P = df (F - K)``
    best fits the minimum-|C-P| quote region, as in paper §8 (short OTM
    strip around the money)."""
    diff = call_mid - put_mid
    # Index of the strike nearest the money (smallest |C-P|).
    closest = np.argsort(np.abs(diff))[: min(5, strikes.size)]
    F = float(np.mean(strikes[closest] + diff[closest] / df))
    return F


def load_snapshot(
    mode: str = "default", *, include_variance_swaps: bool = False,
) -> list[SliceQuotes]:
    """Load the SPX snapshot and return a list of :class:`SliceQuotes`.

    Parameters
    ----------
    mode : {"default", "thin_put"}
        ``thin_put`` generates a surface with a thin put dictionary used by
        experiment 4 to synthesise variance-swap quotes.
    """
    if snapshot_available():
        return _load_cboe_snapshot()
    from .generate_synthetic import generate_surface
    return generate_surface(mode=mode)


def _load_cboe_snapshot() -> list[SliceQuotes]:
    """Parse a Cboe-style CSV from the cache into :class:`SliceQuotes`."""
    df = pd.read_csv(_snapshot_csv(), parse_dates=["expiration"])
    snapshot = pd.Timestamp("2026-11-02 17:05")
    slices: list[SliceQuotes] = []
    # Flat zero curve over the snapshot: with df = exp(-r T) and the same r
    # across expiries, parity still gives a per-expiry forward. The paper
    # describes a joint curve fit; the flat fallback is noted in RESULTS.md.
    r = 0.045
    for (exp_date, settle, cls), group in df.groupby(["expiration", "settle", "class"]):
        settle_time = "09:30" if settle == "AM" else "16:00"
        expire = pd.Timestamp(exp_date.strftime("%Y-%m-%d") + " " + settle_time)
        T = _years_act365(snapshot, expire)
        if T <= 0.0 or T > MAX_MATURITY_YEARS:
            continue
        pivot = (
            group.pivot_table(
                index="strike", columns="option_type", values=["bid", "ask"],
            )
            .dropna(how="any")
        )
        if pivot.empty:
            continue
        K = pivot.index.to_numpy(float)
        c_bid_px = pivot[("bid", "C")].to_numpy(float)
        c_ask_px = pivot[("ask", "C")].to_numpy(float)
        p_bid_px = pivot[("bid", "P")].to_numpy(float)
        p_ask_px = pivot[("ask", "P")].to_numpy(float)
        df_factor = float(np.exp(-r * T))
        F = _parity_forward(K, 0.5 * (c_bid_px + c_ask_px),
                            0.5 * (p_bid_px + p_ask_px), df_factor)
        x = K / F
        # OTM selection: puts for x < 1, calls for x > 1.
        otm_call = x > 1.0
        otm_put = ~otm_call
        c_bid_norm = np.where(otm_call, c_bid_px / F, (p_bid_px - df_factor * (K - F)) / F)
        c_ask_norm = np.where(otm_call, c_ask_px / F, (p_ask_px - df_factor * (K - F)) / F)
        c_bid_norm = np.clip(c_bid_norm, 1e-10, None)
        c_ask_norm = np.clip(c_ask_norm, 1e-10, None)
        c_mid = 0.5 * (c_bid_norm + c_ask_norm)
        from src.bs import implied_vol
        sig_mid = implied_vol(c_mid, x, T)
        sig_bid = implied_vol(c_bid_norm, x, T)
        sig_ask = implied_vol(c_ask_norm, x, T)
        ok = np.isfinite(sig_mid) & np.isfinite(sig_bid) & np.isfinite(sig_ask)
        vega = bs_vega(x[ok], sig_mid[ok], T)
        vega_floor = VEGA_FLOOR_FRACTION * float(vega.max() if vega.size else 0.0)
        keep = vega >= vega_floor
        idx = np.where(ok)[0][keep]
        if idx.size < MIN_QUOTES_STRADDLING_FWD:
            continue
        xs = x[idx]
        if np.sum(xs < 1.0) * np.sum(xs > 1.0) == 0:
            continue
        slices.append(SliceQuotes(
            T=float(T), F=float(F), x=xs,
            c_bid=c_bid_norm[idx], c_ask=c_ask_norm[idx],
            vega=vega[keep], sigma_bid=sig_bid[idx], sigma_ask=sig_ask[idx],
            label=f"{cls}_{settle}_{exp_date.date().isoformat()}",
        ))
    slices.sort(key=lambda s: s.T)
    return slices
