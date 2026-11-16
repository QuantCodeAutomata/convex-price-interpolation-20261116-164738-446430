"""Experiment 2: cubic vs quartic B-spline body in CPI (paper §8).

Fits the same single SPX maturity (paper: PM-settled 30 April 2027, T=0.49,
162 body quotes) with p=3 and p=4 bodies, holds wing construction fixed, and
reports fit metrics, cross-degree implied-vol differences, and density
qualitative check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.bs import implied_vol
from src.config import (
    CALL_B_HI, CALL_B_LO, CONVEXITY_FLOOR_EPSILON, ETA_WING_RIDGE,
    INTERIOR_KNOTS, KNOT_POWER_Q, N_ATOMS_CALL, N_ATOMS_PUT, PUT_B_HI,
    PUT_B_LO, RHO_SMOOTHNESS, WEIGHT_CLIP_QUANTILE,
    WING_TAIL_MARGIN_DELTA_MU,
)
from src.cpi import CPIConfig, calibrate_surface, evaluate_body, evaluate_slice
from src.metrics import slice_fit_stats
from data.data_loader import load_snapshot, snapshot_available


RESULTS = Path(__file__).resolve().parent.parent / "results"
TARGET_T_YEARS = 0.49  # paper §8


def _closest_slice(slices, target_T: float):
    return min(slices, key=lambda s: abs(s.T - target_T))


def _config(degree: int) -> CPIConfig:
    return CPIConfig(
        body_degree=degree,
        n_interior_knots=INTERIOR_KNOTS,
        knot_q=KNOT_POWER_Q,
        put_b_lo=PUT_B_LO, put_b_hi=PUT_B_HI, put_n_atoms=N_ATOMS_PUT,
        call_b_lo=CALL_B_LO, call_b_hi=CALL_B_HI, call_n_atoms=N_ATOMS_CALL,
        rho=RHO_SMOOTHNESS, eta=ETA_WING_RIDGE,
        convexity_floor=CONVEXITY_FLOOR_EPSILON,
        weight_clip_quantile=WEIGHT_CLIP_QUANTILE,
        delta_mu=WING_TAIL_MARGIN_DELTA_MU,
    )


def _fit_single(sl, degree: int):
    sol = calibrate_surface([sl], _config(degree))
    return sol.slices[0]


def _density(sv, x: np.ndarray) -> np.ndarray:
    """Second derivative of the body spline, i.e. the risk-neutral density up
    to the forward factor."""
    Bpp = sv.pieces.basis.basis_matrix(x, nu=2)
    return Bpp @ sv.d.value


def run(out_dir: Path = RESULTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = load_snapshot(mode="default")
    sl = _closest_slice(slices, TARGET_T_YEARS)
    sv_cubic = _fit_single(sl, 3)
    sv_quartic = _fit_single(sl, 4)
    stats_cubic = slice_fit_stats(sv_cubic)
    stats_quartic = slice_fit_stats(sv_quartic)
    # Dense evaluation grid over the body range.
    x_l, x_h = sl.x.min(), sl.x.max()
    x_grid = np.linspace(x_l, x_h, 2000)
    c_cubic = evaluate_body(sv_cubic, x_grid)
    c_quartic = evaluate_body(sv_quartic, x_grid)
    iv_c = implied_vol(c_cubic, x_grid, sl.T)
    iv_q = implied_vol(c_quartic, x_grid, sl.T)
    mask = np.isfinite(iv_c) & np.isfinite(iv_q)
    rmse_cross_bps = float(np.sqrt(np.mean((iv_c[mask] - iv_q[mask]) ** 2)) * 1e4)
    max_cross_bps = float(np.max(np.abs(iv_c[mask] - iv_q[mask])) * 1e4)
    # Densities on the same grid
    rho_cubic = _density(sv_cubic, x_grid)
    rho_quartic = _density(sv_quartic, x_grid)
    peak_cubic = float(np.max(rho_cubic))
    peak_quartic = float(np.max(rho_quartic))
    density_disagreement = float(
        np.max(np.abs(rho_cubic - rho_quartic)) / max(peak_cubic, 1e-12)
    )
    out = {
        "data_source_used": "Cboe snapshot" if snapshot_available() else "synthetic SSVI surface",
        "slice_label": sl.label,
        "T_years": sl.T,
        "n_body_quotes": int(sl.x.size),
        "cubic": {
            "frac_in_spread": stats_cubic.frac_in_spread,
            "rmse_vs_mid_bps": stats_cubic.rmse_bps_vs_mid,
            "n_in_spread": stats_cubic.n_in_spread,
        },
        "quartic": {
            "frac_in_spread": stats_quartic.frac_in_spread,
            "rmse_vs_mid_bps": stats_quartic.rmse_bps_vs_mid,
            "n_in_spread": stats_quartic.n_in_spread,
        },
        "cross_degree_rmse_bps": rmse_cross_bps,
        "cross_degree_max_bps": max_cross_bps,
        "density_peak_cubic": peak_cubic,
        "density_peak_quartic": peak_quartic,
        "density_relative_disagreement_fraction_of_peak": density_disagreement,
    }
    (out_dir / "exp2_spline_order.json").write_text(json.dumps(out, indent=2))
    _save_plot(out_dir, x_grid, iv_c, iv_q, rho_cubic, rho_quartic)
    return out


def _save_plot(out_dir, x, iv_c, iv_q, rho_c, rho_q) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(np.log(x), iv_c * 100, label="cubic p=3")
    ax1.plot(np.log(x), iv_q * 100, label="quartic p=4", linestyle="--")
    ax1.set_xlabel("log-moneyness k"); ax1.set_ylabel("implied vol (%)")
    ax1.set_title("Implied vol: cubic vs quartic")
    ax1.legend()
    ax2.plot(np.log(x), rho_c, label="cubic p=3")
    ax2.plot(np.log(x), rho_q, label="quartic p=4", linestyle="--")
    ax2.set_xlabel("log-moneyness k"); ax2.set_ylabel("density c''(x)")
    ax2.set_title("Risk-neutral density")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_density.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cubic vs quartic CPI body")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    out = run(out_dir=args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
