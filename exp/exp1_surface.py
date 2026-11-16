"""Experiment 1: single convex QP calibration of the full SPX option price
surface with CPI (paper §7, §8).

Running ``python -m exp.exp1_surface`` fits the whole 56-slice SPX surface in
one joint QP and writes fit statistics, runtime, and calendar-verification
results to ``results/``.

The full paper configuration (56 slices, 25 interior knots per slice, cubic
bodies, baseline wing caps) is the default. A reduced configuration driven
from the environment is used by the test suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.config import (
    BODY_DEGREE_CUBIC, CALL_B_HI, CALL_B_LO, CALL_B_LO as _CB,
    CONVEXITY_FLOOR_EPSILON, ETA_WING_RIDGE, INTERIOR_KNOTS, KNOT_POWER_Q,
    MAX_MATURITY_YEARS, N_ATOMS_CALL, N_ATOMS_PUT, PUT_B_HI, PUT_B_LO,
    RHO_SMOOTHNESS, VERIFICATION_EXTENSION_LOG_K, VERIFICATION_GRID_POINTS,
    WEIGHT_CLIP_QUANTILE, WING_TAIL_MARGIN_DELTA_MU,
)
from src.cpi import CPIConfig, calibrate_surface
from src.metrics import surface_fit_summary, verify_calendar
from data.data_loader import load_snapshot, snapshot_available


RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_config(reduced: bool = False) -> CPIConfig:
    """CPI configuration. ``reduced`` is used by the smoke tests only; the
    full paper configuration is the default."""
    if reduced:
        return CPIConfig(
            body_degree=BODY_DEGREE_CUBIC,
            n_interior_knots=10,
            knot_q=KNOT_POWER_Q,
            put_b_lo=PUT_B_LO, put_b_hi=PUT_B_HI, put_n_atoms=4,
            call_b_lo=CALL_B_LO, call_b_hi=CALL_B_HI, call_n_atoms=4,
            rho=RHO_SMOOTHNESS, eta=ETA_WING_RIDGE,
            convexity_floor=CONVEXITY_FLOOR_EPSILON,
            weight_clip_quantile=WEIGHT_CLIP_QUANTILE,
            delta_mu=WING_TAIL_MARGIN_DELTA_MU,
            sandwich_pts_per_s=10.0,
        )
    return CPIConfig(
        body_degree=BODY_DEGREE_CUBIC,
        n_interior_knots=INTERIOR_KNOTS,
        knot_q=KNOT_POWER_Q,
        put_b_lo=PUT_B_LO, put_b_hi=PUT_B_HI, put_n_atoms=N_ATOMS_PUT,
        call_b_lo=CALL_B_LO, call_b_hi=CALL_B_HI, call_n_atoms=N_ATOMS_CALL,
        rho=RHO_SMOOTHNESS, eta=ETA_WING_RIDGE,
        convexity_floor=CONVEXITY_FLOOR_EPSILON,
        weight_clip_quantile=WEIGHT_CLIP_QUANTILE,
        delta_mu=WING_TAIL_MARGIN_DELTA_MU,
        sandwich_pts_per_s=20.0,
    )


def run(reduced: bool = False, out_dir: Path = RESULTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = load_snapshot(mode="default")
    if reduced:
        # Keep the 24 shortest slices for the test smoke run.
        slices = slices[:24]
    cfg = build_config(reduced=reduced)
    sol = calibrate_surface(slices, cfg)
    summary = surface_fit_summary(sol)
    cal = verify_calendar(
        sol,
        n_points=min(VERIFICATION_GRID_POINTS, 8000 if reduced else VERIFICATION_GRID_POINTS),
        pad_log=VERIFICATION_EXTENSION_LOG_K,
    )
    out = {
        "paper_source": "Cboe SPX/SPXW delayed-quote snapshot, 2 Nov 2026 17:05 ET",
        "data_source_used": "Cboe snapshot" if snapshot_available() else "synthetic SSVI surface",
        "n_slices": len(sol.slices),
        "n_quotes_total": int(summary["n_quotes"]),
        "n_in_spread": int(summary["n_in_spread"]),
        "frac_in_spread": summary["frac_in_spread"],
        "median_outside_bps": summary["median_outside_bps"],
        "median_dist_from_mid_bps": summary["median_dist_from_mid_bps"],
        "runtime_s": summary["runtime_s"],
        "solver_status": summary["solver_status"],
        "calendar": cal,
        "reduced": reduced,
    }
    (out_dir / "exp1_surface_summary.json").write_text(json.dumps(out, indent=2))
    # Per-slice detail
    (out_dir / "exp1_per_slice.json").write_text(
        json.dumps(summary["per_slice"], indent=2)
    )
    _save_plots(sol, out_dir)
    return out


def _save_plots(sol, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    from src.cpi import evaluate_slice
    fig, ax = plt.subplots(figsize=(8, 5))
    k_lo = min(sv.k_l for sv in sol.slices) - 0.5
    k_hi = max(sv.k_h for sv in sol.slices) + 0.5
    ks = np.linspace(k_lo, k_hi, 800)
    xs = np.exp(ks)
    for sv in sol.slices[::max(1, len(sol.slices) // 10)]:
        c = evaluate_slice(sv, xs)
        ax.plot(ks, c, lw=0.8, label=f"T={sv.quotes.T:.2f}")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("normalised call c(x)")
    ax.set_title("CPI fitted call-price slices (sampled)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_slices.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPI SPX surface calibration")
    parser.add_argument("--reduced", action="store_true",
                        help="Smoke-size subset for the test suite")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    reduced = args.reduced or bool(os.environ.get("CPI_REDUCED"))
    out = run(reduced=reduced, out_dir=args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
