"""Experiment 3: sensitivity of CPI deep-tail extrapolation to the call-side
Lee-slope cap ``b_hi`` (paper §8).

Refits the full 56-slice surface for each call cap in {0.55, 0.65, 0.80, 1.00}
while keeping every other option identical. Reports (a) global fit statistics
and (b) the deepest call price fit at eight log-moneyness units past the call
seam of the shortest maturity, as in the paper's descriptive table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    CALL_B_LO, CONVEXITY_FLOOR_EPSILON, DEEP_TAIL_LOG_K_PAST_SEAM,
    ETA_WING_RIDGE, EXP3_CALL_B_HI_SWEEP, INTERIOR_KNOTS, KNOT_POWER_Q,
    N_ATOMS_CALL, N_ATOMS_PUT, PUT_B_HI, PUT_B_LO, RHO_SMOOTHNESS,
    WEIGHT_CLIP_QUANTILE, WING_TAIL_MARGIN_DELTA_MU,
)
from src.cpi import CPIConfig, calibrate_surface, evaluate_slice
from src.metrics import surface_fit_summary, verify_calendar
from data.data_loader import load_snapshot, snapshot_available


RESULTS = Path(__file__).resolve().parent.parent / "results"


def _config(call_b_hi: float, n_knots: int, n_call_atoms: int) -> CPIConfig:
    return CPIConfig(
        body_degree=3, n_interior_knots=n_knots, knot_q=KNOT_POWER_Q,
        put_b_lo=PUT_B_LO, put_b_hi=PUT_B_HI, put_n_atoms=N_ATOMS_PUT,
        call_b_lo=CALL_B_LO, call_b_hi=call_b_hi, call_n_atoms=n_call_atoms,
        rho=RHO_SMOOTHNESS, eta=ETA_WING_RIDGE,
        convexity_floor=CONVEXITY_FLOOR_EPSILON,
        weight_clip_quantile=WEIGHT_CLIP_QUANTILE,
        delta_mu=WING_TAIL_MARGIN_DELTA_MU,
    )


def _deep_tail_price(sol, log_k_past: float) -> float:
    """Largest fitted call price at ``log_k_past`` log-moneyness units past
    the call seam, maximised over slices."""
    worst = 0.0
    for sv in sol.slices:
        x = float(np.exp(sv.k_h + log_k_past))
        c = float(evaluate_slice(sv, np.array([x]))[0])
        worst = max(worst, c)
    return worst


def run(reduced: bool = False, out_dir: Path = RESULTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = load_snapshot(mode="default")
    if reduced:
        slices = slices[:16]
    n_knots = 10 if reduced else INTERIOR_KNOTS
    n_atoms = 4 if reduced else N_ATOMS_CALL
    sweep = EXP3_CALL_B_HI_SWEEP
    rows = []
    for b_hi in sweep:
        sol = calibrate_surface(slices, _config(b_hi, n_knots, n_atoms))
        summ = surface_fit_summary(sol)
        cal = verify_calendar(sol, n_points=4000 if reduced else 20000, pad_log=1.0)
        deep = _deep_tail_price(sol, DEEP_TAIL_LOG_K_PAST_SEAM)
        rows.append({
            "call_b_hi": b_hi,
            "frac_in_spread": summ["frac_in_spread"],
            "median_outside_bps": summ["median_outside_bps"],
            "median_dist_from_mid_bps": summ["median_dist_from_mid_bps"],
            "runtime_s": summ["runtime_s"],
            "calendar_violations": cal["violations"],
            "max_fitted_call_8k_past_seam": deep,
        })
    out = {
        "data_source_used": "Cboe snapshot" if snapshot_available() else "synthetic SSVI surface",
        "n_slices_used": len(slices),
        "log_k_past_seam": DEEP_TAIL_LOG_K_PAST_SEAM,
        "sweep": rows,
        "reduced": reduced,
    }
    (out_dir / "exp3_wing_cap.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CPI call-side wing cap sweep")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    out = run(reduced=args.reduced, out_dir=args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
