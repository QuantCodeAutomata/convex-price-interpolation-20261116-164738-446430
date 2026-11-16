"""Experiment 4: add variance swap quotes to CPI to pin the far put wing
(paper §8 and Appendix B).

Steps reproduced:

1. Baseline CPI fit of the surface. Measure the share of the normalised log
   contract ``L`` contributed by the put wing beyond the deepest quoted put.
2. Sweep the put-side cap ``b_hi`` over {1.0, 1.2, 1.4, 1.6, 1.8} with
   options-only data. Report option fit (expected invariant) and
   variance-swap volatility of the shortest slice (expected to move).
3. Synthesise variance-swap quotes from a thin-put (``b_hi = 1.0``) fit, then
   refit with the baseline dictionary (``b_hi = 1.6``) (a) without those
   quotes and (b) with them as weighted least-squares targets. Compare
   variance-swap discrepancies and option fit between (a) and (b), and the
   drop in the median weight on the fattest put atom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import (
    CALL_B_HI, CALL_B_LO, CONVEXITY_FLOOR_EPSILON, ETA_WING_RIDGE,
    EXP4_BASELINE_PUT_B_HI, EXP4_PUT_B_HI_SWEEP, EXP4_THIN_PUT_B_HI,
    INTERIOR_KNOTS, KNOT_POWER_Q, N_ATOMS_CALL, N_ATOMS_PUT, PUT_B_LO,
    RHO_SMOOTHNESS, WEIGHT_CLIP_QUANTILE, WING_TAIL_MARGIN_DELTA_MU,
)
from src.cpi import (
    CPIConfig, calibrate_surface, evaluate_body, fair_variance,
    variance_swap_coefficients,
)
from src.metrics import surface_fit_summary, verify_calendar
from data.data_loader import load_snapshot, snapshot_available


RESULTS = Path(__file__).resolve().parent.parent / "results"


def _config(put_b_hi: float, n_knots: int, n_put: int) -> CPIConfig:
    return CPIConfig(
        body_degree=3, n_interior_knots=n_knots, knot_q=KNOT_POWER_Q,
        put_b_lo=PUT_B_LO, put_b_hi=put_b_hi, put_n_atoms=n_put,
        call_b_lo=CALL_B_LO, call_b_hi=CALL_B_HI, call_n_atoms=N_ATOMS_CALL,
        rho=RHO_SMOOTHNESS, eta=ETA_WING_RIDGE,
        convexity_floor=CONVEXITY_FLOOR_EPSILON,
        weight_clip_quantile=WEIGHT_CLIP_QUANTILE,
        delta_mu=WING_TAIL_MARGIN_DELTA_MU,
    )


def _vs_volatility(L: float, T: float) -> float:
    """Variance-swap volatility ``sqrt(L / T)``. Paper eq. (23) with ``L``
    the normalised log-contract expectation; the forward factor cancels."""
    return float(np.sqrt(max(L, 0.0) / max(T, 1e-12)))


def _put_wing_contribution_fraction(sv) -> float:
    """Fraction of ``L / 2`` from the put wing beyond the deepest quoted put."""
    A, gL, gR, const = variance_swap_coefficients(sv)
    L_body_const = float(A @ sv.d.value + const)
    L_put_wing = float(gL @ sv.aL.value)
    L_call_wing = float(gR @ sv.aR.value)
    total = L_body_const + L_put_wing + L_call_wing
    return float(L_put_wing / max(abs(total), 1e-12))


def run(reduced: bool = False, out_dir: Path = RESULTS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = load_snapshot(mode="default")
    if reduced:
        slices = slices[:12]
    n_knots = 10 if reduced else INTERIOR_KNOTS
    n_put = 4 if reduced else N_ATOMS_PUT

    # --- Step 1: baseline wing importance ---------------------------------
    cfg_base = _config(EXP4_BASELINE_PUT_B_HI, n_knots, n_put)
    sol_base = calibrate_surface(slices, cfg_base)
    short_slices = [sv for sv in sol_base.slices if sv.quotes.T < 3.0 / 52.0]
    wing_share_all = [_put_wing_contribution_fraction(sv) for sv in sol_base.slices]
    wing_share_short = [_put_wing_contribution_fraction(sv) for sv in short_slices]
    wing_importance = {
        "all_median": float(np.median(wing_share_all)),
        "short_median": (float(np.median(wing_share_short))
                         if wing_share_short else None),
        "shortest": wing_share_all[0] if wing_share_all else None,
    }

    # --- Step 2: put cap sweep --------------------------------------------
    sweep_rows = []
    for b_hi in EXP4_PUT_B_HI_SWEEP:
        sol = calibrate_surface(slices, _config(b_hi, n_knots, n_put))
        summ = surface_fit_summary(sol)
        vs_vols = [_vs_volatility(fair_variance(sv), sv.quotes.T) for sv in sol.slices]
        sweep_rows.append({
            "put_b_hi": b_hi,
            "frac_in_spread": summ["frac_in_spread"],
            "vs_vol_shortest": vs_vols[0] if vs_vols else None,
            "vs_vol_median": float(np.median(vs_vols)) if vs_vols else None,
        })

    # --- Step 3: synthetic quotes from thin dict, refit with baseline -----
    cfg_thin = _config(EXP4_THIN_PUT_B_HI, n_knots, n_put)
    sol_thin = calibrate_surface(slices, cfg_thin)
    synthetic_L = [fair_variance(sv) for sv in sol_thin.slices]
    width = 1e-3  # tight synthetic quote: a modeling choice, not specified in the paper

    cfg_no = _config(EXP4_BASELINE_PUT_B_HI, n_knots, n_put)
    sol_no = calibrate_surface(slices, cfg_no)
    L_no = [fair_variance(sv) for sv in sol_no.slices]
    cal_no = verify_calendar(sol_no, n_points=4000 if reduced else 20000, pad_log=1.0)
    summ_no = surface_fit_summary(sol_no)

    vs_quotes = [(i, L, width) for i, L in enumerate(synthetic_L)]
    sol_yes = calibrate_surface(slices, cfg_no, variance_swap_quotes=vs_quotes)
    L_yes = [fair_variance(sv) for sv in sol_yes.slices]
    cal_yes = verify_calendar(sol_yes, n_points=4000 if reduced else 20000, pad_log=1.0)
    summ_yes = surface_fit_summary(sol_yes)

    def _vs_vol_disc_bps(L_fit, L_mkt, T):
        vfit, vmkt = _vs_volatility(L_fit, T), _vs_volatility(L_mkt, T)
        return (vfit - vmkt) * 1e4
    disc_no = np.array([
        _vs_vol_disc_bps(l, lm, sol_no.slices[i].quotes.T)
        for i, (l, lm) in enumerate(zip(L_no, synthetic_L))
    ])
    disc_yes = np.array([
        _vs_vol_disc_bps(l, lm, sol_yes.slices[i].quotes.T)
        for i, (l, lm) in enumerate(zip(L_yes, synthetic_L))
    ])

    fattest_no = np.array([float(sv.aL.value[-1]) for sv in sol_no.slices])
    fattest_yes = np.array([float(sv.aL.value[-1]) for sv in sol_yes.slices])

    out = {
        "data_source_used": "Cboe snapshot" if snapshot_available() else "synthetic SSVI surface",
        "n_slices_used": len(slices),
        "wing_importance_fraction": wing_importance,
        "put_cap_sweep": sweep_rows,
        "augmentation": {
            "no_quotes": {
                "frac_in_spread": summ_no["frac_in_spread"],
                "median_dist_mid_bps": summ_no["median_dist_from_mid_bps"],
                "calendar_violations": cal_no["violations"],
                "median_abs_vs_vol_disc_bps": float(np.median(np.abs(disc_no))),
                "max_abs_vs_vol_disc_bps": float(np.max(np.abs(disc_no))),
                "median_fattest_put_weight": float(np.median(fattest_no)),
            },
            "with_quotes": {
                "frac_in_spread": summ_yes["frac_in_spread"],
                "median_dist_mid_bps": summ_yes["median_dist_from_mid_bps"],
                "calendar_violations": cal_yes["violations"],
                "median_abs_vs_vol_disc_bps": float(np.median(np.abs(disc_yes))),
                "max_abs_vs_vol_disc_bps": float(np.max(np.abs(disc_yes))),
                "median_fattest_put_weight": float(np.median(fattest_yes)),
            },
        },
        "reduced": reduced,
    }
    (out_dir / "exp4_variance_swap.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance-swap-augmented CPI")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    out = run(reduced=args.reduced, out_dir=args.out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
