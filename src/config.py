"""Experiment parameters for the Convex Price Interpolation (CPI) replication.

Each constant carries a brief source note: the paper section/equation that fixes
it, or "not specified in the paper" when an operational choice had to be made.
"""

# --- Snapshot (paper §8, Data paragraph) -------------------------------------

SNAPSHOT_DATE_ET: str = "2026-11-02"  # paper §8: Cboe snapshot 2 Nov 2026 17:05 ET
SNAPSHOT_TIME_ET: str = "17:05"       # paper §8
AM_SETTLE_TIME_ET: str = "09:30"      # paper §8: AM-settled SPX expiry time
PM_SETTLE_TIME_ET: str = "16:00"      # paper §8: PM-settled SPXW expiry time

# --- Quote filtering (paper §8) ----------------------------------------------

VEGA_FLOOR_FRACTION: float = 0.001    # paper §8: keep vega >= 0.1% of slice max
MIN_QUOTES_STRADDLING_FWD: int = 20   # paper §8: at least twenty quotes
MAX_MATURITY_YEARS: float = 2.12      # paper §8: final sample up to 2.12 years
WEIGHT_CLIP_QUANTILE: float = 0.95    # paper §7: "clipped at a high quantile";
                                       # specific quantile not specified in the paper

# --- B-spline body (paper §3, §8) --------------------------------------------

BODY_DEGREE_CUBIC: int = 3            # paper §3, §8: cubic body for main surface
BODY_DEGREE_QUARTIC: int = 4          # paper §8: quartic compared for single slice
INTERIOR_KNOTS: int = 25              # paper §8: twenty-five interior knots per slice
KNOT_POWER_Q: float = 0.5             # paper §3: uniform in sgn(k)|k|^q, q<1;
                                       # specific value not specified in the paper
CONVEXITY_FLOOR_EPSILON: float = 1e-8 # paper §4: D2 d >= epsilon / s_n

# --- Wing dictionary (paper §5.2, §8) ----------------------------------------

N_ATOMS_PUT: int = 6                  # paper §8: six put-side atoms
N_ATOMS_CALL: int = 8                 # paper §8: eight call-side atoms
PUT_B_LO: float = 0.02                # paper §8: put Lee-slope range lower end
PUT_B_HI: float = 1.6                 # paper §8: baseline put cap
CALL_B_LO: float = 0.005              # paper §8: call Lee-slope range lower end
CALL_B_HI: float = 0.65               # paper §8: baseline call cap

# Experiment 3 cap sweep (paper §8)
EXP3_CALL_B_HI_SWEEP = (0.55, 0.65, 0.80, 1.00)

# Experiment 4 put cap sweep (paper §8, variance swap sensitivity)
EXP4_PUT_B_HI_SWEEP = (1.0, 1.2, 1.4, 1.6, 1.8)
EXP4_THIN_PUT_B_HI: float = 1.0       # paper §8: thin dictionary for synthetic quotes
EXP4_BASELINE_PUT_B_HI: float = 1.6   # paper §8: baseline cap in augmentation test

# --- Calendar constraints (paper §6, Appendix A) -----------------------------

OVERLAP_SUBDIVISION: int = 4          # paper §6: each merged knot interval into four
BAND_HERMITE_ERROR_FRACTION: float = 1e-3  # paper §6: epsilon_ij <= 1e-3 h_i
WING_TAIL_MARGIN_DELTA_MU: float = 1e-6    # paper eq. (27): small margin, value not specified

# --- Objective weights (paper §7) --------------------------------------------

RHO_SMOOTHNESS: float = 5e-3          # paper §7: typical smoothness weight
ETA_WING_RIDGE: float = 1e-4          # paper §7: typical wing ridge

# --- Verification grid (paper §8) --------------------------------------------

VERIFICATION_GRID_POINTS: int = 40000 # paper §8: 40,000 verification points
VERIFICATION_EXTENSION_LOG_K: float = 1.0  # paper §8: one log-moneyness unit past seams

# --- Deep-tail extrapolation probe (paper §8 exp_3) --------------------------

DEEP_TAIL_LOG_K_PAST_SEAM: float = 8.0  # paper §8: eight log-moneyness units past call seam

# --- Synthetic surface fallback ---------------------------------------------
# Used only when the Cboe snapshot of 2 November 2026 17:05 ET cannot be
# retrieved from any accessible data source. The synthetic surface mimics the
# SPX dataset size (expiries, strike counts) described in paper §8.

SYNTHETIC_SPOT: float = 5700.0        # approximate SPX level in Nov 2026
SYNTHETIC_R: float = 0.045            # approximate risk-free rate
SYNTHETIC_Q: float = 0.013            # approximate SPX dividend yield
SYNTHETIC_HESTON_V0: float = 0.02     # initial variance (~14.1% vol)
SYNTHETIC_HESTON_THETA: float = 0.035
SYNTHETIC_HESTON_KAPPA: float = 2.5
SYNTHETIC_HESTON_SIGMA: float = 0.9
SYNTHETIC_HESTON_RHO: float = -0.7
SYNTHETIC_BID_ASK_VOL_BPS: float = 14.4  # paper §8: median bid/ask width 14.4 vol bps
