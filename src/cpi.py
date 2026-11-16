"""Convex Price Interpolation: single convex QP that calibrates all maturities
jointly in normalised call-price space (paper §7, eq. 21).

The decision variables of each maturity ``n`` are the body B-spline
coefficients ``d^(n)`` and the non-negative wing weights ``a_L^(n), a_R^(n)``.
The program minimises a weighted chi-square to the quoted mid prices,
regularised by a smoothness Gram ``R`` on the body and a ridge on the wing
weights, subject to body no-arbitrage (eq. 7), C^2 body-wing joins (eq. 18)
and calendar ordering (§6 / Appendix A).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import cvxpy as cp

from .body import BodyNCA
from .bspline import ClampedBSpline, place_knots
from .calendar_constraints import (
    SlicePieces,
    sandwich_grid,
    sandwich_rows,
    wing_tail_rows,
)
from .wings import WingSide, body_variance_swap_row


@dataclass
class SliceQuotes:
    """Quotes for one maturity in normalised call-price space.

    Attributes
    ----------
    T : float
        Time to expiry in years, ACT/365 from snapshot to the settlement
        instant of this contract class.
    F : float
        Forward price for this expiry (from put-call parity).
    x : np.ndarray
        Normalised strikes ``K / F`` of retained OTM quotes.
    c_bid, c_ask : np.ndarray
        Normalised bid/ask call prices ``C / F`` after parity for the puts.
    vega : np.ndarray
        Black-Scholes vega of each quote evaluated at the mid implied
        volatility.
    sigma_bid, sigma_ask : np.ndarray
        Implied volatility of the bid and ask quotes.
    """

    T: float
    F: float
    x: np.ndarray
    c_bid: np.ndarray
    c_ask: np.ndarray
    vega: np.ndarray
    sigma_bid: np.ndarray
    sigma_ask: np.ndarray
    label: str = ""

    @property
    def c_mid(self) -> np.ndarray:
        return 0.5 * (self.c_bid + self.c_ask)

    @property
    def sigma_mid(self) -> np.ndarray:
        return 0.5 * (self.sigma_bid + self.sigma_ask)

    @property
    def delta_sigma(self) -> np.ndarray:
        return self.sigma_ask - self.sigma_bid


@dataclass
class CPIConfig:
    """Hyper-parameters of a CPI calibration (paper §7, §8)."""

    body_degree: int = 3
    n_interior_knots: int = 25
    knot_q: float = 0.5
    put_b_lo: float = 0.02
    put_b_hi: float = 1.6
    put_n_atoms: int = 6
    call_b_lo: float = 0.005
    call_b_hi: float = 0.65
    call_n_atoms: int = 8
    rho: float = 5e-3          # body smoothness penalty
    eta: float = 1e-4          # wing-weight ridge
    convexity_floor: float = 1e-8  # epsilon of eq. (7)
    weight_clip_quantile: float = 0.95
    delta_mu: float = 1e-4     # tail-sum margin of eq. (27)
    sandwich_pts_per_s: float = 20.0  # intervals of s_n / 20 in log-moneyness


# -- Per-slice assembly ----------------------------------------------------


@dataclass
class SliceVars:
    """CVXPY variables and pre-computed matrices for one slice."""

    quotes: SliceQuotes
    pieces: SlicePieces
    d: cp.Variable
    aL: cp.Variable
    aR: cp.Variable
    body_nca: BodyNCA
    omega: np.ndarray
    chi2_scale: float
    s_n: float
    k_l: float
    k_h: float
    n_quotes_in_body: int = 0
    variance_coeffs: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = field(
        default=None
    )


def _build_slice(
    q: SliceQuotes, cfg: CPIConfig,
) -> SliceVars:
    """Build body basis, wings, variables and constraints for one slice."""
    x = q.x
    x_l, x_h = float(x.min()), float(x.max())
    interior = place_knots(x_l, x_h, cfg.n_interior_knots, cfg.knot_q)
    basis = ClampedBSpline(
        degree=cfg.body_degree, x_l=x_l, x_h=x_h, interior_knots=interior,
    )
    put = WingSide.make("put", cfg.put_b_lo, cfg.put_b_hi, cfg.put_n_atoms)
    call = WingSide.make("call", cfg.call_b_lo, cfg.call_b_hi, cfg.call_n_atoms)
    k_l = float(np.log(x_l))
    k_h = float(np.log(x_h))
    pieces = SlicePieces(basis=basis, put=put, call=call, k_l=k_l, k_h=k_h)
    # At-the-money total volatility: interpolate mid variance at x = 1 from
    # the two closest quotes straddling the forward.
    s_n = _atm_total_vol(q)
    body_nca = BodyNCA(basis=basis, s_n=s_n, epsilon=cfg.convexity_floor)
    # Weights: omega_i = (V_i delta sigma_i)^(-2) clipped at a high quantile.
    raw = q.vega * q.delta_sigma
    raw = np.where(raw > 1e-10, raw, 1e-10)
    omega = 1.0 / (raw * raw)
    cap = np.quantile(omega, cfg.weight_clip_quantile)
    omega = np.minimum(omega, cap)
    # Variables
    d = cp.Variable(basis.n_basis, name="d")
    aL = cp.Variable(put.n_atoms, nonneg=True, name="aL")
    aR = cp.Variable(call.n_atoms, nonneg=True, name="aR")
    return SliceVars(
        quotes=q, pieces=pieces, d=d, aL=aL, aR=aR, body_nca=body_nca,
        omega=omega, chi2_scale=1.0 / max(x.size, 1), s_n=s_n,
        k_l=k_l, k_h=k_h, n_quotes_in_body=x.size,
    )


def _atm_total_vol(q: SliceQuotes) -> float:
    """At-the-money total volatility interpolated from the mids (paper §8)."""
    x = q.x
    sig = q.sigma_mid
    mask = np.isfinite(sig)
    if not np.any(mask):
        return 0.1 * np.sqrt(max(q.T, 1e-6))
    x_ok = x[mask]
    sig_ok = sig[mask]
    # Simple log-linear interpolation at x = 1.
    k = np.log(x_ok)
    order = np.argsort(k)
    k_sorted = k[order]
    s_sorted = sig_ok[order]
    sig_atm = float(np.interp(0.0, k_sorted, s_sorted))
    return sig_atm * np.sqrt(max(q.T, 1e-10))


def _slice_constraints(sv: SliceVars) -> list[cp.Constraint]:
    """Linear constraints for one slice: body no-arbitrage (eq. 7), seam joins
    (eq. 18) and wing non-negativity (already in Variable definition)."""
    cons: list[cp.Constraint] = []
    D1 = sv.body_nca.D1()
    D2 = sv.body_nca.D2()
    cons.append(D1 @ sv.d <= 0.0)
    cons.append(D1 @ sv.d >= -1.0)
    cons.append(D2 @ sv.d >= sv.body_nca.convexity_floor())
    d0_floor, dM_floor = sv.body_nca.endpoint_floors()
    cons.append(sv.d[0] >= d0_floor)
    cons.append(sv.d[-1] >= dM_floor)
    # Seam joins (eq. 18): H_s a = (c_0, c_1, c_2)^T with intrinsic subtracted
    # on put wing.
    basis = sv.pieces.basis
    x_l, x_h = basis.x_l, basis.x_h
    Bl = basis.basis_matrix(np.array([x_l]), nu=0)[0]
    Blp = basis.basis_matrix(np.array([x_l]), nu=1)[0]
    Blpp = basis.basis_matrix(np.array([x_l]), nu=2)[0]
    Bh = basis.basis_matrix(np.array([x_h]), nu=0)[0]
    Bhp = basis.basis_matrix(np.array([x_h]), nu=1)[0]
    Bhpp = basis.basis_matrix(np.array([x_h]), nu=2)[0]
    k_l = sv.k_l
    k_h = sv.k_h
    # Body-side seam vectors (c_0, c_1, c_2) as linear functionals of d.
    # c_0 = B d; c_1 = x_s B' d; c_2 = x_s^2 B'' d + x_s B' d.
    c0_l = Bl @ sv.d
    c1_l = x_l * (Blp @ sv.d)
    c2_l = x_l * x_l * (Blpp @ sv.d) + x_l * (Blp @ sv.d)
    c0_h = Bh @ sv.d
    c1_h = x_h * (Bhp @ sv.d)
    c2_h = x_h * x_h * (Bhpp @ sv.d) + x_h * (Bhp @ sv.d)
    Hs_put = sv.pieces.put.seam_matrix(k_l)
    Hs_call = sv.pieces.call.seam_matrix(k_h)
    intr_put = sv.pieces.put.intrinsic_seam(k_l)
    # Put wing: H_s a_L = (c0, c1, c2)^T - intrinsic_seam
    cons.append(Hs_put[0] @ sv.aL == c0_l - intr_put[0])
    cons.append(Hs_put[1] @ sv.aL == c1_l - intr_put[1])
    cons.append(Hs_put[2] @ sv.aL == c2_l - intr_put[2])
    cons.append(Hs_call[0] @ sv.aR == c0_h)
    cons.append(Hs_call[1] @ sv.aR == c1_h)
    cons.append(Hs_call[2] @ sv.aR == c2_h)
    return cons


def _slice_objective(sv: SliceVars, cfg: CPIConfig) -> cp.Expression:
    """Per-slice contribution to the objective (eq. 21)."""
    B = sv.pieces.basis.basis_matrix(sv.quotes.x, nu=0)
    c_mid = sv.quotes.c_mid
    resid = B @ sv.d - c_mid
    data_term = cp.sum(cp.multiply(sv.omega, cp.square(resid))) * sv.chi2_scale
    R = sv.body_nca.R()
    # Numerical symmetrisation in case of rounding noise.
    R = 0.5 * (R + R.T)
    smooth = cfg.rho * cp.quad_form(sv.d, cp.psd_wrap(R))
    ridge = cfg.eta * (cp.sum_squares(sv.aL) + cp.sum_squares(sv.aR))
    return data_term + smooth + ridge


# -- Calendar constraints (shared across slices) ---------------------------


def _calendar_constraints(
    svs: Sequence[SliceVars], cfg: CPIConfig,
) -> list[cp.Constraint]:
    """Calendar ordering via Appendix C (body/body-wing bands) and
    Appendix A (pure wings beyond both seams)."""
    cons: list[cp.Constraint] = []
    for n in range(len(svs) - 1):
        a, b = svs[n], svs[n + 1]
        # Body-and-band sandwich: for simplicity restrict xbar to the body
        # intersection. The tail-sum rows of Appendix A cover the regions
        # outside both seams; the body-wing bands (one seam of a inside the
        # body of b, or vice versa) are covered by the sandwich on the
        # intersection in x, extended up to min(x_h_a, x_h_b) and from
        # max(x_l_a, x_l_b).
        s_n = min(a.s_n, b.s_n)
        grid = sandwich_grid(a.pieces, b.pieces, s_n,
                             pts_per_interval_width=cfg.sandwich_pts_per_s)
        rows = sandwich_rows(a.pieces, b.pieces, grid)
        for row in rows:
            lhs = (
                row["Bd_m"] @ b.d
                - row["Bd_n"] @ a.d
                - row["Pa_n"] @ a.aL
                - row["Ca_n"] @ a.aR
            )
            cons.append(lhs >= row["rhs"])
        # Appendix A: pure wing beyond both seams, each side separately.
        # Put wing: beyond left seam, max(|k_l_a|, |k_l_b|).
        kappa_n_put = abs(a.k_l)
        kappa_m_put = abs(b.k_l)
        TR = wing_tail_rows(a.pieces.put, b.pieces.put, kappa_n_put,
                            kappa_m_put, cfg.delta_mu)
        N_a_put = a.pieces.put.n_atoms
        for row in TR:
            cons.append(row[:N_a_put] @ a.aL + row[N_a_put:] @ b.aL >= 0.0)
        # Call wing
        kappa_n_call = abs(a.k_h)
        kappa_m_call = abs(b.k_h)
        TR = wing_tail_rows(a.pieces.call, b.pieces.call, kappa_n_call,
                            kappa_m_call, cfg.delta_mu)
        N_a_call = a.pieces.call.n_atoms
        for row in TR:
            cons.append(row[:N_a_call] @ a.aR + row[N_a_call:] @ b.aR >= 0.0)
    return cons


# -- Variance swap row (paper eq. 23) --------------------------------------


def variance_swap_coefficients(sv: SliceVars) -> tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, float]:
    """Return ``(A, gL, gR, const)`` such that

        L / 2 = A d + gL . a_L + gR . a_R + const

    for the normalised log-contract expectation ``L`` of this slice (paper
    eq. 28). Unnormalised ``L`` has the same expression; the forward F
    drops out of the normalised expectation."""
    basis = sv.pieces.basis
    A_body, const = body_variance_swap_row(basis, basis.x_l, basis.x_h)
    gL = sv.pieces.put.variance_swap_coeffs(sv.k_l)
    gR = sv.pieces.call.variance_swap_coeffs(sv.k_h)
    return A_body, gL, gR, const


# -- Top-level calibration -------------------------------------------------


@dataclass
class CPISolution:
    """Fitted CPI slices and solver diagnostics."""

    slices: list[SliceVars]
    objective: float
    runtime_s: float
    solver_status: str


def calibrate_surface(
    quotes: Sequence[SliceQuotes], cfg: CPIConfig | None = None,
    variance_swap_quotes: Sequence[tuple[int, float, float]] | None = None,
    verbose: bool = False,
) -> CPISolution:
    """Calibrate the full surface with a single joint QP.

    Parameters
    ----------
    quotes
        Slices in increasing time to expiry.
    cfg
        Hyper-parameters; defaults from :class:`CPIConfig`.
    variance_swap_quotes
        Optional list of ``(slice_index, L_market, width)`` triples. Each
        triple adds a weighted least-squares term
        ``((L_n - L_market) / max(width, |L_market|))^2`` to the objective.
    """
    import time

    cfg = cfg or CPIConfig()
    quotes_sorted = sorted(quotes, key=lambda q: q.T)
    svs = [_build_slice(q, cfg) for q in quotes_sorted]
    cons: list[cp.Constraint] = []
    for sv in svs:
        cons.extend(_slice_constraints(sv))
    cons.extend(_calendar_constraints(svs, cfg))
    obj = sum(_slice_objective(sv, cfg) for sv in svs)
    if variance_swap_quotes:
        for idx, L_mkt, width in variance_swap_quotes:
            sv = svs[idx]
            A, gL, gR, const = variance_swap_coefficients(sv)
            L_expr = 2.0 * (A @ sv.d + gL @ sv.aL + gR @ sv.aR + const)
            scale = max(float(width), abs(float(L_mkt)))
            obj = obj + cp.square((L_expr - L_mkt) / max(scale, 1e-12))
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL, verbose=verbose)
    rt = time.perf_counter() - t0
    return CPISolution(
        slices=svs, objective=float(prob.value or np.nan),
        runtime_s=rt, solver_status=prob.status,
    )


# -- Evaluation helpers ----------------------------------------------------


def evaluate_body(sv: SliceVars, x: np.ndarray) -> np.ndarray:
    """Normalised call price inside the body ``[x_l, x_h]``."""
    B = sv.pieces.basis.basis_matrix(x, nu=0)
    return B @ sv.d.value


def evaluate_slice(sv: SliceVars, x: np.ndarray) -> np.ndarray:
    """Normalised call price across body and wings on any ``x > 0``."""
    x = np.asarray(x, dtype=float)
    k = np.log(x)
    out = np.empty(x.shape)
    body_mask = (x >= sv.pieces.basis.x_l) & (x <= sv.pieces.basis.x_h)
    if np.any(body_mask):
        out[body_mask] = evaluate_body(sv, x[body_mask])
    left = k < sv.k_l
    if np.any(left):
        H = sv.pieces.put.atoms(k[left], sv.k_l)
        out[left] = H @ sv.aL.value + np.maximum(1.0 - x[left], 0.0)
    right = k > sv.k_h
    if np.any(right):
        H = sv.pieces.call.atoms(k[right], sv.k_h)
        out[right] = H @ sv.aR.value
    return out


def fair_variance(sv: SliceVars) -> float:
    """Normalised ``L`` from the fitted variables (paper eq. 23)."""
    A, gL, gR, const = variance_swap_coefficients(sv)
    return 2.0 * float(A @ sv.d.value + gL @ sv.aL.value + gR @ sv.aR.value
                       + const)
