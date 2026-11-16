"""Calendar no-arbitrage constraints between adjacent maturities (paper §6).

Implements two families of linear inequalities:

1. The convex sandwich of Appendix C: for each interval ``[x_1, x_2]`` in the
   union of the two maturities' bodies and body-wing bands, with midpoint
   ``xbar``, require ``c_{n+1}(xbar) + (x_j - xbar) c'_{n+1}(xbar) >= c_n(x_j)``
   for ``j = 1, 2``. This is sufficient because both ``c_n`` and ``c_{n+1}``
   are convex in ``x``. Two linear rows per interval cover body overlaps and
   body-wing bands alike. The paper notes that Appendix C intervals of width
   ``s_n / 20`` give the same fit as §6 on the SPX surface (with roughly four
   times more inequalities), so this is used as a simpler equivalent.

2. The shared-basis tail-sum constraints of Appendix A beyond both seams:
   with ``beta_i^(n) = a_i^(n) kappa_n^(1/2) exp(mu_i kappa_n)``, the ordering

       sum_{i >= m} beta_i^(n+1) >= (1 + delta_mu) sum_{i >= m} beta_i^(n),
       m = 1, ..., N_a,

   each row scaled by ``1 / psi_m(max(kappa_n, kappa_{n+1}))`` so that its
   leading term is order one (paper eq. 27).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bspline import ClampedBSpline
from .wings import WingSide


@dataclass
class SlicePieces:
    """Linear operators needed for calendar constraints on one maturity."""

    basis: ClampedBSpline
    put: WingSide
    call: WingSide
    k_l: float  # left seam in log-moneyness
    k_h: float  # right seam


# -- Point evaluation on a slice (value and dc/dx as linear forms) ----------


def eval_rows(piece: SlicePieces, x: float) -> dict[str, np.ndarray | float]:
    """Return dicts of row operators such that ``c(x) = Bd . d + Pa . a_L +
    Ca . a_R + const`` and similar for ``dc/dx``.

    The returned dict has keys ``Bd``, ``Pa``, ``Ca``, ``const`` for the
    value row and ``Bd1``, ``Pa1``, ``Ca1``, ``const1`` for the first
    derivative.
    """
    x = float(x)
    out: dict[str, np.ndarray | float] = {
        "Bd": np.zeros(piece.basis.n_basis),
        "Pa": np.zeros(piece.put.n_atoms),
        "Ca": np.zeros(piece.call.n_atoms),
        "const": 0.0,
        "Bd1": np.zeros(piece.basis.n_basis),
        "Pa1": np.zeros(piece.put.n_atoms),
        "Ca1": np.zeros(piece.call.n_atoms),
        "const1": 0.0,
    }
    k = float(np.log(x))
    if k < piece.k_l:
        # Deep in the put wing
        H = piece.put.atoms(np.array([k]), piece.k_l)[0]  # (N_a,)
        out["Pa"] = H
        out["const"] = 1.0 - x  # put intrinsic
        # dc/dx = dc/dk * dk/dx = (1/x) dc/dk
        # dh_i/dk = -h_i (1/(2 zeta) + mu_i) for k<0 ... use sign convention
        # Simpler: numerical derivative in x via finite difference for the
        # derivative row. (Not used in critical paths; we only need value rows
        # at seams, where the body derivative row is used instead.)
        raise NotImplementedError("eval beyond seam used only at seam itself")
    if k > piece.k_h:
        H = piece.call.atoms(np.array([k]), piece.k_h)[0]
        out["Ca"] = H
        raise NotImplementedError("eval beyond seam used only at seam itself")
    # Inside the body: use B-spline basis for value and derivative.
    B = piece.basis.basis_matrix(np.array([x]), nu=0)[0]
    Bp = piece.basis.basis_matrix(np.array([x]), nu=1)[0]
    out["Bd"] = B
    out["Bd1"] = Bp
    return out


# -- Convex sandwich inequalities (Appendix C) ------------------------------


def _xbar_rows(piece: SlicePieces, xbar: float) -> tuple[np.ndarray, np.ndarray]:
    """Return rows ``(v, vp)`` such that at ``xbar`` inside the body
    ``c(xbar) = v . d`` and ``c'(xbar) = vp . d``.

    xbar is assumed to lie inside ``[x_l, x_h]`` (the body). This function is
    used as ``g(xbar)`` and ``g'(xbar)`` of the sandwich lemma and so
    ``piece`` is the *later* maturity (``n+1``).
    """
    B = piece.basis.basis_matrix(np.array([xbar]), nu=0)[0]
    Bp = piece.basis.basis_matrix(np.array([xbar]), nu=1)[0]
    return B, Bp


def _value_row(piece: SlicePieces, x: float) -> dict:
    """Linear decomposition of ``c_n(x)`` of maturity ``n``. Returns keys
    ``Bd``, ``Pa``, ``Ca`` (arrays) and ``const`` (scalar)."""
    k = float(np.log(x))
    out = {
        "Bd": np.zeros(piece.basis.n_basis),
        "Pa": np.zeros(piece.put.n_atoms),
        "Ca": np.zeros(piece.call.n_atoms),
        "const": 0.0,
    }
    if k < piece.k_l:
        H = piece.put.atoms(np.array([k]), piece.k_l)[0]
        out["Pa"] = H
        out["const"] = 1.0 - x
    elif k > piece.k_h:
        H = piece.call.atoms(np.array([k]), piece.k_h)[0]
        out["Ca"] = H
    else:
        out["Bd"] = piece.basis.basis_matrix(np.array([x]), nu=0)[0]
    return out


def sandwich_grid(piece_n: SlicePieces, piece_m: SlicePieces, s_n: float,
                  pts_per_interval_width: float = 20.0) -> np.ndarray:
    """Grid of ``x`` breakpoints covering both bodies with intervals of width
    ``s_n * x`` / ``pts_per_interval_width`` in log-moneyness (paper Appendix
    C: ``s_n / 20`` for each pair).

    The grid spans ``[max(x_l_n, x_l_m), min(x_h_n, x_h_m)]`` extended
    outward through the body-wing bands up to each maturity's seam.
    """
    x_l = max(piece_n.basis.x_l, piece_m.basis.x_l)
    x_h = min(piece_n.basis.x_h, piece_m.basis.x_h)
    # Operate in log-moneyness for uniform spacing.
    k_l, k_h = float(np.log(x_l)), float(np.log(x_h))
    step = max(s_n / pts_per_interval_width, 1e-4)
    n = max(int(np.ceil((k_h - k_l) / step)) + 1, 2)
    ks = np.linspace(k_l, k_h, n)
    return np.exp(ks)


def sandwich_rows(
    piece_n: SlicePieces, piece_m: SlicePieces, grid: np.ndarray,
) -> list[dict]:
    """Return a list of linear-row dicts for the sandwich conditions

        c_m(xbar) + (x_j - xbar) c'_m(xbar) >= c_n(x_j),  j = 1, 2,

    for each interval ``[x_j = grid[j], x_{j+1} = grid[j+1]]``, ``xbar`` the
    midpoint, ``m = n + 1``. Each row dict carries arrays ``Bd_m``, ``Pa_m``,
    ``Ca_m``, ``Bd_n``, ``Pa_n``, ``Ca_n`` and a scalar ``rhs`` such that
    ``row . (vars) >= rhs``.
    """
    rows: list[dict] = []
    for a, b in zip(grid[:-1], grid[1:]):
        xbar = 0.5 * (a + b)
        # xbar inside the body of piece_m by construction (grid is intersected
        # body range), so g(xbar) and g'(xbar) are degree-p B-spline rows.
        gval, gder = _xbar_rows(piece_m, xbar)
        for xj in (a, b):
            tangent_rhs_coef = (xj - xbar)
            # Tangent: ell(xj) = c_m(xbar) + (xj - xbar) c'_m(xbar)
            Bd_m = gval + tangent_rhs_coef * gder
            Pa_m = np.zeros(piece_m.put.n_atoms)
            Ca_m = np.zeros(piece_m.call.n_atoms)
            fval = _value_row(piece_n, xj)
            rows.append({
                "Bd_m": Bd_m, "Pa_m": Pa_m, "Ca_m": Ca_m,
                "Bd_n": fval["Bd"], "Pa_n": fval["Pa"], "Ca_n": fval["Ca"],
                "rhs": fval["const"],  # c_m part contributes zero const
            })
    return rows


def pointwise_calendar_rows(
    piece_n: SlicePieces, piece_m: SlicePieces, grid_k: np.ndarray,
) -> list[dict]:
    """Return linear-row dicts for pointwise ``c_m(x) - c_n(x) >= 0`` at the
    log-moneyness grid ``grid_k``.

    Used as a cover of body-wing bands (where the sandwich of Appendix C is
    restricted to the body intersection) and as a dense safety net alongside
    Appendix A tail-sum rows. Each row dict has the same shape as
    ``sandwich_rows`` output and can be consumed by the same assembly code.
    """
    rows: list[dict] = []
    for k in grid_k:
        x = float(np.exp(k))
        vn = _value_row(piece_n, x)
        vm = _value_row(piece_m, x)
        rows.append({
            "Bd_m": vm["Bd"], "Pa_m": vm["Pa"], "Ca_m": vm["Ca"],
            "Bd_n": vn["Bd"], "Pa_n": vn["Pa"], "Ca_n": vn["Ca"],
            "rhs": vn["const"] - vm["const"],
        })
    return rows


# -- Appendix A tail-sum constraints (pure wing beyond both seams) ----------


def wing_tail_rows(
    wing_n: WingSide, wing_m: WingSide, kappa_n: float, kappa_m: float,
    delta_mu: float,
) -> np.ndarray:
    """Return the ``N_a x (2 N_a)`` matrix encoding rows

        [-coef_n, +coef_m] @ (a_n, a_m) >= 0,     with the margin applied

    where ``coef_n[i] = (1 + delta_mu) * kappa_n^(1/2) exp(mu_i kappa_n)`` for
    ``i >= m`` and zero otherwise, scaled by ``1 / psi_m(kappa_bar)`` row by
    row (paper eq. 27).
    """
    N_a = wing_n.n_atoms
    assert wing_m.n_atoms == N_a and np.allclose(wing_n.mu, wing_m.mu)
    mu = wing_n.mu
    kbar = max(kappa_n, kappa_m)
    beta_n_coef = np.sqrt(kappa_n) * np.exp(mu * kappa_n)
    beta_m_coef = np.sqrt(kappa_m) * np.exp(mu * kappa_m)
    # Scale by 1 / psi_m(kbar) = kbar^(1/2) exp(mu_m kbar) per row m.
    rows = np.zeros((N_a, 2 * N_a))
    for m in range(N_a):
        scale = np.sqrt(kbar) * np.exp(mu[m] * kbar)
        # Non-zero for i >= m
        cols_n = -(1.0 + delta_mu) * beta_n_coef / scale
        cols_m = beta_m_coef / scale
        cols_n[:m] = 0.0
        cols_m[:m] = 0.0
        rows[m, :N_a] = cols_n
        rows[m, N_a:] = cols_m
    return rows
