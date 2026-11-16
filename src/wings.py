"""Asymptotic wing atoms, seam matching and variance-swap integrals (paper §5).

A wing side is a non-negative combination of ``N_a`` atoms

    h_i(k) = (|k_s|/|k|)^(1/2) exp(-mu_i (|k| - |k_s|)),     |k| >= |k_s|,

with the put wing carrying an extra intrinsic ``(1 - e^k)^+ = 1 - x`` on k < 0.
The atom decay rates ``mu_i`` are mapped from Lee slopes ``b`` by
``lambda(b) = (2 - b)^2 / (8 b)`` (paper eq. 10); on the put wing
``mu = 1 + lambda(b)`` and on the call wing ``mu = lambda(b)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfcx


def lee_lambda(b: np.ndarray) -> np.ndarray:
    """``lambda(b) = (2 - b)^2 / (8 b)`` of paper eq. (10)."""
    b = np.asarray(b, dtype=float)
    return (2.0 - b) ** 2 / (8.0 * b)


def geometric_slopes(b_lo: float, b_hi: float, n_atoms: int) -> np.ndarray:
    """``N_a`` Lee slopes laid geometrically on ``[b_lo, b_hi]`` (paper §5.2)."""
    return np.geomspace(b_lo, b_hi, n_atoms)


@dataclass
class WingSide:
    """One side of the wing dictionary (``side`` is 'put' or 'call').

    ``mu`` is the array of decay rates, ordered in decreasing mu (thinnest
    first) as required by the shared-basis argument of Appendix A.
    """

    side: str
    b: np.ndarray
    mu: np.ndarray

    @classmethod
    def make(cls, side: str, b_lo: float, b_hi: float, n_atoms: int) -> "WingSide":
        b_asc = geometric_slopes(b_lo, b_hi, n_atoms)
        # Thinnest first means largest mu first: for both sides mu decreases in
        # b (lambda(b) is strictly decreasing on (0, 2)).
        b_desc = b_asc[::-1]
        if side == "put":
            mu = 1.0 + lee_lambda(b_desc)
        elif side == "call":
            mu = lee_lambda(b_desc)
        else:
            raise ValueError(f"side must be 'put' or 'call', got {side}")
        return cls(side=side, b=b_desc, mu=mu)

    @property
    def n_atoms(self) -> int:
        return int(self.mu.size)

    # --- Atom evaluation ----------------------------------------------------

    def atoms(self, k: np.ndarray, k_s: float) -> np.ndarray:
        """Matrix ``H[j, i] = h_i(k_j)`` evaluated at depths ``|k| >= |k_s|``.

        The intrinsic ``(1 - e^k)^+`` is NOT included: on the put wing it is
        added separately so that only ``a_i >= 0`` enter the wing weights.
        """
        k = np.asarray(k, dtype=float)
        kappa = abs(k_s)
        zeta = np.abs(k)
        # Guard against the seam itself: zeta = kappa gives h = 1.
        zeta_safe = np.where(zeta < 1e-14, kappa, zeta)
        pref = np.sqrt(kappa / np.maximum(zeta_safe, 1e-300))
        H = np.empty((k.size, self.n_atoms))
        for i, mu in enumerate(self.mu):
            H[:, i] = pref * np.exp(-mu * (zeta_safe - kappa))
        # Points shallower than the seam are outside the wing's support.
        shallow = zeta < kappa - 1e-14
        if np.any(shallow):
            H[shallow, :] = 0.0
        return H

    def intrinsic(self, k: np.ndarray) -> np.ndarray:
        """Put-side intrinsic ``(1 - e^k)^+`` evaluated at ``k`` (zero on the
        call side). Used to form full wing prices for diagnostics."""
        k = np.asarray(k, dtype=float)
        if self.side == "put":
            x = np.exp(k)
            return np.maximum(1.0 - x, 0.0)
        return np.zeros_like(k)

    # --- Seam matching H_s (paper §5.3) ------------------------------------

    def seam_matrix(self, k_s: float) -> np.ndarray:
        """``3 x N_a`` matrix ``H_s`` whose columns are the atom's seam value,
        first and second ``k``-derivative (paper eq. 17)."""
        kappa = abs(k_s)
        # u_i = mu_i + 1/(2 kappa); sign on first derivative is + on the put
        # wing (k_s < 0, derivative is -d/d|k|) and - on the call wing.
        u = self.mu + 1.0 / (2.0 * kappa)
        sgn = 1.0 if self.side == "put" else -1.0
        row0 = np.ones(self.n_atoms)
        row1 = sgn * u
        row2 = u * u + 1.0 / (2.0 * kappa * kappa)
        return np.stack([row0, row1, row2], axis=0)

    def intrinsic_seam(self, k_s: float) -> np.ndarray:
        """``(c_0, c_1, c_2)`` of the put-side intrinsic at the seam. Zero on
        call side. The intrinsic ``1 - e^k`` is C^infty with derivatives
        ``-e^k`` and ``-e^k`` in ``k``."""
        if self.side == "call":
            return np.zeros(3)
        x_s = float(np.exp(k_s))
        return np.array([1.0 - x_s, -x_s, -x_s])

    # --- Variance swap wing integral (paper eq. 30) ------------------------

    def variance_swap_coeffs(self, k_s: float) -> np.ndarray:
        """``g_i`` of paper eq. (30): the per-atom contribution to the log
        contract ``L`` from this wing. For the put wing ``nu = mu - 1`` and
        the sign in ``e^{+kappa}`` is positive; for the call wing
        ``nu = mu + 1`` and the sign is negative."""
        kappa = abs(k_s)
        if self.side == "put":
            nu = self.mu - 1.0
            sgn_exp = 1.0
        else:
            nu = self.mu + 1.0
            sgn_exp = -1.0
        # Convergence for the put wing requires nu > 0 (b < 2 strictly).
        nu_safe = np.where(nu <= 1e-12, 1e-12, nu)
        return (
            np.sqrt(np.pi / nu_safe)
            * np.sqrt(kappa)
            * np.exp(sgn_exp * kappa)
            * erfcx(np.sqrt(nu_safe * kappa))
        )


# --- Body variance swap row (paper eq. 31) ---------------------------------


def body_variance_swap_row(
    basis,
    x_l: float,
    x_h: float,
) -> tuple[np.ndarray, float]:
    """Linear map ``d -> A d + const`` giving the body contribution to
    ``L / 2`` (eq. 28). On the put branch ``[x_l, 1]`` the integrand is
    ``(c(x) - 1 + x) / x^2`` and on the call branch ``[1, x_h]`` it is
    ``c(x) / x^2``. Each knot span integrates polynomially (eq. 31).

    Returns ``(A, const)`` where ``L/2 = A d + wing terms + const``. The
    extra put intrinsic ``(x - 1)/x^2`` integrates in closed form to
    ``log x + 1/x`` and contributes to ``const``.
    """
    from .bspline import ClampedBSpline  # noqa: F401  (type hint only)

    p = basis.degree
    t = basis.t
    M = basis.n_basis
    # Split intervals at x = 1.
    knots = np.unique(np.concatenate([t[p: -p], np.array([1.0])]))
    knots = knots[(knots >= x_l - 1e-12) & (knots <= x_h + 1e-12)]
    A = np.zeros(M)
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a + 1e-14:
            continue
        # Gauss-Legendre of order p+2 integrates degree-(p + 1) exactly, enough
        # for the ratio polynomial / x^2 split between log and polynomial.
        nodes, weights = np.polynomial.legendre.leggauss(p + 2)
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xq = mid + half * nodes
        wq = half * weights
        B = basis.basis_matrix(xq)  # (nq, M)
        A += (B.T * (wq / (xq * xq))) @ np.ones(xq.size)  # placeholder
        # The above is wrong (it would ignore the basis structure). Replace by
        # a direct quadrature of each basis function.
    # Direct quadrature (overwrites the placeholder above).
    A[:] = 0.0
    const = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a + 1e-14:
            continue
        nodes, weights = np.polynomial.legendre.leggauss(p + 2)
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xq = mid + half * nodes
        wq = half * weights
        B = basis.basis_matrix(xq)
        # Body contribution: int c(x)/x^2 dx = sum_j d_j int B_j(x)/x^2 dx.
        A += B.T @ (wq / (xq * xq))
        if b <= 1.0 + 1e-14:
            # Put branch adds (x - 1)/x^2 = 1/x - 1/x^2. Const contribution:
            const += float(np.sum(wq * (1.0 / xq - 1.0 / (xq * xq))))
    return A, const
