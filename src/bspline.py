"""Clamped B-spline basis and derivative maps used by the CPI body (paper §3).

Implements evaluation of a clamped B-spline basis of arbitrary degree p on a
knot vector t, the sparse derivative-coefficient maps D1 and D2 from the
recursion (6), and the closed-form Gram matrix of third derivatives used by the
smoothness penalty (22). The basis is evaluated with scipy.interpolate.BSpline
on the natural clamped knot vector (multiplicity p+1 at each endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass
class ClampedBSpline:
    """A clamped B-spline basis of degree p on [x_l, x_h] with interior knots.

    Attributes mirror paper §3: ``degree`` is p; ``t`` is the knot vector with
    multiplicity p+1 at both endpoints; ``n_basis`` is M, the number of basis
    functions (degree-p coefficients ``d``).
    """

    degree: int
    x_l: float
    x_h: float
    interior_knots: np.ndarray  # strictly inside (x_l, x_h), ascending

    @property
    def t(self) -> np.ndarray:
        """Clamped knot vector with multiplicity p+1 at each endpoint."""
        p = self.degree
        return np.concatenate(
            [np.full(p + 1, self.x_l), self.interior_knots, np.full(p + 1, self.x_h)]
        )

    @property
    def n_basis(self) -> int:
        """M = number of degree-p basis functions."""
        return len(self.t) - self.degree - 1

    def basis_matrix(self, x: np.ndarray, nu: int = 0) -> np.ndarray:
        """Return the (len(x), M) matrix of degree-p basis values or derivatives.

        ``nu`` is the derivative order (0 for the basis itself, 1 for B', ...).
        Rows at x equal to x_h map onto the last basis using the clamped-knot
        extension convention of scipy.interpolate.BSpline.
        """
        x = np.asarray(x, dtype=float)
        M = self.n_basis
        t = self.t
        out = np.zeros((x.size, M))
        for j in range(M):
            c = np.zeros(M)
            c[j] = 1.0
            spline = BSpline(t, c, self.degree, extrapolate=False)
            out[:, j] = np.nan_to_num(spline(x, nu=nu), nan=0.0)
        # Make sure the right endpoint evaluates to the last basis (clamped).
        right = np.isclose(x, self.x_h)
        if nu == 0 and np.any(right):
            out[right, :] = 0.0
            out[right, -1] = 1.0
        return out

    def D1(self) -> np.ndarray:
        """Sparse map ``e = D1 d`` from degree-p coefficients to degree-(p-1)
        derivative coefficients, eq. (6) of paper §3."""
        p = self.degree
        t = self.t
        M = self.n_basis
        rows = []
        for j in range(M - 1):
            denom = t[j + p + 1] - t[j + 1]
            # The first and last p-1 degree-(p-1) coefficients fall on knot
            # intervals of zero width at the clamped endpoints; drop those rows
            # by returning a finite value only for strictly positive denom.
            if denom <= 0.0:
                continue
            row = np.zeros(M)
            row[j] = -p / denom
            row[j + 1] = p / denom
            rows.append(row)
        return np.asarray(rows)

    def D2(self) -> np.ndarray:
        """Sparse map ``f = D2 d`` from degree-p coefficients to degree-(p-2)
        second-derivative coefficients, eq. (6) of paper §3."""
        p = self.degree
        t = self.t
        M = self.n_basis
        # degree-(p-1) knot vector drops one knot at each end
        t1 = t[1:-1]
        # First derivative in degree-(p-1) basis: coefficients e_j, j=0..M-2
        # Second derivative in degree-(p-2) basis: coefficients f_j, j=0..M-3
        D1 = np.zeros((M - 1, M))
        for j in range(M - 1):
            denom = t[j + p + 1] - t[j + 1]
            if denom <= 0.0:
                continue
            D1[j, j] = -p / denom
            D1[j, j + 1] = p / denom
        D2 = np.zeros((M - 2, M - 1))
        for j in range(M - 2):
            # uses degree-(p-1) knot vector t1
            denom = t1[j + p] - t1[j + 1]
            if denom <= 0.0:
                continue
            D2[j, j] = -(p - 1) / denom
            D2[j, j + 1] = (p - 1) / denom
        out = D2 @ D1
        # Drop zero rows corresponding to collapsed knot spans.
        mask = np.any(out != 0.0, axis=1)
        return out[mask]

    def D1_full(self) -> np.ndarray:
        """``D1`` including zero-width rows, needed for Gram matrix operations."""
        p = self.degree
        t = self.t
        M = self.n_basis
        D = np.zeros((M - 1, M))
        for j in range(M - 1):
            denom = t[j + p + 1] - t[j + 1]
            if denom <= 0.0:
                continue
            D[j, j] = -p / denom
            D[j, j + 1] = p / denom
        return D

    def smoothness_matrix_R(self) -> np.ndarray:
        """Gram matrix of third derivatives, ``R_jj' = int B_j''' B_{j'}''' dx``
        used by the CPI smoothness penalty, closed form on knot spans."""
        p = self.degree
        if p < 3:
            return np.zeros((self.n_basis, self.n_basis))
        t = self.t
        M = self.n_basis
        # Compose three degree-reduction maps to get degree-(p-3) coefficients
        # representing the third derivative.
        def _D(t_vec: np.ndarray, deg: int, n: int) -> np.ndarray:
            D = np.zeros((n - 1, n))
            for j in range(n - 1):
                denom = t_vec[j + deg + 1] - t_vec[j + 1]
                if denom <= 0.0:
                    continue
                D[j, j] = -deg / denom
                D[j, j + 1] = deg / denom
            return D

        D_1 = _D(t, p, M)
        t1 = t[1:-1]
        D_2 = _D(t1, p - 1, M - 1)
        t2 = t1[1:-1]
        D_3 = _D(t2, p - 2, M - 2)
        T = D_3 @ D_2 @ D_1  # (M-3, M)
        t3 = t2[1:-1]  # degree-(p-3) knot vector
        # Gram of degree-(p-3) B-splines on their own knot vector.
        G = _bspline_gram(t3, p - 3)
        return T.T @ G @ T


def _bspline_gram(t: np.ndarray, deg: int) -> np.ndarray:
    """Gram matrix ``G_jk = int B_j^(deg)(x) B_k^(deg)(x) dx`` on the whole span
    of the clamped knot vector ``t``. Uses Gauss-Legendre quadrature of degree
    sufficient for exact integration of degree-``2*deg`` polynomials on each
    non-degenerate knot span."""
    n = len(t) - deg - 1
    G = np.zeros((n, n))
    # Gauss-Legendre of order deg+1 is exact for polynomials of degree 2*deg+1.
    order = max(1, deg + 1)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for a, b in zip(t[:-1], t[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xq = mid + half * nodes
        wq = half * weights
        B = np.zeros((xq.size, n))
        for j in range(n):
            c = np.zeros(n)
            c[j] = 1.0
            spline = BSpline(t, c, deg, extrapolate=False)
            B[:, j] = np.nan_to_num(spline(xq), nan=0.0)
        G += (B.T * wq) @ B
    return G


def place_knots(x_l: float, x_h: float, n_interior: int, q: float) -> np.ndarray:
    """Interior knots concentrated near the money, uniform in
    ``sgn(k)|k|^q`` with ``k = log x`` (paper §3). ``x_l < 1 < x_h`` is
    assumed, matching the paper setup where the quoted range straddles the
    forward. If x=1 is not inside the interval, falls back to uniform in k."""
    k_l, k_h = float(np.log(x_l)), float(np.log(x_h))
    if not (k_l < 0.0 < k_h):
        ks = np.linspace(k_l, k_h, n_interior + 2)[1:-1]
        return np.exp(ks)
    y_l = -((-k_l) ** q)  # sgn * |k|^q
    y_h = k_h ** q
    ys = np.linspace(y_l, y_h, n_interior + 2)[1:-1]
    ks = np.sign(ys) * np.abs(ys) ** (1.0 / q)
    return np.exp(ks)
