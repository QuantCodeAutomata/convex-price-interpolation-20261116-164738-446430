"""Unit tests for the clamped B-spline basis (paper §3)."""

from __future__ import annotations

import numpy as np

from src.bspline import ClampedBSpline, place_knots


def test_partition_of_unity_cubic():
    basis = ClampedBSpline(
        degree=3, x_l=0.5, x_h=1.5,
        interior_knots=place_knots(0.5, 1.5, 10, 0.5),
    )
    x = np.linspace(0.5, 1.5, 11)
    B = basis.basis_matrix(x, nu=0)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-10)


def test_partition_of_unity_quartic():
    basis = ClampedBSpline(
        degree=4, x_l=0.5, x_h=1.5,
        interior_knots=place_knots(0.5, 1.5, 10, 0.5),
    )
    x = np.linspace(0.5, 1.5, 13)
    B = basis.basis_matrix(x, nu=0)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-10)


def test_endpoint_clamping():
    """At clamped endpoints the first basis function is 1, others 0."""
    basis = ClampedBSpline(
        degree=3, x_l=0.3, x_h=2.0,
        interior_knots=place_knots(0.3, 2.0, 8, 0.5),
    )
    Bl = basis.basis_matrix(np.array([0.3]))[0]
    Bh = basis.basis_matrix(np.array([2.0]))[0]
    assert np.isclose(Bl[0], 1.0)
    assert np.isclose(Bh[-1], 1.0)
    assert np.allclose(Bl[1:], 0.0, atol=1e-12)
    assert np.allclose(Bh[:-1], 0.0, atol=1e-12)


def test_derivative_sign_structure():
    """Monotone-decreasing body coefficients yield ``D1 d <= 0`` and
    convex (strictly increasing in index) coefficients yield ``D2 d >= 0``
    from the recursion of paper eq. (6)."""
    basis = ClampedBSpline(
        degree=3, x_l=0.1, x_h=2.0,
        interior_knots=place_knots(0.1, 2.0, 7, 0.4),
    )
    M = basis.n_basis
    t = basis.t
    p = basis.degree
    # Greville abscissae x*_j = (t[j+1] + ... + t[j+p]) / p; coefficients
    # d_j = f(x*_j) give a good B-spline fit for linear and quadratic f.
    greville = np.array([t[j + 1:j + p + 1].mean() for j in range(M)])
    d_dec = -greville  # strictly decreasing (f(x) = -x)
    assert np.all(basis.D1() @ d_dec <= 1e-10)
    d_conv = greville ** 2  # strictly convex (f(x) = x^2)
    assert np.all(basis.D2() @ d_conv >= -1e-8)


def test_smoothness_matrix_available():
    basis = ClampedBSpline(
        degree=3, x_l=0.1, x_h=2.0,
        interior_knots=place_knots(0.1, 2.0, 7, 0.4),
    )
    R = basis.smoothness_matrix_R()
    assert R.shape == (basis.n_basis, basis.n_basis)


def test_place_knots_monotone_and_inside():
    k = place_knots(0.3, 2.0, 15, 0.5)
    assert np.all(np.diff(k) > 0)
    assert k[0] > 0.3 and k[-1] < 2.0


def test_smoothness_matrix_symmetric_psd():
    basis = ClampedBSpline(
        degree=3, x_l=0.3, x_h=1.7,
        interior_knots=place_knots(0.3, 1.7, 10, 0.5),
    )
    R = basis.smoothness_matrix_R()
    assert np.allclose(R, R.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(0.5 * (R + R.T))
    assert eigs.min() > -1e-8
