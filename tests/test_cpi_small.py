"""End-to-end CPI tests on a small synthetic surface.

These tests exercise the shared pipeline (data preparation, QP assembly, solve,
evaluation) on a handful of slices of the synthetic SSVI surface. They verify
the paper's structural properties: body no-arbitrage, C^2 join, non-negative
density, and that a positive solver status is returned.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cpi import (
    CPIConfig, calibrate_surface, evaluate_body, evaluate_slice, fair_variance,
    variance_swap_coefficients,
)


@pytest.fixture(scope="module")
def tiny_solution(small_surface):
    cfg = CPIConfig(
        n_interior_knots=10, put_n_atoms=4, call_n_atoms=4,
        sandwich_pts_per_s=8.0, delta_mu=0.0,
    )
    return calibrate_surface(small_surface, cfg)


def test_solver_returns_optimal(tiny_solution):
    assert tiny_solution.solver_status.startswith("optimal")


def test_body_monotone_and_convex(tiny_solution):
    """``c(x)`` is non-increasing on ``[x_l, x_h]`` and convex (paper eq. 7)."""
    for sv in tiny_solution.slices:
        x = np.linspace(sv.pieces.basis.x_l, sv.pieces.basis.x_h, 200)
        c = evaluate_body(sv, x)
        assert np.all(np.diff(c) <= 1e-7)
        cpp = sv.pieces.basis.basis_matrix(x, nu=2) @ sv.d.value
        assert np.all(cpp >= -1e-6)


def test_body_slope_bounds(tiny_solution):
    """``-1 <= c'(x) <= 0`` across the body (paper eq. 7)."""
    for sv in tiny_solution.slices:
        x = np.linspace(sv.pieces.basis.x_l, sv.pieces.basis.x_h, 100)
        cp = sv.pieces.basis.basis_matrix(x, nu=1) @ sv.d.value
        assert np.all(cp <= 1e-7)
        assert np.all(cp >= -1.0 - 1e-7)


def test_endpoint_intrinsic(tiny_solution):
    """``c(x_l) >= 1 - x_l`` and ``c(x_h) >= 0`` (paper §4)."""
    for sv in tiny_solution.slices:
        d = sv.d.value
        assert d[0] >= (1.0 - sv.pieces.basis.x_l) - 1e-7
        assert d[-1] >= -1e-7


def test_wing_continuous_at_seam(tiny_solution):
    """The body and wing agree at both seams (paper eq. 18 C^2 join)."""
    for sv in tiny_solution.slices:
        x_l = sv.pieces.basis.x_l
        x_h = sv.pieces.basis.x_h
        # Evaluate just inside and just outside each seam.
        eps = 1e-5
        c_body_l = float(evaluate_body(sv, np.array([x_l]))[0])
        c_wing_l = float(evaluate_slice(sv, np.array([x_l * (1.0 - eps)]))[0])
        assert np.isclose(c_body_l, c_wing_l, atol=1e-4, rtol=1e-3)
        c_body_h = float(evaluate_body(sv, np.array([x_h]))[0])
        c_wing_h = float(evaluate_slice(sv, np.array([x_h * (1.0 + eps)]))[0])
        assert np.isclose(c_body_h, c_wing_h, atol=1e-4, rtol=1e-3)


def test_wing_weights_non_negative(tiny_solution):
    for sv in tiny_solution.slices:
        assert np.all(sv.aL.value >= -1e-10)
        assert np.all(sv.aR.value >= -1e-10)


def test_fair_variance_positive(tiny_solution):
    """The fair log-contract expectation is non-negative (paper Appendix B)."""
    for sv in tiny_solution.slices:
        L = fair_variance(sv)
        assert L > -1e-8


def test_variance_swap_affine_mapping(tiny_solution):
    """``L / 2 = A d + gL . aL + gR . aR + const`` by construction."""
    for sv in tiny_solution.slices:
        A, gL, gR, const = variance_swap_coefficients(sv)
        L_affine = 2.0 * (
            A @ sv.d.value + gL @ sv.aL.value + gR @ sv.aR.value + const
        )
        L_fit = fair_variance(sv)
        assert np.isclose(L_affine, L_fit, atol=1e-10)
