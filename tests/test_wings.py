"""Tests for wing dictionary and body-wing C^2 join (paper §5)."""

from __future__ import annotations

import numpy as np

from src.wings import WingSide, lee_lambda


def test_lee_lambda_values():
    """``lambda(b) = (2 - b)^2 / (8 b)`` (paper §5): decreasing for b in
    (0, 2), equals 1/8 at b = 1, equals 0 at b = 2."""
    assert np.isclose(lee_lambda(1.0), 1.0 / 8.0)
    assert np.isclose(lee_lambda(2.0), 0.0)
    bs = np.linspace(0.01, 1.8, 200)
    lam = np.array([lee_lambda(b) for b in bs])
    # Strictly decreasing on (0, 2).
    assert np.all(np.diff(lam) < 0)


def test_put_wing_atom_value_at_seam_is_one():
    put = WingSide.make("put", 0.02, 1.6, 6)
    k_s = -0.3
    H = put.atoms(np.array([k_s]), k_s)
    assert np.allclose(H[0], 1.0)


def test_call_wing_atom_value_at_seam_is_one():
    call = WingSide.make("call", 0.005, 0.65, 8)
    k_s = 0.2
    H = call.atoms(np.array([k_s]), k_s)
    assert np.allclose(H[0], 1.0)


def test_put_wing_decays_further_from_seam():
    put = WingSide.make("put", 0.02, 1.6, 6)
    k_s = -0.3
    H_near = put.atoms(np.array([k_s - 0.1]), k_s)
    H_far = put.atoms(np.array([k_s - 1.0]), k_s)
    # Every atom decays monotonically away from the seam.
    assert np.all(H_far <= H_near + 1e-12)


def test_seam_matrix_shape():
    put = WingSide.make("put", 0.02, 1.6, 6)
    H_s = put.seam_matrix(-0.3)
    assert H_s.shape == (3, 6)


def test_mu_from_b_mapping():
    """``mu = 1 + lambda(b)`` on the put wing, ``mu = lambda(b)`` on the
    call wing (paper §5)."""
    b = 1.0
    put = WingSide.make("put", b, b, 1)
    call = WingSide.make("call", b, b, 1)
    assert np.isclose(put.mu[0], 1.0 + 1.0 / 8.0)
    assert np.isclose(call.mu[0], 1.0 / 8.0)
