"""Body no-arbitrage constraints and quadratic smoothness penalty (paper §4).

Given a :class:`src.bspline.ClampedBSpline` body basis on ``[x_l, x_h]``, this
module assembles the derivative maps ``D1``, ``D2``, the smoothness Gram matrix
``R``, and the endpoint intrinsic bounds imposed on the degree-p coefficients
``d``. Per paper eq. (7): ``D1 d <= 0``, ``D1 d >= -1``, ``D2 d >= epsilon / s_n``,
``d_0 >= 1 - x_l`` and ``d_{M-1} >= 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bspline import ClampedBSpline


@dataclass
class BodyNCA:
    """Body no-arbitrage pieces for one maturity (paper §4)."""

    basis: ClampedBSpline
    s_n: float            # at-the-money total volatility (eq. 22)
    epsilon: float = 1e-8  # convexity floor (paper §4)

    def D1(self) -> np.ndarray:
        return self.basis.D1_full()

    def D2(self) -> np.ndarray:
        return self.basis.D2()

    def R(self) -> np.ndarray:
        """Smoothness Gram ``R = s_n^3 int B_j''' B_k''' dx`` (paper eq. 22)."""
        return (self.s_n ** 3) * self.basis.smoothness_matrix_R()

    def convexity_floor(self) -> float:
        """Right-hand side of the strict convexity bound ``epsilon / s_n``."""
        return self.epsilon / max(self.s_n, 1e-12)

    def endpoint_floors(self) -> tuple[float, float]:
        """``d_0 >= 1 - x_l`` and ``d_{M-1} >= 0``."""
        return max(1.0 - self.basis.x_l, 0.0), 0.0
