"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "market_data: tests that depend on live market-data credentials",
    )


@pytest.fixture(scope="session")
def small_surface():
    """A compact synthetic surface used by multiple tests."""
    from data.generate_synthetic import generate_surface
    slices = generate_surface()
    return slices[:6]
