"""P16 (#1): infinities and log(0) inside the fastmath-compiled grid.

``calc_lnlike_grid`` is compiled with ``fastmath=True`` yet deliberately writes
``-inf`` and evaluates ``log(fB)`` / ``log(1 - fB)``, which are ``-inf`` at
``fB = 0`` / ``fB = 1``.  fastmath lets the compiler assume no infs/NaNs, so
these edges are formally undefined.  These tests pin the intended values at the
edges so a compiler/flag change cannot silently alter them (#5 covers
``fB = 0``; ``fB = 1`` was not covered).
"""

import math

import numpy as np
import pytest

from isochrones.cluster_utils import calc_lnlike_grid
from isochrones.priors import powerlaw_lnpdf

ALPHA, GAMMA = -2.3, 0.3
MASS_LO, MASS_HI, Q_LO = 0.2, 100.0, 0.1


def _grid(fB, masses=(0.8, 1.0, 1.2)):
    n = len(masses)
    model_mags = np.array([[10.0 + 0.3 * i] for i in range(n)])
    eeps = np.arange(n, dtype=float)
    return calc_lnlike_grid(
        np.zeros((1, n)),
        model_mags,
        1,
        np.array(masses),
        np.zeros(n),
        eeps,
        np.array([[10.0]]),
        np.array([[0.05]]),
        ALPHA,
        GAMMA,
        fB,
        MASS_LO,
        MASS_HI,
        Q_LO,
    ), model_mags


def _expected_cell(model_mags, masses, j, k, fB):
    f1, f2 = 10 ** (-0.4 * model_mags[j, 0]), 10 ** (-0.4 * model_mags[k, 0])
    binary = -0.5 * ((-2.5 * math.log10(f1 + f2) - 10.0) / 0.05) ** 2
    single = -0.5 * ((model_mags[j, 0] - 10.0) / 0.05) ** 2
    with np.errstate(divide="ignore"):
        phot = np.logaddexp(np.log(fB) + binary, np.log(1 - fB) + single)
    mass = powerlaw_lnpdf(masses[j], ALPHA, MASS_LO, MASS_HI)
    ratio = powerlaw_lnpdf(masses[k] / masses[j], GAMMA, Q_LO, 1.0)
    return phot + mass + ratio


@pytest.mark.parametrize("fB", [0.0, 1.0, 0.5])
def test_lower_triangle_matches_reference_at_fB_edges(fB):
    masses = (0.8, 1.0, 1.2)
    grid, model_mags = _grid(fB, masses)
    for j in range(3):
        for k in range(j + 1):
            assert np.isfinite(grid[0, j, k])
            assert np.isclose(grid[0, j, k], _expected_cell(model_mags, masses, j, k, fB), rtol=1e-10)


def test_mass_ratio_below_q_lo_is_exactly_negative_infinity():
    grid, _ = _grid(0.5, masses=(0.8, 1.0, 20.0))
    assert grid[0, 2, 0] == -np.inf
    assert grid[0, 2, 1] == -np.inf
    assert not np.isnan(grid).any()
