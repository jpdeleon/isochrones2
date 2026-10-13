"""P12 (#1): the property likelihood of an unresolved binary uses both stars.

Valls-Gabaud (2016), Sec. 3.1: luminosity adds (Eq. 22) and the effective
temperature follows from the total luminosity and total radius (Eq. 23), not
from a luminosity-weighted mean of the temperatures.
"""

import numpy as np
import pytest

from isochrones.cluster_utils import add_binary_property_term, binary_teff


def test_binary_teff_of_identical_stars_is_unchanged():
    teff = np.array([5000.0, 5000.0])
    radius = np.array([1.0, 1.0])

    assert np.allclose(binary_teff(teff, radius), 5000.0)


def test_binary_teff_matches_eq_23():
    teff = np.array([6000.0, 4000.0])
    radius = np.array([1.2, 0.7])

    got = binary_teff(teff, radius)

    # independent: L = R^2 T^4 (common constant), Teff^4 = L_tot / R_tot^2
    L = radius**2 * teff**4
    expected = ((L[0] + L[1]) / (radius[0] ** 2 + radius[1] ** 2)) ** 0.25
    assert np.isclose(got[0, 1], expected)
    assert np.isclose(got[1, 0], expected)  # symmetric in the pair


def test_binary_teff_is_not_the_luminosity_weighted_mean():
    teff = np.array([6000.0, 4000.0])
    radius = np.array([1.2, 0.7])
    L = radius**2 * teff**4
    lum_weighted = (L * teff).sum() / L.sum()

    assert not np.isclose(binary_teff(teff, radius)[0, 1], lum_weighted, rtol=1e-3)


def test_binary_teff_lies_between_the_components():
    teff = np.array([6500.0, 3800.0])
    radius = np.array([1.5, 0.5])

    val = binary_teff(teff, radius)[0, 1]

    assert 3800.0 < val < 6500.0


def test_property_term_is_the_fB_mixture():
    grid = np.zeros((1, 2, 2))
    sgl = np.array([[-1.0, -2.0]])
    bin_ = np.array([[[-0.5, -3.0], [-1.5, -0.2]]])
    fB = 0.3

    out = add_binary_property_term(grid, sgl, bin_, fB)

    for j in range(2):
        for k in range(2):
            expected = np.log(fB * np.exp(bin_[0, j, k]) + (1 - fB) * np.exp(sgl[0, j]))
            assert np.isclose(out[0, j, k], expected)


@pytest.mark.parametrize("fB, source", [(0.0, "sgl"), (1.0, "bin")])
def test_property_term_edge_fractions_pick_one_branch(fB, source):
    sgl = np.array([[-1.0, -2.0]])
    bin_ = np.array([[[-0.5, -3.0], [-1.5, -0.2]]])

    out = add_binary_property_term(np.zeros((1, 2, 2)), sgl, bin_, fB)

    expected = np.broadcast_to(sgl[:, :, None], bin_.shape) if source == "sgl" else bin_
    assert np.all(np.isfinite(out))
    assert np.allclose(out, expected)


def test_property_term_does_not_modify_the_input_grid():
    grid = np.zeros((1, 2, 2))

    add_binary_property_term(grid, np.zeros((1, 2)), np.zeros((1, 2, 2)), 0.5)

    assert np.all(grid == 0.0)
