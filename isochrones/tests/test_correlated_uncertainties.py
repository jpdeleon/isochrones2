"""P13 (#1): correlated observables (Valls-Gabaud 2016, Eq. 14)."""

import math

import numpy as np
import pandas as pd
import pytest

from isochrones.cluster import StarClusterModel
from isochrones.likelihood import bivariate_gauss_lnprob, correlation_correction, gauss_lnprob
from isochrones.tests.test_cluster import MINEEP, MAXEEP, PARAMS, FakeIC


def test_bivariate_reduces_to_two_independent_gaussians_at_rho_zero():
    got = bivariate_gauss_lnprob(1.0, 0.5, 0.2, 3.0, 2.0, 2.5, 0.0)

    assert np.isclose(got, gauss_lnprob(1.0, 0.5, 0.2) + gauss_lnprob(3.0, 2.0, 2.5))


def test_bivariate_matches_covariance_matrix_form():
    va, ua, ma, vb, ub, mb, rho = 1.0, 0.5, 0.2, 3.0, 2.0, 2.5, 0.6
    cov = np.array([[ua**2, rho * ua * ub], [rho * ua * ub, ub**2]])
    d = np.array([va - ma, vb - mb])
    expected = -0.5 * d @ np.linalg.solve(cov, d) - math.log(2 * math.pi * math.sqrt(np.linalg.det(cov)))

    assert np.isclose(bivariate_gauss_lnprob(va, ua, ma, vb, ub, mb, rho), expected)


def test_positive_correlation_favours_residuals_of_the_same_sign():
    same = correlation_correction(1.0, 1.0, 0.8)
    opposite = correlation_correction(1.0, -1.0, 0.8)

    assert same > 0 > opposite


def test_correlation_correction_is_zero_at_rho_zero():
    assert np.allclose(correlation_correction(np.array([0.3, -1.2]), np.array([2.0, 0.1]), 0.0), 0.0)


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5])
def test_invalid_rho_is_rejected(rho):
    with pytest.raises(ValueError):
        correlation_correction(1.0, 1.0, rho)
    with pytest.raises(ValueError):
        bivariate_gauss_lnprob(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, rho)


class _TwoPropIC(FakeIC):
    def logg(self, eeps, age, feh):
        return 4.5 - 0.02 * (np.asarray(eeps) - MINEEP)


def _model(correlations=None):
    df = pd.DataFrame(
        {
            "Teff": [5710.0, 5750.0, 5690.0],
            "Teff_unc": [80.0, 90.0, 100.0],
            "logg": [4.5, 4.4, 4.52],
            "logg_unc": [0.05, 0.05, 0.05],
            "g_mag": [5.0, 5.1, 5.2],
            "g_mag_unc": [0.01, 0.01, 0.01],
            "r_mag": [5.5, 5.6, 5.7],
            "r_mag_unc": [0.01, 0.01, 0.01],
        }
    )
    return StarClusterModel(
        _TwoPropIC(),
        df,
        props=["Teff", "logg"],
        eep_bounds=(MINEEP, MAXEEP),
        mass_bounds=(0.2, 100.0),
        correlations=correlations,
    )


def test_cluster_lnlike_without_correlations_is_unchanged_by_the_option():
    assert _model().lnlike(PARAMS) == _model({}).lnlike(PARAMS)


def test_cluster_lnlike_changes_with_a_correlation():
    base = _model().lnlike(PARAMS)
    corr = _model({("Teff", "logg"): 0.7}).lnlike(PARAMS)

    assert np.isfinite(corr)
    assert not np.isclose(base, corr)


def test_cluster_correlation_with_rho_zero_matches_independent():
    assert np.isclose(_model().lnlike(PARAMS), _model({("Teff", "logg"): 0.0}).lnlike(PARAMS))


@pytest.mark.parametrize(
    "corr",
    [{("Teff", "nope"): 0.3}, {("Teff", "Teff"): 0.3}, {("Teff", "logg"): 1.0}],
)
def test_cluster_rejects_bad_correlations(corr):
    with pytest.raises(ValueError):
        _model(corr)
