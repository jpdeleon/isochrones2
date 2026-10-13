"""P14 (#1): per-component ages and the coevality diagnostic."""

import numpy as np
import pytest

from isochrones.coevality import age_difference_samples, coeval_age_posterior, coevality_summary


def _normal(mu, sigma, n=20000, seed=0):
    return np.random.default_rng(seed).normal(mu, sigma, n)


def test_coeval_posterior_of_equal_gaussians_is_narrower_by_root_two():
    grid, pdf = coeval_age_posterior(_normal(9.0, 0.1, seed=1), _normal(9.0, 0.1, seed=2))

    mean = np.trapezoid(grid * pdf, grid)
    sd = np.sqrt(np.trapezoid((grid - mean) ** 2 * pdf, grid))
    assert np.isclose(mean, 9.0, atol=0.01)
    assert np.isclose(sd, 0.1 / np.sqrt(2.0), rtol=0.05)


def test_coeval_posterior_is_the_precision_weighted_mean():
    # N(9.0, 0.1) x N(9.3, 0.2): mean = (9.0/0.01 + 9.3/0.04) / (1/0.01 + 1/0.04) = 9.06
    grid, pdf = coeval_age_posterior(_normal(9.0, 0.1, seed=1), _normal(9.3, 0.2, seed=2))

    assert np.isclose(np.trapezoid(grid * pdf, grid), 9.06, atol=0.01)
    assert np.isclose(np.trapezoid(pdf, grid), 1.0)


def test_summary_consistent_with_coeval_components():
    s = coevality_summary(_normal(9.0, 0.1, seed=1), _normal(9.0, 0.1, seed=2), seed=0)

    assert s["zero_in_interval"] == 1.0
    assert s["overlap"] > 0.85
    assert abs(s["delta_median"]) < 0.01


def test_summary_flags_discrepant_components():
    s = coevality_summary(_normal(8.6, 0.05, seed=1), _normal(9.2, 0.05, seed=2), seed=0)

    assert s["zero_in_interval"] == 0.0
    assert s["overlap"] < 0.01
    assert s["delta_hi"] < 0.0
    assert np.isclose(s["delta_median"], -0.6, atol=0.02)


def test_difference_samples_are_reproducible_with_a_seed():
    a, b = _normal(9.0, 0.1, 500, seed=1), _normal(9.1, 0.1, 500, seed=2)

    assert np.array_equal(
        age_difference_samples(a, b, n=100, seed=5), age_difference_samples(a, b, n=100, seed=5)
    )


def test_non_overlapping_components_cannot_be_combined():
    with pytest.raises(ValueError):
        coeval_age_posterior(_normal(5.0, 0.01, seed=1), _normal(12.0, 0.01, seed=2))


@pytest.mark.parametrize("bad", [np.array([1.0]), np.array([2.0, 2.0, 2.0]), np.array([np.nan, np.nan])])
def test_degenerate_samples_are_rejected(bad):
    good = _normal(9.0, 0.1, 100)
    with pytest.raises(ValueError):
        coevality_summary(bad, good)


def test_alpha_is_validated():
    good = _normal(9.0, 0.1, 100)
    with pytest.raises(ValueError):
        coevality_summary(good, good, alpha=1.5)
