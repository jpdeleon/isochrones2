"""P6 (#1): CMD binning and the simulated-vs-real W comparison."""

import numpy as np
import pytest

from isochrones.wstat import bin_cmd, log_w_poisson, w_goodness_of_fit

COLOR_EDGES = np.array([0.0, 1.0, 2.0])
MAG_EDGES = np.array([10.0, 11.0, 12.0, 13.0])


def test_bin_cmd_counts_stars_per_cell_and_drops_outside():
    color = [0.5, 0.5, 1.5, 1.5, 1.5, 5.0]  # last star is outside the edges
    mag = [10.5, 10.6, 12.5, 12.4, 10.5, 11.5]

    counts = bin_cmd(color, mag, COLOR_EDGES, MAG_EDGES)

    assert counts.shape == (6,)
    assert counts.sum() == 5
    # row-major over (colour bin, mag bin): (0,0)=2, (1,0)=1, (1,2)=2
    assert counts[0] == 2
    assert counts[3] == 1
    assert counts[5] == 2


def test_bin_cmd_weights_scale_model_counts():
    counts = bin_cmd([0.5, 0.5], [10.5, 10.5], COLOR_EDGES, MAG_EDGES, weights=[0.25, 0.5])

    assert np.isclose(counts[0], 0.75)


def test_bin_cmd_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bin_cmd([0.5, 0.5], [10.5], COLOR_EDGES, MAG_EDGES)


def test_goodness_of_fit_matches_log_w_poisson_for_observed():
    m = np.array([12.0, 30.0, 25.0, 8.0, 2.0])
    s = np.array([11.0, 33.0, 24.0, 7.0, 3.0])

    result = w_goodness_of_fit(m, s, n_sim=50, seed=1)

    assert np.isclose(result.log_w_observed, log_w_poisson(m, s))
    assert result.log_w_simulated.shape == (50,)


def test_goodness_of_fit_data_drawn_from_model_is_typical():
    rng = np.random.default_rng(3)
    m = np.array([40.0, 60.0, 30.0, 20.0, 10.0, 5.0])
    s = rng.poisson(m).astype(float)

    result = w_goodness_of_fit(m, s, n_sim=500, seed=7)

    assert 0.05 < result.p_value < 1.0


def test_goodness_of_fit_flags_a_wrong_model():
    m = np.array([40.0, 60.0, 30.0, 20.0, 10.0, 5.0])
    s = np.array([5.0, 10.0, 20.0, 30.0, 60.0, 40.0])  # reversed shape

    result = w_goodness_of_fit(m, s, n_sim=500, seed=7)

    assert result.p_value < 0.01


def test_goodness_of_fit_is_reproducible_with_a_seed():
    m, s = np.array([10.0, 20.0, 5.0]), np.array([9.0, 22.0, 6.0])

    a = w_goodness_of_fit(m, s, n_sim=20, seed=11)
    b = w_goodness_of_fit(m, s, n_sim=20, seed=11)

    assert np.array_equal(a.log_w_simulated, b.log_w_simulated)


@pytest.mark.parametrize(
    "m, s, n_sim",
    [([1.0, -1.0], [1.0, 1.0], 10), ([1.0, 2.0], [1.0], 10), ([1.0], [1.0], 0), ([], [], 10)],
)
def test_goodness_of_fit_validates_input(m, s, n_sim):
    with pytest.raises(ValueError):
        w_goodness_of_fit(m, s, n_sim=n_sim)
