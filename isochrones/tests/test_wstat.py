import math

import numpy as np
import pytest
from scipy.special import gammaln

from isochrones.wstat import log_w, log_w_poisson


def test_log_w_single_bin():
    # one bin holding both counts: log(20!/(10! 10!))
    assert np.isclose(log_w([10], [10]), math.lgamma(21) - math.lgamma(11) - math.lgamma(11), rtol=1e-12)
    # bins cannot disagree within a single bin: no evidence -> W = 1
    assert log_w([0], [0]) == 0.0
    assert np.isclose(log_w([10], [0]), 0.0, rtol=1e-12)
    assert np.isclose(log_w([0], [10]), 0.0, rtol=1e-12)


def test_log_w_known_product():
    # (2!/(2!0!)) * (2!/(0!2!)) * (2!/(1!1!)) = 1 * 1 * 2 = 2
    assert np.isclose(log_w([2, 0, 1], [0, 2, 1]), math.log(2), rtol=1e-12)


def test_log_w_two_bins():
    # (2!/(1!1!))^2 = 4
    assert np.isclose(log_w([1, 1], [1, 1]), math.log(4), rtol=1e-12)


def test_log_w_zero_counts_everywhere():
    assert np.isclose(log_w([0, 0, 0], [0, 0, 0]), 0.0)


def test_log_w_matches_factorial_product():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 4, size=5)
    s = rng.integers(0, 4, size=5)
    expected = sum(
        math.log(math.factorial(mi + si) / (math.factorial(mi) * math.factorial(si)))
        for mi, si in zip(m, s)
    )
    assert np.isclose(log_w(m, s), expected)


def test_log_w_poisson_matches_factorial_formula():
    m = [1, 2]
    s = [2, 1]
    M, S, B = 3.0, 3.0, 2
    product = 1.0
    for mi, si in zip(m, s):
        product *= math.factorial(mi + si) / (math.factorial(mi) * math.factorial(si))
    expected = (
        math.exp(-M)
        * M**M
        * math.exp(-S)
        * S**S
        * math.factorial(B - 1)
        / math.factorial(int(M + S + B - 1))
        * product
    )
    assert np.isclose(log_w_poisson(m, s), math.log(expected), rtol=1e-12)


def test_log_w_poisson_equals_fixed_plus_poisson_norm():
    m = [0.5, 0.2, 1.0]
    s = [1, 0, 2]
    M, S, B = sum(m), sum(s), 3.0
    poisson_term = (
        -M
        + M * math.log(M)
        - S
        + S * math.log(S)
        + math.lgamma(B)
        - math.lgamma(M + S + B)
    )
    assert np.isclose(log_w_poisson(m, s), log_w(m, s) + poisson_term, rtol=1e-10)


def test_log_w_accepts_real_model_counts():
    m = np.array([0.4, 0.6])
    s = np.array([1, 2])
    expected = gammaln(2.4) - gammaln(1.4) - gammaln(2) + gammaln(3.6) - gammaln(1.6) - gammaln(3)
    assert np.isclose(log_w(m, s), expected, rtol=1e-12)


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        log_w([1, 2], [1])
    with pytest.raises(ValueError):
        log_w_poisson([1], [1, 2])


def test_empty_raises():
    with pytest.raises(ValueError):
        log_w([], [])
    with pytest.raises(ValueError):
        log_w_poisson([], [])