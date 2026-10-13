"""P15 (#1): the minimum credible interval may be disconnected.

Valls-Gabaud (2016), Sec. 3: the MCI is "built in such a way that the
posterior PDF is always larger inside the MCI than outside ... may not be
connected".  ``summary.hpd`` returns one shortest interval, which for a
bimodal posterior spans the gap; ``summary.mci_intervals`` returns every
connected component.
"""

import numpy as np

from isochrones.summary import hpd, mci_intervals

MU, SIGMA = 3.0, 0.3


def _bimodal_samples(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-MU, SIGMA, n), rng.normal(MU, SIGMA, n)])


def _mixture_mass(intervals):
    """Probability of the 50/50 two-Gaussian mixture inside the intervals."""
    from scipy.stats import norm

    total = 0.0
    for lo, hi in intervals:
        for m in (-MU, MU):
            total += 0.5 * (norm.cdf(hi, m, SIGMA) - norm.cdf(lo, m, SIGMA))
    return total


def test_bimodal_pdf_gives_two_intervals_separated_by_the_gap():
    intervals = mci_intervals(_bimodal_samples(), alpha=0.9)

    assert len(intervals) == 2
    (lo1, hi1), (lo2, hi2) = intervals
    assert lo1 < -MU < hi1 < 0.0 < lo2 < MU < hi2


def test_bimodal_intervals_enclose_alpha_of_the_true_posterior():
    intervals = mci_intervals(_bimodal_samples(), alpha=0.9)

    assert np.isclose(_mixture_mass(intervals), 0.9, atol=0.02)


def test_hpd_single_interval_spans_the_gap_that_mci_intervals_excludes():
    x = _bimodal_samples()
    lo, hi = hpd(x, alpha=0.9)

    assert lo < 0.0 < hi  # contains the low-density gap
    assert len(mci_intervals(x, alpha=0.9)) == 2


def test_unimodal_pdf_gives_one_interval_matching_the_analytic_hpd():
    x = np.random.default_rng(1).normal(0.0, 1.0, 20000)

    (lo, hi), = mci_intervals(x, alpha=0.68)

    half_width = 0.9944578832  # N(0, 1): 68% HPD is (-z, z), z = Phi^-1(0.84)
    assert np.isclose(lo, -half_width, atol=0.06)
    assert np.isclose(hi, half_width, atol=0.06)


def test_intervals_are_ordered_and_disjoint():
    intervals = mci_intervals(_bimodal_samples(), alpha=0.95)

    flat = [v for iv in intervals for v in iv]
    assert flat == sorted(flat)


def test_degenerate_and_empty_samples():
    assert mci_intervals(np.array([2.0, 2.0, 2.0])) == [(2.0, 2.0)]
    assert mci_intervals(np.array([np.nan, np.inf])) == []
