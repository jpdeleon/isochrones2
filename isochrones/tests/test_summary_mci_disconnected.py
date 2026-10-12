"""P15 (#1): the minimum credible interval may be disconnected.

Valls-Gabaud (2016), Sec. 3: the MCI is "built in such a way that the
posterior PDF is always larger inside the MCI than outside ... may not be
connected".  ``summary.hpd`` returns the shortest single interval, which for a
bimodal posterior spans the gap and so includes low-density values while
leaving higher-density values outside.
"""

import numpy as np
import pytest

from isochrones.summary import hpd


def _mixture_pdf(x, mu=3.0, sigma=0.3):
    def g(v, m):
        return np.exp(-0.5 * ((v - m) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return 0.5 * g(x, -mu) + 0.5 * g(x, mu)


@pytest.mark.xfail(
    strict=True,
    reason="P15 (#1): hpd returns one shortest interval; the MCI of a bimodal PDF is two intervals",
)
def test_mci_density_is_larger_inside_than_outside_for_bimodal_pdf():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3.0, 0.3, 5000), rng.normal(3.0, 0.3, 5000)])

    lo, hi = hpd(x, alpha=0.9)

    grid = np.linspace(-5.0, 5.0, 4001)
    pdf = _mixture_pdf(grid)
    inside = (grid >= lo) & (grid <= hi)

    assert pdf[inside].min() >= pdf[~inside].max()
