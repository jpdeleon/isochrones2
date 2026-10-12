"""P10 (#1): the IMF must be normalised over the isochrone's own mass range.

Valls-Gabaud (2016), Eq. 18-19: ``K = int_{m_lim}^{m_top} rho(m) dm`` with
``m_lim`` / ``m_top`` the lower / upper mass of the isochrone being evaluated.
``StarClusterModel.lnlike`` instead normalises over the fixed grid-wide
``(ic.minmass, ic.maxmass)``, so the weight of an isochrone does not track how
much of the mass range actually exists at that age.

Offline: synthetic interpolator, no MIST files.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from isochrones import cluster
from isochrones.cluster import StarClusterModel

MINEEP, MAXEEP = 300, 304
TURNOFF_EEP = 302  # EEPs above this do not exist at the test age


class TurnoffIC:
    """Stand-in interpolator whose isochrone ends at TURNOFF_EEP."""

    mineep = MINEEP
    maxeep = MAXEEP
    minmass = 0.2
    maxmass = 100.0

    def initial_mass(self, eeps, age, feh):
        eeps = np.asarray(eeps, dtype=float)
        masses = 0.8 + 0.1 * (eeps - MINEEP)
        return np.where(eeps <= TURNOFF_EEP, masses, np.nan)

    def interp_value(self, x, columns):
        return 1.0 + 0.01 * (np.asarray(x[0]) - MINEEP)

    def interp_mag(self, x, bands):
        eeps = np.asarray(x[0])
        mags = np.zeros((len(eeps), len(bands)))
        for b in range(len(bands)):
            mags[:, b] = 5.1 + 0.5 * b + 0.01 * (eeps - MINEEP)
        return None, None, None, mags


def _model():
    df = pd.DataFrame(
        {
            "g_mag": [5.0, 5.1],
            "g_mag_unc": [0.01, 0.01],
            "r_mag": [5.5, 5.6],
            "r_mag_unc": [0.01, 0.01],
        }
    )
    return StarClusterModel(TurnoffIC(), df, props=[], eep_bounds=(MINEEP, MAXEEP))


@pytest.mark.xfail(
    strict=True,
    reason="P10 (#1): IMF normalised over grid-wide mass bounds, not [m_lim, m_top] of the isochrone",
)
def test_imf_is_normalised_over_the_isochrone_mass_range():
    model = _model()
    captured = {}
    real = cluster.calc_lnlike_grid

    def spy(*args):
        captured["mass_lo"], captured["mass_hi"] = args[11], args[12]
        return real(*args)

    with mock.patch.object(cluster, "calc_lnlike_grid", spy):
        model.lnlike([9.0, 0.0, 100.0, 0.1, -2.3, 0.3, 0.3])

    ic = TurnoffIC()
    finite = ic.initial_mass(np.arange(MINEEP, MAXEEP + 1), 9.0, 0.0)
    finite = finite[np.isfinite(finite)]

    assert np.isclose(captured["mass_lo"], finite.min())
    assert np.isclose(captured["mass_hi"], finite.max())
