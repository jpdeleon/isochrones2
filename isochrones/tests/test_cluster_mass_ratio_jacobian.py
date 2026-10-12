"""P11 (#1): the mass-ratio term needs the secondary's change of variables.

``calc_lnlike_grid`` integrates over the secondary's EEP (``EEP_k``) but
weights each cell with a density in ``q = m_k / m_j``.  Converting that density
to the EEP_k grid requires ``dq/dEEP_k = (dm_k/dEEP_k) / m_j``.  Without it the
single-star branch ``(1 - fB)`` is multiplied by ``int p(q) dEEP_k`` instead of
``int p(q) dq`` (= 1), so its weight depends on the EEP grid.

Setup chosen so the correct answer is analytic: flat IMF, flat q-density,
perfect photometric match for every EEP, ``fB = 0``.  Then

    L = int dEEP_j (dm/dEEP) phi * P_j,   P_j = (1 - m_0/m_j) / (1 - q_lo)

Offline: calls ``calc_lnlike_grid`` / ``integrate_over_eeps`` directly.
"""

import numpy as np
import pytest

from isochrones.cluster_utils import calc_lnlike_grid, integrate_over_eeps

M0 = 0.8
Q_LO = 0.05


def _single_branch_marginal(n, slope):
    eeps = np.arange(n, dtype=float)
    masses = M0 + slope * eeps
    mass_hi = masses[-1]
    grid = calc_lnlike_grid(
        np.zeros((1, n)),  # lnlike_prop
        np.full((n, 1), 10.0),  # model_mags: always matches the data
        1,
        masses,
        np.full(n, np.log(slope)),  # ln dm/dEEP
        eeps,
        np.array([[10.0]]),
        np.array([[0.05]]),
        0.0,  # alpha: flat IMF
        0.0,  # gamma: flat q density
        0.0,  # fB: single-star branch only
        M0,
        mass_hi,
        Q_LO,
    )
    got = integrate_over_eeps(grid, eeps, 1)[0]

    phi = 1.0 / (mass_hi - M0)
    p_j = (1.0 - M0 / masses) / (1.0 - Q_LO)
    ref = np.trapz(slope * phi * p_j, eeps)
    return got, ref


@pytest.mark.xfail(
    strict=True,
    reason="P11 (#1): secondary EEP integral lacks the dm_k/dEEP_k / m_j Jacobian",
)
def test_single_branch_marginal_matches_analytic_mass_ratio_integral():
    got, ref = _single_branch_marginal(n=61, slope=0.02)
    assert np.isclose(got, ref, rtol=2e-2)


@pytest.mark.xfail(
    strict=True,
    reason="P11 (#1): result depends on how finely the EEP grid samples the same mass range",
)
def test_single_branch_marginal_independent_of_eep_grid_resolution():
    # Same mass range 0.8 -> 1.6, sampled with 31 vs 61 EEP steps.
    coarse, _ = _single_branch_marginal(n=31, slope=0.8 / 30)
    fine, _ = _single_branch_marginal(n=61, slope=0.8 / 60)
    assert np.isclose(coarse, fine, rtol=2e-2)
