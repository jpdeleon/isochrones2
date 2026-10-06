"""Regression tests for the cluster photometric likelihood.

The property-likelihood array ``lnlike_prop`` is produced by
``StarClusterModel.lnlike`` (isochrones/cluster.py) and consumed by
``calc_lnlike_grid`` (isochrones/cluster_utils.py).  Both document it as
``(Nstars, Neep)`` and the consumer indexes it as ``lnlike_prop[i, j]`` with
``i`` the star and ``j`` the EEP.  An earlier merge left the producer
allocating ``(Neep, Nstars)`` instead, which silently read the wrong
per-star likelihoods (Numba does no bounds checking, so nothing raised).

These tests use a synthetic interpolator and catalog so they need no MIST
model files and run offline.
"""

import numpy as np
import pandas as pd

from isochrones.cluster import StarClusterModel
from isochrones.cluster_utils import calc_lnlike_grid, integrate_over_eeps

MINEEP, MAXEEP = 300, 304
EEPS = np.arange(MINEEP, MAXEEP + 1)

NSTARS = 3
NBANDS = 2

# age, feh, distance, AV, alpha, gamma, fB
PARAMS = [9.0, 0.0, 100.0, 0.1, -2.3, 0.3, 0.3]


class FakeIC:
    """Minimal stand-in for an IsochroneInterpolator."""

    mineep = MINEEP
    maxeep = MAXEEP
    minmass = 0.2
    maxmass = 100.0

    def initial_mass(self, eeps, age, feh):
        return 0.8 + 0.1 * (np.asarray(eeps) - MINEEP)

    def interp_value(self, x, columns):
        return 1.0 + 0.01 * (np.asarray(x[0]) - MINEEP)

    def interp_mag(self, x, bands):
        eeps = np.asarray(x[0])
        mags = np.zeros((len(eeps), len(bands)))
        for b in range(len(bands)):
            mags[:, b] = 5.1 + 0.5 * b + 0.01 * (eeps - MINEEP)
        return None, None, None, mags

    def Teff(self, eeps, age, feh):
        return 5700.0 + 10.0 * (np.asarray(eeps) - MINEEP)


def _make_model():
    df = pd.DataFrame(
        {
            "Teff": [5710.0, 5750.0, 5690.0],
            "Teff_unc": [80.0, 90.0, 100.0],
            "g_mag": [5.0, 5.1, 5.2],
            "g_mag_unc": [0.01, 0.01, 0.01],
            "r_mag": [5.5, 5.6, 5.7],
            "r_mag_unc": [0.01, 0.01, 0.01],
        }
    )
    return StarClusterModel(
        FakeIC(),
        df,
        props=["Teff"],
        eep_bounds=(MINEEP, MAXEEP),
        mass_bounds=(0.2, 100.0),
    )


def test_lnlike_prop_layout_is_nstars_by_neep():
    """StarClusterModel.lnlike must agree with an independently built (Nstars, Neep) grid.

    Rebuilding lnlike_prop here with the documented layout pins both the
    allocation and the broadcast in cluster.py: a transposed producer gives a
    different answer here rather than failing loudly.
    """
    model = _make_model()
    age, feh, distance, AV, alpha, gamma, fB = PARAMS

    got = model.lnlike(PARAMS)
    assert np.isfinite(got)

    eeps = np.arange(MINEEP, MAXEEP + 1)
    model_masses = model.ic.initial_mass(eeps, age, feh)
    ok = np.isfinite(model_masses)
    model_masses = model_masses[ok]
    eeps = eeps[ok]

    dm_deeps = model.ic.interp_value([eeps, age, feh], ["dm_deep"])
    ln_dm_deeps = np.log(np.absolute(dm_deeps))
    _, _, _, model_mags = model.ic.interp_mag(
        [eeps, age, feh, distance, AV], model.stars.bands
    )

    Nstars = len(model.stars.df)
    Nbands = len(model.stars.bands)

    # The contract under test: (Nstars, Neep), with star as the slow axis.
    lnlike_prop = np.zeros((Nstars, len(eeps)))
    for prop, (vals, uncs) in model.stars.iter_props(values=True):
        model_vals = getattr(model.ic, prop)(eeps, age, feh)
        lnlike_prop += -0.5 * (vals[:, None] - model_vals[None, :]) ** 2 / uncs[:, None] ** 2

    vals_arr = np.empty((Nstars, Nbands))
    uncs_arr = np.empty((Nstars, Nbands))
    for i, (b, (vals, uncs)) in enumerate(model.stars.iter_bands(values=True)):
        vals_arr[:, i] = vals
        uncs_arr[:, i] = uncs

    mass_lo, mass_hi = model.bounds("mass")
    grid = calc_lnlike_grid(
        lnlike_prop,
        model_mags,
        Nbands,
        model_masses,
        ln_dm_deeps,
        eeps,
        vals_arr,
        uncs_arr,
        alpha,
        gamma,
        fB,
        mass_lo,
        mass_hi,
        model.minq,
    )
    ref = np.log(integrate_over_eeps(grid, eeps, Nstars)).sum()

    assert np.isfinite(ref)
    assert np.isclose(got, ref, rtol=1e-12)


def test_calc_lnlike_grid_reads_star_then_eep():
    """calc_lnlike_grid must add lnlike_prop[i, j] to the (star i, eep j) cell.

    Sentinel values are asymmetric under transpose, so a transposed
    ``lnlike_prop`` produces different numbers instead of a shape error.
    """
    Nstars, Neep, Nbands = 3, 5, 2
    sentinel = np.array([[100 * i + j for j in range(Neep)] for i in range(Nstars)])

    model_mags = 5.1 + 0.01 * np.arange(Neep)[:, None] + 0.5 * np.arange(Nbands)[None, :]
    masses = 0.8 + 0.1 * np.arange(Neep)
    ln_dm_deeps = np.zeros(Neep)
    eeps = np.arange(Neep)
    mag_values = np.array([[5.0, 5.5]] * Nstars)
    mag_uncs = np.array([[0.01, 0.01]] * Nstars)

    def run(prop):
        return calc_lnlike_grid(
            prop,
            model_mags,
            Nbands,
            masses,
            ln_dm_deeps,
            eeps,
            mag_values,
            mag_uncs,
            -2.3,
            0.3,
            0.3,
            0.2,
            100.0,
            0.1,
        )

    with_sentinel = run(sentinel)
    without_sentinel = run(np.zeros_like(sentinel))

    # k == j is always written by the (j, k) loops, so the diagonal always
    # carries the full lnlike_prop[i, j] contribution.
    for i in range(Nstars):
        for j in range(Neep):
            assert np.isclose(
                with_sentinel[i, j, j] - without_sentinel[i, j, j], sentinel[i, j]
            )
