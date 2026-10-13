"""Likelihood functions for stellar parameter inference.

This module provides JIT-compiled likelihood functions for computing the
log-likelihood of stellar models given observational data.

Functions:
    gauss_lnprob: Gaussian log-probability for a single observation.
    bivariate_gauss_lnprob: Correlated bivariate Gaussian log-probability.
    correlation_correction: Change in a pair of independent Gaussian terms
        when the observables are correlated.
    star_lnlike: Total log-likelihood for stellar model fitting.
"""

from __future__ import annotations

from isochrones.mags import interp_mag
from .utils import fast_addmags
import numba as nb
from math import pi, log, sqrt
from typing import Any

import numpy as np


LOG_ONE_OVER_ROOT_2PI = log(1.0 / sqrt(2 * pi))


@nb.jit(nopython=True)
def gauss_lnprob(val: float, unc: float, model_val: float) -> float:
    """Compute Gaussian log-probability.

    Args:
        val: Observed value.
        unc: Uncertainty (1-sigma).
        model_val: Model predicted value.

    Returns:
        Log-probability.
    """
    resid = val - model_val
    return LOG_ONE_OVER_ROOT_2PI - log(unc) - 0.5 * resid * resid / (unc * unc)


def _check_rho(rho: float) -> None:
    if not -1.0 < rho < 1.0:
        raise ValueError("correlation coefficient must satisfy -1 < rho < 1, got {}".format(rho))


def bivariate_gauss_lnprob(
    val_a: float,
    unc_a: float,
    model_a: float,
    val_b: float,
    unc_b: float,
    model_b: float,
    rho: float,
) -> float:
    """Log of the correlated bivariate Gaussian (Valls-Gabaud 2016, Eq. 14).

    With ``D = (value - model) / sigma`` for each observable,

        ln L = -[D_a^2 + D_b^2 - 2 rho D_a D_b] / [2 (1 - rho^2)]
               - ln(2 pi sigma_a sigma_b sqrt(1 - rho^2))

    Args:
        val_a, unc_a, model_a: Observed value, 1-sigma uncertainty and model
            value of the first observable.
        val_b, unc_b, model_b: Same for the second observable.
        rho: Correlation coefficient, ``cov(a, b) = rho sigma_a sigma_b``.

    Returns:
        Log-probability.

    Raises:
        ValueError: If ``rho`` is not strictly between -1 and 1.
    """
    _check_rho(rho)
    da = (val_a - model_a) / unc_a
    db = (val_b - model_b) / unc_b
    quad = (da * da + db * db - 2.0 * rho * da * db) / (2.0 * (1.0 - rho * rho))
    return float(-quad - log(2.0 * pi * unc_a * unc_b * sqrt(1.0 - rho * rho)))


def correlation_correction(scaled_resid_a: Any, scaled_resid_b: Any, rho: float) -> Any:
    """Additive change to ``-0.5 (D_a^2 + D_b^2)`` when ``rho != 0``.

    Add this to a log-likelihood that already contains the independent terms
    ``-0.5 D_a^2 - 0.5 D_b^2`` to make it the exponent of Eq. 14.  The
    normalisation ``sqrt(1 - rho^2)`` is constant for fixed ``rho`` and is
    left out, like the other constants in the cluster likelihood.

    Args:
        scaled_resid_a: ``(value - model) / sigma`` of the first observable.
        scaled_resid_b: Same for the second (broadcastable against the first).
        rho: Correlation coefficient.

    Returns:
        Array (or float) shaped like the broadcast residuals.

    Raises:
        ValueError: If ``rho`` is not strictly between -1 and 1.
    """
    _check_rho(rho)
    da = np.asarray(scaled_resid_a, dtype=float)
    db = np.asarray(scaled_resid_b, dtype=float)
    corr = -(da * da + db * db - 2.0 * rho * da * db) / (2.0 * (1.0 - rho * rho))
    return corr + 0.5 * (da * da + db * db)


@nb.jit(nopython=True)
def star_lnlike(
    pars: Any,
    index_order: Any,
    spec_vals: Any,
    spec_uncs: Any,
    mag_vals: Any,
    mag_uncs: Any,
    i_mags: Any,
    model_grid: Any,
    i_Teff: Any,
    i_logg: Any,
    i_feh: Any,
    i_Mbol: Any,
    model_ii0: Any,
    model_ii1: Any,
    model_ii2: Any,
    bc_grid: Any,
    bc_ii0: Any,
    bc_ii1: Any,
    bc_ii2: Any,
    bc_ii3: Any,
) -> float:
    """Compute log-likelihood for stellar model.

    This function computes the total log-likelihood combining spectroscopic
    constraints (Teff, logg, [Fe/H]) and photometric constraints (magnitudes).
    Supports single, binary, and triple star systems.

    Args:
        pars: Model parameters (mass, age, feh, distance, AV, and optionally
            secondary/tertiary masses).
        index_order: Index order for parameter mapping.
        spec_vals: Spectroscopic values [Teff, logg, feh].
        spec_uncs: Spectroscopic uncertainties.
        mag_vals: Magnitude values.
        mag_uncs: Magnitude uncertainties.
        i_mags: Indices for magnitude arrays.
        model_grid: Interpolated model grid.
        i_Teff: Teff index.
        i_logg: log(g) index.
        i_feh: [Fe/H] index.
        i_Mbol: Bolometric magnitude index.
        model_ii0: Model interpolation indices 0.
        model_ii1: Model interpolation indices 1.
        model_ii2: Model interpolation indices 2.
        bc_grid: Bolometric correction grid.
        bc_ii0: BC interpolation indices 0.
        bc_ii1: BC interpolation indices 1.
        bc_ii2: BC interpolation indices 2.
        bc_ii3: BC interpolation indices 3.

    Returns:
        Total log-likelihood.
    """
    n_pars = len(pars)
    has_binary = False
    has_triple = False
    if n_pars == 5:
        single_pars = [pars[0], pars[1], pars[2], pars[3], pars[4]]
    elif n_pars == 6:
        single_pars = [pars[0], pars[2], pars[3], pars[4], pars[5]]
        binary_pars = [pars[1], pars[2], pars[3], pars[4], pars[5]]
        has_binary = True
    elif n_pars == 7:
        single_pars = [pars[0], pars[3], pars[4], pars[5], pars[6]]
        binary_pars = [pars[1], pars[3], pars[4], pars[5], pars[6]]
        triple_pars = [pars[2], pars[3], pars[4], pars[5], pars[6]]
        has_binary = True
        has_triple = True

    Teff, logg, feh, mags = interp_mag(
        single_pars,
        index_order,
        model_grid,
        i_Teff,
        i_logg,
        i_feh,
        i_Mbol,
        model_ii0,
        model_ii1,
        model_ii2,
        bc_grid,
        i_mags,
        bc_ii0,
        bc_ii1,
        bc_ii2,
        bc_ii3,
    )

    if has_binary:
        _, _, _, mags_binary = interp_mag(
            binary_pars,
            index_order,
            model_grid,
            i_Teff,
            i_logg,
            i_feh,
            i_Mbol,
            model_ii0,
            model_ii1,
            model_ii2,
            bc_grid,
            i_mags,
            bc_ii0,
            bc_ii1,
            bc_ii2,
            bc_ii3,
        )

    if has_triple:
        _, _, _, mags_triple = interp_mag(
            triple_pars,
            index_order,
            model_grid,
            i_Teff,
            i_logg,
            i_feh,
            i_Mbol,
            model_ii0,
            model_ii1,
            model_ii2,
            bc_grid,
            i_mags,
            bc_ii0,
            bc_ii1,
            bc_ii2,
            bc_ii3,
        )

    if n_pars == 6:
        for i in range(len(mags)):
            mags[i] = fast_addmags([mags[i], mags_binary[i]])
    elif n_pars == 7:
        for i in range(len(mags)):
            mags[i] = fast_addmags([mags[i], mags_binary[i], mags_triple[i]])

    lnlike = 0.0

    val = spec_vals[0]
    unc = spec_uncs[0]
    if val == val:
        lnlike += gauss_lnprob(val, unc, Teff)

    val = spec_vals[1]
    unc = spec_uncs[1]
    if val == val:
        lnlike += gauss_lnprob(val, unc, logg)

    val = spec_vals[2]
    unc = spec_uncs[2]
    if val == val:
        lnlike += gauss_lnprob(val, unc, feh)

    for i in range(len(mag_vals)):
        val = mag_vals[i]
        unc = mag_uncs[i]
        lnlike += gauss_lnprob(val, unc, mags[i])

    return lnlike
