"""Equal Evolutionary Point (EEP) functions for stellar evolution tracks.

This module provides functions for working with Equal Evolutionary Points (EEPs),
which are standardized points along stellar evolution tracks that enable
consistent interpolation between different stellar models.

The EEP approach defines specific evolutionary stages (pre-main sequence,
main sequence, subgiant, red giant, etc.) as fixed points, allowing
for more accurate interpolation of stellar properties.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from typing import Optional, List


@nb.jit(nopython=True)
def eep_fn(
    x: float,
    p5: float,
    p4: float,
    p3: float,
    p2: float,
    p1: float,
    p0: float,
    A: float,
    x0: float,
    tau: float,
    order: int = 5
) -> float:
    """Polynomial + exponential function to approximate EEP(age) for a given track.

    This function combines a polynomial with an exponential term to model
    the relationship between stellar age and the Equal Evolutionary Point (EEP).

    Args:
        x: Age value.
        p5-p0: Polynomial coefficients (highest to lowest power).
        A: Exponential amplitude.
        x0: Exponential center.
        tau: Exponential timescale.
        order: Polynomial order (1-5). Defaults to 5.

    Returns:
        EEP value at the given age.
    """
    if order < 5:
        p5 = 0
        if order < 4:
            p4 = 0
            if order < 3:
                p3 = 0
                if order < 2:
                    p2 = 0

    return p5 * x ** 5 + p4 * x ** 4 + p3 * x ** 3 + p2 * x ** 2 + p1 * x + p0 + A * np.exp((x - x0) / tau)


@nb.jit(nopython=True)
def eep_jac(
    x: np.ndarray,
    p5: float,
    p4: float,
    p3: float,
    p2: float,
    p1: float,
    p0: float,
    A: float,
    x0: float,
    tau: float,
    order: int = 5
) -> np.ndarray:
    """Jacobian of the EEP function with respect to parameters.

    Args:
        x: Array of age values.
        p5-p0: Polynomial coefficients.
        A: Exponential amplitude.
        x0: Exponential center.
        tau: Exponential timescale.
        order: Polynomial order. Defaults to 5.

    Returns:
        2D array of Jacobian values (n_points, 9 parameters).
    """
    if order < 5:
        p5 = 0
        if order < 4:
            p4 = 0
            if order < 3:
                p3 = 0
                if order < 2:
                    p2 = 0

    n = len(x)

    result = np.empty((n, 9), dtype=nb.float64)
    for i in range(n):
        xi = x[i]
        result[i, 0] = xi * xi * xi * xi * xi
        result[i, 1] = xi * xi * xi * xi
        result[i, 2] = xi * xi * xi
        result[i, 3] = xi * xi
        result[i, 4] = xi
        result[i, 5] = 0.0
        result[i, 6] = np.exp((xi - x0) / tau)
        result[i, 7] = -1.0 / tau * A * np.exp((xi - x0) / tau)
        result[i, 8] = -1.0 / tau ** 2 * (xi - x0) * A * np.exp((xi - x0) / tau)
    return result


def eep_fn_p0(ages: np.ndarray, eeps: np.ndarray, order: int = 5) -> List[float]:
    """Get initial parameter guesses for EEP function fitting.

    Fits a simple linear model to the early EEPs (EEP < 300) to get
    initial parameter estimates.

    Args:
        ages: Array of ages.
        eeps: Array of EEP values.
        order: Polynomial order. Defaults to 5.

    Returns:
        List of initial parameter guesses [p5, p4, p3, p2, p1, p0, A, x0, tau].
    """
    m = eeps < 300
    p1, p0 = np.polyfit(ages[m], eeps[m], 1)
    return [0, 0, 0, 0, p1, p0, 1, ages.max() - 0.3, 0.05]


def fit_section_poly(
    age: np.ndarray,
    eep: np.ndarray,
    a: float,
    b: float,
    order: int = 3
) -> np.ndarray:
    """Fit a polynomial to a section of the EEP-age relationship.

    Args:
        age: Array of ages.
        eep: Array of EEP values.
        a: Lower EEP bound.
        b: Upper EEP bound.
        order: Polynomial order. Defaults to 3.

    Returns:
        Polynomial coefficients.

    Raises:
        ValueError: If not enough points in the specified range.
    """
    m = (a < eep) & (eep < b)
    if m.sum() < order + 1:
        raise ValueError("Not enough points in the specified EEP range")
    return np.polyfit(age[m], eep[m], order)
