"""Utility functions for the isochrones package.

This module provides general-purpose utilities including:
- Magnitude arithmetic (adding magnitudes with errors)
- File downloading
- Numerical utilities (trapezoidal integration, polynomial evaluation)
- Coordinate transformations

Functions:
    band_pairs: Generate pairs of photometric bands for color calculations.
    download_file: Download a file from a URL to a local path.
    addmags: Add magnitudes or magnitudes with uncertainties.
    fast_addmags: Fast JIT-compiled magnitude addition.
    distance: Calculate angular separation between two positions.
    trapz: Trapezoidal integration.
    polyval: Evaluate a polynomial.
"""

from __future__ import print_function, division
from __future__ import annotations

import requests  # type: ignore[import-untyped]
import os
from typing import Union, Tuple, List, Optional

import numpy as np
import numba as nb
from math import log10

from .logger import getLogger
from .config import ISOCHRONES


def band_pairs(bands: List[str]) -> List[Tuple[str, str]]:
    """Generate pairs of photometric bands for color calculations.

    Creates pairs of bands where each band is paired with the last band
    in the list (typically used for calculating colors).

    Args:
        bands: List of photometric band names.

    Returns:
        List of tuples (band_i, last_band) for i in range(len(bands)-1).

    Examples:
        >>> band_pairs(['G', 'BP', 'RP', 'J', 'H', 'K'])
        [('G', 'K'), ('BP', 'K'), ('RP', 'K'), ('J', 'K'), ('H', 'K')]
    """
    return [(bands[i], bands[-1]) for i in range(len(bands) - 1)]


def download_file(url: str, path: Optional[str] = None, clobber: bool = False) -> str:
    """Download a file from a URL to a local path.

    Args:
        url: URL of the file to download.
        path: Local path to save the file. If None, uses the URL filename
            in the ISOCHRONES data directory. Defaults to None.
        clobber: If True, overwrite existing files. Defaults to False.

    Returns:
        str: Path to the downloaded file.

    Raises:
        requests.RequestException: If the download fails.
    """
    if path is None:
        local_filename = os.path.join(ISOCHRONES, url.split("/")[-1])
    else:
        local_filename = path

    if os.path.exists(local_filename) and not clobber:
        getLogger().info("{} exists; not downloading.".format(local_filename))
        return local_filename

    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return local_filename


def addmags(
    *mags: Union[float, Tuple[float, float]]
) -> Union[float, Tuple[float, float]]:
    """Add magnitudes or magnitudes with uncertainties.

    Performs magnitude addition using the inverse relationship between
    magnitude and flux. Supports both plain magnitudes and magnitude-error
    pairs.

    Args:
        *mags: Either a list of magnitude values (float) or a list of
            (magnitude, error) tuples.

    Returns:
        If input contains only magnitudes: total magnitude as float.
        If input contains magnitude-error pairs: tuple of (total_magnitude, total_error).

    Examples:
        >>> addmags(5.0, 4.0, 3.0)  # Just magnitudes
        >>> addmags((5.0, 0.1), (4.0, 0.2))  # With errors
    """
    tot = 0.0
    uncs: List[float] = []
    for mag in mags:
        if isinstance(mag, tuple):
            m, dm = mag
            f = 10 ** (-0.4 * m)
            tot += f
            unc = f * (1 - 10 ** (-0.4 * dm))
            uncs.append(unc)
        else:
            tot += 10 ** (-0.4 * mag)

    totmag = -2.5 * np.log10(tot)
    if len(uncs) > 0:
        f_unc = np.sqrt(np.array([u ** 2 for u in uncs]).sum())
        return totmag, -2.5 * np.log10(1 - f_unc / tot)
    else:
        return totmag


@nb.jit(nopython=True)
def fast_addmags(mags: np.ndarray) -> float:
    """Fast JIT-compiled magnitude addition.

    Args:
        mags: Array of magnitude values.

    Returns:
        float: Total magnitude from adding all input magnitudes.
    """
    tot = 0.0
    for mag in mags:
        tot += 10 ** (-0.4 * mag)
    return -2.5 * log10(tot)


def distance(
    pos0: Tuple[float, float], pos1: Tuple[float, float]
) -> float:
    """Calculate angular separation between two positions.

    Positions are defined by (separation, position angle) pairs in
    (r, PA) format, where r is angular separation in degrees and PA
    is position angle in degrees.

    Args:
        pos0: First position as (separation, PA) tuple in degrees.
        pos1: Second position as (separation, PA) tuple in degrees.

    Returns:
        float: Angular separation between the two positions.
    """
    r0, pa0 = pos0
    ra0 = r0 * np.sin(pa0 * np.pi / 180)
    dec0 = r0 * np.cos(pa0 * np.pi / 180)

    r1, pa1 = pos1
    ra1 = r1 * np.sin(pa1 * np.pi / 180)
    dec1 = r1 * np.cos(pa1 * np.pi / 180)

    dra = ra1 - ra0
    ddec = dec1 - dec0
    return np.sqrt(dra ** 2 + ddec ** 2)


@nb.jit(nopython=True)
def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration.

    Args:
        y: Function values at points in x.
        x: x-coordinates of the points.

    Returns:
        float: Approximate integral of y with respect to x.
    """
    n = len(y)
    tot = 0.0

    for i in range(n - 1):
        i2 = i + 1
        tot = tot + 0.5 * (y[i] + y[i2]) * (x[i2] - x[i])

    return tot


@nb.jit(nopython=True)
def polyval(p: np.ndarray, x: float) -> float:
    """Evaluate a polynomial.

    Evaluates p[0]*x^(N-1) + p[1]*x^(N-2) + ... + p[N-1] where N is
    the length of p.

    Args:
        p: Polynomial coefficients, highest power first.
        x: Value at which to evaluate the polynomial.

    Returns:
        float: Polynomial evaluated at x.
    """
    N = len(p)
    result = 0.0
    for i in range(N):
        result += p[i] * x ** (N - 1 - i)
    return result
