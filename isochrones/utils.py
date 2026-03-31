"""Utility functions for the isochrones package.

This module provides various utility functions including:
- Magnitude manipulation (adding magnitudes, conversions)
- Distance calculations
- File download utilities
- Numerical utilities (interpolation, polynomial evaluation)
"""

import requests
import os

import numpy as np
import numba as nb
from math import log10

from .logger import getLogger


def band_pairs(bands):
    """Generate all adjacent band pairs for color calculations.

    Parameters
    ----------
    bands : list
        List of photometric band names.

    Returns
    -------
    list
        List of (band1, band2) tuples for all adjacent pairs.
    """
    return [(bands[i], bands[-1]) for i in range(len(bands) - 1)]


def download_file(url, path=None, clobber=False):
    """Download a file from a URL.

    Parameters
    ----------
    url : str
        URL to download from.
    path : str, optional
        Local path to save to. If not provided, uses current directory.
    clobber : bool, optional
        If True, overwrite existing file (default: False).

    Returns
    -------
    str
        Local path where file was saved.
    """
    if path is None:
        local_filename = url.split("/")[-1]
    else:
        local_filename = path

    if os.path.exists(local_filename) and not clobber:
        getLogger().info("{} exists; not downloading.".format(local_filename))
        return local_filename

    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return local_filename


def addmags(*mags):
    """Add magnitudes from multiple sources.

    Parameters
    ----------
    *mags : float or tuple
        Either a list of magnitudes or a list of (mag, err) pairs.

    Returns
    -------
    float or tuple
        If input is magnitudes only, returns total magnitude.
        If input includes uncertainties, returns (total_mag, total_err).
    """
    tot = 0
    uncs = []
    for mag in mags:
        if isinstance(mag, (tuple, list)):
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
def fast_addmags(mags):
    """Add magnitudes using numba JIT compilation.

    Parameters
    ----------
    mags : array_like
        List of magnitudes to add.

    Returns
    -------
    float
        Total magnitude.
    """
    tot = 0
    for mag in mags:
        tot += 10 ** (-0.4 * mag)
    return -2.5 * log10(tot)


def distance(pos0, pos1):
    """Calculate angular distance between two positions.

    Parameters
    ----------
    pos0 : tuple
        Position as (separation, PA) in arcsec and degrees.
    pos1 : tuple
        Position as (separation, PA) in arcsec and degrees.

    Returns
    -------
    float
        Angular distance between positions.
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
def trapz(y, x):
    n = len(y)
    tot = 0

    for i in range(n - 1):
        i2 = i + 1
        tot = tot + 0.5 * (y[i] + y[i2]) * (x[i2] - x[i])

    return tot


@nb.jit(nopython=True)
def polyval(p, x):
    N = len(p)
    result = 0
    for i in range(N):
        result += p[i] * x ** (N - 1 - i)
    return result
