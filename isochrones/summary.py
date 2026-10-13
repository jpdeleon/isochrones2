"""Summary statistics utilities for stellar model fitting results.

This module provides functions for computing summary statistics from MCMC
sampling results of stellar parameter estimation.  Following
vallsgabaud2016, the quantum of interest is the (posterior) mode plus the
central credible interval (CCI, equal tails each holding ``(1-alpha)/2`` of
the posterior volume) and the minimum credible interval (MCI, the shortest
interval containing a fraction ``alpha`` of the posterior volume, built so
the posterior PDF is always larger inside than outside).
"""

from __future__ import annotations

import os
import sys
import re
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Callable, Tuple
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .starmodel import StarModel, BasicStarModel


def mode_of_samples(x: np.ndarray, grid_size: int = 2000) -> float:
    """Estimate the mode of a 1D sample via the peak of a KDE.

    The mode is the value of the parameter where the posterior PDF reaches
    a maximum, and is preferred over the median and the (often badly
    biased) mean as a point estimate (vallsgabaud2016).

    Args:
        x: 1D array of samples.
        grid_size: Number of points for the KDE evaluation grid.

    Returns:
        The estimated mode.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.ptp(x) == 0.0:
        return float(x[0])
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), grid_size)
    return float(grid[np.argmax(kde(grid))])


def central_credible_interval(x: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """Central credible interval (CCI).

    The intervals ``(-inf, low)`` and ``(high, +inf)`` each contain
    ``(1-alpha)/2`` of the posterior volume, so the CCI always contains the
    median.

    Args:
        x: 1D array of samples.
        alpha: Posterior fraction enclosed (default 0.95).

    Returns:
        ``(low, high)``.
    """
    tail = (1.0 - alpha) / 2.0
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return float(lo), float(hi)


def hpd(x: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    """Minimum (or highest posterior density) credible interval.

    The MCI is the shortest interval enclosing a fraction ``alpha`` of the
    samples, built so the posterior PDF is always larger inside than
    outside.  For a bounded/unimodal 1D posterior this coincides with the
    highest-density interval and always contains the mode.  For a
    multimodal posterior the MCI is disconnected and this single interval
    spans the gap between modes; use :func:`mci_intervals` to get every
    component.

    Args:
        x: 1D array of samples.
        alpha: Posterior fraction enclosed (default 0.95).

    Returns:
        ``(low, high)``.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n_alpha = max(1, int(np.ceil(alpha * x.size)))
    if n_alpha >= x.size:
        raise ValueError("alpha too large for the number of samples")
    spans = x[n_alpha - 1 :] - x[: x.size - n_alpha + 1]
    i = int(np.argmin(spans))
    return float(x[i]), float(x[i + n_alpha - 1])


_CV_FACTOR_SCALES = (0.03, 0.06, 0.12, 0.25, 0.5, 1.0)
_CV_MAX_SAMPLES = 2000


def _cv_bandwidth_factor(x: np.ndarray) -> float:
    """Gaussian-KDE bandwidth factor chosen by leave-one-out likelihood.

    Scott's rule uses the overall standard deviation, which oversmooths
    well-separated modes into one broad bump and so merges the components of
    a disconnected MCI.  The factor is picked from a few multiples of Scott's
    value by maximising the leave-one-out log-likelihood on a subsample.
    """
    rng = np.random.default_rng(0)
    if x.size > _CV_MAX_SAMPLES:
        x = rng.choice(x, size=_CV_MAX_SAMPLES, replace=False)
    n = x.size
    std = x.std(ddof=1)
    scott = n ** (-1.0 / 5.0)
    best_factor, best_score = scott, -np.inf
    for scale in _CV_FACTOR_SCALES:
        factor = scale * scott
        h = factor * std
        kde = gaussian_kde(x, bw_method=factor)
        self_term = 1.0 / (np.sqrt(2.0 * np.pi) * h * n)
        loo = (kde(x) - self_term) * n / (n - 1)
        score = np.log(np.maximum(loo, 1e-300)).sum()
        if score > best_score:
            best_factor, best_score = factor, score
    return best_factor


def mci_intervals(x: np.ndarray, alpha: float = 0.95, grid_size: int = 2000) -> List[Tuple[float, float]]:
    """Minimum credible interval (MCI) as a list of disjoint intervals.

    The MCI is the set where the posterior PDF exceeds the level enclosing a
    fraction ``alpha`` of the posterior volume, so the PDF is larger inside
    than outside.  For a multimodal posterior this set is not connected
    (vallsgabaud2016, Sec. 3); :func:`hpd` instead returns the shortest single
    interval holding ``alpha`` of the samples, which spans the gaps.

    The PDF is a Gaussian KDE of the samples evaluated on a regular grid, so
    the interval edges are resolved to the grid spacing.

    Args:
        x: 1D array of samples.
        alpha: Posterior fraction enclosed (default 0.95).
        grid_size: Number of points for the KDE evaluation grid.

    Returns:
        List of ``(low, high)`` tuples in increasing order, one per connected
        component.  Empty if ``x`` has no finite samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    if np.ptp(x) == 0.0:
        return [(float(x[0]), float(x[0]))]

    kde = gaussian_kde(x, bw_method=_cv_bandwidth_factor(x))
    pad = 3.0 * kde.factor * x.std(ddof=1)
    grid = np.linspace(x.min() - pad, x.max() + pad, grid_size)
    pdf = kde(grid)
    mass = pdf * (grid[1] - grid[0])
    mass = mass / mass.sum()

    order = np.argsort(pdf)[::-1]
    n_keep = int(np.searchsorted(np.cumsum(mass[order]), alpha)) + 1
    inside = np.zeros(grid.size, dtype=bool)
    inside[order[:n_keep]] = True

    edges = np.diff(np.concatenate(([0], inside.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return [(float(grid[a]), float(grid[b])) for a, b in zip(starts, stops)]


def summary_statistics(x: np.ndarray, qs: Optional[List[float]] = None, alpha: float = 0.95) -> Dict[str, float]:
    """Compute quantiles plus mode, CCI, and MCI for a 1D sample.

    Args:
        x: 1D array of samples.
        qs: Quantiles to compute.  Defaults to
            ``[0.05, 0.16, 0.5, 0.84, 0.95]``.
        alpha: Coverage of the CCI and MCI (default 0.95).

    Returns:
        Dict keyed by column suffix, e.g. ``"50"`` (median), ``"mode"``,
        ``"cci_lo_95"``, ``"mci_hi_95"``.
    """
    if qs is None:
        qs = [0.05, 0.16, 0.5, 0.84, 0.95]
    pct = "{:02d}".format(int(round(alpha * 100)))
    stats: Dict[str, float] = {}
    # Drop non-finite samples, as pandas' Series.quantile did previously.
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        keys = ["{:02.0f}".format(q * 100) for q in qs]
        keys += ["mode"] + [k + pct for k in ("cci_lo_", "cci_hi_", "mci_lo_", "mci_hi_")]
        return {k: float("nan") for k in keys}
    for q in qs:
        stats["{:02.0f}".format(q * 100)] = float(np.quantile(x, q))
    stats["mode"] = mode_of_samples(x)
    lo, hi = central_credible_interval(x, alpha)
    stats["cci_lo_" + pct] = lo
    stats["cci_hi_" + pct] = hi
    lo, hi = hpd(x, alpha)
    stats["mci_lo_" + pct] = lo
    stats["mci_hi_" + pct] = hi
    return stats


def get_quantiles(
    name: str,
    rootdir: str = ".",
    columns: Optional[List[str]] = None,
    qs: Optional[List[float]] = None,
    modelname: str = "mist_starmodel_single",
    verbose: bool = False,
    raise_exceptions: bool = False,
    alpha: float = 0.95,
) -> pd.DataFrame:
    """Get parameter statistics for a star model.

    This function loads a saved StarModel from an HDF5 file and computes,
    for each derived parameter, the requested quantiles plus the mode,
    central credible interval (CCI), and minimum credible interval (MCI)
    of the marginalized posterior samples.

    Args:
        name: Name of the star/folder.
        rootdir: Root directory containing the star folders.
        columns: List of parameter names to extract. Defaults to
            ['eep', 'mass', 'radius', 'age', 'feh', 'distance', 'AV'].
        qs: Quantile values to compute. Defaults to [0.05, 0.16, 0.5, 0.84, 0.95].
        modelname: Base name of the model file. Defaults to 'mist_starmodel_single'.
        verbose: If True, print loading errors. Defaults to False.
        raise_exceptions: If True, raise exceptions instead of returning empty DataFrame.
        alpha: Coverage of the CCI and MCI columns (default 0.95).

    Returns:
        DataFrame with quantiles, mode, CCI, and MCI for each parameter.

    Example:
        >>> df = get_quantiles('Betelgeuse', rootdir='./results/')
    """
    if columns is None:
        columns = ["eep", "mass", "radius", "age", "feh", "distance", "AV"]
    if qs is None:
        qs = [0.05, 0.16, 0.5, 0.84, 0.95]

    modfile = os.path.join(rootdir, name, "{}.h5".format(modelname))
    try:
        mod = BasicStarModel.load_hdf(modfile)
    except Exception:
        if verbose:
            print("cannnot load starmodel! ({})".format(modfile))
        if raise_exceptions:
            raise
        return pd.DataFrame()

    true_cols: List[str] = []
    for c1 in mod.derived_samples.columns:
        for c2 in columns:
            if re.search(c2, c1):
                true_cols.append(c1)

    df = pd.DataFrame(index=[name])
    for c in true_cols:
        for suffix, value in summary_statistics(mod.derived_samples[c].values, qs, alpha).items():
            df.loc[name, c + "_" + suffix] = value

    return df


class quantile_worker:
    """Worker class for parallel quantile computation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with keyword arguments for get_quantiles."""
        self.kwargs = kwargs

    def __call__(self, name: str) -> pd.DataFrame:
        """Call get_quantiles with stored keyword arguments."""
        return get_quantiles(name, **self.kwargs)


def get_summary_df(
    names: Optional[List[str]] = None,
    pool: Optional[Pool] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Get summary statistics for multiple stars.

    This function computes quantile statistics for multiple stars in parallel
    using multiprocessing.

    Args:
        names: List of star names to process.
        pool: Optional multiprocessing pool for parallel execution.
        **kwargs: Keyword arguments passed to get_quantiles.

    Returns:
        DataFrame with quantiles for all stars.

    Example:
        >>> df = get_summary_df(['star1', 'star2', 'star3'], rootdir='./results/')
    """
    if names is None:
        names = []

    if pool is None:
        map_fn: Callable = map
    else:
        map_fn = pool.map

    worker = quantile_worker(**kwargs)
    dfs = map_fn(worker, names)

    df = pd.concat(dfs)
    return df
