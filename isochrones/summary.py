"""Summary statistics utilities for stellar model fitting results.

This module provides functions for computing summary statistics (quantiles)
from MCMC sampling results of stellar parameter estimation.
"""

from __future__ import annotations

import os
import sys
import re
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Callable
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd

from .starmodel import StarModel, BasicStarModel


def get_quantiles(
    name: str,
    rootdir: str = ".",
    columns: Optional[List[str]] = None,
    qs: Optional[List[float]] = None,
    modelname: str = "mist_starmodel_single",
    verbose: bool = False,
    raise_exceptions: bool = False,
) -> pd.DataFrame:
    """Get parameter quantiles for a star model.

    This function loads a saved StarModel from an HDF5 file and computes
    quantile statistics for the derived parameters.

    Args:
        name: Name of the star/folder.
        rootdir: Root directory containing the star folders.
        columns: List of parameter names to extract. Defaults to
            ['eep', 'mass', 'radius', 'age', 'feh', 'distance', 'AV'].
        qs: Quantile values to compute. Defaults to [0.05, 0.16, 0.5, 0.84, 0.95].
        modelname: Base name of the model file. Defaults to 'mist_starmodel_single'.
        verbose: If True, print loading errors. Defaults to False.
        raise_exceptions: If True, raise exceptions instead of returning empty DataFrame.

    Returns:
        DataFrame with quantile values for each parameter.

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

    q_df = mod.derived_samples[true_cols].quantile(qs)

    df = pd.DataFrame(index=[name])
    for c in true_cols:
        for q in qs:
            col = c + "_{:02.0f}".format(q * 100)
            df.loc[name, col] = q_df.loc[q, c]

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
