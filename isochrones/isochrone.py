"""Base isochrone classes and utilities.

This module provides the base classes and factory function for loading
isochrone/interpolator objects for various stellar evolution models.
"""

from __future__ import annotations

from .config import on_rtd

import os
import re
import sys
import warnings
import itertools
import pickle
from typing import TYPE_CHECKING, Optional, List, Union, Any, Type

from .logger import getLogger

try:
    import holoviews as hv
except ImportError:
    getLogger().warning("Holoviews not imported. Some visualizations will not be available.")
    pass

if not on_rtd:
    import pandas as pd
    import numpy as np

    from scipy.interpolate import LinearNDInterpolator as interpnd
    from scipy.optimize import newton, minimize
    import numpy.random as rand
    import matplotlib.pyplot as plt
    from numba import jit

    from astropy import constants as const

    G = const.G.cgs.value
    MSUN = const.M_sun.cgs.value
    RSUN = const.R_sun.cgs.value

    from .models import ModelGridInterpolator

else:
    G = 6.67e-11
    MSUN = 1.99e33
    RSUN = 6.96e10

from .config import ISOCHRONES


def get_ichrone(
    models: Union[str, Type, "ModelGridInterpolator"],
    bands: Optional[List[str]] = None,
    default: bool = False,
    tracks: bool = False,
    basic: bool = False,
    **kwargs: Any
) -> "ModelGridInterpolator":
    """Get an Isochrone object by name or type with the specified bands.

    This is the main factory function for loading stellar model interpolators.
    It supports MIST (MESA Isochrones and Stellar Tracks) models and can
    return either isochrones or evolution tracks.

    Args:
        models: Either a string name ('mist'), a ModelGridInterpolator subclass,
            or an existing ModelGridInterpolator instance.
        bands: List of photometric band names for which to compute magnitudes.
            If None, uses default bands for the model type.
        default: If True, union of provided bands with default bands.
            Defaults to False.
        tracks: If True, return evolution tracks instead of isochrones.
            Defaults to False.
        basic: If True, return basic MIST isochrones (simpler interpolation).
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the isochrone constructor.

    Returns:
        ModelGridInterpolator: An isochrone or evolution track interpolator object.

    Raises:
        ValueError: If the models argument is not recognized.

    Examples:
        >>> from isochrones import get_ichrone
        >>> mist = get_ichrone('mist', bands=['G', 'BP', 'RP', 'J', 'H', 'K'])
        >>> tracks = get_ichrone('mist', tracks=True)
    """
    if not bands:
        bands = None

    if isinstance(models, ModelGridInterpolator):
        return models

    if isinstance(models, type):
        ichrone = models(bands)
    elif models == "mist":
        if tracks:
            from isochrones.mist import MIST_EvolutionTrack

            ichrone = MIST_EvolutionTrack(bands=bands, **kwargs)
        else:
            if basic:
                from isochrones.mist import MIST_BasicIsochrone

                ichrone = MIST_BasicIsochrone(bands=bands, **kwargs)
            else:
                from isochrones.mist import MIST_Isochrone

                ichrone = MIST_Isochrone(bands=bands, **kwargs)
    else:
        raise ValueError("Unknown stellar models: {}".format(models))
    return ichrone
