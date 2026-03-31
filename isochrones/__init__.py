"""Isochrone interpolation and stellar model fitting package.

This package provides tools for fitting stellar evolution models to
observational data using Bayesian inference methods.

Main components:
- Isochrone interpolation (MIST models)
- Stellar model fitting (single, binary, binary stars)
- Prior distributions for stellar parameters
- Cluster modeling and fitting

Typical usage::

    >>> from isochrones import get_ichrone, StarModel
    >>> ic = get_ichrone('mist', bands=['J', 'H', 'K'])
    >>> mod = StarModel(ic, J=(10.0, 0.05), H=(9.5, 0.05), K=(9.0, 0.05))
"""

from .version import __version__
from .isochrone import get_ichrone
from .starmodel import StarModel, SingleStarModel, BinaryStarModel, TripleStarModel
