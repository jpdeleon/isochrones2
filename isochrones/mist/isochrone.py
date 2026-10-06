"""MIST isochrone and evolution track classes.

This module provides the MIST-specific isochrone and evolution track
classes that inherit from the base interpolator classes.
"""

from __future__ import annotations

from ..models import EvolutionTrackInterpolator, IsochroneInterpolator
from .models import MISTIsochroneGrid, MISTBasicIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid


class MIST_Isochrone(IsochroneInterpolator):
    """MIST (MESA Isochrones and Stellar Tracks) isochrone interpolator.

    This class provides interpolation for MIST isochrones, allowing
    stellar properties to be computed at any point on the grid.

    Attributes:
        grid_type: The grid class (MISTIsochroneGrid).
        bc_type: The bolometric correction grid class.
        eep_bounds: Bounds for Equal Evolutionary Points.
    """

    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_BasicIsochrone(IsochroneInterpolator):
    """Basic MIST isochrone interpolator.

    A simplified version of the MIST isochrone interpolator.
    """

    grid_type = MISTBasicIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_EvolutionTrack(EvolutionTrackInterpolator):
    """MIST evolution track interpolator.

    This class provides interpolation for MIST stellar evolution tracks,
    allowing properties to be computed at any point along the track.
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_BasicEvolutionTrack(EvolutionTrackInterpolator):
    """Basic MIST evolution track interpolator.

    A simplified version of the MIST evolution track interpolator.
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


MIST_Isochrone._track_type = MIST_EvolutionTrack
MIST_BasicIsochrone._track_type = MIST_BasicEvolutionTrack
MIST_EvolutionTrack._iso_type = MIST_Isochrone
MIST_BasicEvolutionTrack._iso_type = MIST_BasicIsochrone
