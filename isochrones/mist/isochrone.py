"""MIST isochrone and evolution track classes.

This module provides the MIST-specific isochrone and evolution track
classes that inherit from the base interpolator classes.
"""

from __future__ import annotations

from ..models import EvolutionTrackInterpolator, IsochroneInterpolator
from .models import MISTIsochroneGrid, MISTBasicIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid


class MIST_Isochrone(IsochroneInterpolator):
<<<<<<< HEAD
    """MESA Isochrones and Stellar Tracks (MIST) full isochrone.

    This class provides interpolation of MIST stellar evolution models
    using isochrone parameters (EEP, age, [Fe/H], distance, A_V).

    The EEP (Equivalent Evolutionary Phase) replaces mass as the
    primary evolutionary parameter.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
=======
    """MIST (MESA Isochrones and Stellar Tracks) isochrone interpolator.

    This class provides interpolation for MIST isochrones, allowing
    stellar properties to be computed at any point on the grid.

    Attributes:
        grid_type: The grid class (MISTIsochroneGrid).
        bc_type: The bolometric correction grid class.
        eep_bounds: Bounds for Equal Evolutionary Points.
>>>>>>> update
    """

    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_BasicIsochrone(IsochroneInterpolator):
<<<<<<< HEAD
    """MIST basic isochrone with simplified physics.

    Similar to MIST_Isochrone but with simplified atmospheric
    and mixing length treatment.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
=======
    """Basic MIST isochrone interpolator.

    A simplified version of the MIST isochrone interpolator.
>>>>>>> update
    """

    grid_type = MISTBasicIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_EvolutionTrack(EvolutionTrackInterpolator):
    """MIST evolution track interpolator.

<<<<<<< HEAD
    This class provides interpolation of MIST stellar evolution models
    using track parameters (mass, age, [Fe/H], distance, A_V).

    The evolution track parameterization uses mass directly rather
    than EEP.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
=======
    This class provides interpolation for MIST stellar evolution tracks,
    allowing properties to be computed at any point along the track.
>>>>>>> update
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


class MIST_BasicEvolutionTrack(EvolutionTrackInterpolator):
<<<<<<< HEAD
    """MIST basic evolution track with simplified physics.

    Similar to MIST_EvolutionTrack but with simplified atmospheric
    and mixing length treatment.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
=======
    """Basic MIST evolution track interpolator.

    A simplified version of the MIST evolution track interpolator.
>>>>>>> update
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds: tuple = (0, 1710)


MIST_Isochrone._track_type = MIST_EvolutionTrack
MIST_BasicIsochrone._track_type = MIST_BasicEvolutionTrack
MIST_EvolutionTrack._iso_type = MIST_Isochrone
MIST_BasicEvolutionTrack._iso_type = MIST_BasicIsochrone
