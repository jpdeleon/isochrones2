<<<<<<< HEAD
"""MIST (MESA Isochrones and Stellar Tracks) isochrone models.

This module provides access to MIST stellar evolution models,
including:
- Full isochrones with bolometric corrections
- Basic isochrones (simpler physics)
- Evolution tracks

References
----------
MESA Isochrones and Stellar Tracks:
    http://waps.cfa.harvard.edu/MIST/
    (Choi et al. 2016, Dotter 2016)
=======
"""MIST (MESA Isochrones and Stellar Tracks) model implementations.

This subpackage provides classes for working with MIST stellar evolution
models, including isochrones, evolution tracks, and bolometric corrections.

Classes:
    MIST_Isochrone: Full MIST isochrone interpolator.
    MIST_BasicIsochrone: Basic MIST isochrone interpolator.
    MIST_EvolutionTrack: MIST evolution track interpolator.
    MIST_BasicEvolutionTrack: Basic MIST evolution track interpolator.
    MISTIsochroneGrid: MIST isochrone grid.
    MISTEvolutionTrackGrid: MIST evolution track grid.
>>>>>>> update
"""

from .isochrone import MIST_Isochrone, MIST_BasicIsochrone, MIST_EvolutionTrack, MIST_BasicEvolutionTrack
from .models import MISTIsochroneGrid, MISTEvolutionTrackGrid
