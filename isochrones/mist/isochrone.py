from ..models import EvolutionTrackInterpolator, IsochroneInterpolator
from .models import MISTIsochroneGrid, MISTBasicIsochroneGrid, MISTEvolutionTrackGrid
from .bc import MISTBolometricCorrectionGrid


class MIST_Isochrone(IsochroneInterpolator):
    """MESA Isochrones and Stellar Tracks (MIST) full isochrone.

    This class provides interpolation of MIST stellar evolution models
    using isochrone parameters (EEP, age, [Fe/H], distance, A_V).

    The EEP (Equivalent Evolutionary Phase) replaces mass as the
    primary evolutionary parameter.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
    """

    grid_type = MISTIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_BasicIsochrone(IsochroneInterpolator):
    """MIST basic isochrone with simplified physics.

    Similar to MIST_Isochrone but with simplified atmospheric
    and mixing length treatment.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
    """

    grid_type = MISTBasicIsochroneGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_EvolutionTrack(EvolutionTrackInterpolator):
    """MIST evolution track interpolator.

    This class provides interpolation of MIST stellar evolution models
    using track parameters (mass, age, [Fe/H], distance, A_V).

    The evolution track parameterization uses mass directly rather
    than EEP.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


class MIST_BasicEvolutionTrack(EvolutionTrackInterpolator):
    """MIST basic evolution track with simplified physics.

    Similar to MIST_EvolutionTrack but with simplified atmospheric
    and mixing length treatment.

    Attributes
    ----------
    eep_bounds : tuple
        Valid range of EEP values (0, 1710).
    """

    grid_type = MISTEvolutionTrackGrid
    bc_type = MISTBolometricCorrectionGrid
    eep_bounds = (0, 1710)


MIST_Isochrone._track_type = MIST_EvolutionTrack
MIST_BasicIsochrone._track_type = MIST_BasicEvolutionTrack
MIST_EvolutionTrack._iso_type = MIST_Isochrone
MIST_BasicEvolutionTrack._iso_type = MIST_BasicIsochrone
