"""Utility functions for MIST model handling.

This module provides utility functions for working with MIST (MESA Isochrones
and Stellar Tracks) models, particularly for determining maximum EEP
(Equal Evolutionary Point) values.
"""

from __future__ import annotations


def default_max_eep(mass: float) -> int:
    """Get default maximum EEP for a given mass (MIST v1.2).

    This function returns the default maximum EEP value for a stellar
    mass when specific metallicity is not provided.

    Args:
        mass: Stellar mass in solar masses.

    Returns:
        Maximum EEP value.

    Examples:
        >>> default_max_eep(0.5)
        454
        >>> default_max_eep(1.0)
        1710
    """
    if mass < 0.6:
        return 454
    elif mass == 0.6:
        return 605
    elif mass == 0.65:
        return 808
    elif mass < 6.0:
        return 1710
    else:
        return 808


def max_eep(mass: float, feh: float) -> int:
    """Get maximum EEP for a given mass and metallicity (MIST v1.2).

    This function returns the maximum EEP (Equal Evolutionary Point)
    value for a star of given mass and metallicity. The maximum EEP
    represents the end of the evolution track.

    Args:
        mass: Stellar mass in solar masses.
        feh: Metallicity [Fe/H].

    Returns:
        Maximum EEP value.

    Examples:
        >>> max_eep(1.0, 0.0)
        1710
        >>> max_eep(0.5, -3.0)
        454
    """
    eep: int = 0
    if feh == -4.0:
        if mass < 0.6:
            eep = 454
        elif mass <= 0.94:
            eep = 631
        elif mass < 3.8:
            eep = 808
        elif mass <= 4.4:
            eep = 1409
        elif mass >= 18:
            eep = 631
    elif feh == -3.5:
        if mass == 0.65:
            eep = 631
        elif 0.65 < mass < 1.78:
            eep = 808
        elif mass == 1.78:
            eep = 1409
        elif 1.78 < mass <= 3.4:
            eep = 808
        elif mass >= 19:
            eep = 707
    elif feh == -3.0:
        if 0.7 <= mass <= 2.48:
            eep = 808
        elif 2.5 <= mass <= 4.4:
            eep = 1409
    elif feh == -2.5:
        if 0.7 <= mass <= 2.32:
            eep = 808
        elif 2.32 < mass <= 5.8:
            eep = 1409
    elif feh == 0.5:
        if 0.7 <= mass <= 0.75:
            eep = 808

    if eep == 0:
        return default_max_eep(mass)
    else:
        return eep
