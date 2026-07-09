"""Extinction calculations for stellar observations.

This module provides functions for querying and calculating interstellar
extinction (dust extinction) along specific lines of sight using the
NED (NASA/IPAC Extragalactic Database) service.

The primary function queries the V-band extinction (A_V) at infinity
for a given sky position.
"""

from __future__ import annotations

import re
from typing import Optional

from .config import on_rtd

if not on_rtd:
    from astropy.coordinates import SkyCoord
    from six.moves import urllib  # type: ignore[import-untyped]


def get_AV_infinity(ra: float, dec: float, frame: str = "icrs") -> float:
    """Get the V-band extinction at infinity for a given line of sight.

    Queries the NED (NASA/IPAC Extragalactic Database) to obtain the
    total V-band extinction (A_V) for a given sky position. This represents
    the total extinction along the line of sight through the Galaxy.

    Args:
        ra: Right ascension in degrees.
        dec: Declination in degrees.
        frame: Coordinate frame of input coordinates. Options include
            'icrs' (default), 'galactic', 'fk5', etc.

    Returns:
        V-band extinction A_V in magnitudes.

    Raises:
        RuntimeError: If the extinction query fails.

    Example:
        >>> get_AV_infinity(83.633, -5.391)  # Coordinates of Betelgeuse
        0.56
    """
    coords = SkyCoord(ra, dec, unit="deg", frame=frame).transform_to("icrs")

    rah, ram, ras = coords.ra.hms
    decd, decm, decs = coords.dec.dms
    if decd > 0:
        decsign = "%2B"
    else:
        decsign = "%2D"
    url = (
        "http://ned.ipac.caltech.edu/cgi-bin/nph-calc?in_csys=Equatorial&in_equinox=J2000.0&obs_epoch=2010&lon="
        + "%i" % rah
        + "%3A"
        + "%i" % ram
        + "%3A"
        + "%05.2f" % ras
        + "&lat=%s" % decsign
        + "%i" % abs(decd)
        + "%3A"
        + "%i" % abs(decm)
        + "%3A"
        + "%05.2f" % abs(decs)
        + "&pa=0.0&out_csys=Equatorial&out_equinox=J2000.0"
    )

    AV: Optional[float] = None
    for line in urllib.request.urlopen(url).readlines():
        m = re.search(b"^Landolt V \\(0.54\\)\\s+(\\d+\\.\\d+)", line)
        if m:
            AV = float(m.group(1))
            break

    if AV is None:
        raise RuntimeError("AV query fails!  URL is {}".format(url))

    return AV
