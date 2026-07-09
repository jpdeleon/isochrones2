"""Base query class for stellar catalog queries.

This module provides the base Query class for querying stellar catalogs
to obtain photometric and spectroscopic data.
"""

from __future__ import annotations

from typing import Union, Optional
from astropy.coordinates import SkyCoord
import astropy.units as u


class EmptyQueryError(ValueError):
    """Error raised when a catalog query returns no results."""
    pass


class Query:
    """Base class for stellar catalog queries.

    This class represents a query to a stellar catalog at a specific
    sky position with optional proper motion information.

    Attributes:
        ra: Right ascension in decimal degrees.
        dec: Declination in decimal degrees.
        pmra: Proper motion in RA direction (mas/yr).
        pmdec: Proper motion in Dec direction (mas/yr).
        epoch: Epoch of the coordinates (default: 2000.0).
        radius: Search radius for catalog queries.

    Example:
        >>> from isochrones import Query
        >>> q = Query(ra=83.633, dec=-5.391)
        >>> print(q)
    """

    def __init__(
        self,
        ra: float,
        dec: float,
        pmra: float = 0.0,
        pmdec: float = 0.0,
        epoch: float = 2000.0,
        radius: Union[float, u.Quantity] = 5 * u.arcsec
    ) -> None:
        """Initialize a Query.

        Args:
            ra: Right ascension in decimal degrees.
            dec: Declination in decimal degrees.
            pmra: Proper motion in RA (mas/yr). Defaults to 0.0.
            pmdec: Proper motion in Dec (mas/yr). Defaults to 0.0.
            epoch: Epoch of coordinates. Defaults to 2000.0.
            radius: Search radius. Can be a float (arcsec) or Quantity.
                Defaults to 5 arcsec.
        """
        self.ra = ra
        self.dec = dec
        self.pmra = pmra
        self.pmdec = pmdec
        self.epoch = epoch

        if type(radius) in [type(1), type(1.0)]:
            self.radius = radius * u.arcsec
        else:
            self.radius = radius

        self._coords: Optional[SkyCoord] = None

    def __str__(self) -> str:
        return "({0.ra}, {0.dec}), pm=({0.pmra}, {0.pmdec}), epoch={0.epoch}, radius={0.radius}".format(self)

    def __repr__(self) -> str:
        return "Query(ra={0.ra}, dec={0.dec}, pmra={0.pmra}, ".format(
            self
        ) + "pmdec={0.pmdec}, epoch={0.epoch}, radius={0.radius})".format(self)

    @property
    def coords(self) -> SkyCoord:
        """Get SkyCoord object for the query position."""
        if self._coords is None:
            self._coords = SkyCoord(self.ra, self.dec, unit="deg")
        return self._coords
