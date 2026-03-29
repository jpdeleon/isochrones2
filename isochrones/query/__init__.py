"""Stellar catalog query utilities.

This subpackage provides classes for querying stellar catalogs to obtain
photometric and spectroscopic data for stars.

Classes:
    Query: Base class for catalog queries.
    EmptyQueryError: Error raised when query returns no results.
    WISE: Query class for WISE photometry.
    TwoMASS: Query class for 2MASS photometry.
    Tycho2: Query class for Tycho-2 catalog.
    Gaia: Query class for Gaia data.
"""

from .query import Query, EmptyQueryError
from .vizier import WISE, TwoMASS, Tycho2, Gaia
