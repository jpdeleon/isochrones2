"""Bolometric correction grid classes.

This module provides the BolometricCorrectionGrid class for managing
bolometric correction tables used in stellar parameter estimation.

Bolometric corrections convert absolute bolometric magnitudes to
magnitudes in specific photometric bands.
"""

from __future__ import annotations

import pandas as pd
import os
import glob
from typing import Optional, Dict, List, Set, Tuple, Any

from .config import ISOCHRONES
from .grid import Grid


class BolometricCorrectionGrid(Grid):
    """Bolometric corrections in different bands as a function of stellar parameters.

    Stores bolometric corrections computed on a grid of stellar atmospheric
    parameters (Teff, logg, [Fe/H]), Av, and Rv. These corrections are used
    to convert between bolometric magnitude and magnitudes in specific
    photometric bands.

    Specific implementations of this grid should subclass this
    (e.g., `MISTBolometricCorrectionGrid`).

    Attributes:
        index_cols: Column names for the grid index (Teff, logg, [Fe/H], Av, Rv).
        name: Name identifier for the grid.
        is_full: Whether the grid is full (vs. sparse).
        bands: List of band names for which corrections are available.

    Example:
        >>> from isochrones.mist import MISTBolometricCorrectionGrid
        >>> bc = MISTBolometricCorrectionGrid(bands=['G', 'BP', 'RP'])
    """

    index_cols: Tuple[str, str, str, str, str] = ("Teff", "logg", "[Fe/H]", "Av", "Rv")
    name: Optional[str] = None
    is_full: bool = True
    default_bands: List[str] = []

    def __init__(self, bands: Optional[List[str]] = None) -> None:
        """Initialize the bolometric correction grid.

        Args:
            bands: List of band names. If None, uses default_bands.
        """
        self.bands = bands if bands is not None else list(self.default_bands)

        self._band_map: Optional[Dict[str, str]] = None
        self._phot_systems: Optional[Set[str]] = None

        self._df: Optional[pd.DataFrame] = None
        self._interp: Optional[Any] = None

    def get_band(self, *args: Any, **kwargs: Any) -> Tuple[str, str]:
        """Parse a band name into photometric system and band.

        Must be implemented by subclasses.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tuple of (photometric_system, band_name).
        """
        raise NotImplementedError

    def _make_band_map(self) -> None:
        """Create mapping between band names and photometric systems."""
        phot_systems: Set[str] = set()
        band_map: Dict[str, str] = {}
        for b in self.bands:
            phot, band = self.get_band(b)
            phot_systems.add(phot)
            band_map[b] = band
        self._band_map = band_map
        self._phot_systems = phot_systems

    @property
    def band_map(self) -> Dict[str, str]:
        """Mapping from band names to internal band names."""
        if self._band_map is None:
            self._make_band_map()
        return self._band_map  # type: ignore

    @property
    def phot_systems(self) -> Set[str]:
        """Set of photometric system names."""
        if self._phot_systems is None:
            self._make_band_map()
        return self._phot_systems  # type: ignore

    @property
    def datadir(self) -> str:
        """Directory where bolometric correction data is stored."""
        return os.path.join(ISOCHRONES, "BC", self.name or "")

    def get_filename(self, phot: str, feh: float) -> str:
        """Get the filename for a metallicity table.

        Args:
            phot: Photometric system name.
            feh: Metallicity [Fe/H].

        Returns:
            Path to the table file.
        """
        rootdir = self.datadir
        sign_str = "m" if feh < 0 else "p"
        filename = "feh{0}{1:03.0f}.{2}".format(sign_str, abs(feh) * 100, phot)
        return os.path.join(rootdir, filename)

    def parse_table(self, filename: str) -> pd.DataFrame:
        """Read a bolometric correction table into a DataFrame.

        Args:
            filename: Path to the table file.

        Returns:
            DataFrame with the table data.
        """
        with open(filename) as fin:
            for i, line in enumerate(fin):
                if i == 5:
                    names = line[1:].split()
                    break
        return pd.read_csv(
            filename, names=names, sep=r'\s+', comment="#", index_col=self.index_cols
        )

    def get_table(self, phot: str, feh: float) -> pd.DataFrame:
        """Get the bolometric correction table for a given metallicity.

        Args:
            phot: Photometric system name.
            feh: Metallicity [Fe/H].

        Returns:
            DataFrame with bolometric corrections.
        """
        return self.parse_table(self.get_filename(phot, feh))

    def get_hdf_filename(self, phot: str = "", **kwargs: Any) -> str:
        """Get the HDF5 cache filename for a photometric system.

        Args:
            phot: Photometric system name.

        Returns:
            Path to the HDF5 file.
        """
        return os.path.join(self.datadir, "{}.h5".format(phot))

    def get_tarball_url(self, phot: str = "", **kwargs: Any) -> str:
        """Get the URL for downloading the bolometric correction tables.

        Args:
            phot: Photometric system name.

        Returns:
            URL of the tarball.
        """
        url = "http://waps.cfa.harvard.edu/MIST/BC_tables/{}.txz".format(phot)
        return url

    def get_tarball_file(self, phot: str = "", **kwargs: Any) -> str:
        """Get the local path for the downloaded tarball.

        Args:
            phot: Photometric system name.

        Returns:
            Path to the tarball file.
        """
        return os.path.join(self.datadir, "{}.txz".format(phot))

    def get_df(self, orig: bool = False) -> pd.DataFrame:
        """Get the combined bolometric correction DataFrame.

        Returns:
            DataFrame with bolometric corrections for all bands.
        """
        df_all = pd.DataFrame()
        for phot in self.phot_systems:
            hdf_filename = self.get_hdf_filename(phot=phot)
            if not os.path.exists(hdf_filename):
                filenames = glob.glob(os.path.join(self.datadir, "*.{}".format(phot)))
                if not filenames:
                    self.extract_tarball(phot=phot)
                    filenames = glob.glob(os.path.join(self.datadir, "*.{}".format(phot)))
                df = pd.concat([self.parse_table(f) for f in filenames]).sort_index()
                df.to_hdf(hdf_filename, key="df")
            df = pd.read_hdf(hdf_filename, key="df")
            df_all = pd.concat([df_all, df], axis=1)

        df_all = df_all.rename(columns={v: k for k, v in self.band_map.items()})
        for col in list(df_all.columns):
            if col not in self.bands:
                del df_all[col]

        return df_all
