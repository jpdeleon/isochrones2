"""Base grid classes for stellar model management.

This module provides the base `Grid` class for managing stellar evolution
model grids, including downloading, caching, and interpolation functionality.
"""

from __future__ import annotations

import pandas as pd
import os
import tarfile
from typing import Optional, Dict, Any, Tuple

from .utils import download_file
from .interp import DFInterpolator
from .logger import getLogger


class Grid:
    """Generalized model grid manager object.

    This is the base class for model grids, defined as a set of
    simulated or computed numbers as a function of a few regularly gridded
    parameters. The canonical example for which this was developed
    is grids of stellar evolution models. This object manages all of the following:

    * Downloading and unpacking raw grids from remote URLs
        (usually as many files in multiple folders),
    * Re-formatting full grid into a single dataframe with multi-level index,
      and writing grid to disk in binary (HDF5) format.
    * Defining a `DFInterpolator` object that allows for fast linear interpolation in this grid.

    `Grid` itself is the base class for two other more specific base classes implemented in
    **isochrones**, each of which is subclassed for specific cases: `StellarModelGrid`
    and `BolometricCorrectionGrid`.

    Arbitrary keywords may be passed, and will be stored in the `.kwargs` attribute.

    The key attributes are

    * `.df`: the grid data as a single dataframe, and
    * `.interp`: `DFInterpolator` object of the grid data.

    To subclass this, please see `StellarModelGrid` and `BolometricCorrectionGrid`
    as examples for the various methods that need to be implemented.

    Attributes:
        index_cols: Column names used for the grid index.
        is_full: Whether the grid is full (vs. sparse).
        bounds: Tuple of (min, max) bounds for grid parameters.
    """

    index_cols: Any = None
    is_full: bool = False
    bounds: tuple = tuple()

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Grid with optional keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments stored in `.kwargs` attribute.
        """
        self.kwargs: Dict[str, Any] = {}
        if hasattr(self, "default_kwargs"):
            self.kwargs = self.default_kwargs.copy()
        self.kwargs.update(kwargs)

        self._df: Optional[pd.DataFrame] = None
        self._df_orig: Optional[pd.DataFrame] = None
        self._interp: Optional[DFInterpolator] = None
        self._interp_orig: Optional[DFInterpolator] = None
        self._limits: Dict[str, Tuple[float, float]] = dict(self.bounds)

    def get_limits(self, prop: str) -> Tuple[float, float]:
        """Get the min/max limits for a given property.

        Args:
            prop: Name of the property/column.

        Returns:
            Tuple of (min, max) values for the property.
        """
        if prop not in self._limits:
            self._limits[prop] = (self.df[prop].min(), self.df[prop].max())
        return self._limits[prop]

    @property
    def datadir(self) -> str:
        """Directory where grid data is stored.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_hdf_filename(self, **kwargs: Any) -> str:
        """Get the HDF5 filename for storing the grid.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            str: Path to the HDF5 file.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def hdf_filename(self) -> str:
        """Path to the HDF5 file for this grid."""
        return self.get_hdf_filename()

    def get_tarball_url(self, **kwargs: Any) -> str:
        """Get the URL for downloading the grid tarball.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            str: URL of the tarball.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_tarball_file(self, **kwargs: Any) -> str:
        """Get the local path for the downloaded tarball.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            str: Local path to the tarball file.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def download_tarball(self, **kwargs: Any) -> None:
        """Download the grid tarball from the remote URL.

        Creates the data directory if it doesn't exist and downloads
        the tarball if not already present.

        Args:
            **kwargs: Additional keyword arguments passed to get_tarball_file.
        """
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        tarball = self.get_tarball_file(**kwargs)
        if not os.path.exists(tarball):
            url = self.get_tarball_url(**kwargs)
            getLogger().info("Downloading {}...".format(url))
            download_file(url, tarball)

    def extract_tarball(self, **kwargs: Any) -> None:
        """Extract the downloaded grid tarball.

        If the tarball doesn't exist, first downloads it. Handles corrupted
        tarballs by re-downloading.

        Args:
            **kwargs: Additional keyword arguments passed to get_tarball_file.
        """
        tarball = self.get_tarball_file(**kwargs)
        if not os.path.exists(tarball):
            self.download_tarball(**kwargs)

        try:
            with tarfile.open(tarball) as tar:
                getLogger().info("Extracting {}...".format(tarball))
                tar.extractall(self.datadir)
        except EOFError:
            getLogger().error("{} corrupted; deleting and re-downloading.".format(tarball))
            os.remove(tarball)
            self.extract_tarball(**kwargs)

    def read_hdf(self, orig: bool = False) -> pd.DataFrame:
        """Read the grid data from HDF5 format.

        Args:
            orig: If True, read the original (unprocessed) grid data.
                Defaults to False.

        Returns:
            pd.DataFrame: The grid data.
        """
        h5file = self.hdf_filename
        try:
            path = "orig" if orig else "df"
            df = pd.read_hdf(h5file, key=path)
        except (FileNotFoundError, KeyError):
            df = self.write_hdf(orig=orig)
        return df

    def write_hdf(self, orig: bool = False) -> pd.DataFrame:
        """Write the grid data to HDF5 format.

        Args:
            orig: If True, write the original (unprocessed) grid data.
                Defaults to False.

        Returns:
            pd.DataFrame: The grid data that was written.
        """
        df = self.get_df(orig=orig)
        h5file = self.hdf_filename
        path = "orig" if orig else "df"
        df.to_hdf(h5file, key=path)
        getLogger().info("{} written to {}.".format(path, h5file))
        return df

    @property
    def df(self) -> pd.DataFrame:
        """The grid data as a pandas DataFrame.

        Loads the data on first access, caching the result.
        """
        if self._df is None:
            self._df = self.get_df()
        return self._df

    @property
    def df_orig(self) -> pd.DataFrame:
        """The original (unprocessed) grid data as a pandas DataFrame.

        Loads the data on first access, caching the result.
        """
        if self._df_orig is None:
            self._df_orig = self.read_hdf(orig=True)
        return self._df_orig

    @property
    def interp(self) -> DFInterpolator:
        """The interpolation object for the grid data.

        Creates the interpolator on first access, caching the result.
        """
        if self._interp is None:
            filename = getattr(self, "interp_grid_npz_filename", None)
            self._interp = DFInterpolator(self.df, filename=filename, is_full=self.is_full)
        return self._interp

    @property
    def interp_orig(self) -> DFInterpolator:
        """The interpolation object for the original grid data.

        Creates the interpolator on first access, caching the result.
        """
        if self._interp_orig is None:
            filename = getattr(self, "interp_grid_orig_npz_filename", None)
            self._interp_orig = DFInterpolator(
                self.df_orig, filename=filename, is_full=self.is_full
            )
        return self._interp_orig

    def get_df(self, orig: bool = False) -> pd.DataFrame:
        """Get the grid data as a DataFrame.

        Args:
            orig: If True, return the original (unprocessed) grid data.
                Defaults to False.

        Returns:
            pd.DataFrame: The grid data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
