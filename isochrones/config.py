"""Configuration settings for the isochrones package.

This module provides paths and settings for data directories used by
the isochrones package, including the main isochrones cache directory
and optional PolyChord installation path.

Environment Variables:
    ISOCHRONES: Path to the isochrones data directory (default: ~/.isochrones)
    POLYCHORD: Path to PolyChord installation (default: ~/PolyChord)
    READTHEDOCS: Set to 'True' when building docs on ReadTheDocs
"""

from __future__ import annotations

import os

on_rtd: bool = os.environ.get("READTHEDOCS") == "True"

ISOCHRONES: str = os.getenv("ISOCHRONES", os.path.expanduser(os.path.join("~", ".isochrones")))

POLYCHORD: str = os.getenv("POLYCHORD", os.path.expanduser(os.path.join("~", "PolyChord")))
