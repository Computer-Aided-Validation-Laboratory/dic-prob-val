"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Package data directory access helpers.
"""

from pathlib import Path


def get_data_path(name: str = "elastic2d") -> Path:
    """Get the path to a package data directory.

    Parameters
    ----------
    name : str, optional
        Name of the data subfolder, by default "elastic2d"

    Returns
    -------
    Path
        Absolute path to the data folder.
    """
    data_dir = Path(__file__).resolve().parent / name
    if not data_dir.exists():
        raise FileNotFoundError(f"Package data directory '{name}' not found.")
    return data_dir
