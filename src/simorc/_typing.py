"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Internal type definitions.
"""

from os import PathLike
from pathlib import Path
from typing import TypeAlias, Union
import numpy as np

PathType: TypeAlias = Union[str, Path, PathLike]
FloatArray: TypeAlias = np.ndarray
IntArray: TypeAlias = np.ndarray
