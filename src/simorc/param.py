"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Physical parameter, parameter space, and evaluated parameter values.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .uncertainty import DistNormal, DistUniform, Interval, Uncertainty


@dataclass(slots=True)
class Param:
    """Physical parameter definition with optional uncertainty specification."""

    name: str
    nominal: float
    uncertainty: Uncertainty | None = None
    unit: str | None = None

    def calc_bounds(self, n_sigma: float = 3.0) -> tuple[float, float]:
        """Calculate the practical parameter bounds."""
        if self.uncertainty is None:
            return (self.nominal, self.nominal)

        if isinstance(self.uncertainty, Interval):
            return (self.uncertainty.lower, self.uncertainty.upper)

        if isinstance(self.uncertainty, DistUniform):
            lower_bound = (
                self.uncertainty.lower.lower
                if isinstance(self.uncertainty.lower, Interval)
                else float(self.uncertainty.lower)
            )
            upper_bound = (
                self.uncertainty.upper.upper
                if isinstance(self.uncertainty.upper, Interval)
                else float(self.uncertainty.upper)
            )
            return (lower_bound, upper_bound)

        if isinstance(self.uncertainty, DistNormal):
            mean_min = (
                self.uncertainty.mean.lower
                if isinstance(self.uncertainty.mean, Interval)
                else float(self.uncertainty.mean)
            )
            mean_max = (
                self.uncertainty.mean.upper
                if isinstance(self.uncertainty.mean, Interval)
                else float(self.uncertainty.mean)
            )
            std_max = (
                self.uncertainty.std.upper
                if isinstance(self.uncertainty.std, Interval)
                else float(self.uncertainty.std)
            )
            return (
                mean_min - n_sigma * std_max,
                mean_max + n_sigma * std_max,
            )

        return (self.nominal, self.nominal)


@dataclass(slots=True)
class ParamValues:
    """Deterministic matrix of parameter values across simulation samples."""

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim == 1:
            self.values = self.values.reshape(1, -1)
        if self.values.shape[1] != len(self.names):
            raise ValueError(
                f"Mismatch: names has {len(self.names)} entries but values has "
                f"{self.values.shape[1]} columns."
            )

    def get_num_samples(self) -> int:
        """Get the number of parameter sample rows."""
        return self.values.shape[0]

    def get_num_params(self) -> int:
        """Get the number of parameters."""
        return len(self.names)

    def extract_dict(self, sample_idx: int = 0) -> dict[str, float]:
        """Convert a specific sample row to a name-to-value dictionary."""
        return {
            name: float(self.values[sample_idx, ii])
            for ii, name in enumerate(self.names)
        }

    def get_param_value(self, name: str, sample_idx: int = 0) -> float:
        """Get the value of a named parameter for a given sample index."""
        col_idx = self.names.index(name)
        return float(self.values[sample_idx, col_idx])


@dataclass(slots=True)
class ParamSpace:
    """Collection of physical parameters defining the design space."""

    params: tuple[Param, ...]

    def __init__(self, params: Sequence[Param]) -> None:
        self.params = tuple(params)

    def get_names(self) -> tuple[str, ...]:
        """Get tuple of all parameter names."""
        return tuple(p.name for p in self.params)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return len(self.params)

    def calc_bounds(self, n_sigma: float = 3.0) -> np.ndarray:
        """Calculate array of bounds of shape (num_params, 2)."""
        bounds_list = [p.calc_bounds(n_sigma) for p in self.params]
        return np.array(bounds_list, dtype=np.float64)

    def get_nominal_values(self) -> np.ndarray:
        """Get vector of nominal parameter values."""
        return np.array([p.nominal for p in self.params], dtype=np.float64)

    def validate_param_values(self, values: ParamValues | np.ndarray) -> None:
        """Validate shape and bounds for parameter values."""
        arr = values.values if isinstance(values, ParamValues) else values
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} columns, got {arr.shape[1]}."
            )
