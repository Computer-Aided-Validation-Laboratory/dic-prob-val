"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Simulation results: scalar quantities and spatial field data.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import numpy as np


class EFieldLocation(Enum):
    """Spatial location of field data."""

    node = "node"
    element = "element"
    quadrature = "quadrature"


@dataclass(slots=True)
class ResultScalar:
    """Scalar simulation output value."""

    name: str
    value: float
    unit: str | None = None


@dataclass(slots=True)
class ResultField:
    """Spatial field simulation output data."""

    name: str
    values: np.ndarray
    coords: np.ndarray
    components: tuple[str, ...] = ("val",)
    location: EFieldLocation = EFieldLocation.node

    def __post_init__(self) -> None:
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if self.coords.shape[0] != self.values.shape[0]:
            raise ValueError(
                f"Field '{self.name}' has {self.coords.shape[0]} coordinates "
                f"but {self.values.shape[0]} value points."
            )

    def get_num_points(self) -> int:
        """Get number of spatial points."""
        return self.coords.shape[0]

    def get_num_components(self) -> int:
        """Get number of field components."""
        return len(self.components)


@dataclass(slots=True)
class SimResult:
    """Complete simulation result containing scalars and spatial fields."""

    scalars: tuple[ResultScalar, ...]
    fields: tuple[ResultField, ...]

    def __init__(
        self,
        scalars: Sequence[ResultScalar] = (),
        fields: Sequence[ResultField] = (),
    ) -> None:
        self.scalars = tuple(scalars)
        self.fields = tuple(fields)

    def get_scalar_names(self) -> tuple[str, ...]:
        """Get names of all scalar outputs."""
        return tuple(s.name for s in self.scalars)

    def get_field_names(self) -> tuple[str, ...]:
        """Get names of all field outputs."""
        return tuple(f.name for f in self.fields)

    def get_scalar_value(self, name: str) -> float:
        """Get value of named scalar output."""
        for s in self.scalars:
            if s.name == name:
                return s.value
        raise KeyError(
            f"Scalar output '{name}' not found in simulation result."
        )

    def get_field(self, name: str) -> ResultField:
        """Get field data by name."""
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(
            f"Field output '{name}' not found in simulation result."
        )
