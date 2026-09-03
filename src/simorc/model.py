"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Deterministic physics model interfaces and specifications.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from .param import ParamValues
from .result import SimResult


@dataclass(slots=True)
class OutputSpec:
    """Specification of a simulation output quantity to extract."""

    name: str
    output_type: str  # "scalar" or "field"
    target_name: str  # MOOSE postprocessor or variable name
    unit: str | None = None


class IModel(ABC):
    """Abstract interface for a deterministic simulation model."""

    @abstractmethod
    def run(
        self,
        params: ParamValues,
        work_dir: Path,
    ) -> SimResult:
        """Execute simulation deterministically for given parameter values.

        Parameters
        ----------
        params : ParamValues
            Evaluated parameter values for this simulation run.
        work_dir : Path
            Dedicated working directory for this run.

        Returns
        -------
        SimResult
            Extracted scalar and spatial field simulation outputs.
        """
        ...
