"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Uncertainty representations: Interval (epistemic), DistNormal, DistUniform.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy import stats


@dataclass(slots=True)
class Interval:
    """Epistemic interval representation with lower and upper bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"Interval lower bound ({self.lower}) cannot exceed upper "
                f"bound ({self.upper})."
            )
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError(
                f"Interval bounds must be finite numbers, got "
                f"[{self.lower}, {self.upper}]."
            )

    def calc_midpoint(self) -> float:
        """Calculate the midpoint of the interval."""
        return 0.5 * (self.lower + self.upper)

    def calc_width(self) -> float:
        """Calculate the total width of the interval."""
        return self.upper - self.lower

    def sample(
        self,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw samples uniformly spanning the epistemic interval."""
        if rng is None:
            rng = np.random.default_rng()
        if self.lower == self.upper:
            return np.full(num_samples, self.lower, dtype=np.float64)
        return rng.uniform(self.lower, self.upper, size=num_samples)


@dataclass(slots=True)
class DistNormal:
    """Normal distribution supporting pure aleatory or mixed uncertainty."""

    mean: float | Interval
    std: float | Interval

    def __post_init__(self) -> None:
        if isinstance(self.std, float | int) and self.std <= 0.0:
            raise ValueError(f"Standard deviation must be > 0, got {self.std}.")
        if isinstance(self.std, Interval) and self.std.lower <= 0.0:
            raise ValueError(
                f"Standard deviation interval lower bound must be > 0, got "
                f"{self.std.lower}."
            )

    def has_epistemic(self) -> bool:
        """Check if mean or standard deviation contains an interval."""
        return isinstance(self.mean, Interval) or isinstance(
            self.std, Interval
        )

    def resolve(
        self,
        mean_val: float | None = None,
        std_val: float | None = None,
    ) -> "DistNormal":
        """Resolve any interval parameters to specific scalar values."""
        resolved_mean = (
            mean_val
            if isinstance(self.mean, Interval)
            else float(self.mean)
        )
        resolved_std = (
            std_val if isinstance(self.std, Interval) else float(self.std)
        )
        if resolved_mean is None or resolved_std is None:
            raise ValueError("All interval values must be supplied to resolve.")
        return DistNormal(mean=resolved_mean, std=resolved_std)

    def sample_aleatory(
        self,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample from the resolved (pure aleatory) distribution."""
        if self.has_epistemic():
            raise RuntimeError(
                "Cannot sample aleatory from mixed distribution without "
                "resolving epistemic states first."
            )
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(
            loc=float(self.mean), scale=float(self.std), size=num_samples
        )

    def cdf(self, val: np.ndarray | float) -> np.ndarray:
        """Compute the cumulative distribution function."""
        if self.has_epistemic():
            raise RuntimeError("Resolve epistemic parameters before cdf().")
        return stats.norm.cdf(
            val, loc=float(self.mean), scale=float(self.std)
        )

    def ppf(self, quant: np.ndarray | float) -> np.ndarray:
        """Compute the percent point (quantile) function."""
        if self.has_epistemic():
            raise RuntimeError("Resolve epistemic parameters before ppf().")
        return stats.norm.ppf(
            quant, loc=float(self.mean), scale=float(self.std)
        )


@dataclass(slots=True)
class DistUniform:
    """Uniform distribution supporting pure aleatory or mixed uncertainty."""

    lower: float | Interval
    upper: float | Interval

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, float | int)
            and isinstance(self.upper, float | int)
            and self.lower >= self.upper
        ):
            raise ValueError(
                f"Lower bound ({self.lower}) must be < upper bound "
                f"({self.upper})."
            )

    def has_epistemic(self) -> bool:
        """Check if bounds contain an interval."""
        return isinstance(self.lower, Interval) or isinstance(
            self.upper, Interval
        )

    def resolve(
        self,
        lower_val: float | None = None,
        upper_val: float | None = None,
    ) -> "DistUniform":
        """Resolve any interval parameters to specific scalar values."""
        resolved_lower = (
            lower_val
            if isinstance(self.lower, Interval)
            else float(self.lower)
        )
        resolved_upper = (
            upper_val
            if isinstance(self.upper, Interval)
            else float(self.upper)
        )
        if resolved_lower is None or resolved_upper is None:
            raise ValueError("All interval values must be supplied to resolve.")
        return DistUniform(lower=resolved_lower, upper=resolved_upper)

    def sample_aleatory(
        self,
        num_samples: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample from the resolved (pure aleatory) distribution."""
        if self.has_epistemic():
            raise RuntimeError(
                "Cannot sample aleatory from mixed distribution without "
                "resolving epistemic states first."
            )
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(
            low=float(self.lower), high=float(self.upper), size=num_samples
        )

    def cdf(self, val: np.ndarray | float) -> np.ndarray:
        """Compute the cumulative distribution function."""
        if self.has_epistemic():
            raise RuntimeError("Resolve epistemic parameters before cdf().")
        scale = float(self.upper) - float(self.lower)
        return stats.uniform.cdf(val, loc=float(self.lower), scale=scale)

    def ppf(self, quant: np.ndarray | float) -> np.ndarray:
        """Compute the percent point (quantile) function."""
        if self.has_epistemic():
            raise RuntimeError("Resolve epistemic parameters before ppf().")
        scale = float(self.upper) - float(self.lower)
        return stats.uniform.ppf(quant, loc=float(self.lower), scale=scale)


Uncertainty = Union[Interval, DistNormal, DistUniform]
