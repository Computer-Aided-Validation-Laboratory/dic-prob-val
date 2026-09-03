"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Design of Experiments (DOE) sampling algorithms using SciPy QMC.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
from scipy.stats import qmc
from .param import ParamSpace, ParamValues


@dataclass(slots=True)
class DoeLatinHypercube:
    """Latin Hypercube sampling design configuration."""

    num_samples: int
    seed: int = 42
    scramble: bool = True


@dataclass(slots=True)
class DoeSobol:
    """Sobol quasi-random sequence design configuration."""

    num_samples: int
    seed: int = 42
    scramble: bool = True


@dataclass(slots=True)
class DoeRandom:
    """Standard uniform random Monte Carlo design configuration."""

    num_samples: int
    seed: int = 42


DoeConfig = Union[DoeLatinHypercube, DoeSobol, DoeRandom]


def build_doe(
    param_space: ParamSpace,
    config: DoeConfig,
    bounds: np.ndarray | None = None,
) -> ParamValues:
    """Build a Design of Experiments matrix spanning the parameter space.

    Parameters
    ----------
    param_space : ParamSpace
        Target parameter space.
    config : DoeConfig
        Sampling scheme configuration (LHS, Sobol, or Random).
    bounds : np.ndarray | None, optional
        Custom bounds of shape (num_params, 2), by default None.

    Returns
    -------
    ParamValues
        Generated parameter values matrix.
    """
    dim = param_space.get_num_params()
    if bounds is None:
        bounds = param_space.calc_bounds()

    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    if isinstance(config, DoeLatinHypercube):
        sampler = qmc.LatinHypercube(
            d=dim, scramble=config.scramble, seed=config.seed
        )
        unit_samples = sampler.random(n=config.num_samples)

    elif isinstance(config, DoeSobol):
        sampler = qmc.Sobol(d=dim, scramble=config.scramble, seed=config.seed)
        unit_samples = sampler.random(n=config.num_samples)

    elif isinstance(config, DoeRandom):
        rng = np.random.default_rng(config.seed)
        unit_samples = rng.uniform(0.0, 1.0, size=(config.num_samples, dim))

    else:
        raise TypeError(f"Unsupported DOE configuration type: {type(config)}")

    scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
    return ParamValues(names=param_space.get_names(), values=scaled_samples)
