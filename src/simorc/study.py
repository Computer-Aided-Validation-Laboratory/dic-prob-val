"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Study workflow container coordinating parameters, runs, surrogates, and UQ.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import numpy as np
from .dataset import TrainingData, build_training_data
from .model import IModel
from .param import ParamSpace, ParamValues
from .probabilistic import ProbConfig, ProbResult, run_probabilistic
from .runner import IRunner, RunnerLocal, RunSet
from .sampling import DoeConfig, DoeLatinHypercube, build_doe
from .sensitivity import SensitivityResult, calc_sensitivity
from .surrogate import (
    ConfigGaussianProcess,
    ISurrogate,
    SurrogateGaussianProcess,
    build_surrogate,
)


@dataclass(slots=True)
class Study:
    """High-level simulation orchestration study container."""

    param_space: ParamSpace
    model: IModel | None = None
    runner: IRunner = field(default_factory=lambda: RunnerLocal(num_workers=1))
    work_dir: Path = field(default_factory=lambda: Path("./study_work"))
    doe: ParamValues | None = None
    runs: RunSet | None = None
    training_data: TrainingData | None = None
    surrogate: ISurrogate | None = None
    sensitivity: SensitivityResult | None = None
    prob_result: ProbResult | None = None

    def build_doe(
        self,
        config: DoeConfig | None = None,
        num_samples: int = 20,
    ) -> ParamValues:
        """Generate parameter sample design of experiments."""
        if config is None:
            config = DoeLatinHypercube(num_samples=num_samples)
        self.doe = build_doe(self.param_space, config)
        return self.doe

    def run_samples(self, work_dir: Path | None = None) -> RunSet:
        """Execute simulation runs for current DOE samples."""
        if self.doe is None:
            raise RuntimeError("Build DOE before running samples.")
        if self.model is None:
            raise RuntimeError(
                "A simulation model must be configured in Study."
            )

        target_dir = work_dir if work_dir is not None else self.work_dir
        self.runs = self.runner.run_samples(self.model, self.doe, target_dir)
        return self.runs

    def build_training_data(
        self, scalar_names: Sequence[str] | None = None
    ) -> TrainingData:
        """Extract training data from completed simulation runs."""
        if self.runs is None or self.doe is None:
            raise RuntimeError(
                "Execute simulation runs before building training data."
            )
        completed_results = self.runs.get_completed_results()
        self.training_data = build_training_data(
            self.doe, completed_results, scalar_names=scalar_names
        )
        return self.training_data

    def build_surrogate(
        self,
        config: ConfigGaussianProcess | None = None,
    ) -> ISurrogate:
        """Fit a Gaussian Process surrogate to the training data."""
        if self.training_data is None:
            self.build_training_data()
        if self.training_data is None:
            raise RuntimeError("Training data is required to fit surrogate.")
        self.surrogate = build_surrogate(self.training_data, config=config)
        return self.surrogate

    def calc_sensitivity(
        self, num_samples: int = 2048, seed: int = 42
    ) -> SensitivityResult:
        """Perform Sobol global sensitivity analysis using the surrogate."""
        if self.surrogate is None:
            self.build_surrogate()
        if self.surrogate is None:
            raise RuntimeError(
                "Surrogate is required for sensitivity analysis."
            )
        self.sensitivity = calc_sensitivity(
            self.surrogate, self.param_space, num_samples=num_samples, seed=seed
        )
        return self.sensitivity

    def run_probabilistic(
        self, config: ProbConfig | None = None
    ) -> ProbResult:
        """Propagate uncertainties to generate Probability Boxes."""
        if self.surrogate is None:
            self.build_surrogate()
        if self.surrogate is None:
            raise RuntimeError(
                "Surrogate is required for probabilistic propagation."
            )
        self.prob_result = run_probabilistic(
            self.surrogate, self.param_space, config=config
        )
        return self.prob_result
