"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Dataset structures for surrogate model training and validation.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .param import ParamValues
from .result import SimResult


@dataclass(slots=True)
class TrainingData:
    """Surrogate dataset mapping parameter inputs to scalar outputs."""

    param_names: tuple[str, ...]
    output_names: tuple[str, ...]
    inputs: np.ndarray
    outputs: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim == 1:
            self.inputs = self.inputs.reshape(-1, 1)
        if self.outputs.ndim == 1:
            self.outputs = self.outputs.reshape(-1, 1)
        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError(
                f"Sample count mismatch: inputs has {self.inputs.shape[0]} "
                f"rows while outputs has {self.outputs.shape[0]} rows."
            )

    def get_num_samples(self) -> int:
        """Get the number of training samples."""
        return self.inputs.shape[0]

    def get_num_inputs(self) -> int:
        """Get the number of input parameters."""
        return len(self.param_names)

    def get_num_outputs(self) -> int:
        """Get the number of output quantities."""
        return len(self.output_names)


def build_training_data(
    param_values: ParamValues,
    sim_results: Sequence[SimResult],
    scalar_names: Sequence[str] | None = None,
) -> TrainingData:
    """Construct TrainingData matrix from parameters and results.

    Parameters
    ----------
    param_values : ParamValues
        Input parameters for each simulation run.
    sim_results : Sequence[SimResult]
        List of simulation outputs.
    scalar_names : Sequence[str] | None, optional
        Names of scalar outputs to extract, by default extracts all scalars.

    Returns
    -------
    TrainingData
        Structured training data object.
    """
    if len(sim_results) != param_values.get_num_samples():
        raise ValueError(
            f"Expected {param_values.get_num_samples()} results, got "
            f"{len(sim_results)}."
        )

    if scalar_names is None:
        scalar_names = sim_results[0].get_scalar_names()

    out_mat = np.zeros(
        (len(sim_results), len(scalar_names)), dtype=np.float64
    )
    for ii, res in enumerate(sim_results):
        for jj, name in enumerate(scalar_names):
            out_mat[ii, jj] = res.get_scalar_value(name)

    return TrainingData(
        param_names=param_values.names,
        output_names=tuple(scalar_names),
        inputs=param_values.values.copy(),
        outputs=out_mat,
    )


def split_training_data(
    data: TrainingData,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TrainingData, TrainingData]:
    """Split training dataset into training and validation sets.

    Parameters
    ----------
    data : TrainingData
        Source dataset.
    test_fraction : float, optional
        Fraction of samples for testing/validation, by default 0.2.
    seed : int, optional
        Random seed for shuffling, by default 42.

    Returns
    -------
    tuple[TrainingData, TrainingData]
        (train_data, val_data)
    """
    num_samples = data.get_num_samples()
    num_test = max(1, int(np.round(num_samples * test_fraction)))

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)
    test_indices = shuffled_indices[:num_test]
    train_indices = shuffled_indices[num_test:]

    train_data = TrainingData(
        param_names=data.param_names,
        output_names=data.output_names,
        inputs=data.inputs[train_indices],
        outputs=data.outputs[train_indices],
    )
    val_data = TrainingData(
        param_names=data.param_names,
        output_names=data.output_names,
        inputs=data.inputs[test_indices],
        outputs=data.outputs[test_indices],
    )
    return (train_data, val_data)
