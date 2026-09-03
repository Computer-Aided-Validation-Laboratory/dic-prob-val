"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Variance-based Sobol global sensitivity analysis and parameter down-selection.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.stats import qmc
from .param import ParamSpace
from .surrogate import ISurrogate


@dataclass(slots=True)
class SensitivityResult:
    """Sobol global sensitivity indices (first-order S1 and total-order ST)."""

    param_names: tuple[str, ...]
    output_names: tuple[str, ...]
    s1: np.ndarray  # shape: (num_outputs, num_params)
    st: np.ndarray  # shape: (num_outputs, num_params)
    s1_conf: np.ndarray | None = None
    st_conf: np.ndarray | None = None

    def get_ranked_params(
        self, output_idx: int = 0, metric: str = "st"
    ) -> list[tuple[str, float]]:
        """Get parameters ranked by sensitivity index descending."""
        arr = (
            self.st[output_idx]
            if metric.lower() == "st"
            else self.s1[output_idx]
        )
        sorted_indices = np.argsort(arr)[::-1]
        return [(self.param_names[i], float(arr[i])) for i in sorted_indices]


def calc_sensitivity(
    surrogate: ISurrogate,
    param_space: ParamSpace,
    num_samples: int = 2048,
    seed: int = 42,
) -> SensitivityResult:
    """Calculate first-order (S1) and total-order (ST) Sobol indices.

    Parameters
    ----------
    surrogate : ISurrogate
        Trained surrogate model.
    param_space : ParamSpace
        Parameter space with bounds.
    num_samples : int, optional
        Base sample size N for Saltelli matrices, by default 2048.
    seed : int, optional
        Quasi-random sequence seed, by default 42.

    Returns
    -------
    SensitivityResult
        Calculated sensitivity indices.
    """
    dim = param_space.get_num_params()
    bounds = param_space.calc_bounds()
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    sampler = qmc.Sobol(d=2 * dim, scramble=True, seed=seed)
    raw_samples = sampler.random(n=num_samples)

    scaled_all = qmc.scale(
        raw_samples,
        np.concatenate([lower_bounds, lower_bounds]),
        np.concatenate([upper_bounds, upper_bounds]),
    )

    mat_a = scaled_all[:, :dim]
    mat_b = scaled_all[:, dim:]

    y_a, _ = surrogate.predict(mat_a)
    y_b, _ = surrogate.predict(mat_b)

    num_outputs = y_a.shape[1]
    s1_mat = np.zeros((num_outputs, dim), dtype=np.float64)
    st_mat = np.zeros((num_outputs, dim), dtype=np.float64)

    y_all = np.vstack([y_a, y_b])
    var_y = np.var(y_all, axis=0) + 1e-12

    for ii in range(dim):
        mat_ab_i = mat_a.copy()
        mat_ab_i[:, ii] = mat_b[:, ii]

        y_ab_i, _ = surrogate.predict(mat_ab_i)

        for out_idx in range(num_outputs):
            cov_term = np.mean(
                y_b[:, out_idx] * (y_ab_i[:, out_idx] - y_a[:, out_idx])
            )
            s1_val = cov_term / var_y[out_idx]
            diff_term = 0.5 * np.mean(
                (y_a[:, out_idx] - y_ab_i[:, out_idx]) ** 2
            )
            st_val = diff_term / var_y[out_idx]

            s1_mat[out_idx, ii] = max(0.0, float(s1_val))
            st_mat[out_idx, ii] = max(0.0, float(st_val))

    output_names = getattr(
        surrogate, "output_names", tuple(f"out_{i}" for i in range(num_outputs))
    )

    return SensitivityResult(
        param_names=param_space.get_names(),
        output_names=output_names,
        s1=s1_mat,
        st=st_mat,
    )


def select_params(
    sensitivity: SensitivityResult,
    threshold: float = 0.05,
    metric: str = "st",
) -> tuple[str, ...]:
    """Select significant parameters exceeding the sensitivity threshold.

    Parameters
    ----------
    sensitivity : SensitivityResult
        Sensitivity result containing indices.
    threshold : float, optional
        Cutoff threshold (e.g. 0.05 for 5% variance contribution), default 0.05.
    metric : str, optional
        Sensitivity metric to use ('st' or 's1'), by default 'st'.

    Returns
    -------
    tuple[str, ...]
        Names of influential parameters.
    """
    arr = sensitivity.st if metric.lower() == "st" else sensitivity.s1
    max_scores = np.max(arr, axis=0)
    selected = [
        sensitivity.param_names[i]
        for i, score in enumerate(max_scores)
        if score >= threshold
    ]
    return tuple(selected)
