"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Epistemic-outer / aleatory-inner probabilistic sampling & P-box propagation.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.integrate import trapezoid
from .field import FieldLayout, restore_field_layout
from .param import ParamSpace
from .surrogate import ISurrogate, SurrogateField
from .uncertainty import DistNormal, DistUniform, Interval


@dataclass(slots=True)
class ProbConfig:
    """Probabilistic uncertainty propagation sampling configuration."""

    num_epistemic: int = 50
    num_aleatory: int = 500
    quantiles: tuple[float, ...] = (0.05, 0.50, 0.95)
    seed: int = 42


@dataclass(slots=True)
class PBox:
    """Probability Box (P-Box) defining epistemic bounds on the CDF."""

    eval_points: np.ndarray
    cdf_lower: np.ndarray
    cdf_upper: np.ndarray
    cdf_median: np.ndarray
    quantile_levels: np.ndarray
    quantile_lower: np.ndarray
    quantile_upper: np.ndarray

    def calc_area(self) -> float:
        """Calculate the total epistemic area between CDF bounds."""
        diff = self.cdf_upper - self.cdf_lower
        return float(trapezoid(diff, self.eval_points))


@dataclass(slots=True)
class ProbResult:
    """Outcome of scalar probabilistic uncertainty propagation."""

    output_names: tuple[str, ...]
    pboxes: dict[str, PBox]


@dataclass(slots=True)
class ProbFieldResult:
    """Outcome of spatial field probabilistic uncertainty propagation."""

    layout: FieldLayout
    quantile_levels: np.ndarray
    quantile_maps: np.ndarray
    mean_field: np.ndarray | None = None
    std_field: np.ndarray | None = None


def run_probabilistic(
    surrogate: ISurrogate,
    param_space: ParamSpace,
    config: ProbConfig | None = None,
) -> ProbResult:
    """Propagate mixed uncertainties through surrogate model.

    Parameters
    ----------
    surrogate : ISurrogate
        Trained surrogate model.
    param_space : ParamSpace
        Parameter space with uncertainty specifications.
    config : ProbConfig | None, optional
        Sampling configuration, by default ProbConfig().

    Returns
    -------
    ProbResult
        Computed probability boxes for each output QoI.
    """
    if config is None:
        config = ProbConfig()

    rng = np.random.default_rng(config.seed)
    params = param_space.params
    dim = len(params)

    # 1. Sample epistemic parameters across epistemic space
    epistemic_samples = []
    for _ in range(config.num_epistemic):
        ep_state = []
        for p in params:
            if p.uncertainty is None:
                ep_state.append(p.nominal)
            elif isinstance(p.uncertainty, Interval):
                ep_state.append(float(p.uncertainty.sample(1, rng=rng)[0]))
            elif isinstance(p.uncertainty, DistNormal):
                mean_val = (
                    float(p.uncertainty.mean.sample(1, rng=rng)[0])
                    if isinstance(p.uncertainty.mean, Interval)
                    else float(p.uncertainty.mean)
                )
                std_val = (
                    float(p.uncertainty.std.sample(1, rng=rng)[0])
                    if isinstance(p.uncertainty.std, Interval)
                    else float(p.uncertainty.std)
                )
                ep_state.append((mean_val, std_val))
            elif isinstance(p.uncertainty, DistUniform):
                low_val = (
                    float(p.uncertainty.lower.sample(1, rng=rng)[0])
                    if isinstance(p.uncertainty.lower, Interval)
                    else float(p.uncertainty.lower)
                )
                high_val = (
                    float(p.uncertainty.upper.sample(1, rng=rng)[0])
                    if isinstance(p.uncertainty.upper, Interval)
                    else float(p.uncertainty.upper)
                )
                ep_state.append((low_val, high_val))
        epistemic_samples.append(ep_state)

    # 2. For each epistemic state, sample aleatory conditional distributions
    output_names = getattr(
        surrogate, "output_names", ("output_0",)
    )
    num_outputs = len(output_names)

    # Collect conditional outputs per epistemic state: (n_ep, n_ale, n_out)
    cond_outputs = np.zeros(
        (config.num_epistemic, config.num_aleatory, num_outputs),
        dtype=np.float64,
    )

    for ep_idx, ep_state in enumerate(epistemic_samples):
        al_mat = np.zeros((config.num_aleatory, dim), dtype=np.float64)
        for p_idx, p in enumerate(params):
            if p.uncertainty is None:
                al_mat[:, p_idx] = p.nominal
            elif isinstance(p.uncertainty, Interval):
                al_mat[:, p_idx] = float(ep_state[p_idx])
            elif isinstance(p.uncertainty, DistNormal):
                m_v, s_v = ep_state[p_idx]
                dist = DistNormal(mean=m_v, std=s_v)
                al_mat[:, p_idx] = dist.sample_aleatory(
                    config.num_aleatory, rng=rng
                )
            elif isinstance(p.uncertainty, DistUniform):
                l_v, u_v = ep_state[p_idx]
                dist = DistUniform(lower=l_v, upper=u_v)
                al_mat[:, p_idx] = dist.sample_aleatory(
                    config.num_aleatory, rng=rng
                )

        pred_means, _ = surrogate.predict(al_mat)
        cond_outputs[ep_idx, :, :] = pred_means

    # 3. Construct P-Boxes for each output QoI
    pbox_dict = {}
    num_eval_pts = 200

    for out_idx, name in enumerate(output_names):
        y_all = cond_outputs[:, :, out_idx]
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        padding = 0.05 * (y_max - y_min + 1e-12)
        eval_x = np.linspace(y_min - padding, y_max + padding, num_eval_pts)

        cdf_mat = np.zeros(
            (config.num_epistemic, num_eval_pts), dtype=np.float64
        )
        q_levels = np.array(config.quantiles, dtype=np.float64)
        q_mat = np.zeros(
            (config.num_epistemic, len(q_levels)), dtype=np.float64
        )

        for ep_idx in range(config.num_epistemic):
            y_sorted = np.sort(y_all[ep_idx, :])
            # Empirical CDF
            cdf_vals = np.searchsorted(y_sorted, eval_x, side="right") / float(
                config.num_aleatory
            )
            cdf_mat[ep_idx, :] = cdf_vals
            q_mat[ep_idx, :] = np.quantile(y_sorted, q_levels)

        cdf_low = np.min(cdf_mat, axis=0)
        cdf_high = np.max(cdf_mat, axis=0)
        cdf_med = np.median(cdf_mat, axis=0)

        q_low = np.min(q_mat, axis=0)
        q_high = np.max(q_mat, axis=0)

        pbox_dict[name] = PBox(
            eval_points=eval_x,
            cdf_lower=cdf_low,
            cdf_upper=cdf_high,
            cdf_median=cdf_med,
            quantile_levels=q_levels,
            quantile_lower=q_low,
            quantile_upper=q_high,
        )

    return ProbResult(output_names=output_names, pboxes=pbox_dict)


def run_field_probabilistic(
    field_surrogate: SurrogateField,
    param_space: ParamSpace,
    config: ProbConfig | None = None,
) -> ProbFieldResult:
    """Propagate uncertainties to generate spatial quantile field maps.

    Parameters
    ----------
    field_surrogate : SurrogateField
        Trained field POD-GP surrogate model.
    param_space : ParamSpace
        Parameter space with uncertainty specifications.
    config : ProbConfig | None, optional
        Sampling configuration, by default ProbConfig().

    Returns
    -------
    ProbFieldResult
        Computed 2D spatial quantile field maps.
    """
    if config is None:
        config = ProbConfig()

    rng = np.random.default_rng(config.seed)
    params = param_space.params
    dim = len(params)
    total_samples = config.num_epistemic * config.num_aleatory

    samples = np.zeros((total_samples, dim), dtype=np.float64)
    for ii, p in enumerate(params):
        if p.uncertainty is None:
            samples[:, ii] = p.nominal
        elif isinstance(p.uncertainty, Interval):
            samples[:, ii] = p.uncertainty.sample(total_samples, rng=rng)
        elif isinstance(p.uncertainty, DistNormal):
            mean_v = (
                p.uncertainty.mean.calc_midpoint()
                if isinstance(p.uncertainty.mean, Interval)
                else float(p.uncertainty.mean)
            )
            std_v = (
                p.uncertainty.std.calc_midpoint()
                if isinstance(p.uncertainty.std, Interval)
                else float(p.uncertainty.std)
            )
            dist = DistNormal(mean=mean_v, std=std_v)
            samples[:, ii] = dist.sample_aleatory(total_samples, rng=rng)
        elif isinstance(p.uncertainty, DistUniform):
            low_v = (
                p.uncertainty.lower.calc_midpoint()
                if isinstance(p.uncertainty.lower, Interval)
                else float(p.uncertainty.lower)
            )
            high_v = (
                p.uncertainty.upper.calc_midpoint()
                if isinstance(p.uncertainty.upper, Interval)
                else float(p.uncertainty.upper)
            )
            dist = DistUniform(lower=low_v, upper=high_v)
            samples[:, ii] = dist.sample_aleatory(total_samples, rng=rng)

    field_recon, _ = field_surrogate.predict(samples)
    layout = field_surrogate.basis.layout

    q_levels = np.array(config.quantiles, dtype=np.float64)
    q_maps_list = []
    for q_val in q_levels:
        q_raw = np.quantile(field_recon, q_val, axis=0)
        q_maps_list.append(restore_field_layout(q_raw, layout))

    q_maps_arr = np.array(q_maps_list)
    f_mean = restore_field_layout(np.mean(field_recon, axis=0), layout)
    f_std = restore_field_layout(np.std(field_recon, axis=0), layout)

    return ProbFieldResult(
        layout=layout,
        quantile_levels=q_levels,
        quantile_maps=q_maps_arr,
        mean_field=f_mean,
        std_field=f_std,
    )
