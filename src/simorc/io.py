"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Reversible file persistence for every workflow stage using compressed NPZ.
"""

from pathlib import Path
from typing import Any
import numpy as np
from .dataset import TrainingData
from .field import FieldLayout
from .fieldreduce import EFieldReductionCenter, FieldBasis
from .param import ParamValues
from .probabilistic import PBox, ProbFieldResult, ProbResult
from .result import EFieldLocation
from .sensitivity import SensitivityResult
from .surrogate import (
    ConfigGaussianProcess,
    SingleOutputGP,
    SurrogateGaussianProcess,
)


def save_param_values(param_values: ParamValues, file_path: Path) -> None:
    """Save ParamValues object to a compressed NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file_path,
        names=np.array(param_values.names),
        values=param_values.values,
    )


def load_param_values(file_path: Path) -> ParamValues:
    """Load ParamValues object from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    names = tuple(str(n) for n in data["names"])
    values = np.array(data["values"], dtype=np.float64)
    return ParamValues(names=names, values=values)


def save_training_data(data: TrainingData, file_path: Path) -> None:
    """Save TrainingData object to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file_path,
        param_names=np.array(data.param_names),
        output_names=np.array(data.output_names),
        inputs=data.inputs,
        outputs=data.outputs,
    )


def load_training_data(file_path: Path) -> TrainingData:
    """Load TrainingData object from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    param_names = tuple(str(n) for n in data["param_names"])
    output_names = tuple(str(n) for n in data["output_names"])
    return TrainingData(
        param_names=param_names,
        output_names=output_names,
        inputs=np.array(data["inputs"], dtype=np.float64),
        outputs=np.array(data["outputs"], dtype=np.float64),
    )


def save_field_basis(basis: FieldBasis, file_path: Path) -> None:
    """Save FieldBasis object to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "coords": basis.layout.coords,
        "components": np.array(basis.layout.components),
        "original_shape": np.array(basis.layout.original_shape),
        "location": basis.layout.location.value,
        "modes": basis.modes,
        "singular_values": basis.singular_values,
        "num_modes": basis.num_modes,
        "reduction_center": basis.reduction_center.value,
    }
    if basis.mean is not None:
        save_dict["mean"] = basis.mean
    np.savez_compressed(file_path, **save_dict)


def load_field_basis(file_path: Path) -> FieldBasis:
    """Load FieldBasis object from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    layout = FieldLayout(
        coords=np.array(data["coords"], dtype=np.float64),
        components=tuple(str(c) for c in data["components"]),
        original_shape=tuple(int(s) for s in data["original_shape"]),
        location=EFieldLocation(str(data["location"])),
    )
    mean = np.array(data["mean"], dtype=np.float64) if "mean" in data else None
    return FieldBasis(
        layout=layout,
        mean=mean,
        modes=np.array(data["modes"], dtype=np.float64),
        singular_values=np.array(data["singular_values"], dtype=np.float64),
        num_modes=int(data["num_modes"]),
        reduction_center=EFieldReductionCenter(str(data["reduction_center"])),
    )


def save_surrogate_gp(
    surrogate: SurrogateGaussianProcess, file_path: Path
) -> None:
    """Save SurrogateGaussianProcess model to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "param_names": np.array(surrogate.param_names),
        "output_names": np.array(surrogate.output_names),
        "kernel": surrogate.config.kernel,
        "noise_level": surrogate.config.noise_level,
        "normalize_inputs": surrogate.config.normalize_inputs,
        "normalize_outputs": surrogate.config.normalize_outputs,
        "n_restarts": surrogate.config.n_restarts,
        "seed": surrogate.config.seed,
        "num_models": len(surrogate.models),
    }
    for idx, gp in enumerate(surrogate.models):
        if gp.x_train is not None:
            save_dict[f"gp_{idx}_x_train"] = gp.x_train
        if gp.y_train is not None:
            save_dict[f"gp_{idx}_y_train"] = gp.y_train
        if gp.x_mean is not None:
            save_dict[f"gp_{idx}_x_mean"] = gp.x_mean
        if gp.x_std is not None:
            save_dict[f"gp_{idx}_x_std"] = gp.x_std
        save_dict[f"gp_{idx}_y_mean"] = gp.y_mean
        save_dict[f"gp_{idx}_y_std"] = gp.y_std
        if gp.length_scale is not None:
            save_dict[f"gp_{idx}_length_scale"] = gp.length_scale
        save_dict[f"gp_{idx}_sigma_f2"] = gp.sigma_f2
        if gp.k_inv is not None:
            save_dict[f"gp_{idx}_k_inv"] = gp.k_inv
        if gp.alpha is not None:
            save_dict[f"gp_{idx}_alpha"] = gp.alpha

    np.savez_compressed(file_path, **save_dict)


def load_surrogate_gp(file_path: Path) -> SurrogateGaussianProcess:
    """Load SurrogateGaussianProcess model from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    param_names = tuple(str(n) for n in data["param_names"])
    output_names = tuple(str(n) for n in data["output_names"])
    cfg = ConfigGaussianProcess(
        kernel=str(data["kernel"]),
        noise_level=float(data["noise_level"]),
        normalize_inputs=bool(data["normalize_inputs"]),
        normalize_outputs=bool(data["normalize_outputs"]),
        n_restarts=int(data["n_restarts"]),
        seed=int(data["seed"]),
    )
    num_models = int(data["num_models"])
    models = []
    for idx in range(num_models):
        gp = SingleOutputGP(
            config=cfg,
            x_train=np.array(data[f"gp_{idx}_x_train"], dtype=np.float64),
            y_train=np.array(data[f"gp_{idx}_y_train"], dtype=np.float64),
            x_mean=np.array(data[f"gp_{idx}_x_mean"], dtype=np.float64),
            x_std=np.array(data[f"gp_{idx}_x_std"], dtype=np.float64),
            y_mean=float(data[f"gp_{idx}_y_mean"]),
            y_std=float(data[f"gp_{idx}_y_std"]),
            length_scale=np.array(
                data[f"gp_{idx}_length_scale"], dtype=np.float64
            ),
            sigma_f2=float(data[f"gp_{idx}_sigma_f2"]),
            k_inv=np.array(data[f"gp_{idx}_k_inv"], dtype=np.float64),
            alpha=np.array(data[f"gp_{idx}_alpha"], dtype=np.float64),
        )
        models.append(gp)

    return SurrogateGaussianProcess(
        config=cfg,
        param_names=param_names,
        output_names=output_names,
        models=tuple(models),
    )


def save_sensitivity(sensitivity: SensitivityResult, file_path: Path) -> None:
    """Save SensitivityResult to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file_path,
        param_names=np.array(sensitivity.param_names),
        output_names=np.array(sensitivity.output_names),
        s1=sensitivity.s1,
        st=sensitivity.st,
    )


def load_sensitivity(file_path: Path) -> SensitivityResult:
    """Load SensitivityResult from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    return SensitivityResult(
        param_names=tuple(str(n) for n in data["param_names"]),
        output_names=tuple(str(n) for n in data["output_names"]),
        s1=np.array(data["s1"], dtype=np.float64),
        st=np.array(data["st"], dtype=np.float64),
    )


save_sensitivity_result = save_sensitivity
load_sensitivity_result = load_sensitivity


def save_prob_result(prob_result: ProbResult, file_path: Path) -> None:
    """Save ProbResult to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        "output_names": np.array(prob_result.output_names)
    }
    for name, pbox in prob_result.pboxes.items():
        save_dict[f"pbox_eval_points_{name}"] = pbox.eval_points
        save_dict[f"pbox_cdf_lower_{name}"] = pbox.cdf_lower
        save_dict[f"pbox_cdf_upper_{name}"] = pbox.cdf_upper
        save_dict[f"pbox_cdf_median_{name}"] = pbox.cdf_median
        save_dict[f"pbox_q_levels_{name}"] = pbox.quantile_levels
        save_dict[f"pbox_q_lower_{name}"] = pbox.quantile_lower
        save_dict[f"pbox_q_upper_{name}"] = pbox.quantile_upper
    np.savez_compressed(file_path, **save_dict)


def load_prob_result(file_path: Path) -> ProbResult:
    """Load ProbResult from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    output_names = tuple(str(n) for n in data["output_names"])
    pboxes = {}
    for name in output_names:
        pboxes[name] = PBox(
            eval_points=np.array(
                data[f"pbox_eval_points_{name}"], dtype=np.float64
            ),
            cdf_lower=np.array(
                data[f"pbox_cdf_lower_{name}"], dtype=np.float64
            ),
            cdf_upper=np.array(
                data[f"pbox_cdf_upper_{name}"], dtype=np.float64
            ),
            cdf_median=np.array(
                data[f"pbox_cdf_median_{name}"], dtype=np.float64
            ),
            quantile_levels=np.array(
                data[f"pbox_q_levels_{name}"], dtype=np.float64
            ),
            quantile_lower=np.array(
                data[f"pbox_q_lower_{name}"], dtype=np.float64
            ),
            quantile_upper=np.array(
                data[f"pbox_q_upper_{name}"], dtype=np.float64
            ),
        )
    return ProbResult(output_names=output_names, pboxes=pboxes)


def save_prob_field_result(
    prob_field: ProbFieldResult, file_path: Path
) -> None:
    """Save ProbFieldResult to an NPZ file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file_path,
        quantile_maps=prob_field.quantile_maps,
        quantile_levels=prob_field.quantile_levels,
        coords=prob_field.layout.coords,
        components=np.array(prob_field.layout.components),
        original_shape=np.array(prob_field.layout.original_shape),
        location=prob_field.layout.location.value,
        mean_field=prob_field.mean_field,
        std_field=prob_field.std_field,
    )


def load_prob_field_result(file_path: Path) -> ProbFieldResult:
    """Load ProbFieldResult from an NPZ file."""
    data = np.load(file_path, allow_pickle=True)
    layout = FieldLayout(
        coords=np.array(data["coords"], dtype=np.float64),
        components=tuple(str(c) for c in data["components"]),
        original_shape=tuple(int(s) for s in data["original_shape"]),
        location=EFieldLocation(str(data["location"])),
    )
    mean_f = (
        np.array(data["mean_field"], dtype=np.float64)
        if "mean_field" in data
        else None
    )
    std_f = (
        np.array(data["std_field"], dtype=np.float64)
        if "std_field" in data
        else None
    )
    return ProbFieldResult(
        layout=layout,
        quantile_levels=np.array(data["quantile_levels"], dtype=np.float64),
        quantile_maps=np.array(data["quantile_maps"], dtype=np.float64),
        mean_field=mean_f,
        std_field=std_f,
    )
