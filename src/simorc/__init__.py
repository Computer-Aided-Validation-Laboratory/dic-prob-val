"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Public package interface.
"""

from pathlib import Path
from .data import get_data_path
from .dataset import (
    TrainingData,
    build_training_data,
    split_training_data,
)
from .field import (
    FieldLayout,
    build_grid_layout,
    flatten_field,
    restore_field_layout,
    standardise_field_grid,
)
from .fieldreduce import (
    EFieldReductionCenter,
    FieldBasis,
    FieldReductionValidation,
    calc_field_basis,
    calc_field_mode_errors,
)
from .io import (
    load_field_basis,
    load_param_values,
    load_prob_field_result,
    load_prob_result,
    load_sensitivity,
    load_sensitivity_result,
    load_surrogate_gp,
    load_training_data,
    save_field_basis,
    save_param_values,
    save_prob_field_result,
    save_prob_result,
    save_sensitivity,
    save_sensitivity_result,
    save_surrogate_gp,
    save_training_data,
)
from .model import (
    IModel,
    OutputSpec,
)
from .moose import ModelMoose
from .param import (
    Param,
    ParamSpace,
    ParamValues,
)
from .probabilistic import (
    PBox,
    ProbConfig,
    ProbFieldResult,
    ProbResult,
    run_field_probabilistic,
    run_probabilistic,
)
from .runner import (
    ERunStatus,
    IRunner,
    RunnerLocal,
    RunResult,
    RunSet,
)
from .sampling import (
    DoeConfig,
    DoeLatinHypercube,
    DoeRandom,
    DoeSobol,
    build_doe,
)
from .sensitivity import (
    SensitivityResult,
    calc_sensitivity,
    select_params,
)
from .study import Study
from .surrogate import (
    ConfigGaussianProcess,
    ISurrogate,
    SurrogateField,
    SurrogateGaussianProcess,
    SurrogateValidation,
    build_field_surrogate,
    build_surrogate,
)
from .uncertainty import (
    DistNormal,
    DistUniform,
    Interval,
    Uncertainty,
)


def build_example_param_space() -> ParamSpace:
    """Construct benchmark parameter space for the 2D hole-notch specimen."""
    params = [
        Param(
            name="EMod",
            nominal=195e3,
            uncertainty=DistNormal(mean=Interval(190e3, 200e3), std=5e3),
            unit="MPa",
        ),
        Param(
            name="PRatio",
            nominal=0.30,
            uncertainty=DistUniform(lower=0.28, upper=0.32),
            unit="-",
        ),
        Param(
            name="c1_bot",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        Param(
            name="c2_bot",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
        Param(
            name="c1_top",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        Param(
            name="c2_top",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
    ]
    return ParamSpace(params)


def build_example_model(
    executable: str | Path | None = None,
) -> ModelMoose:
    """Construct benchmark MOOSE model pointing to package data."""
    if executable is None:
        if Path("/home/lloydf/proteus/proteus-opt").exists():
            executable_path = Path("/home/lloydf/proteus/proteus-opt")
        else:
            executable_path = Path("proteus-opt")
    else:
        executable_path = Path(executable)

    data_dir = get_data_path("elastic2d")
    out_specs = (
        OutputSpec(
            name="react_y_top",
            output_type="scalar",
            target_name="react_y_top",
            unit="N",
        ),
        OutputSpec(
            name="stress_vm_max",
            output_type="scalar",
            target_name="stress_vm_max",
            unit="MPa",
        ),
        OutputSpec(
            name="disp_y_max",
            output_type="scalar",
            target_name="disp_y_max",
            unit="mm",
        ),
        OutputSpec(
            name="disp_y",
            output_type="field",
            target_name="disp_y",
            unit="mm",
        ),
        OutputSpec(
            name="vonmises_stress",
            output_type="field",
            target_name="vonmises_stress",
            unit="MPa",
        ),
    )
    common_includes = (
        data_dir / "common_load_time.i",
        data_dir / "common_elas_props.i",
        data_dir / "common_solver.i",
        data_dir / "common_outputs.i",
    )
    return ModelMoose(
        executable=executable_path,
        input_template=data_dir / "base.i",
        mesh_template=data_dir / "hole_notch_2d.msh",
        common_files=common_includes,
        output_specs=out_specs,
        sim_name="hole_notch_2d_elas_prob",
    )


def build_ludwik_param_space() -> ParamSpace:
    """Construct parameter space for the 2D Ludwik plasticity specimen."""
    params = [
        Param(
            name="EMod",
            nominal=195e3,
            uncertainty=DistNormal(mean=Interval(190e3, 200e3), std=4e3),
            unit="MPa",
        ),
        Param(
            name="PRatio",
            nominal=0.30,
            uncertainty=DistUniform(lower=0.28, upper=0.32),
            unit="-",
        ),
        Param(
            name="Yield",
            nominal=280.0,
            uncertainty=DistNormal(mean=Interval(270.0, 290.0), std=8.0),
            unit="MPa",
        ),
        Param(
            name="c1_bot",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        Param(
            name="c2_bot",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
        Param(
            name="c1_top",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        Param(
            name="c2_top",
            nominal=0.0,
            uncertainty=DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
    ]
    return ParamSpace(params)


def build_ludwik_model(
    executable: str | Path | None = None,
) -> ModelMoose:
    """Construct Ludwik plasticity MOOSE model pointing to package data."""
    if executable is None:
        if Path("/home/lloydf/proteus/proteus-opt").exists():
            executable_path = Path("/home/lloydf/proteus/proteus-opt")
        else:
            executable_path = Path("proteus-opt")
    else:
        executable_path = Path(executable)

    data_dir = get_data_path("ludwik2d")
    out_specs = (
        OutputSpec(
            name="react_y_top",
            output_type="scalar",
            target_name="react_y_top",
            unit="N",
        ),
        OutputSpec(
            name="stress_vm_max",
            output_type="scalar",
            target_name="stress_vm_max",
            unit="MPa",
        ),
        OutputSpec(
            name="plastic_strain_eq_max",
            output_type="scalar",
            target_name="plastic_strain_eq_max",
            unit="-",
        ),
        OutputSpec(
            name="disp_y_max",
            output_type="scalar",
            target_name="disp_y_max",
            unit="mm",
        ),
        OutputSpec(
            name="disp_y",
            output_type="field",
            target_name="disp_y",
            unit="mm",
        ),
        OutputSpec(
            name="vonmises_stress",
            output_type="field",
            target_name="vonmises_stress",
            unit="MPa",
        ),
        OutputSpec(
            name="effective_plastic_strain_out",
            output_type="field",
            target_name="effective_plastic_strain_out",
            unit="-",
        ),
    )
    common_includes = (
        data_dir / "common_load_time.i",
        data_dir / "common_plas_ludwik_props.i",
        data_dir / "common_solver.i",
        data_dir / "common_outputs.i",
    )
    return ModelMoose(
        executable=executable_path,
        input_template=data_dir / "base.i",
        mesh_template=data_dir / "hole_notch_2d.msh",
        common_files=common_includes,
        output_specs=out_specs,
        sim_name="hole_notch_2d_plas_ludwik_prob",
    )


__all__ = [
    "Study",
    "Param",
    "ParamSpace",
    "ParamValues",
    "Interval",
    "DistNormal",
    "DistUniform",
    "Uncertainty",
    "DoeConfig",
    "DoeLatinHypercube",
    "DoeSobol",
    "DoeRandom",
    "build_doe",
    "IModel",
    "OutputSpec",
    "ModelMoose",
    "IRunner",
    "RunnerLocal",
    "RunResult",
    "RunSet",
    "ERunStatus",
    "ResultScalar",
    "ResultField",
    "SimResult",
    "EFieldLocation",
    "TrainingData",
    "build_training_data",
    "split_training_data",
    "FieldLayout",
    "build_grid_layout",
    "flatten_field",
    "restore_field_layout",
    "standardise_field_grid",
    "EFieldReductionCenter",
    "FieldBasis",
    "FieldReductionValidation",
    "calc_field_basis",
    "calc_field_mode_errors",
    "ConfigGaussianProcess",
    "SurrogateValidation",
    "ISurrogate",
    "SurrogateGaussianProcess",
    "SurrogateField",
    "build_surrogate",
    "build_field_surrogate",
    "SensitivityResult",
    "calc_sensitivity",
    "select_params",
    "ProbConfig",
    "PBox",
    "ProbResult",
    "ProbFieldResult",
    "run_probabilistic",
    "run_field_probabilistic",
    "save_param_values",
    "load_param_values",
    "save_training_data",
    "load_training_data",
    "save_field_basis",
    "load_field_basis",
    "save_surrogate_gp",
    "load_surrogate_gp",
    "save_sensitivity",
    "load_sensitivity",
    "save_sensitivity_result",
    "load_sensitivity_result",
    "save_prob_result",
    "load_prob_result",
    "save_prob_field_result",
    "load_prob_field_result",
    "get_data_path",
    "build_example_param_space",
    "build_example_model",
    "build_ludwik_param_space",
    "build_ludwik_model",
]
