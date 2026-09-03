"""
================================================================================
KC4 Modern Workflow: Stage 1 - Mixed Epistemic-Aleatory Uncertainty Model
================================================================================
Defines mixed aleatory-epistemic distributions for Ludwik plasticity.
Outputs saved to ./out/kc4modern/stage1_param_space.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 1: Build modern mixed uncertainty parameter model."""
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[KC4-MODERN Stage 1] Defining mixed uncertainty model...")
    params = [
        so.Param(
            name="EMod",
            nominal=195.0e3,
            uncertainty=so.DistNormal(
                mean=so.Interval(190.0e3, 200.0e3),
                std=4.0e3,
            ),
            unit="MPa",
        ),
        so.Param(
            name="PRatio",
            nominal=0.30,
            uncertainty=so.DistUniform(
                lower=so.Interval(0.28, 0.29),
                upper=so.Interval(0.31, 0.32),
            ),
            unit="-",
        ),
        so.Param(
            name="Yield",
            nominal=280.0,
            uncertainty=so.DistNormal(
                mean=so.Interval(270.0, 290.0),
                std=6.0,
            ),
            unit="MPa",
        ),
        so.Param(
            name="c1_bot",
            nominal=0.0,
            uncertainty=so.DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        so.Param(
            name="c2_bot",
            nominal=0.0,
            uncertainty=so.DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
        so.Param(
            name="c1_top",
            nominal=0.0,
            uncertainty=so.DistNormal(mean=0.0, std=0.002),
            unit="mm/mm",
        ),
        so.Param(
            name="c2_top",
            nominal=0.0,
            uncertainty=so.DistNormal(mean=0.0, std=0.0002),
            unit="mm/mm^2",
        ),
    ]
    param_space = so.ParamSpace(params)

    sobol_doe = so.build_doe(
        param_space,
        so.DoeSobol(
            num_samples=cfg.NUM_DOE_SAMPLES,
            seed=cfg.DOE_SEED,
            scramble=True,
        ),
    )

    save_path = cfg.OUT_DIR / "stage1_param_space.npz"
    so.save_param_values(sobol_doe, save_path)
    print(
        f"[KC4-MODERN Stage 1] Configured {param_space.get_num_params()} "
        f"parameters with mixed aleatory-epistemic distributions.\n"
        f"[KC4-MODERN Stage 1] Generated {cfg.NUM_DOE_SAMPLES} Sobol "
        f"design points.\n"
        f"[KC4-MODERN Stage 1] Saved initial DOE to {save_path}"
    )


if __name__ == "__main__":
    main()
