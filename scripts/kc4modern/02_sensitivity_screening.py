"""
================================================================================
KC4 Modern Workflow: Stage 2 - Global Sensitivity Analysis & Down-Selection
================================================================================
Evaluates Sobol total-order indices (ST) to identify influential parameters.
Outputs saved to ./out/kc4modern/stage2_sensitivity.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 2: Sensitivity analysis & parameter screening."""
    param_space = so.build_ludwik_param_space()

    print("[KC4-MODERN Stage 2] Building exploratory DOE for sensitivity...")
    pilot_doe = so.build_doe(
        param_space,
        so.DoeSobol(
            num_samples=cfg.NUM_DOE_SAMPLES,
            seed=cfg.DOE_SEED,
            scramble=True,
        ),
    )

    model = so.build_ludwik_model()
    work_dir = cfg.SIM_RUNS_DIR
    runner = so.RunnerLocal(
        num_workers=cfg.NUM_WORKERS,
        num_threads_per_sim=cfg.NUM_THREADS_PER_SIM,
        restart=cfg.RESTART,
        verbose=False,
    )
    run_set = runner.run_samples(model, pilot_doe, work_dir)

    train_data = so.build_training_data(
        pilot_doe,
        run_set.get_completed_results(),
        scalar_names=("react_y_top", "stress_vm_max"),
    )
    surrogate = so.build_surrogate(train_data)

    print("[KC4-MODERN Stage 2] Computing Sobol global sensitivity indices...")
    sens_res = so.calc_sensitivity(
        surrogate,
        param_space,
        num_samples=cfg.SENSITIVITY_SAMPLES,
        seed=cfg.SENSITIVITY_SEED,
    )

    influential_params = so.select_params(
        sens_res, threshold=cfg.SCREENING_THRESHOLD, metric="st"
    )

    print(
        f"[KC4-MODERN Stage 2] Influential parameters (ST >= "
        f"{cfg.SCREENING_THRESHOLD}): {influential_params}"
    )
    for idx, out_name in enumerate(sens_res.output_names):
        print(f"  Rankings for {out_name}:")
        for p_name, st_val in sens_res.get_ranked_params(idx, metric="st"):
            print(f"    - {p_name:10s} : ST = {st_val:.4f}")

    save_path = cfg.OUT_DIR / "stage2_sensitivity.npz"
    so.save_sensitivity_result(sens_res, save_path)
    print(f"[KC4-MODERN Stage 2] Saved sensitivity results to {save_path}")


if __name__ == "__main__":
    main()
