"""
================================================================================
KC4 Modern Workflow: Stage 4 - Mean-Centered POD Spatial Field Reduction
================================================================================
Standardises spatial fields and extracts mean-centered POD modal bases.
Outputs saved to ./out/kc4modern/stage4_pod_basis.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 4: Mean-centered POD modal reduction."""
    work_dir = cfg.SIM_RUNS_DIR
    param_path = cfg.OUT_DIR / "stage1_param_space.npz"

    doe_samples = so.load_param_values(param_path)
    model = so.build_ludwik_model()
    runner = so.RunnerLocal(
        num_workers=cfg.NUM_WORKERS,
        num_threads_per_sim=cfg.NUM_THREADS_PER_SIM,
        restart=cfg.RESTART,
        verbose=False,
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    completed_results = run_set.get_completed_results()

    field_layout = so.build_grid_layout(
        grid_shape=cfg.GRID_SHAPE,
        bounds=cfg.GRID_BOUNDS,
    )

    stress_snapshots = []
    for res in completed_results:
        stress_field = res.get_field("vonmises_stress")
        grid_stress = so.standardise_field_grid(stress_field, field_layout)
        stress_snapshots.append(grid_stress.values.flatten())

    stress_mat = np.array(stress_snapshots)

    print("[KC4-MODERN Stage 4] Computing mean-centered POD modal basis...")
    pod_basis = so.calc_field_basis(
        stress_mat,
        layout=field_layout,
        center=so.EFieldReductionCenter.mean,
        num_modes=cfg.MAX_POD_MODES,
    )

    val_res = so.calc_field_mode_errors(stress_mat, pod_basis)
    rmse_err = val_res.reconstruction_rmse

    save_path = cfg.OUT_DIR / "stage4_pod_basis.npz"
    so.save_field_basis(pod_basis, save_path)
    np.save(cfg.OUT_DIR / "stage4_mode_errors.npy", rmse_err)
    np.save(cfg.OUT_DIR / "stage4_stress_snapshots.npy", stress_mat)

    print(
        f"[KC4-MODERN Stage 4] Retained {pod_basis.num_modes} POD modes "
        f"(reconstruction RMSE = {rmse_err[-1]:.4e} MPa).\n"
        f"[KC4-MODERN Stage 4] Saved POD basis to {save_path}"
    )


if __name__ == "__main__":
    main()
