"""
================================================================================
KC4 Standard Workflow: Stage 3 - Common-Grid Field Standardisation
================================================================================
Standardises spatial FE fields onto a common regular DIC validation grid.
Outputs saved to ./out/kc4std/stage3_standardised_fields.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 3: Interpolate FE fields onto common DIC grid."""
    work_dir = cfg.SIM_RUNS_DIR
    doe_path = cfg.OUT_DIR / "stage1_doe.npz"

    doe_samples = so.load_param_values(doe_path)
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
    plastic_snapshots = []

    print("[KC4-STD Stage 3] Standardising spatial fields across all runs...")
    for res in completed_results:
        stress_field = res.get_field("vonmises_stress")
        grid_stress = so.standardise_field_grid(stress_field, field_layout)
        stress_snapshots.append(grid_stress.values.flatten())

        plas_field = res.get_field("effective_plastic_strain_out")
        grid_plas = so.standardise_field_grid(plas_field, field_layout)
        plastic_snapshots.append(grid_plas.values.flatten())

    stress_mat = np.array(stress_snapshots)
    plastic_mat = np.array(plastic_snapshots)

    save_path = cfg.OUT_DIR / "stage3_standardised_fields.npz"
    np.savez_compressed(
        save_path,
        stress_snapshots=stress_mat,
        plastic_snapshots=plastic_mat,
        grid_shape=np.array(field_layout.original_shape),
        bounds=np.array(cfg.GRID_BOUNDS),
    )
    print(
        f"[KC4-STD Stage 3] Standardised {stress_mat.shape[0]} field "
        f"snapshots with {stress_mat.shape[1]} grid points.\n"
        f"[KC4-STD Stage 3] Saved standardised fields to {save_path}"
    )


if __name__ == "__main__":
    main()
