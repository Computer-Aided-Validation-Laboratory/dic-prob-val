"""
================================================================================
KC4 Standard Workflow: Stage 4 - SVD Field Modal Reduction & Mode Selection
================================================================================
Performs uncentered SVD decomposition on standardized spatial stress fields.
Outputs saved to ./out/kc4std/stage4_modal_basis.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 4: SVD modal reduction on field snapshots."""
    fields_path = cfg.OUT_DIR / "stage3_standardised_fields.npz"

    if not fields_path.exists():
        raise FileNotFoundError(
            f"Fields file not found at {fields_path}. Run Stage 3 first."
        )

    data = np.load(fields_path)
    stress_snapshots = data["stress_snapshots"]

    layout = so.build_grid_layout(
        grid_shape=cfg.GRID_SHAPE, bounds=cfg.GRID_BOUNDS
    )

    print("[KC4-STD Stage 4] Computing classical SVD modal basis...")
    basis = so.calc_field_basis(
        stress_snapshots,
        layout=layout,
        center=so.EFieldReductionCenter.none,
        num_modes=cfg.MAX_MODES,
    )

    val_res = so.calc_field_mode_errors(stress_snapshots, basis)
    rmse_err = val_res.reconstruction_rmse

    print(
        f"[KC4-STD Stage 4] Basis retained {basis.num_modes} modes.\n"
        f"  - Singular values: {basis.singular_values[:basis.num_modes]}\n"
        f"  - Reconstruction RMSE (max modes): {rmse_err[-1]:.4e} MPa"
    )

    save_path = cfg.OUT_DIR / "stage4_modal_basis.npz"
    so.save_field_basis(basis, save_path)
    np.save(cfg.OUT_DIR / "stage4_mode_errors.npy", rmse_err)
    print(f"[KC4-STD Stage 4] Saved modal basis to {save_path}")


if __name__ == "__main__":
    main()
