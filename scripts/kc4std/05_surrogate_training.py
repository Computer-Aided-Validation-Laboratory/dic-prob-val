"""
================================================================================
KC4 Standard Workflow: Stage 5 - Surrogate Model Construction
================================================================================
Fits Gaussian Process surrogates for scalar QoIs and modal coefficients.
Outputs saved to ./out/kc4std/stage5_surrogates.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 5: Train GP surrogates."""
    train_path = cfg.OUT_DIR / "stage2_training_data.npz"
    fields_path = cfg.OUT_DIR / "stage3_standardised_fields.npz"
    basis_path = cfg.OUT_DIR / "stage4_modal_basis.npz"

    train_data = so.load_training_data(train_path)
    fields_data = np.load(fields_path)
    stress_snapshots = fields_data["stress_snapshots"]
    basis = so.load_field_basis(basis_path)

    print("[KC4-STD Stage 5] Fitting scalar GP surrogate...")
    scalar_gp = so.build_surrogate(
        train_data,
        config=so.ConfigGaussianProcess(
            n_restarts=cfg.GP_RESTARTS, seed=cfg.GP_SEED
        ),
    )

    print("[KC4-STD Stage 5] Fitting modal field GP surrogate...")
    field_gp = so.build_field_surrogate(
        data_inputs=train_data.inputs,
        snapshots=stress_snapshots,
        basis=basis,
        config=so.ConfigGaussianProcess(
            n_restarts=cfg.GP_RESTARTS, seed=cfg.GP_SEED
        ),
    )

    save_path = cfg.OUT_DIR / "stage5_surrogates.npz"
    so.save_surrogate_gp(scalar_gp, save_path)
    print(f"[KC4-STD Stage 5] Saved trained surrogate models to {save_path}")


if __name__ == "__main__":
    main()
