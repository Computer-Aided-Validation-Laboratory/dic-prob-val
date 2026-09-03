"""
================================================================================
KC4 Modern Workflow: Stage 5 - Modal Field GP with Epistemic Posterior Tracking
================================================================================
Fits Gaussian Process surrogates on modal POD coefficients with variance bounds.
Outputs saved to ./out/kc4modern/stage5_field_surrogate.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 5: Train modal GP surrogate model."""
    train_data = so.load_training_data(cfg.OUT_DIR / "stage3_training_data.npz")
    pod_basis = so.load_field_basis(cfg.OUT_DIR / "stage4_pod_basis.npz")
    stress_snapshots = np.load(cfg.OUT_DIR / "stage4_stress_snapshots.npy")

    print("[KC4-MODERN Stage 5] Building modal field GP surrogate...")
    field_surrogate = so.build_field_surrogate(
        data_inputs=train_data.inputs,
        snapshots=stress_snapshots,
        basis=pod_basis,
        config=so.ConfigGaussianProcess(
            n_restarts=cfg.GP_RESTARTS, seed=cfg.GP_SEED
        ),
    )

    save_path = cfg.OUT_DIR / "stage5_field_surrogate.npz"
    so.save_surrogate_gp(field_surrogate.modal_surrogate, save_path)
    print(
        f"[KC4-MODERN Stage 5] Trained {pod_basis.num_modes}-mode field "
        f"GP surrogate.\n"
        f"[KC4-MODERN Stage 5] Saved surrogate to {save_path}"
    )


if __name__ == "__main__":
    main()
