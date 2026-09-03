"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 4: Spatial Field SVD/POD Modal Decomposition and Field Surrogate.
Outputs saved to ./out/4_field_surrogate/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run spatial field surrogate example."""
    out_dir = Path("./out/4_field_surrogate")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Run DOE samples
    doe_samples = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=8, seed=77)
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    completed_results = run_set.get_completed_results()

    # 2. Extract spatial field snapshots (von Mises stress)
    first_field = completed_results[0].get_field("vonmises_stress")
    layout = so.FieldLayout(
        coords=first_field.coords,
        components=first_field.components,
        original_shape=first_field.values.shape,
        location=first_field.location,
    )

    num_samples = len(completed_results)
    num_dofs = layout.get_total_dofs()
    snapshots = np.zeros((num_samples, num_dofs), dtype=np.float64)

    for ii, res in enumerate(completed_results):
        field_obj = res.get_field("vonmises_stress")
        snapshots[ii, :] = so.flatten_field(field_obj.values)

    # 3. Calculate POD Field Basis
    basis = so.calc_field_basis(
        snapshots,
        layout,
        energy_fraction=0.999,
        center=so.EFieldReductionCenter.mean,
    )
    so.save_field_basis(basis, out_dir / "field_basis.npz")
    print(f"Calculated Field Basis: Retained {basis.num_modes} modes.")

    # 4. Build Field Surrogate
    field_surrogate = so.build_field_surrogate(
        doe_samples, snapshots, basis
    )

    # 5. Evaluate on query point and reconstruct
    query_param = so.ParamValues(
        names=param_space.get_names(),
        values=param_space.get_nominal_values().reshape(1, -1),
    )
    pred_recon, pred_vars = field_surrogate.predict(query_param)
    pred_field_map = so.restore_field_layout(pred_recon[0], layout)
    true_field_map = completed_results[0].get_field("vonmises_stress").values

    # 6. Visualise Basis Decay and Field Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Singular value decay & Energy
    ax1 = axes[0]
    ranks = np.arange(1, len(basis.singular_values) + 1)
    ax1.semilogy(ranks, basis.singular_values, "bo-", label="Singular Value")
    ax1.set_xlabel("Mode Index")
    ax1.set_ylabel("Singular Value $\\sigma_k$")
    ax1.set_title(f"SVD Spectrum (Retained {basis.num_modes} modes)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Actual FE field (triangulation scatter / contour)
    ax2 = axes[1]
    coords_x = layout.coords[:, 0]
    coords_y = layout.coords[:, 1]
    sc1 = ax2.scatter(
        coords_x,
        coords_y,
        c=true_field_map[:, 0],
        cmap="viridis",
        s=8,
    )
    plt.colorbar(sc1, ax=ax2, label="Von Mises Stress (MPa)")
    ax2.set_title("Actual FE Field (Sample 0)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis("equal")

    # Plot 3: Field Surrogate Prediction
    ax3 = axes[2]
    sc2 = ax3.scatter(
        coords_x,
        coords_y,
        c=pred_field_map[:, 0],
        cmap="viridis",
        s=8,
    )
    plt.colorbar(sc2, ax=ax3, label="Von Mises Stress (MPa)")
    ax3.set_title("POD-GP Surrogate Prediction")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.axis("equal")

    plt.tight_layout()
    fig_path = out_dir / "field_surrogate_comparison.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
