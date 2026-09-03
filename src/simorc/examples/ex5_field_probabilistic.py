"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 5: Spatial Field Probabilistic Propagation and Quantile Maps.
Outputs saved to ./out/5_field_probabilistic/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run spatial field probabilistic propagation example."""
    out_dir = Path("./out/5_field_probabilistic")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Run samples & extract field
    doe_samples = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=8, seed=42)
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    completed_results = run_set.get_completed_results()

    first_field = completed_results[0].get_field("vonmises_stress")
    layout = so.FieldLayout(
        coords=first_field.coords,
        components=first_field.components,
        original_shape=first_field.values.shape,
        location=first_field.location,
    )

    snapshots = np.zeros(
        (len(completed_results), layout.get_total_dofs()), dtype=np.float64
    )
    for ii, res in enumerate(completed_results):
        snapshots[ii, :] = so.flatten_field(
            res.get_field("vonmises_stress").values
        )

    # 2. Fit Field Basis & Surrogate
    basis = so.calc_field_basis(snapshots, layout, energy_fraction=0.999)
    field_surrogate = so.build_field_surrogate(
        doe_samples, snapshots, basis
    )

    # 3. Propagate Uncertainties for Spatial Field
    prob_config = so.ProbConfig(num_epistemic=20, num_aleatory=100, seed=55)
    field_maps, _ = so.run_field_probabilistic(
        field_surrogate, param_space, config=prob_config
    )

    # 4. Plot 4-panel Spatial Quantile Maps (Q05, Median Q50, Q95, Std Dev)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    coords_x = layout.coords[:, 0]
    coords_y = layout.coords[:, 1]

    vmin = np.min(field_maps["q05"][:, 0])
    vmax = np.max(field_maps["q95"][:, 0])

    # 5th Percentile
    ax0 = axes[0]
    sc0 = ax0.scatter(
        coords_x,
        coords_y,
        c=field_maps["q05"][:, 0],
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        s=6,
    )
    plt.colorbar(sc0, ax=ax0, label="MPa")
    ax0.set_title("5th Percentile ($Q_{0.05}$)")
    ax0.axis("equal")

    # Median (50th Percentile)
    ax1 = axes[1]
    sc1 = ax1.scatter(
        coords_x,
        coords_y,
        c=field_maps["q50"][:, 0],
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        s=6,
    )
    plt.colorbar(sc1, ax=ax1, label="MPa")
    ax1.set_title("Median ($Q_{0.50}$)")
    ax1.axis("equal")

    # 95th Percentile
    ax2 = axes[2]
    sc2 = ax2.scatter(
        coords_x,
        coords_y,
        c=field_maps["q95"][:, 0],
        vmin=vmin,
        vmax=vmax,
        cmap="coolwarm",
        s=6,
    )
    plt.colorbar(sc2, ax=ax2, label="MPa")
    ax2.set_title("95th Percentile ($Q_{0.95}$)")
    ax2.axis("equal")

    # Spatial Standard Deviation
    ax3 = axes[3]
    sc3 = ax3.scatter(
        coords_x,
        coords_y,
        c=field_maps["std"][:, 0],
        cmap="magma",
        s=6,
    )
    plt.colorbar(sc3, ax=ax3, label="MPa (Std Dev)")
    ax3.set_title("Spatial Uncertainty ($\\sigma$)")
    ax3.axis("equal")

    for ax in axes:
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    fig_path = out_dir / "spatial_field_probabilistic_quantiles.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
