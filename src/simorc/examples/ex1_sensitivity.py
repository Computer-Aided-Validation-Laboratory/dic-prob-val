"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 1: Global Sobol Sensitivity Analysis and Parameter Down-selection.
Outputs saved to ./out/1_sensitivity/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run sensitivity analysis example."""
    out_dir = Path("./out/1_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Sample DOE & Run simulations
    doe_samples = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=8, seed=123)
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    train_data = so.build_training_data(
        doe_samples,
        run_set.get_completed_results(),
        scalar_names=("react_y_top", "stress_vm_max"),
    )

    # 2. Fit GP surrogate
    surrogate = so.build_surrogate(train_data)

    # 3. Calculate Sobol indices
    sensitivity = so.calc_sensitivity(surrogate, param_space, num_samples=2048)
    so.save_sensitivity(sensitivity, out_dir / "sensitivity_results.npz")

    # 4. Down-select parameters with threshold = 0.05
    influential = so.select_params(sensitivity, threshold=0.05, metric="st")
    print(f"Influential parameters (ST >= 0.05): {influential}")

    # 5. Visualise sensitivity bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    param_names = param_space.get_names()
    x_indices = np.arange(len(param_names))
    width = 0.35

    for ii, out_name in enumerate(sensitivity.output_names):
        ax = axes[ii]
        ax.bar(
            x_indices - width / 2,
            sensitivity.s1[ii],
            width,
            label="First-order (S1)",
            color="royalblue",
        )
        ax.bar(
            x_indices + width / 2,
            sensitivity.st[ii],
            width,
            label="Total-order (ST)",
            color="darkorange",
        )
        ax.axhline(
            0.05,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Threshold (0.05)",
        )
        ax.set_xticks(x_indices)
        ax.set_xticklabels(param_names, rotation=30)
        ax.set_ylabel("Sensitivity Index")
        ax.set_title(f"Sobol Indices for {out_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "sensitivity_analysis.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
