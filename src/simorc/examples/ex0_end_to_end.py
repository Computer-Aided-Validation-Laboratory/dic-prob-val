"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 0: Complete End-to-End Workflow Demonstration.
Outputs saved to ./out/0_end_to_end/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run end-to-end orchestration pipeline."""
    out_dir = Path("./out/0_end_to_end")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    print("--- 1. Setting up Parameter Space & Model ---")
    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    print("--- 2. Generating DOE Samples (Latin Hypercube) ---")
    doe_config = so.DoeLatinHypercube(num_samples=6, seed=42)
    doe_samples = so.build_doe(param_space, doe_config)
    print(f"Generated {doe_samples.get_num_samples()} samples for parameters:")
    for name in doe_samples.names:
        print(f"  - {name}")

    print("--- 3. Executing Deterministic Simulations ---")
    run_set = runner.run_samples(model, doe_samples, work_dir)
    print(f"Completed runs: {run_set.get_num_complete()}/{len(run_set.runs)}")

    print("--- 4. Extracting Training Data ---")
    completed_results = run_set.get_completed_results()
    train_data = so.build_training_data(
        doe_samples,
        completed_results,
        scalar_names=("react_y_top", "stress_vm_max"),
    )
    print(
        f"Training data shape: inputs {train_data.inputs.shape}, "
        f"outputs {train_data.outputs.shape}"
    )

    print("--- 5. Training Gaussian Process Surrogate ---")
    surrogate = so.build_surrogate(train_data)

    # Evaluate surrogate on fine 1D slice across EMod
    e_mod_slice = np.linspace(180e3, 210e3, 100)
    query_pts = np.tile(param_space.get_nominal_values(), (100, 1))
    query_pts[:, 0] = e_mod_slice  # EMod is column 0
    pred_means, pred_vars = surrogate.predict(query_pts)
    pred_stds = np.sqrt(pred_vars)

    print("--- 6. Computing Sobol Sensitivity Indices ---")
    sensitivity = so.calc_sensitivity(surrogate, param_space, num_samples=1024)
    for out_idx, out_name in enumerate(sensitivity.output_names):
        print(f"Sensitivity rankings for {out_name}:")
        ranked = sensitivity.get_ranked_params(out_idx, metric="st")
        for p_name, st_val in ranked:
            print(f"  {p_name:10s} : ST = {st_val:.4f}")

    print("--- 7. Propagating Uncertainties (P-Box Generation) ---")
    prob_config = so.ProbConfig(num_epistemic=30, num_aleatory=200, seed=42)
    prob_res = so.run_probabilistic(surrogate, param_space, config=prob_config)

    print("--- 8. Plotting Figures ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Surrogate fit slice
    ax1 = axes[0]
    ax1.plot(e_mod_slice / 1e3, pred_means[:, 0], "b-", label="GP Mean")
    ax1.fill_between(
        e_mod_slice / 1e3,
        pred_means[:, 0] - 1.96 * pred_stds[:, 0],
        pred_means[:, 0] + 1.96 * pred_stds[:, 0],
        color="b",
        alpha=0.2,
        label="95% CI",
    )
    ax1.scatter(
        train_data.inputs[:, 0] / 1e3,
        train_data.outputs[:, 0],
        color="r",
        zorder=5,
        label="FE Samples",
    )
    ax1.set_xlabel("Young's Modulus E (GPa)")
    ax1.set_ylabel("Reaction Force (N)")
    ax1.set_title("GP Surrogate Slice")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sobol Sensitivity Bar Chart
    ax2 = axes[1]
    x_indices = np.arange(len(param_space.params))
    width = 0.35
    ax2.bar(
        x_indices - width / 2,
        sensitivity.s1[0],
        width,
        label="S1 (First-order)",
        color="steelblue",
    )
    ax2.bar(
        x_indices + width / 2,
        sensitivity.st[0],
        width,
        label="ST (Total-order)",
        color="darkorange",
    )
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(param_space.get_names(), rotation=30)
    ax2.set_ylabel("Sensitivity Index")
    ax2.set_title("Sobol Indices (Reaction Force)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Probability Box (P-Box)
    ax3 = axes[2]
    pbox = prob_res.pboxes["react_y_top"]
    ax3.fill_betweenx(
        pbox.cdf_lower,
        pbox.eval_points,
        np.interp(pbox.cdf_lower, pbox.cdf_upper, pbox.eval_points),
        color="purple",
        alpha=0.25,
        label="Epistemic P-Box Area",
    )
    ax3.plot(pbox.eval_points, pbox.cdf_lower, "m--", label="Lower CDF")
    ax3.plot(pbox.eval_points, pbox.cdf_upper, "m-.", label="Upper CDF")
    ax3.plot(pbox.eval_points, pbox.cdf_median, "k-", label="Median CDF")
    ax3.set_xlabel("Reaction Force (N)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Reaction Force P-Box")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "end_to_end_summary.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
