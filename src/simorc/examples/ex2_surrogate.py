"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 2: Surrogate Model Training, Parity Validation, and Error Metrics.
Outputs saved to ./out/2_surrogate/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run surrogate validation example."""
    out_dir = Path("./out/2_surrogate")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Generate 10 samples
    doe_samples = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=10, seed=99)
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    full_data = so.build_training_data(
        doe_samples,
        run_set.get_completed_results(),
        scalar_names=("react_y_top", "stress_vm_max"),
    )

    # 2. Split into Train (80%) and Test (20%)
    train_data, test_data = so.split_training_data(
        full_data, test_fraction=0.3, seed=42
    )
    print(f"Train samples: {train_data.get_num_samples()}, "
          f"Test samples: {test_data.get_num_samples()}")

    # 3. Fit GP Surrogate
    surrogate = so.build_surrogate(train_data)
    validation = surrogate.validate(test_data)

    print("Surrogate Validation Metrics on Test Set:")
    for ii, name in enumerate(validation.output_names):
        print(f"  [{name}]")
        print(f"    RMSE         : {validation.rmse[ii]:.4e}")
        print(f"    NRMSE        : {validation.nrmse[ii]:.4%}")
        print(f"    Max Abs Error: {validation.max_abs_error[ii]:.4e}")
        print(f"    R^2 Score    : {validation.r2_score[ii]:.4f}")

    # 4. Predict on Test Set
    pred_means, pred_vars = surrogate.predict(test_data.inputs)
    pred_stds = np.sqrt(pred_vars)

    # 5. Parity Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ii, name in enumerate(validation.output_names):
        ax = axes[ii]
        y_true = test_data.outputs[:, ii]
        y_pred = pred_means[:, ii]
        err_bar = 1.96 * pred_stds[:, ii]

        min_val = min(np.min(y_true), np.min(y_pred)) * 0.98
        max_val = max(np.max(y_true), np.max(y_pred)) * 1.02

        ax.errorbar(
            y_true,
            y_pred,
            yerr=err_bar,
            fmt="o",
            color="crimson",
            ecolor="gray",
            capsize=4,
            label="Predictions (95% CI)",
        )
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            label="Ideal 1:1 Parity",
        )
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(
            f"Parity Plot: {name}\n$R^2 = {validation.r2_score[ii]:.3f}$, "
            f"NRMSE = {validation.nrmse[ii]:.2%}"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "surrogate_parity.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
