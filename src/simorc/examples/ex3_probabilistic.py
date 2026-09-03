"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 3: Mixed Epistemic-Aleatory Uncertainty Propagation and P-Boxes.
Outputs saved to ./out/3_probabilistic/
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run probabilistic P-Box propagation example."""
    out_dir = Path("./out/3_probabilistic")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "sim_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Sample and fit surrogate
    doe_samples = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=8, seed=42)
    )
    run_set = runner.run_samples(model, doe_samples, work_dir)
    train_data = so.build_training_data(
        doe_samples,
        run_set.get_completed_results(),
        scalar_names=("react_y_top", "stress_vm_max"),
    )
    surrogate = so.build_surrogate(train_data)

    # 2. Propagate nested uncertainties
    prob_config = so.ProbConfig(
        num_epistemic=50,
        num_aleatory=500,
        quantiles=(0.05, 0.50, 0.95),
        seed=101,
    )
    prob_result = so.run_probabilistic(
        surrogate, param_space, config=prob_config
    )
    so.save_prob_result(prob_result, out_dir / "prob_result.npz")

    # 3. Plot Probability Boxes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ii, name in enumerate(prob_result.output_names):
        ax = axes[ii]
        pbox = prob_result.pboxes[name]

        area = pbox.calc_area()
        print(f"P-Box area for {name}: {area:.4f}")

        # Plot CDF envelope
        ax.plot(
            pbox.eval_points,
            pbox.cdf_lower,
            "b--",
            label="CDF Lower Envelope ($F_{lower}$)",
        )
        ax.plot(
            pbox.eval_points,
            pbox.cdf_upper,
            "r--",
            label="CDF Upper Envelope ($F_{upper}$)",
        )
        ax.plot(
            pbox.eval_points,
            pbox.cdf_median,
            "k-",
            label="Median CDF ($F_{50}$)",
        )
        ax.fill_between(
            pbox.eval_points,
            pbox.cdf_lower,
            pbox.cdf_upper,
            color="purple",
            alpha=0.2,
            label=f"P-Box (Area = {area:.2f})",
        )

        # Annotate 5th and 95th quantile bounds
        q05_low = pbox.quantile_lower[0]
        q95_high = pbox.quantile_upper[2]
        ax.axvline(
            q05_low,
            color="gray",
            linestyle=":",
            label="5% - 95% Quantile Bounds",
        )
        ax.axvline(q95_high, color="gray", linestyle=":")

        ax.set_xlabel(name)
        ax.set_ylabel(r"Cumulative Probability $P(Y \leq y)$")
        ax.set_title(f"Probability Box: {name}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "pbox_propagation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
