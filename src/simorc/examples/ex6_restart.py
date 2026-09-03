"""
================================================================================
simorc: simulation orchestrator
License: MIT
================================================================================
Example 6: Campaign Persistence, Cache Resumption, and Incremental Runs.
Outputs saved to ./out/6_restart/
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Run campaign restart and caching demonstration."""
    out_dir = Path("./out/6_restart")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "campaign_runs"

    param_space = so.build_example_param_space()
    model = so.build_example_model()
    runner = so.RunnerLocal(num_workers=4, restart=True)

    # 1. Initial Campaign: 4 samples
    print("--- Step 1: Initial Campaign (4 samples) ---")
    initial_doe = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=4, seed=10)
    )
    so.save_param_values(initial_doe, out_dir / "campaign_doe.npz")

    t_start = time.perf_counter()
    run_set_1 = runner.run_samples(model, initial_doe, work_dir)
    t_initial = time.perf_counter() - t_start
    print(
        f"Initial run completed: {run_set_1.get_num_complete()}/4 runs in "
        f"{t_initial:.2f} s"
    )

    # 2. Resuming Same Campaign (Cache verification)
    print("\n--- Step 2: Resuming Same Campaign (Cache Check) ---")
    t_start = time.perf_counter()
    loaded_doe = so.load_param_values(out_dir / "campaign_doe.npz")
    run_set_cached = runner.run_samples(model, loaded_doe, work_dir)
    t_cached = time.perf_counter() - t_start
    print(
        f"Cached run completed: {run_set_cached.get_num_complete()}/4 runs in "
        f"{t_cached:.4f} s (Instant cache hit!)"
    )

    # 3. Augmenting Campaign with 2 Additional Samples
    print("\n--- Step 3: Augmenting Campaign with 2 Additional Samples ---")
    new_doe = so.build_doe(
        param_space, so.DoeLatinHypercube(num_samples=6, seed=10)
    )
    t_start = time.perf_counter()
    run_set_aug = runner.run_samples(model, new_doe, work_dir)
    t_aug = time.perf_counter() - t_start
    print(
        f"Augmented campaign: {run_set_aug.get_num_complete()}/6 runs "
        f"in {t_aug:.2f} s"
    )

    # 4. Save Final Training Data
    completed_res = run_set_aug.get_completed_results()
    train_data = so.build_training_data(new_doe, completed_res)
    so.save_training_data(train_data, out_dir / "final_campaign_data.npz")

    # 5. Visualise Execution Timing Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stages = [
        "Initial\n(4 runs)",
        "Re-run\n(4 cached)",
        "Augmented\n(2 new + 4 cached)",
    ]
    timings = [t_initial, t_cached, t_aug]
    colors = ["cornflowerblue", "forestgreen", "coral"]

    bars = ax.bar(stages, timings, color=colors, width=0.5)
    for bar in bars:
        y_val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_val + max(timings) * 0.02,
            f"{y_val:.2f} s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Campaign Restart & Caching Benchmark")
    ax.set_ylim(0, max(timings) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "campaign_timing_benchmark.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
