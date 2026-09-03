"""
================================================================================
KC4 Modern Workflow: Stage 3 - Enriched DOE & Multi-Core Execution
================================================================================
Executes space-filling simulation campaign with automatic restart caching.
Outputs saved to ./out/kc4modern/sim_runs/ and stage3_training_data.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import simorc as so


def main() -> None:
    """Execute Stage 3: Enriched DOE execution."""
    work_dir = cfg.SIM_RUNS_DIR
    param_path = cfg.OUT_DIR / "stage1_param_space.npz"

    if not param_path.exists():
        raise FileNotFoundError(
            f"DOE file not found at {param_path}. Run Stage 1 first."
        )

    doe_samples = so.load_param_values(param_path)
    model = so.build_ludwik_model()

    runner = so.RunnerLocal(
        num_workers=cfg.NUM_WORKERS,
        num_threads_per_sim=cfg.NUM_THREADS_PER_SIM,
        restart=cfg.RESTART,
        verbose=cfg.VERBOSE,
    )

    print("[KC4-MODERN Stage 3] Executing enriched simulation campaign...")
    run_set = runner.run_samples(model, doe_samples, work_dir)

    completed_results = run_set.get_completed_results()
    train_data = so.build_training_data(
        doe_samples,
        completed_results,
        scalar_names=cfg.SCALAR_OUTPUTS,
    )

    train_path = cfg.OUT_DIR / "stage3_training_data.npz"
    so.save_training_data(train_data, train_path)
    print(
        f"[KC4-MODERN Stage 3] Successfully completed "
        f"{len(completed_results)}/{len(run_set.runs)} runs.\n"
        f"[KC4-MODERN Stage 3] Saved training dataset to {train_path}"
    )


if __name__ == "__main__":
    main()
