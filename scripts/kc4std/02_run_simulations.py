"""
================================================================================
KC4 Standard Workflow: Stage 2 - Finite Element Simulation Execution
================================================================================
Executes MOOSE Ludwik plasticity simulations across the DOE parameter set.
Outputs saved to ./out/kc4std/stage2_training_data.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import simorc as so


def main() -> None:
    """Execute Stage 2: Run MOOSE simulations for DOE samples."""
    work_dir = cfg.SIM_RUNS_DIR
    doe_path = cfg.OUT_DIR / "stage1_doe.npz"

    if not doe_path.exists():
        raise FileNotFoundError(
            f"DOE file not found at {doe_path}. Run Stage 1 first."
        )

    doe_samples = so.load_param_values(doe_path)
    model = so.build_ludwik_model()

    runner = so.RunnerLocal(
        num_workers=cfg.NUM_WORKERS,
        num_threads_per_sim=cfg.NUM_THREADS_PER_SIM,
        restart=cfg.RESTART,
        verbose=cfg.VERBOSE,
    )

    print("[KC4-STD Stage 2] Launching MOOSE simulation campaign...")
    run_set = runner.run_samples(model, doe_samples, work_dir)

    completed_results = run_set.get_completed_results()
    print(
        f"[KC4-STD Stage 2] Finished: {len(completed_results)}/"
        f"{len(run_set.runs)} runs completed successfully."
    )

    train_data = so.build_training_data(
        doe_samples,
        completed_results,
        scalar_names=cfg.SCALAR_OUTPUTS,
    )

    train_path = cfg.OUT_DIR / "stage2_training_data.npz"
    so.save_training_data(train_data, train_path)
    print(f"[KC4-STD Stage 2] Saved scalar training data to {train_path}")


if __name__ == "__main__":
    main()
