import time
from pathlib import Path
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)

USER_DIR = Path.home()


def main() -> None:
    config = {
        "main_path": USER_DIR / "moose",
        "app_path": USER_DIR / "proteus",
        "app_name": "proteus-opt",
    }

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    # Set run options - redirect_out=True might be better for batch runs
    # to keep the terminal clean, but sticking to False to match original
    moose_runner.set_run_opts(
        n_tasks=2,
        n_threads=1,
        redirect_out=False,
    )

    # Find all .i files in the current directory
    script_dir = Path(__file__).resolve().parent
    moose_files = sorted(list(script_dir.glob("*.i")))

    if not moose_files:
        print(
            "No MOOSE input (.i) files found in the "
            "current directory."
        )
        return

    print("=" * 80)
    print(f"Running {len(moose_files)} MOOSE simulations...")
    print("=" * 80)

    total_start = time.perf_counter()

    for moose_file in moose_files:
        print(f"\nProcessing: {moose_file.name}")
        moose_start = time.perf_counter()

        try:
            from run_helpers import run_moose_and_tee
            run_moose_and_tee(moose_runner, moose_file)
            moose_run_time = time.perf_counter() - moose_start
            print(
                f"Finished {moose_file.name} in "
                f"{moose_run_time:.2f} seconds"
            )
        except Exception as e:
            print(f"Error running {moose_file.name}: {e}")

    total_run_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print(
        f"Total MOOSE run time for all files = "
        f"{total_run_time:.2f} seconds"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
