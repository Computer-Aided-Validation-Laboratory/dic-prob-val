import time
from pathlib import Path
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)
MOOSE_FILES = [
    "platehole3d_elas.i",
    "platehole3d_elas_het.i",
    "platenotch3d_elas.i",
    "platenotch3d_elas_het.i",
]

USER_DIR = Path.home()


def main() -> None:
    config = {
        "main_path": USER_DIR / "moose",
        "app_path": USER_DIR / "proteus",
        "app_name": "proteus-opt",
    }

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(
        n_tasks=2,
        n_threads=1,
        redirect_out=False,
    )

    print("=" * 80)
    print(f"Running {len(MOOSE_FILES)} selected MOOSE simulations...")
    print("=" * 80)

    total_start = time.perf_counter()

    for filename in MOOSE_FILES:
        moose_file = Path(__file__).resolve().parent / filename
        if not moose_file.exists():
            print(f"\nFile not found: {filename}")
            continue

        print(f"\nProcessing: {filename}")
        moose_start = time.perf_counter()

        try:
            from run_helpers import run_moose_and_tee
            run_moose_and_tee(moose_runner, moose_file)
            moose_run_time = time.perf_counter() - moose_start
            print(
                f"Finished {filename} in "
                f"{moose_run_time:.2f} seconds"
            )
        except Exception as e:
            print(f"Error running {filename}: {e}")

    total_run_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print(
        f"Total MOOSE run time for selected files = "
        f"{total_run_time:.2f} seconds"
    )
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
