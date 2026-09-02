import argparse
import time
from pathlib import Path
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)

USER_DIR = Path.home()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="platehole3d_elas.i",
        help="Input file name or path"
    )
    args = parser.parse_args()

    moose_path = Path(args.input)
    if not moose_path.is_absolute():
        moose_path = Path(__file__).resolve().parent / moose_path

    config = {
        "main_path": USER_DIR / "moose",
        "app_path": USER_DIR / "proteus",
        "app_name": "proteus-opt"
    }

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(
        n_tasks=1,
        n_threads=1,
        redirect_out=False
    )

    from run_helpers import run_moose_and_tee
    moose_start_time = time.perf_counter()
    run_moose_and_tee(moose_runner, moose_path)
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("=" * 80)
    print(f"MOOSE run time = {moose_run_time:.3f} seconds")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
