"""
================================================================================
License: MIT
Copyright (C) 2026 The Computer Aided Validation Team
================================================================================
"""
import time
from pathlib import Path
from pyvale.mooseherder import GmshRunner

PARSE_ONLY = False
USER_DIR = Path.home()
GMSH_BIN = USER_DIR / "gmsh/bin/gmsh"


def main() -> None:
    gmsh_runner = GmshRunner(GMSH_BIN)

    # Find all 3D .geo files
    script_dir = Path(__file__).resolve().parent
    geo_files = sorted(list(script_dir.glob("mesh3d_*.geo")))

    if not geo_files:
        print("No 3D .geo files found in the data directory.")
        return

    print("=" * 80)
    print(f"Running {len(geo_files)} 3D Gmsh scripts...")
    print("=" * 80)

    total_start = time.perf_counter()

    for geo_file in geo_files:
        print(f"\nProcessing: {geo_file.name}")
        gmsh_start = time.perf_counter()

        try:
            gmsh_runner.run(geo_file, parse_only=PARSE_ONLY)
            gmsh_run_time = time.perf_counter() - gmsh_start
            print(
                f"Finished {geo_file.name} in "
                f"{gmsh_run_time:.2f} seconds"
            )
        except Exception as e:
            print(f"Error running {geo_file.name}: {e}")

    total_run_time = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print(f"Total run time for all files = {total_run_time:.2f} seconds")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
