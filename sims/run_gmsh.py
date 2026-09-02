"""
================================================================================
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import argparse
import time
from pathlib import Path
from pyvale.mooseherder import GmshRunner

PARSE_ONLY = False
USER_DIR = Path.home()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="mesh3d_holeplate.geo",
        help="Input geo file name or path"
    )
    args = parser.parse_args()

    gmsh_path = Path(args.input)
    if not gmsh_path.is_absolute():
        gmsh_path = Path(__file__).resolve().parent / gmsh_path

    gmsh_runner = GmshRunner(USER_DIR / "gmsh/bin/gmsh")
    print(gmsh_path)
    gmsh_start = time.perf_counter()
    gmsh_runner.run(gmsh_path, parse_only=PARSE_ONLY)
    gmsh_run_time = time.perf_counter() - gmsh_start

    print()
    print("=" * 80)
    print(f"Gmsh run time = {gmsh_run_time:.2f} seconds")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
