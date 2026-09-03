"""
================================================================================
KC4 Modern Workflow: Master Runner
================================================================================
Runs all 7 stages of the modern probabilistic validation pipeline.
Outputs saved to ./out/kc4modern/
"""

from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    """Execute all KC4 modern pipeline stages sequentially."""
    print("=" * 70)
    print("STARTING FULL KC4 MODERN PROBABILISTIC WORKFLOW")
    print("=" * 70)
    t_start = time.perf_counter()

    script_dir = Path(__file__).parent
    stages = [
        "01_uncertainty_definition.py",
        "02_sensitivity_screening.py",
        "03_adaptive_doe_and_runs.py",
        "04_centered_pod_reduction.py",
        "05_surrogate_field_model.py",
        "06_field_probabilistic_propagation.py",
        "07_plot_diagnostics.py",
    ]

    for stage_script in stages:
        script_path = script_dir / stage_script
        print(f"\n>>> Executing {stage_script}...")
        subprocess.run([sys.executable, str(script_path)], check=True)

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"KC4 MODERN WORKFLOW COMPLETE in {t_total:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
