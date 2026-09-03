from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    """Execute all KC4 standard pipeline stages sequentially."""
    print("=" * 70)
    print("STARTING FULL KC4 STANDARD PROBABILISTIC WORKFLOW")
    print("=" * 70)
    t_start = time.perf_counter()

    script_dir = Path(__file__).parent
    stages = [
        "01_doe_sampling.py",
        "02_run_simulations.py",
        "03_field_standardisation.py",
        "04_modal_reduction.py",
        "05_surrogate_training.py",
        "06_probabilistic_pbox.py",
        "07_plot_results.py",
    ]

    for stage_script in stages:
        script_path = script_dir / stage_script
        print(f"\n>>> Executing {stage_script}...")
        subprocess.run([sys.executable, str(script_path)], check=True)

    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"KC4 STANDARD WORKFLOW COMPLETE in {t_total:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
