"""
================================================================================
KC4 Standard Workflow: Stage 1 - Parameter Space & Latin Hypercube DOE Sampling
================================================================================
Generates Latin Hypercube samples for Ludwik plasticity parameters.
Outputs saved to ./out/kc4std/stage1_doe.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import simorc as so


def main() -> None:
    """Execute Stage 1: Build DOE parameter samples."""
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[KC4-STD Stage 1] Building parameter space for Ludwik model...")
    param_space = so.build_ludwik_param_space()

    doe_config = so.DoeLatinHypercube(
        num_samples=cfg.NUM_DOE_SAMPLES, seed=cfg.DOE_SEED
    )
    doe_samples = so.build_doe(param_space, doe_config)

    save_path = cfg.OUT_DIR / "stage1_doe.npz"
    so.save_param_values(doe_samples, save_path)
    print(
        f"[KC4-STD Stage 1] Generated {doe_samples.get_num_samples()} LHS "
        f"samples spanning {param_space.get_num_params()} parameters."
    )
    for name, bounds in zip(
        doe_samples.names, param_space.calc_bounds()
    ):
        print(f"  - {name:10s} : range [{bounds[0]:.4e}, {bounds[1]:.4e}]")
    print(f"[KC4-STD Stage 1] Saved DOE to {save_path}")


if __name__ == "__main__":
    main()
