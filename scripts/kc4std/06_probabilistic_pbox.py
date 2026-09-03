"""
================================================================================
KC4 Standard Workflow: Stage 6 - Epistemic/Aleatory Propagation & P-Boxes
================================================================================
Propagates nested epistemic-aleatory uncertainties to compute Probability Boxes.
Outputs saved to ./out/kc4std/stage6_prob_result.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import simorc as so


def main() -> None:
    """Execute Stage 6: Propagate nested uncertainties."""
    surrogate_path = cfg.OUT_DIR / "stage5_surrogates.npz"

    scalar_gp = so.load_surrogate_gp(surrogate_path)
    param_space = so.build_ludwik_param_space()

    print("[KC4-STD Stage 6] Propagating epistemic/aleatory uncertainties...")
    prob_config = so.ProbConfig(
        num_epistemic=cfg.NUM_EPISTEMIC,
        num_aleatory=cfg.NUM_ALEATORY,
        quantiles=cfg.QUANTILES,
        seed=cfg.UQ_SEED,
    )

    prob_result = so.run_probabilistic(
        scalar_gp, param_space, config=prob_config
    )

    save_path = cfg.OUT_DIR / "stage6_prob_result.npz"
    so.save_prob_result(prob_result, save_path)
    print(
        f"[KC4-STD Stage 6] Generated P-Boxes for outputs: "
        f"{prob_result.output_names}\n"
        f"[KC4-STD Stage 6] Saved probabilistic result to {save_path}"
    )


if __name__ == "__main__":
    main()
