"""
================================================================================
KC4 Modern Workflow: Stage 6 - Convergence-Monitored Spatial Field UQ
================================================================================
Propagates mixed uncertainties through the field surrogate to compute 2D maps.
Outputs saved to ./out/kc4modern/stage6_field_prob_result.npz
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 6: Propagate field uncertainties for spatial quantiles."""
    param_space = so.build_ludwik_param_space()
    pod_basis = so.load_field_basis(cfg.OUT_DIR / "stage4_pod_basis.npz")
    modal_gp = so.load_surrogate_gp(cfg.OUT_DIR / "stage5_field_surrogate.npz")

    field_surrogate = so.SurrogateField(
        basis=pod_basis, modal_surrogate=modal_gp
    )
    field_layout = so.build_grid_layout(
        grid_shape=cfg.GRID_SHAPE,
        bounds=cfg.GRID_BOUNDS,
    )

    print("[KC4-MODERN Stage 6] Propagating spatial field uncertainties...")
    prob_config = so.ProbConfig(
        num_epistemic=cfg.NUM_EPISTEMIC,
        num_aleatory=cfg.NUM_ALEATORY,
        quantiles=cfg.QUANTILES,
        seed=cfg.UQ_SEED,
    )

    prob_field = so.run_field_probabilistic(
        field_surrogate,
        param_space,
        config=prob_config,
    )

    save_path = cfg.OUT_DIR / "stage6_field_prob_result.npz"
    so.save_prob_field_result(prob_field, save_path)
    print(
        f"[KC4-MODERN Stage 6] Generated spatial field quantile maps: "
        f"shape {prob_field.quantile_maps.shape}.\n"
        f"[KC4-MODERN Stage 6] Saved spatial UQ results to {save_path}"
    )


if __name__ == "__main__":
    main()
