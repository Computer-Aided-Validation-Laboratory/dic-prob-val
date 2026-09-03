"""
================================================================================
KC4 Modern Workflow: Stage 7 - Comprehensive Modern Diagnostic Visualizations
================================================================================
Generates diagnostic figures for POD reduction, sensitivity, and 2D spatial UQ.
Outputs saved to ./out/kc4modern/*.png
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 7: Generate modern diagnostic figures."""
    out_dir = cfg.OUT_DIR

    # 1. Plot Sobol Global Sensitivity Bar Chart
    sens_res = so.load_sensitivity_result(out_dir / "stage2_sensitivity.npz")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_pos = np.arange(len(sens_res.param_names))
    width = 0.35

    ax.bar(
        x_pos - width / 2,
        sens_res.s1[0],
        width,
        label="S1 (First-order)",
        color="royalblue",
    )
    ax.bar(
        x_pos + width / 2,
        sens_res.st[0],
        width,
        label="ST (Total-order)",
        color="darkorange",
    )
    ax.axhline(
        cfg.SCREENING_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Screening Threshold ({cfg.SCREENING_THRESHOLD*100:.0f}%)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sens_res.param_names, rotation=30)
    ax.set_ylabel("Sobol Sensitivity Index")
    ax.set_title("KC4-MODERN: Global Sensitivity Screening (Reaction Force)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = out_dir / "01_sensitivity_screening.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[KC4-MODERN Stage 7] Saved sensitivity plot to {fig1_path}")

    # 2. Plot 2D Spatial Quantile & Epistemic Uncertainty Field Maps
    prob_field = so.load_prob_field_result(
        out_dir / "stage6_field_prob_result.npz"
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    titles = [
        "5% Quantile $Q_{0.05}(\\mathbf{r})$ [MPa]",
        "Median Field $Q_{0.50}(\\mathbf{r})$ [MPa]",
        "95% Quantile $Q_{0.95}(\\mathbf{r})$ [MPa]",
        "Pointwise P-Box Width $Q_{0.95} - Q_{0.05}$ [MPa]",
    ]

    pbox_width = (
        prob_field.quantile_maps[2, :, :]
        - prob_field.quantile_maps[0, :, :]
    )
    field_maps = [
        prob_field.quantile_maps[0, :, :],
        prob_field.quantile_maps[1, :, :],
        prob_field.quantile_maps[2, :, :],
        pbox_width,
    ]

    x_min, x_max = cfg.GRID_BOUNDS[0]
    y_min, y_max = cfg.GRID_BOUNDS[1]

    for idx, (ax, t, f_map) in enumerate(zip(axes.flat, titles, field_maps)):
        im = ax.imshow(
            f_map,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="viridis" if idx < 3 else "magma",
        )
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig2_path = out_dir / "02_spatial_probabilistic_fields.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[KC4-MODERN Stage 7] Saved spatial UQ maps to {fig2_path}")


if __name__ == "__main__":
    main()
