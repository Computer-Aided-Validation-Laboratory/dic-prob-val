"""
================================================================================
KC4 Standard Workflow: Stage 7 - Diagnostic & Validation Plots
================================================================================
Generates publication-quality validation figures in the style of the KC4 study.
Outputs saved to ./out/kc4std/*.png
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import param as cfg
import matplotlib.pyplot as plt
import numpy as np
import simorc as so


def main() -> None:
    """Execute Stage 7: Generate diagnostic plots."""
    out_dir = cfg.OUT_DIR

    # 1. Plot SVD Mode Convergence
    basis = so.load_field_basis(out_dir / "stage4_modal_basis.npz")
    mode_errors = np.load(out_dir / "stage4_mode_errors.npy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    modes = np.arange(1, len(basis.singular_values) + 1)

    ax1.semilogy(
        modes[:8], basis.singular_values[:8] / basis.singular_values[0], "bo-"
    )
    ax1.set_xlabel("Mode Index $r$")
    ax1.set_ylabel(r"Normalized Singular Value $\sigma_r / \sigma_1$")
    ax1.set_title("KC4-STD: SVD Singular Value Spectrum")
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, len(mode_errors) + 1), mode_errors, "rs--")
    ax2.set_xlabel("Retained Modes")
    ax2.set_ylabel("Field Reconstruction RMSE (MPa)")
    ax2.set_title("KC4-STD: Modal Reconstruction Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = out_dir / "01_svd_modal_convergence.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[KC4-STD Stage 7] Saved SVD plot to {fig1_path}")

    # 2. Plot Scalar Probability Boxes
    prob_res = so.load_prob_result(out_dir / "stage6_prob_result.npz")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ii, name in enumerate(prob_res.output_names):
        ax = axes[ii]
        pbox = prob_res.pboxes[name]
        area = pbox.calc_area()

        ax.plot(pbox.eval_points, pbox.cdf_lower, "b--", label="Lower CDF")
        ax.plot(pbox.eval_points, pbox.cdf_upper, "r--", label="Upper CDF")
        ax.plot(pbox.eval_points, pbox.cdf_median, "k-", label="Median CDF")
        ax.fill_between(
            pbox.eval_points,
            pbox.cdf_lower,
            pbox.cdf_upper,
            color="purple",
            alpha=0.25,
            label=f"P-Box (Area={area:.2f})",
        )
        ax.set_xlabel(name)
        ax.set_ylabel(r"Cumulative Probability $P(Y \leq y)$")
        ax.set_title(f"KC4-STD P-Box: {name}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = out_dir / "02_scalar_pboxes.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[KC4-STD Stage 7] Saved P-box plot to {fig2_path}")


if __name__ == "__main__":
    main()
