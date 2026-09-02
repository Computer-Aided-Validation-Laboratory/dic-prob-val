"""
================================================================================
Compare constitutive hardening laws for SS316L at room temperature.

Laws evaluated:
1. Pure Elastic
2. Linear Hardening
3. Ludwik Hardening (power law)
4. Voce Hardening (exponential saturation)

Computes uniaxial monotonic tension response up to 20% strain and saves
comparison plots to ./out/0_hardeninglaws/.
================================================================================
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Elastic parameters (SS316L at room temperature)
    e_mod = 195.0e3  # MPa (195 GPa)
    nu = 0.30  # Poisson ratio
    yield_stress = 280.0  # MPa (0.2% proof stress)

    # Plastic hardening parameters
    # Linear: sigma = sigma_0 + H * eps_p
    h_linear = 1200.0  # MPa

    # Ludwik: sigma = sigma_0 + K * eps_p^n
    k_ludwik = 630.0  # MPa
    n_ludwik = 0.60

    # Voce: sigma = sigma_0 + R_inf * (1 - exp(-b * eps_p))
    r_inf_voce = 290.0  # MPa (saturation increment, sigma_sat = 570 MPa)
    b_voce = 8.5

    # Strain range up to 20% total strain
    max_strain = 0.20
    num_pts = 1000
    eps_total = np.linspace(0.0, max_strain, num_pts)

    # 1. Pure elastic response
    sigma_elastic = e_mod * eps_total

    # Plastic strain calculations
    eps_yield = yield_stress / e_mod

    sigma_linear = np.zeros_like(eps_total)
    sigma_ludwik = np.zeros_like(eps_total)
    sigma_voce = np.zeros_like(eps_total)

    for ii, eps in enumerate(eps_total):
        if eps <= eps_yield:
            sigma_linear[ii] = e_mod * eps
            sigma_ludwik[ii] = e_mod * eps
            sigma_voce[ii] = e_mod * eps
        else:
            # Linear hardening
            sigma_linear[ii] = (eps + yield_stress / h_linear) / (
                1.0 / e_mod + 1.0 / h_linear
            )

            # Ludwik hardening via Newton-Raphson
            ep_l = eps - eps_yield
            for _ in range(20):
                sig_val = yield_stress + k_ludwik * (ep_l**n_ludwik)
                res = ep_l + sig_val / e_mod - eps
                dres = (
                    1.0
                    + (k_ludwik * n_ludwik * (ep_l ** (n_ludwik - 1.0)))
                    / e_mod
                )
                dep = -res / dres
                ep_l += dep
                if abs(dep) < 1e-12:
                    break
            sigma_ludwik[ii] = yield_stress + k_ludwik * (ep_l**n_ludwik)

            # Voce hardening via Newton-Raphson
            ep_v = eps - eps_yield
            for _ in range(20):
                sig_val = yield_stress + r_inf_voce * (
                    1.0 - np.exp(-b_voce * ep_v)
                )
                res = ep_v + sig_val / e_mod - eps
                dres = (
                    1.0
                    + (r_inf_voce * b_voce * np.exp(-b_voce * ep_v))
                    / e_mod
                )
                dep = -res / dres
                ep_v += dep
                if abs(dep) < 1e-12:
                    break
            sigma_voce[ii] = yield_stress + r_inf_voce * (
                1.0 - np.exp(-b_voce * ep_v)
            )

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Full strain range (0 to 20%)
    ax1.plot(
        eps_total * 100,
        sigma_linear,
        label=f"Linear ($H={h_linear:.0f}$ MPa)",
        color="tab:blue",
        linewidth=2,
    )
    ax1.plot(
        eps_total * 100,
        sigma_ludwik,
        label=f"Ludwik ($K={k_ludwik:.0f}$ MPa, $n={n_ludwik:.2f}$)",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )
    ax1.plot(
        eps_total * 100,
        sigma_voce,
        label=f"Voce ($R_\\infty={r_inf_voce:.0f}$ MPa, $b={b_voce:.1f}$)",
        color="tab:green",
        linewidth=2,
        linestyle="-.",
    )
    ax1.axhline(
        yield_stress,
        color="gray",
        linestyle=":",
        label=f"Yield $\\sigma_0 = {yield_stress:.0f}$ MPa",
    )

    ax1.set_xlabel("Total Strain $\\epsilon$ (%)")
    ax1.set_ylabel("True Stress $\\sigma$ (MPa)")
    ax1.set_title("SS316L Hardening Laws Comparison (0 - 20% Strain)")
    ax1.set_xlim([0, 20])
    ax1.set_ylim([0, 600])
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Zoomed into yield transition (0 to 2% strain)
    idx_zoom = eps_total <= 0.02
    ax2.plot(
        eps_total[idx_zoom] * 100,
        sigma_elastic[idx_zoom],
        label=f"Elastic ($E={e_mod/1e3:.0f}$ GPa)",
        color="tab:red",
        linewidth=1.5,
        linestyle=":",
    )
    ax2.plot(
        eps_total[idx_zoom] * 100,
        sigma_linear[idx_zoom],
        label="Linear",
        color="tab:blue",
        linewidth=2,
    )
    ax2.plot(
        eps_total[idx_zoom] * 100,
        sigma_ludwik[idx_zoom],
        label="Ludwik",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )
    ax2.plot(
        eps_total[idx_zoom] * 100,
        sigma_voce[idx_zoom],
        label="Voce",
        color="tab:green",
        linewidth=2,
        linestyle="-.",
    )

    ax2.set_xlabel("Total Strain $\\epsilon$ (%)")
    ax2.set_ylabel("True Stress $\\sigma$ (MPa)")
    ax2.set_title("Yield Transition Region (0 - 2% Strain)")
    ax2.set_xlim([0, 2.0])
    ax2.set_ylim([0, 400])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "out/0_hardeninglaws"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hardening_laws_comparison.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Plot successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
