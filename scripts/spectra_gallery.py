"""Gallery of the spectrum types atomx can synthesize.

Six panels, one per spectrum family:

  A. Elementary line profiles      Gaussian / Lorentzian / Voigt
  B. Ab-initio bound-bound lines   He I MCDHF transitions -> emission spectrum
  C. CRM plasma emission           Ne steady-state lines + bremsstrahlung
  D. Continuum emission            free-free emissivity vs T_e
  E. Unresolved transition array   dense line forest vs UTA Gaussian moments
  F. Photoionization cross-section Kramers background x Fano autoionization
  G. Auger electron spectrum       Fe XVII KLL ab-initio widths vs FAC

Run:  python scripts/spectra_gallery.py
Writes scripts/spectra_gallery.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atomx import Atom
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMLevel,
    CRMTransition,
)
from atomx.scattering.autoionization import auger_rate_csf_average
from atomx.scattering.photoionization import FanoResonance, apply_resonances
from atomx.spectrum import Spectrum, crm_emission_spectrum, synthesize_from_transitions
from atomx.spectrum.continuum import free_free_emissivity
from atomx.spectrum.uta import add_uta_array_to_spectrum

HC_EV_NM = 1239.841984


# ---------------------------------------------------------------- A
def panel_profiles(ax):
    """Same integrated intensity, three broadening mechanisms."""
    for label, kw in [
        ("Gaussian (Doppler)", dict(sigma_nm=0.10)),
        ("Lorentzian (natural/pressure)", dict(gamma_nm=0.10)),
        ("Voigt (both)", dict(sigma_nm=0.07, gamma_nm=0.05)),
    ]:
        s = Spectrum.linear(98.0, 102.0, 4001)
        s.add_line(100.0, intensity=1.0, **kw)
        ax.plot(s.wavelength_nm, s.values, label=label, lw=1.4)
    ax.set_title("A. Line profiles (unit area each)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalized intensity")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- B
def panel_ab_initio_lines(ax):
    """He I MCDHF (1s2 + 1s2p) transitions rendered as an emission spectrum."""
    he = Atom(Z=2, configs=["1s2", "1s1 2p1"])
    he.solve(max_iter=40, tol=1e-4)
    trs = he.transitions(multipoles=["E1"])
    spec = synthesize_from_transitions(
        trs, lam_min_nm=35.0, lam_max_nm=80.0, n_points=6001,
        default_sigma_nm=0.25, default_gamma_nm=0.08,
        intensity_key="A_rate_per_s",
    )
    ax.semilogy(spec.wavelength_nm, spec.values, color="tab:blue", lw=1.2)
    ymax = spec.values.max()
    for t in trs:
        if 35.0 < t.wavelength_nm < 80.0:
            ax.annotate(f"{t.wavelength_nm:.1f} nm\nA = {t.A_rate_per_s:.1e}/s",
                        xy=(t.wavelength_nm, ymax), fontsize=7,
                        xytext=(t.wavelength_nm + 1.5, ymax * 0.1))
    ax.set_ylim(ymax * 1e-6, ymax * 3)
    ax.set_title(f"B. Ab-initio He I emission ({len(trs)} E1 lines, MCDHF)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("A-weighted emissivity (arb., log)")


# ---------------------------------------------------------------- C
def panel_crm(ax):
    """3-level H CRM: steady-state populations -> lines + free-free continuum.

    NIST Einstein A's: Ly-a 4.699e8/s, Ly-b 5.575e7/s, H-a 4.410e7/s.
    """
    levels = [
        CRMLevel("1s", 0.0, g=2),
        CRMLevel("2p", 10.199, g=8),
        CRMLevel("3", 12.088, g=18),
    ]
    transitions = [
        CRMTransition(upper=1, lower=0, A_rate=4.699e8, gf=0.8324),
        CRMTransition(upper=2, lower=0, A_rate=5.575e7, gf=0.1583),
        CRMTransition(upper=2, lower=1, A_rate=4.410e7, gf=5.126),
    ]
    crm = CollisionalRadiativeModel(levels, transitions)
    for Te_eV, color in [(1.0, "tab:blue"), (5.0, "tab:red")]:
        spec = crm_emission_spectrum(
            crm, 1.0e14, Te_eV,
            lam_min_nm=80.0, lam_max_nm=750.0, n_points=8001,
            ion_mass_amu=1.008, profile="voigt",
            include_free_free=True, ff_Z_eff=1.0,
        )
        ax.semilogy(spec.wavelength_nm, np.maximum(spec.values, 1e-300),
                    color=color, lw=1.1,
                    label=f"$T_e$ = {Te_eV:.0f} eV")
    for lam, name in [(121.6, "Ly$\\alpha$"), (102.6, "Ly$\\beta$"),
                      (656.3, "H$\\alpha$")]:
        ax.annotate(name, xy=(lam, ax.get_ylim()[1]), fontsize=8,
                    xytext=(lam + 8, ax.get_ylim()[1] * 0.25))
    ax.set_title("C. H CRM emission ($n_e$=$10^{14}$): lines + free-free")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("emissivity (arb., log)")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(ymax * 1e-10, ymax)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- D
def panel_free_free(ax):
    """Thermal bremsstrahlung: exponential cutoff moves with T_e."""
    E = np.geomspace(10.0, 3.0e4, 400)          # photon energy, eV
    for Te in (100.0, 1000.0, 5000.0):
        eps = free_free_emissivity(E, Te, Z_eff=1.0,
                                   n_e_cm3=1e14, n_i_cm3=1e14)
        ax.loglog(E, eps, lw=1.4, label=f"$T_e$ = {Te:.0f} eV")
    ax.set_title("D. Free-free (bremsstrahlung) continuum, H plasma")
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel(r"$\varepsilon_E$ (eV s$^{-1}$ cm$^{-3}$ eV$^{-1}$)")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- E
def panel_uta(ax, rng):
    """Heavy-element line forest: resolved sum vs UTA statistical envelope."""
    n_lines = 300
    E_lines = rng.normal(62.0, 1.6, n_lines)               # eV, e.g. W 4d-4f band
    strengths = rng.lognormal(0.0, 1.0, n_lines)
    lines = list(zip(E_lines, strengths))

    lam = HC_EV_NM / E_lines
    lo, hi = 17.0, 23.0

    resolved = Spectrum.linear(lo, hi, 8001)
    for lam0, s in zip(lam, strengths):
        resolved.add_line(lam0, s * HC_EV_NM / lam0, sigma_nm=0.004)

    uta = Spectrum.linear(lo, hi, 8001)
    moments = add_uta_array_to_spectrum(uta, lines, lam_min_nm=lo, lam_max_nm=hi)

    ax.vlines(lam, 0, strengths * E_lines / (strengths * E_lines).max()
              * resolved.values.max() * 0.35,
              color="0.75", lw=0.5, label=f"{n_lines} sticks")
    ax.plot(resolved.wavelength_nm, resolved.values, color="tab:blue",
            lw=1.0, label="resolved sum")
    ax.plot(uta.wavelength_nm, uta.values, color="crimson", lw=1.8,
            label="UTA Gaussian (moments)")
    ax.set_title(f"E. UTA: centroid {moments.centroid_nm:.2f} nm, "
                 f"$\\sigma$ = {moments.std_eV:.2f} eV")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("emissivity (arb.)")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- F
def panel_fano(ax):
    """He photoionization: Kramers background x 2s2p(1P) Fano series."""
    E = np.linspace(56.0, 66.0, 6000)                       # eV
    sigma_bg = 7.4 * (24.59 / E) ** 3                       # Mb, Kramers-like
    resonances = [
        FanoResonance(E_res_eV=60.15, gamma_eV=0.037, q=-2.77,
                      label="2s2p ${}^1P^o$ (n=2)"),
        FanoResonance(E_res_eV=63.66, gamma_eV=0.010, q=-2.5,
                      label="n=3"),
        FanoResonance(E_res_eV=64.46, gamma_eV=0.004, q=-2.5,
                      label="n=4"),
    ]
    sigma = apply_resonances(sigma_bg, E, resonances)
    ax.plot(E, sigma_bg, "--", color="0.6", lw=1.0, label="Kramers background")
    ax.plot(E, sigma, color="tab:green", lw=1.2,
            label="with Fano autoionization")
    for r in resonances:
        ax.axvline(r.E_res_eV, color="crimson", alpha=0.25, lw=0.8)
    ax.annotate("He 2s2p $^1P^o$\n(q = -2.77, $\\Gamma$ = 37 meV)",
                xy=(60.15, sigma.max()), fontsize=8, ha="left",
                xytext=(60.6, 0.85 * sigma.max()))
    ax.set_title("F. Photoionization $\\sigma(E)$ with Fano resonances")
    ax.set_xlabel("photon energy (eV)")
    ax.set_ylabel("cross-section (Mb)")
    ax.legend(fontsize=8, loc="lower left")


# ---------------------------------------------------------------- G
HBAR_EV_S = 6.582119569e-16


def panel_auger(ax):
    """Fe XVII KLL Auger electron spectrum, ab-initio widths vs FAC.

    Solves Ne-like Fe XVII with the doubly-excited 2p^5 3l 3l' configs and
    computes each KLL channel's Auger width with auger_rate_csf_average
    (delta(E)-normalised Dirac shooting continuum).  Each resonance emits an
    Auger electron at its energy above the Fe XVI ground state, rendered as
    a Lorentzian of natural width Gamma_total; FAC AITable widths from the
    benchmark fixture are overlaid as sticks (equivalent-Lorentzian peak
    height).  The config-average angular factor over-counts some magnetic
    substates, so per-channel agreement is order-of-magnitude to ~10 %
    depending on the channel (see tests/test_auger_kll_vs_fac.py).
    """
    fixture = json.loads(
        (Path(__file__).resolve().parent.parent
         / "tests" / "data" / "fe17_kll_auger_fac.json").read_text())
    atom = Atom(
        Z=26, charge=16,
        configs=["1s2 2s2 2p6", "1s2 2s2 2p5 3s1",
                 "1s2 2s2 2p5 3p1", "1s2 2s2 2p5 3d1"],
        j_resolved_csfs=False, breit=False, qed=False,
    )
    atom.solve()

    E_res = [row["E_resonance_eV"] for row in fixture["resonances"]]
    E = np.linspace(min(E_res) - 12.0, max(E_res) + 12.0, 12000)
    total = np.zeros_like(E)
    for i, row in enumerate(fixture["resonances"]):
        out = auger_rate_csf_average(
            atom._result,
            hole_label=row["atomx_hole_label"],
            source_label=row["atomx_source_label"],
            source2_label=row["atomx_source2_label"],
            resonance_energy_eV=row["E_resonance_eV"],
        )
        gamma_a = out["Gamma_per_s"]
        E0 = row["E_resonance_eV"]
        hwhm = HBAR_EV_S * (gamma_a + row["a_radiative_per_s"]) / 2.0
        total += gamma_a * (hwhm / np.pi) / ((E - E0) ** 2 + hwhm ** 2)

        gamma_fac = row["gamma_auger_per_s"]
        hwhm_fac = HBAR_EV_S * (gamma_fac + row["a_radiative_per_s"]) / 2.0
        ax.vlines(E0, 0, gamma_fac / (np.pi * hwhm_fac),
                  color="crimson", lw=1.4, alpha=0.75)
        ax.annotate(
            f"{row['atomx_source_label']}{row['atomx_source2_label']}"
            f" $\\to$ {row['atomx_hole_label']}$^{{-1}}$\n"
            f"ratio {gamma_a / gamma_fac:.2f}",
            xy=(E0, gamma_fac / (np.pi * hwhm_fac)),
            xytext=(E0 + 2.0,
                    gamma_fac / (np.pi * hwhm_fac) * (0.85 - 0.25 * (i % 2))),
            fontsize=7,
        )

    ax.plot(E, total, color="tab:blue", lw=1.2,
            label="atomx ab-initio $\\Gamma_A$ (Lorentzian, natural width)")
    ax.vlines([], 0, 0, color="crimson", lw=1.4,
              label="FAC AITable width (equivalent peak)")
    ax.set_title("G. Fe XVII KLL Auger electron spectrum "
                 "($2p^5 3l 3l' \\to 2p^6 + e^-$)")
    ax.set_xlabel("Auger electron energy (eV)")
    ax.set_ylabel("electrons s$^{-1}$ eV$^{-1}$ per resonance")
    ax.legend(fontsize=8)


def main():
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(14, 20))
    gs = fig.add_gridspec(4, 2)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)]
                     for r in range(3)])
    ax_auger = fig.add_subplot(gs[3, :])
    panel_profiles(axes[0, 0])
    panel_ab_initio_lines(axes[0, 1])
    panel_crm(axes[1, 0])
    panel_free_free(axes[1, 1])
    panel_uta(axes[2, 0], rng)
    panel_fano(axes[2, 1])
    panel_auger(ax_auger)
    fig.suptitle("atomx spectrum gallery", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
