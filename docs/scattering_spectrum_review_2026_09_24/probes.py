"""Reproduce the 2026-09-24 review findings without changing production code.

Run from the repository root with the project virtualenv. Results describe
current behavior, including defects; this is an audit, not an acceptance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch
import warnings

import numpy as np
from scipy.special import k0e

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def spectrum_probes():
    from atomx.spectrum.crm_synth import crm_emission_spectrum
    from atomx.spectrum.spectrum import Spectrum
    from atomx.spectrum.continuum import gaunt_ff_born, free_free_emissivity
    from atomx.scattering.polarisation import (
        alignment_from_msublevel_cross_sections as alignment,
        linear_polarisation, polarised_emission_intensity,
    )
    from atomx.scattering.msub import m_resolved_weights
    from atomx.properties.transitions.two_photon import two_photon_spectrum

    def crm(rates):
        return NS(levels=[NS(energy_eV=0), NS(energy_eV=0),
                          NS(energy_eV=1239.8419843320026 / 100)],
                  transitions=[NS(upper=2, lower=i, A_rate=a)
                               for i, a in enumerate(rates)],
                  steady_state=lambda *a, **kw: np.array([0., 0., 1.]))

    args = dict(n_e_cm3=1e10, T_e_eV=10, lam_min_nm=99.9,
                lam_max_nm=100.1, n_points=2001, T_ion_K=1)
    spectra = {mode: crm_emission_spectrum(crm([1e8]), **args,
                                         line_resolution=mode, ci_result=object())
               for mode in ("uta", "ci", "invalid")}

    def widths(rates, **kw):
        calls = []
        with patch.object(Spectrum, "add_line",
                          lambda self, center, intensity, **width:
                          calls.append(dict(center_nm=center, **width))):
            crm_emission_spectrum(crm(rates), **(args | kw))
        return calls

    branches = widths([1e12, 9e12])
    expected_gamma = branches[0]["center_nm"] ** 2 / 2.99792458e17 * 1e13 / (4*np.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.array([100., 1500., 2000.])
        gaunt = gaunt_ff_born(x)
        ff = free_free_emissivity(x, 1., 1.)
    A = alignment([1., 0.], 1.)
    return dict(
        ci_resolution=dict(ci_equals_uta=bool(np.array_equal(spectra["ci"].values, spectra["uta"].values)),
                           invalid_equals_uta=bool(np.array_equal(spectra["invalid"].values, spectra["uta"].values)),
                           ci_groups_present="ci_groups" in spectra["ci"].metadata),
        widths=dict(branches=branches, expected_both_natural_gamma_nm=expected_gamma,
                    with_stark=widths([1e12], stark_hwhm_nm=.01),
                    expected_single_branch_plus_stark_nm=expected_gamma/10+.01,
                    grid_201=widths([1e8], n_points=201),
                    grid_2001=widths([1e8], n_points=2001)),
        polarisation=dict(pure_M0_alignment=A, pure_M0_P=linear_polarisation(A, 1.),
                          isotropic_combined_M_alignment=alignment([1., 2.], 1.),
                          I_90_for_P_half=polarised_emission_intensity(.5, np.pi/2),
                          msub_J0_to_J1=m_resolved_weights(0., 1.)),
        free_free=dict(x=x.tolist(), gaunt=[None if not np.isfinite(v) else float(v) for v in gaunt],
                       scaled_bessel_gaunt=(np.sqrt(3)/np.pi*k0e(x/2)).tolist(),
                       emissivity=[None if not np.isfinite(v) else float(v) for v in ff]),
        known_two_photon_shape=dict(y=[.25, .5, .75],
                                   values=two_photon_spectrum(np.array([.25, .5, .75]), 1).tolist()),
    )


def scattering_probes():
    from atomx import Atom, Basis
    from atomx.constants import HARTREE_EV
    from atomx.scattering.distorted_wave import dw_excitation_cross_section
    from atomx.scattering.electron_ionization import eii_cross_section
    from atomx.scattering.r_matrix import compute_multichannel, heplus_target_states
    from atomx.scattering.r_matrix.cc_outer import S_matrix_from_K
    g = Atom(2, charge=1, config="1s1", engine="bspline", corrections="none", basis=Basis(ns=45))
    e = Atom(2, charge=1, config="2p1", engine="bspline", corrections="none", basis=Basis(ns=30))
    g.solve()
    e.solve()
    try:
        dw_excitation_cross_section(g, e, np.array([100.]), kappa_max=1, topup=False)
        grid = dict(error=None)
    except Exception as exc:
        grid = dict(error=type(exc).__name__, message=str(exc))
    cutoff = -5 * max(v for v in g._result.orbital_energies.values() if v < 0) * HARTREE_EV
    E = cutoff * np.array([1-1e-6, 1+1e-6])
    auto = eii_cross_section(g._result, E)
    # Isolate the detailed-balance factor from radial/multipole accuracy.
    from atomx.scattering.radiative_recombination import rr_cross_section
    from atomx.constants import ALPHA
    high_eV = np.array([1e4, 1e5, 5e5])
    with patch("atomx.scattering.radiative_recombination.photoionization_cross_section",
               return_value=np.ones(3)):
        rr = rr_cross_section(g._result, high_eV, multipoles=("E1",),
                              g_initial=1, g_final=2)
    kinetic = high_eV / HARTREE_EV
    binding = -g._result.orbital_energies["1s1/2"]
    expected = 2 * ALPHA**2 * (kinetic+binding)**2 / (kinetic*(2+ALPHA**2*kinetic))
    cc = compute_multichannel(list(heplus_target_states(Z=2).values()),
                              L_total=0, S_total=0, parity=1,
                              energies_au=np.array([-.8, -.78]),
                              n_basis=40, n_gauss=8, l_c_max=1,
                              include_l2_pseudostates=True)
    diffs = [float(np.max(np.abs(m.S_matrix-S_matrix_from_K(m.K_matrix)))) for m in cc.matchings]
    return dict(mixed_bound_grids=grid,
                eii_auto_seam=dict(energy_eV=E.tolist(), sigma_cm2=auto.sigma_total_cm2.tolist(),
                                   jump_ratio=float(auto.sigma_total_cm2[1]/auto.sigma_total_cm2[0])),
                rr_milne=dict(energy_eV=high_eV.tolist(),
                              implemented_factor=rr.sigma_total_a0_sq.tolist(),
                              relativistic_factor=expected.tolist(),
                              ratio=(rr.sigma_total_a0_sq/expected).tolist(),
                              probe="PI set to unit cross section to isolate phase-space factor"),
                r_matrix=dict(energy_au=cc.energies_au.tolist(), S_K_max_abs_mismatch=diffs,
                              injected_poles=cc.provenance["l2_pseudostates"]))


def auger_failure_probe():
    fixture = runpy.run_path(str(ROOT / "tests/test_auger_mscheme_ci.py"))["manifold_j72"].__wrapped__
    normal = fixture()
    with warnings.catch_warnings(record=True) as caught:
        with patch("atomx.scattering.autoionization.mscheme_ci._continuum_on_grid",
                   side_effect=RuntimeError("audit injected continuum failure")) as injected:
            failed = fixture()
    return dict(normal_widths_per_s=[v.gamma_auger_per_s for v in normal],
                failed_widths_per_s=[v.gamma_auger_per_s for v in failed],
                injected_failures=injected.call_count,
                warnings=[str(w.message) for w in caught])


def performance_probe():
    from atomx.spectrum import Spectrum
    from atomx.lineshapes.profiles import gaussian
    grid = np.linspace(1., 101., 10001)
    rng = np.random.default_rng(17)
    centers = rng.uniform(2., 100., 2000)
    intensities = rng.uniform(.1, 1., len(centers))
    width = .02

    def full():
        s = Spectrum.from_grid(grid)
        for center, intensity in zip(centers, intensities):
            s.add_line(center, intensity, sigma_nm=width)
        return s.values

    def windowed():
        out = np.zeros_like(grid)
        for center, intensity in zip(centers, intensities):
            lo, hi = np.searchsorted(grid, [center-8*width, center+8*width])
            out[lo:hi] += intensity * gaussian(grid[lo:hi], center, width)
        return out

    full(); windowed()  # warm imports and code paths
    timings = {}
    arrays = {}
    for name, fn in (("full_grid", full), ("eight_sigma_window", windowed)):
        times = []
        for _ in range(5):
            start = time.perf_counter()
            arrays[name] = fn()
            times.append(time.perf_counter()-start)
        timings[name] = float(np.median(times))
    baseline = arrays["full_grid"]
    candidate = arrays["eight_sigma_window"]
    return dict(n_grid=len(grid), n_lines=len(centers), sigma_nm=width,
                median_seconds=timings,
                speedup=timings["full_grid"]/timings["eight_sigma_window"],
                max_error_relative_to_peak=float(np.max(np.abs(baseline-candidate))/np.max(baseline)),
                integrated_relative_error=float(abs(np.trapezoid(baseline-candidate, grid))/np.trapezoid(baseline, grid)),
                scope="Gaussian only; Lorentzian/Voigt tails require a different error budget")


def fac_atomx_probe(live_json):
    from bench_fac_dw_excitation import _threshold_eV, run_atomx, SEQUENCES, ENERGY_FACTORS
    fac = json.loads(Path(live_json).read_text())
    for Z in (10, 26, 54):
        start = time.perf_counter()
        threshold = _threshold_eV(Z, SEQUENCES["he-like"])
        omega, converged = run_atomx(Z, SEQUENCES["he-like"], np.array(ENERGY_FACTORS)*threshold, threshold)
        f = fac[str(Z)]
        f.update(atomx_omega=omega.tolist(), atomx_threshold_eV=threshold,
                 atomx_converged=bool(converged), atomx_seconds=time.perf_counter()-start,
                 relative_error=(omega/np.array(f["omega"])-1).tolist())
    return fac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-fac-json", type=Path)
    args = parser.parse_args()
    results = dict(python=sys.version, numpy=np.__version__)
    for name, fn in (("spectrum", spectrum_probes), ("scattering", scattering_probes),
                     ("auger_failure", auger_failure_probe), ("performance", performance_probe)):
        start = time.perf_counter()
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = dict(probe_error=repr(exc))
        results[name]["elapsed_seconds"] = time.perf_counter()-start
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        print(name, "completed", flush=True)
    if args.live_fac_json:
        results["live_fac_comparison"] = fac_atomx_probe(args.live_fac_json)
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
