#!/usr/bin/env python3
"""Run a conditional fixed-charge tin source with preheat/energy audit.

Outputs refuse overwrites. This CLI does not submit a platform task or claim
experimental CE. See docs/EUV_SOURCE_MODEL.md for missing multi-ion physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomx.plasma.euv_source import (
    SlabSourceConfig,
    evolve_lte_slab,
    integrated_escaped_spectrum,
)
from atomx.spectrum.slab_transfer import slab_escaped_flux
from atomx.workflows.tin_source import load_tin_lte_plasma


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ["src/atomx/constants.py", "src/atomx/plasma/euv_source.py",
             "src/atomx/plasma/laser_absorption.py", "src/atomx/spectrum/slab_transfer.py",
             "src/atomx/workflows/tin_source.py", "tools/run_tin_source.py"]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rho-g-cm3", type=float, default=1e-4)
    parser.add_argument("--thickness-um", type=float, default=30)
    parser.add_argument("--temperature-eV", type=float, default=20)
    parser.add_argument("--laser-wavelength-um", type=float, default=10.6)
    parser.add_argument("--peak-intensity-W-cm2", type=float, default=1e8)
    parser.add_argument("--duration-ns", type=float, default=5)
    parser.add_argument("--bin-width-eV", type=float, default=0.05)
    parser.add_argument("--fwhm-eV", type=float, default=0.2)
    parser.add_argument("--coulomb-log", type=float, default=5)
    parser.add_argument("--rtol", type=float, default=2e-7)
    parser.add_argument("--no-expansion", action="store_true")
    parser.add_argument("--memory-gib", type=float, default=2)
    args = parser.parse_args()
    if (not np.isfinite(args.peak_intensity_W_cm2) or args.peak_intensity_W_cm2 <= 0
            or not np.isfinite(args.memory_gib) or args.memory_gib <= 0):
        parser.error("positive finite intensity and memory cap required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = source_hashes()
    config_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "input.json").write_text(json.dumps(dict(arguments=config_args, source_sha256=sources,
        thread_environment={k: os.environ.get(k) for k in
                            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")}), indent=2) + "\n")
    try:
        plasma, atomic, endpoints, folds = load_tin_lte_plasma(
            args.atomic_run, bin_width_eV=args.bin_width_eV, fwhm_eV=args.fwhm_eV,
            memory_cap_bytes=int(args.memory_gib * 1024**3))
        load_seconds = time.perf_counter() - started
        ni = args.rho_g_cm3 * 1000 / plasma.mass_kg
        config = SlabSourceConfig(ion_density_per_m3=ni, thickness_m=args.thickness_um * 1e-6,
            temperature_eV=args.temperature_eV, laser_wavelength_m=args.laser_wavelength_um * 1e-6,
            coulomb_log=args.coulomb_log, duration_s=args.duration_ns * 1e-9,
            expansion=not args.no_expansion, rtol=args.rtol)
        (args.output / "atomic_input.json").write_text(json.dumps(atomic, indent=2) + "\n")
        # Smooth sin² pulse with known incident fluence I_peak * duration / 2.
        intensity = args.peak_intensity_W_cm2 * 1e4
        def pulse(t):
            return intensity * np.sin(np.pi * t / config.duration_s)**2
        print(f"Loaded {atomic['levels']} partition states, {atomic['lines']} E1 lines, {atomic['bins']} bins", flush=True)
        sample_times = np.linspace(0, config.duration_s, 401)
        on = evolve_lte_slab(plasma, config, pulse, sample_times_s=sample_times)
        print(f"Laser-on complete: {on['rhs_evaluations']} RHS evaluations", flush=True)
        off = evolve_lte_slab(plasma, config, lambda t: 0.0, sample_times_s=sample_times)
        print(f"Laser-off preheat control complete: {off['rhs_evaluations']} RHS evaluations", flush=True)
        y, dark = on["state"], off["state"]
        fluence = y[3, -1]
        expected_fluence = intensity * config.duration_s / 2
        if abs(fluence / expected_fluence - 1) > 5e-6:
            raise ArithmeticError("incident fluence failed analytic sin²-pulse integral")
        snapshots, quadrature = {}, {}
        for label, r in (("on", on), ("off", off)):
            T, L = r["state"][:2, -1]
            p = r["final_populations"]
            e, a = plasma.coefficients(p, ni * config.thickness_m / L)
            snapshots[f"{label}_emissivity_W_m3_bin"] = e
            snapshots[f"{label}_absorption_per_m"] = a
            snapshots[f"{label}_escaped_W_m2_bin"] = slab_escaped_flux(e, a, L)
            snapshots[f"{label}_final_population"] = p
            snapshots[f"{label}_state"] = r["state"]
            snapshots[f"{label}_internal_J_m2"] = r["internal_J_m2"]
            snapshots[f"{label}_kinetic_J_m2"] = r["kinetic_J_m2"]
            snapshots[f"{label}_ledger_residual_J_m2"] = r["ledger_residual_J_m2"]
            spectrum, audit = integrated_escaped_spectrum(plasma, r, ni * config.thickness_m)
            snapshots[f"{label}_integrated_escaped_J_m2_bin"] = spectrum
            quadrature[label] = audit
        np.savez_compressed(args.output / "trajectory.npz", time_s=on["time_s"],
                            photon_energy_edges_eV=plasma.edges, **snapshots)

        def summary(r):
            last = r["state"][:, -1]
            return dict(final_temperature_eV=float(last[0]), final_thickness_um=float(last[1] * 1e6),
                        edge_velocity_m_s=float(last[2]),
                        initial_internal_J_m2=r["initial_internal_J_m2"],
                        energy_J_m2={name: float(value) for name, value in zip(r["state_rows"][3:], last[3:], strict=True)},
                        final_internal_J_m2=float(r["internal_J_m2"][-1]),
                        final_kinetic_J_m2=float(r["kinetic_J_m2"][-1]),
                        maximum_relative_energy_defect=r["maximum_relative_energy_defect"],
                        final_population_outside_radiative_endpoints=float(r["final_populations"][~endpoints].sum()),
                        final_fold_populations={str(k): float(r["final_populations"][folds == k].sum())
                                                for k in np.unique(folds)},
                        rhs_evaluations=r["rhs_evaluations"])

        result = dict(status="CONDITIONAL_FIXED_CHARGE_LTE_SOURCE_NOT_EXPERIMENTAL_CE",
            config=asdict(config), atomic_input_source_id=atomic["source_id"],
            source_unchanged=sources == source_hashes(), laser_on=summary(on), laser_off=summary(off),
            band_nm=[13.365, 13.635], angle_convention="one slab face = hemisphere; both faces = two hemispheres",
            gross_band_one_face_over_incident=float(0.5 * y[8, -1] / fluence),
            added_pulse_band_one_face_over_incident=float(0.5 * (y[8, -1] - dark[8, -1]) / fluence),
            absorbed_fraction=float(y[4, -1] / fluence), initial_critical_density_ratio=on["initial_critical_density_ratio"],
            sampled_spectrum_energy_audit=quadrature,
            load_and_table_seconds=load_seconds, wall_seconds=time.perf_counter() - started,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            limits=["fixed Sn12+; no ionization/recombination charge balance",
                    "preheated corona; charge preparation/prepulse energy excluded from incident denominator",
                    "bare DC CI; no core MBPT/Breit/QED; saved endpoint window only",
                    "bound-bound radiation only; no free-free/free-bound opacity or emissivity",
                    "one-zone one-temperature ideal EOS; phenomenological width and explicit Coulomb log",
                    "quasi-static homogeneous transfer; no spatial ablation, conduction or radiation pressure"])
        if not result["source_unchanged"]:
            raise RuntimeError("source changed during run; certification refused")
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(type=type(error).__name__, error=str(error),
            elapsed_seconds=time.perf_counter() - started), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
