"""Native configuration energies and consistent fields against the saved FAC panel.

No external thresholds or photon energies enter the native calculations.
External FAC/XATOM values are used only to evaluate ratios after calculation.
The HF and Gu panels remain distinct model choices.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src/atomx").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.methods.dirac_fock import DFSolver
from atomx.properties.transitions.fluorescence import fluorescence_rates
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization import subshell_photoionization


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(args):
    refs = json.loads((args.reference / "references.json").read_text())
    output = []
    for el in args.elements:
        Z, shells = b.CASES[el]
        for exchange in args.exchanges:
            cache = {}

            def solve(configuration, cache=cache, Z=Z, exchange=exchange, el=el):
                if configuration not in cache:
                    s = DFSolver.from_configuration(
                        Z,
                        configuration,
                        exchange=exchange,
                        ns=args.ns,
                        ks=9,
                        r_max=80,
                        nuclear="point",
                        starting_potential="tfda",
                    )
                    r = s.solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
                    if not r.converged:
                        raise RuntimeError(
                            f"{el} {exchange} {configuration} unconverged"
                        )
                    cache[configuration] = (s, r)
                return cache[configuration]

            start = time.perf_counter()
            s0, neutral = solve(b.config(shells))
            hole_states = {}
            for hole in ("1s", "2s", "2p"):
                initial = dict(shells)
                initial[hole] -= 1
                hole_states[hole] = solve(b.config(initial))
            photons = refs[el]["photons_eV"]
            thresholds = {"orbital": None, "frozen": {}, "relaxed": {}}
            for sh in neutral.subshells:
                shell = f"{sh.n}{'spdf'[sh.l]}"
                if shell not in hole_states:
                    continue
                final = dict(shells)
                final[shell] -= 1
                thresholds["frozen"][sh.label] = (
                    s0.frozen_configuration_energy(neutral, b.config(final))
                    - neutral.total_energy
                ) * HARTREE_EV
                thresholds["relaxed"][sh.label] = (
                    hole_states[shell][1].total_energy - neutral.total_energy
                ) * HARTREE_EV
            selected = [sh.label for sh in neutral.subshells if sh.n in (1, 2)]
            pirows = []
            for field, model in [
                (
                    "residual",
                    ContinuumModel.fac_residual_core(
                        n_grid=args.shooting_grid, r_max=80
                    ),
                ),
                (
                    "frozen",
                    ContinuumModel.fac_frozen_core(n_grid=args.shooting_grid, r_max=80),
                ),
            ]:
                for energy_mode, threshold in thresholds.items():
                    if field == "residual" and energy_mode != "orbital":
                        continue
                    result = subshell_photoionization(
                        neutral,
                        photons,
                        continuum_model=model,
                        orbitals=selected,
                        thresholds_eV=threshold,
                    )
                    shell_sigmas = {
                        shell: np.zeros(len(photons)) for shell in ("1s", "2s", "2p")
                    }
                    for sh in neutral.subshells:
                        if sh.label in result.sigma_by_orbital_a0_sq:
                            shell = f"{sh.n}{'spdf'[sh.l]}"
                            shell_sigmas[shell] += result.sigma_by_orbital_a0_sq[
                                sh.label
                            ] * (BOHR_RADIUS_M**2 / 1e-22)
                    for shell, values in shell_sigmas.items():
                        for i, (E, value) in enumerate(
                            zip(photons, values, strict=True)
                        ):
                            fac = refs[el]["fac_pi"][shell][i]
                            pirows.append(
                                dict(
                                    element=el,
                                    Z=Z,
                                    exchange=exchange,
                                    field=field,
                                    energy_mode=energy_mode,
                                    shell=shell,
                                    photon_eV=E,
                                    atomx_Mb=float(value),
                                    fac_Mb=fac,
                                    ratio=float(value / fac),
                                )
                            )
            radrows = []
            for hole, (solver, initial) in hole_states.items():
                initial_shells = dict(shells)
                initial_shells[hole] -= 1
                energies = {"orbital": None, "frozen": {}, "relaxed": {}}
                # Allowed downward sources; do not let unrelated reverse TR
                # lines or intrinsic open-valence vacancies enter this width.
                orbital = b.radiative(initial, hole)
                for channel in orbital["fluorescence"]:
                    source = channel.split("->")[0]
                    final = dict(shells)
                    final[source] -= 1
                    final_config = b.config(final)
                    _, relaxed = solve(final_config)
                    frozen = solver.frozen_configuration_energy(initial, final_config)
                    for tag, Ef in [
                        ("frozen", frozen),
                        ("relaxed", relaxed.total_energy),
                    ]:
                        energy = (initial.total_energy - Ef) * HARTREE_EV
                        if energy > 0:
                            energies[tag][b.nl(source)] = energy
                for mode, photon_map in energies.items():
                    rates = fluorescence_rates(
                        initial, b.nl(hole), photon_energies_eV=photon_map
                    )
                    fac = refs[el]["decay"][hole]["fac"]["radiative_per_s"]
                    radrows.append(
                        dict(
                            element=el,
                            Z=Z,
                            exchange=exchange,
                            hole=hole,
                            energy_mode=mode,
                            atomx_per_s=rates["per_vacancy_per_s"],
                            fac_per_s=fac,
                            ratio=rates["per_vacancy_per_s"] / fac if fac else None,
                        )
                    )
            record = dict(
                element=el,
                exchange=exchange,
                ns=args.ns,
                shooting_grid=args.shooting_grid,
                seconds=time.perf_counter() - start,
                configurations={
                    cfg: dict(
                        total_Ha=float(r.total_energy),
                        iterations=r.iterations,
                        orbital_eV={
                            k: v * HARTREE_EV for k, v in r.orbital_energies.items()
                        },
                    )
                    for cfg, (_, r) in cache.items()
                },
                thresholds_eV=thresholds,
                pi=pirows,
                fluorescence=radrows,
                occupation_model="statistical_jj_moments",
            )
            b.dump(args.output / f"{el}-{exchange}.json", record)
            output.append(record)
            write_csv(
                args.output / "pi.csv", [row for item in output for row in item["pi"]]
            )
            write_csv(
                args.output / "fluorescence.csv",
                [row for item in output for row in item["fluorescence"]],
            )
            print(
                el,
                exchange,
                "relaxed fluorescence/FAC",
                [
                    (r["hole"], r["ratio"])
                    for r in radrows
                    if r["energy_mode"] == "relaxed"
                ],
                "frozen/relaxed PI range",
                min(
                    r["ratio"]
                    for r in pirows
                    if r["field"] == "frozen" and r["energy_mode"] == "relaxed"
                ),
                max(
                    r["ratio"]
                    for r in pirows
                    if r["field"] == "frozen" and r["energy_mode"] == "relaxed"
                ),
                flush=True,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=ROOT / "docs/atomic_consistency_2026_10_02"
    )
    p.add_argument(
        "--reference", type=Path, default=ROOT / "docs/systematic_xray_2026_10_02"
    )
    p.add_argument(
        "--elements", nargs="+", default=list(b.CASES), choices=list(b.CASES)
    )
    p.add_argument("--exchanges", nargs="+", default=["hf", "gu"], choices=["hf", "gu"])
    p.add_argument("--ns", type=int, default=180)
    p.add_argument("--shooting-grid", type=int, default=16000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
