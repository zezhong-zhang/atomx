"""Per-source Gu/FAC radial E1 and bound-grid controls at native energies."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b

from atomx.constants import HARTREE_EV
from atomx.methods.dirac_fock import DFSolver
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.fluorescence import fluorescence_rates
from atomx.properties.transitions.rates import einstein_A_eL

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "atomic_consistency_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02"


def run(ns):
    rows, validations = [], []
    strengths = list(csv.DictReader((HERE / "fluorescence_channels.csv").open()))
    for el, (Z, shells) in b.CASES.items():
        archive = json.loads((OLD / f"{el}-gu.json").read_text())
        for hole in ("1s", "2s", "2p"):
            channels = [
                r
                for r in strengths
                if r["element"] == el and r["hole"] == hole and r["exchange"] == "gu"
            ]
            if not channels:
                continue
            start = time.perf_counter()
            initial = dict(shells)
            initial[hole] -= 1
            config = b.config(initial)
            solver = DFSolver.from_configuration(
                Z,
                config,
                exchange="gu",
                ns=ns,
                ks=9,
                r_max=80,
                nuclear="point",
                starting_potential="tfda",
            )
            result = solver.solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
            assert result.converged
            energies = {
                b.nl(r["source"]): float(r["native_photon_eV"]) for r in channels
            }
            native = fluorescence_rates(result, b.nl(hole), photon_energies_eV=energies)
            raw = (
                REF / "raw" / el / "fac-k-radial-control"
                if hole == "1s"
                else HERE / "raw" / el / f"fac-{hole}-radial"
            )
            if hole != "1s":
                script = b.fac_header(el) + [
                    f"Config('{config}',group='initial')",
                    "ConfigEnergy(0)",
                    "OptimizeRadial(['initial'])",
                    "ConfigEnergy(1)",
                ]
                for sh in result.subshells:
                    script.append(
                        f"WaveFuncTable('wave-{sh.n}-{sh.kappa}.txt',{sh.n},{sh.kappa})"
                    )
                b.fac_job(raw, script)
            sampled = {}
            for sh in result.subshells:
                data = np.loadtxt(raw / f"wave-{sh.n}-{sh.kappa}.txt")
                sampled[sh.n, sh.kappa] = (data[:, 1], data[:, 4], data[:, 5])
            targets = [s for s in result.subshells if (s.n, s.l) == b.nl(hole)]
            for channel in channels:
                nl = b.nl(channel["source"])
                fac_radial_rate = 0.0
                for sh in result.subshells:
                    if (sh.n, sh.l) != nl:
                        continue
                    r, P, Q = sampled[sh.n, sh.kappa]
                    for h in targets:
                        rh, Ph, Qh = sampled[h.n, h.kappa]
                        a = angular_eL(sh.kappa, h.kappa, 1)
                        radial = simpson(
                            r
                            * (
                                P * np.interp(r, rh, Ph, left=0, right=0)
                                + Q * np.interp(r, rh, Qh, left=0, right=0)
                            ),
                            x=r,
                        )
                        fac_radial_rate += (
                            sh.occupation
                            / (2 * (2 * h.l + 1))
                            * einstein_A_eL(
                                energies[nl] / HARTREE_EV,
                                (a * radial) ** 2,
                                1,
                                2 * abs(sh.kappa) - 1,
                            )
                        )
                value = native["per_source_per_s"][nl]
                collapsed = float(channel["fac_same_native_photon_per_s"])
                rows.append(
                    dict(
                        element=el,
                        hole=hole,
                        source=channel["source"],
                        ns=ns,
                        native_photon_eV=energies[nl],
                        native_source_per_s=value,
                        fac_radial_same_photon_per_s=fac_radial_rate,
                        fac_resolved_strength_same_photon_per_s=collapsed,
                        native_over_fac_radial=value / fac_radial_rate,
                        fac_radial_over_fac_resolved_strength=fac_radial_rate
                        / collapsed,
                        native_over_fac_resolved_strength=value / collapsed,
                    )
                )
            old = next(
                r["atomx_per_s"]
                for r in archive["fluorescence"]
                if r["hole"] == hole and r["energy_mode"] == "relaxed"
            )
            validations.append(
                dict(
                    element=el,
                    hole=hole,
                    ns=ns,
                    native_per_s=native["per_vacancy_per_s"],
                    native_over_archived=native["per_vacancy_per_s"] / old,
                    energy_delta_Ha=result.total_energy
                    - archive["configurations"][config]["total_Ha"],
                )
            )
            print(
                el,
                hole,
                ns,
                time.perf_counter() - start,
                "native / archived",
                native["per_vacancy_per_s"] / old,
                flush=True,
            )
    for name, data in [
        ("fluorescence_radial", rows),
        ("fluorescence_grid", validations),
    ]:
        with (HERE / f"{name}_ns{ns}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=180)
    run(parser.parse_args().ns)
