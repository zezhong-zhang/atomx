"""Gu neutral controls on archived native thresholds; no production changes."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b
from fac_runner import parse_levels

import atomx.scattering.radiative_recombination as rr
from atomx.constants import BOHR_RADIUS_M
from atomx.methods.dirac_fock import DFSolver
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization import subshell_photoionization

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "systematic_xray_2026_10_02"
OLD = HERE.parent / "atomic_consistency_2026_10_02"
refs = json.loads((REF / "references.json").read_text())


def run(el, ns=180):
    Z, shells = b.CASES[el]
    start = time.perf_counter()
    archive = json.loads((OLD / f"{el}-gu.json").read_text())
    solver = DFSolver.from_configuration(
        Z,
        b.config(shells),
        exchange="gu",
        ns=ns,
        ks=9,
        r_max=80,
        nuclear="point",
        starting_potential="tfda",
    )
    result = solver.solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    selected = [s.label for s in result.subshells if s.n <= 2]
    native = archive["thresholds_eV"]["relaxed"]
    common, resolved = {}, {}
    x_orbitals = b.parse_xatom(REF / "raw" / el / "xatom-neutral/result.out")[
        "orbitals_eV"
    ]
    x_thresholds = {
        s.label: -x_orbitals[f"{s.n}{'spdf'[s.l]}"]
        for s in result.subshells
        if s.label in selected
    }
    for shell in ("1s", "2s", "2p"):
        levels = parse_levels(REF / "raw" / el / f"fac-pi-{shell}/levels.txt")
        ni = b.csf_count(shells)
        initial, final = levels[:ni], levels[ni:]
        Ei = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in initial) / sum(
            r["TWOJ"] + 1 for r in initial
        )
        Ef = sum(r["ENERGY_eV"] * (r["TWOJ"] + 1) for r in final) / sum(
            r["TWOJ"] + 1 for r in final
        )
        for sh in result.subshells:
            if f"{sh.n}{'spdf'[sh.l]}" != shell:
                continue
            common[sh.label] = Ef - Ei
            # For a closed-shell J=0 neutral, ion J uniquely identifies
            # the j of the hole. Open shells do not permit this mapping.
            if len(initial) == 1 and initial[0]["TWOJ"] == 0:
                candidates = [r for r in final if r["TWOJ"] == 2 * abs(sh.kappa) - 1]
                assert len(candidates) == 1
                resolved[sh.label] = candidates[0]["ENERGY_eV"] - Ei
            else:
                resolved[sh.label] = Ef - Ei
    variants = {}
    for tag, threshold, ngrid in [
        ("native_16000", native, 16000),
        ("native_32000", native, 32000),
        ("fac_centroid_control", common, 16000),
        ("fac_jj_threshold_control", resolved, 16000),
        ("xatom_threshold_control", x_thresholds, 16000),
    ]:
        pi = subshell_photoionization(
            result,
            refs[el]["photons_eV"],
            continuum_model=ContinuumModel.fac_frozen_core(n_grid=ngrid, r_max=80),
            orbitals=selected,
            thresholds_eV=threshold,
        )
        variants[tag] = {s: [0.0] * 3 for s in ("1s", "2s", "2p")}
        for sh in result.subshells:
            if sh.label not in selected:
                continue
            shell = f"{sh.n}{'spdf'[sh.l]}"
            values = pi.sigma_by_orbital_a0_sq[sh.label] * (BOHR_RADIUS_M**2 / 1e-22)
            variants[tag][shell] = (values + variants[tag][shell]).tolist()
    # Swap only the radial operator in this process, restoring it afterwards.
    # This removes finite-wavelength effects for comparison with NR XATOM.
    original = rr.radial_eL
    try:

        def static(*args, **kwargs):
            kwargs["omega"] = None
            return radial_eL(*args, **kwargs)

        rr.radial_eL = static
        for tag, thresholds in [
            ("long_wavelength_control", native),
            ("xatom_threshold_static_control", x_thresholds),
        ]:
            pi = subshell_photoionization(
                result,
                refs[el]["photons_eV"],
                continuum_model=ContinuumModel.fac_frozen_core(n_grid=16000, r_max=80),
                orbitals=selected,
                thresholds_eV=thresholds,
            )
            variants[tag] = {s: [0.0] * 3 for s in ("1s", "2s", "2p")}
            for sh in result.subshells:
                if sh.label in selected:
                    shell = f"{sh.n}{'spdf'[sh.l]}"
                    values = pi.sigma_by_orbital_a0_sq[sh.label] * (
                        BOHR_RADIUS_M**2 / 1e-22
                    )
                    variants[tag][shell] = (values + variants[tag][shell]).tolist()
    finally:
        rr.radial_eL = original
    output = dict(
        element=el,
        variants_Mb=variants,
        native_thresholds_eV=native,
        fac_centroid_thresholds_eV=common,
        fac_jj_thresholds_eV=resolved,
        xatom_thresholds_eV=x_thresholds,
        resolved_jj_control_valid=(el not in ("C", "Fe")),
        neutral_energy_Ha=result.total_energy,
        neutral_minus_archived_Ha=result.total_energy
        - archive["configurations"][b.config(shells)]["total_Ha"],
        ns=ns,
        seconds=time.perf_counter() - start,
        control_only=True,
    )
    suffix = "" if ns == 180 else f"-ns{ns}"
    (HERE / f"pi-{el}{suffix}.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        el,
        output["seconds"],
        "s; neutral energy delta",
        output["neutral_minus_archived_Ha"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, default=180)
    parser.add_argument("elements", nargs="*", default=list(b.CASES))
    args = parser.parse_args()
    for el in args.elements:
        run(el, args.ns)
