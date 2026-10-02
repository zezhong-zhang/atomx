#!/usr/bin/env python3
"""Live, auditable DBSR_HF/RADIAL/FAC structure comparisons.

Raw inputs/outputs are retained. Point nuclei and corrections off isolate
Dirac–Coulomb/local-exchange mechanisms. FAC comparisons explicitly retain
its different orbital optimization model. This is a structure panel, not
validation of the entire spectrum/property/scattering API.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from atomx import Atom
from atomx.config.subshell import Subshell
from atomx.constants import HARTREE_EV, SPEED_OF_LIGHT, chemical_symbols
from atomx.numerics.shooting.local import ShootingSolver
from atomx.reference_data.structure_outputs import (
    orbital_differences,
    parse_dbsr_configurations,
    parse_dbsr_summary,
    parse_radial_summary,
)
from benchmarks.bench_shooting_vs_radial import parse_dhfs_input
from benchmarks.fac_runner import parse_e0, parse_levels

#: sfac on this Mac needs the relocated gcc runtime on DYLD_LIBRARY_PATH
#: (same mechanism as benchmarks/fac_runner.py; set FAC_DYLD_LIB).
FAC_DYLD_LIB = os.environ.get('FAC_DYLD_LIB', '')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_reference(command, directory, *, stdin=None, timeout=180):
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=directory,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        for name, content in (("stdout", exc.stdout), ("stderr", exc.stderr)):
            if isinstance(content, bytes):
                content = content.decode(errors="replace")
            (directory / f"{name}.txt").write_text(content or "")
        (directory / "process.json").write_text(
            json.dumps(
                dict(
                    status="timeout",
                    elapsed_s=time.perf_counter() - start,
                    limit_s=timeout,
                )
            )
        )
        raise
    elapsed = time.perf_counter() - start
    (directory / "stdout.txt").write_text(proc.stdout)
    (directory / "stderr.txt").write_text(proc.stderr)
    (directory / "process.json").write_text(
        json.dumps(dict(status="exited", elapsed_s=elapsed, exit_code=proc.returncode))
    )
    if proc.returncode:
        raise RuntimeError(f"reference exit {proc.returncode}: {proc.stderr[-1000:]}")
    return elapsed


def run_atomx(factory, repeats):
    timings = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        result = factory()
        timings.append(time.perf_counter() - start)
    return result, dict(
        first_solve_s=timings[0],
        warm_s=timings[1:],
        warm_median_s=statistics.median(timings[1:]),
    )


def compare(result, reference):
    delta = result.total_energy - reference["total_energy_Ha"]
    return dict(
        energy_difference_Ha=delta,
        energy_relative_difference=abs(delta / reference["total_energy_Ha"]),
        atomx_converged=bool(result.converged),
        orbitals=orbital_differences(
            result.orbital_energies, reference.get("orbital_energies_Ha", {})
        ),
    )


def dbsr_case(
    z,
    binary,
    folder,
    engine,
    repeats,
    *,
    grid=None,
    reference_c=None,
    reference_orbital_tol=1e-9,
    reference_tail_tol=1e-12,
    atomx_orbital_tol=1e-9,
    atomx_n_grid=2000,
):
    symbol = chemical_symbols[z]
    # Nuclear input belongs in the knot file: nuclear=point on the command
    # line is ignored by the supplied DBSR_HF Read_nuclear routine.
    grid = {} if grid is None else grid
    if any(
        key not in ("hi", "he", "hmax") or not math.isfinite(value) or value <= 0
        for key, value in grid.items()
    ):
        raise ValueError("DBSR grid controls must be positive hi, he or hmax")
    for value in (reference_orbital_tol, reference_tail_tol, atomx_orbital_tol):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "DF orbital and tail tolerances must be finite and positive"
            )
    if reference_tail_tol >= reference_orbital_tol:
        raise ValueError("DBSR tail cutoff must be smaller than its orbital tolerance")
    if not isinstance(atomx_n_grid, int) or atomx_n_grid < 50:
        raise ValueError("atomx_n_grid must be an integer >= 50")
    (folder / f"{symbol}.knot").write_text(
        "nuclear = point\n"
        + "".join(f"{key} = {value}\n" for key, value in grid.items())
    )
    command = [
        str(binary),
        f"an={z}",
        "term=AV",
        "mbreit=0",
        "scf_tol=1e-12",
        f"orb_tol={reference_orbital_tol:g}",
        f"end_tol={reference_tail_tol:g}",
        "max_it=100",
    ]
    elapsed = run_reference(command, folder)
    if not re.search(r"nuclear\s*=\s*point", (folder / f"{symbol}.knot").read_text()):
        raise ValueError("DBSR did not retain the requested point nucleus")
    ref = parse_dbsr_summary((folder / f"{symbol}.log").read_text())
    shells, ensemble = parse_dbsr_configurations(
        (folder / f"{symbol}.conf").read_text()
    )
    if abs(sum(s.occupation for s in shells) - z) > 1e-10:
        raise ValueError(
            "DBSR electron count differs from the requested neutral system"
        )

    def solve():
        if engine == "shooting":
            from atomx.numerics.shooting.hartree_fock import ShootingHFSolver

            solver = ShootingHFSolver(
                z, shells, jj_configs=ensemble, nuclear="point", n_grid=atomx_n_grid
            )
        else:
            from atomx.methods.dirac_fock import DFSolver
            from atomx.numerics.bspline.basis import prepare_bspline_data
            from atomx.numerics.bspline.knots import make_physics_knots

            knots, count = make_physics_knots(Z=z, ks=9, nuclear="point", **grid)
            solver = DFSolver(
                z,
                shells,
                ns=count,
                ks=9,
                jj_configs=ensemble,
                nuclear="point",
                include_breit=False,
                include_qed=False,
            )
            solver.t = knots
            solver.bsp_data = prepare_bspline_data(
                knots, count, 9, kinetic_balance=True
            )
            solver.kinetic_balance = True
            solver.r_max = float(knots[-1])
            solver._precompute()
        return solver.solve(tol=1e-9, max_iter=100, orbital_tolerance=atomx_orbital_tol)

    result, timing = run_atomx(solve, repeats)
    waves = result.waves
    radii = {
        label: orbital.expectation_r(waves.grid)
        for label, orbital in waves.orbitals.items()
    }
    radius_difference = {
        label: radii[label] - value
        for label, value in ref["orbital_mean_radius_bohr"].items()
        if label in radii
    }
    return dict(
        reference=ref,
        atomx_energy_Ha=result.total_energy,
        atomx_calculation=result.calculation_spec.to_dict(),
        atomx_solver_diagnostics=result.solver_diagnostics,
        final_fock_orbitals=(
            orbital_differences(
                result.orbital_energies, ref["orbital_fock_expectations_Ha"]
            )
            if ref["orbital_fock_expectations_Ha"]
            else None
        ),
        mean_radius=dict(
            atomx_bohr=radii,
            difference_bohr=radius_difference,
            missing_calculated=sorted(
                set(ref["orbital_mean_radius_bohr"]) - radii.keys()
            ),
            all_within_reference_printing=(
                bool(radius_difference)
                and len(radius_difference) == len(ref["orbital_mean_radius_bohr"])
                and all(
                    abs(value) <= ref["printed_radius_rounding_bohr"][label]
                    for label, value in radius_difference.items()
                )
            ),
        ),
        atomx_timing=timing,
        reference_s=elapsed,
        **compare(result, ref),
        reference_converged=(
            ref["residuals"]["orbital"] <= reference_orbital_tol
            and ref["residuals"]["energy"] <= 1e-12
        ),
        contract=dict(
            comparison="same_configuration_average_Dirac_Fock",
            nuclear="point",
            corrections="none",
            state="neutral integer-jj ensemble from DBSR .conf",
            engine=engine,
            requested_reference_grid=grid,
            reference_controls=dict(
                scf_tolerance=1e-12,
                orbital_tolerance=reference_orbital_tol,
                tail_cutoff=reference_tail_tol,
                max_iter=100,
            ),
            orbital_tolerance_semantics="DBSR uses a relative maximum P-coefficient update; "
            "Atomx uses a dimensionless P/Q L2 update. Equal numeric thresholds do not "
            "establish equal error; compare observables and radial refinements.",
            orbital_comparison="Printed e(nl) may be a retired-iteration eigenvalue; compare final Fock expectations separately when available. Closed-shell occupied rotations may change diagonal expectations.",
            speed_of_light_au=dict(
                atomx=SPEED_OF_LIGHT,
                reference=reference_c,
                matched=None if reference_c is None else reference_c == SPEED_OF_LIGHT,
            ),
            ensemble=[
                dict(
                    weight=c.weight,
                    shells=[(s.n, s.kappa, s.occupation) for s in c.subshells],
                )
                for c in ensemble
            ],
        ),
    )


def radial_case(z, directory, folder, repeats):
    inp = parse_dhfs_input(directory / f"Z{z:03d}.in")
    raw = re.sub(r"^C4\s*\([^)]*\)", "C4 (  0.000000)", inp.raw, flags=re.M)
    (folder / "input.in").write_text(raw)
    elapsed = run_reference([str(directory / "DHFS")], folder, stdin=raw)
    text = (folder / "dhfs.dat").read_text()
    if "Nuclear model: point nucleus" not in text:
        raise ValueError("RADIAL did not retain the requested point nucleus")
    ref = parse_radial_summary(text)
    shells = [
        Subshell(n, -(ell + 1) if twoj == 2 * ell + 1 else ell, occ)
        for n, ell, twoj, occ in inp.shells
    ]

    def solve():
        return ShootingSolver(
            z,
            shells,
            n_grid=inp.n_grid,
            r_max=inp.r_max,
            alpha_x=inp.alpha_x,
            aw=0.0,
            equation="dirac",
        ).solve(max_iter=300, tol=1e-9)

    result, timing = run_atomx(solve, repeats)
    stdout = (folder / "stdout.txt").read_text()
    residuals = re.findall(r"CHI\s*=\s*([-+0-9.DE]+)", stdout)
    ref["last_printed_scf_residual"] = (
        float(residuals[-1].replace("D", "E")) if residuals else None
    )
    return dict(
        reference=ref,
        atomx_energy_Ha=result.total_energy,
        atomx_timing=timing,
        reference_s=elapsed,
        **compare(result, ref),
        reference_converged=(
            ref["last_printed_scf_residual"] is not None
            and ref["last_printed_scf_residual"] <= 1e-9
        ),
        contract=dict(
            comparison="same_local_Slater_model",
            nuclear="point",
            corrections="none",
            alpha_x=inp.alpha_x,
            n_grid=inp.n_grid,
            r_max=inp.r_max,
            state="neutral fixed occupations",
            energy="HF expectation on local-exchange orbitals",
        ),
    )


def fac_case(z, binary, folder, engine, repeats):
    script = (
        f"SetAtom('{chemical_symbols[z]}', {z}, -1, 0)\n"
        "SetBreit(0)\nSetSE(0)\nSetVP(0)\nSetMS(0,0)\nConfig('g','1s2')\n"
        "OptimizeRadial(['g'])\nStructure('levels.b',['g'])\nPrintTable('levels.b','levels.txt',1)\n"
    )
    (folder / "run.sf").write_text(script)
    elapsed = run_reference([str(binary), "run.sf"], folder)
    levels = parse_levels(folder / "levels.txt")
    e0 = parse_e0(folder / "levels.txt")
    if e0 is None or len(levels) != 1 or levels[0]["TWOJ"] != 0 or levels[0]["P"] != 0:
        raise ValueError("FAC He-like ground state identity was not established")
    ref = dict(
        total_energy_Ha=(e0 + levels[0]["ENERGY_eV"]) / HARTREE_EV, levels=levels
    )
    result, timing = run_atomx(
        lambda: Atom(
            z,
            charge=z - 2,
            config="1s2",
            engine=engine,
            nuclear="point",
            corrections="none",
        ).solve(tol=1e-9, max_iter=100),
        repeats,
    )
    return dict(
        reference=ref,
        atomx_energy_Ha=result.total_energy,
        atomx_timing=timing,
        reference_s=elapsed,
        **compare(result, ref),
        reference_converged=None,
        contract=dict(
            comparison="different_orbital_models",
            nuclear="point",
            corrections="none",
            state="He-like 1s2 J=0 even",
            atomx="nonlocal DF",
            fac="FAC optimized central potential",
            limitation="FAC output does not certify the radial residual; this is not a numerical-equivalence gate",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel", type=int, nargs="+", default=[2, 6, 10, 54])
    parser.add_argument(
        "--references",
        nargs="+",
        choices=["dbsr", "radial", "fac"],
        default=["dbsr", "radial", "fac"],
    )
    parser.add_argument("--dbsr", type=Path, default=os.environ.get("DBSR_HF_BIN"))
    parser.add_argument(
        "--radial", type=Path, default=os.environ.get("RADIAL_DHFS_DIR")
    )
    parser.add_argument("--fac", type=Path, default=os.environ.get("FAC_SFAC_PATH"))
    parser.add_argument("--engine", choices=["shooting", "bspline"], default="shooting")
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument(
        "--dbsr-grid", type=float, nargs=3, metavar=("HI", "HE", "HMAX")
    )
    parser.add_argument(
        "--dbsr-c",
        type=float,
        help="Verified c in the supplied DBSR build (atomic units)",
    )
    parser.add_argument("--dbsr-orbital-tol", type=float, default=1e-9)
    parser.add_argument("--dbsr-tail-tol", type=float, default=1e-12)
    parser.add_argument(
        "--dbsr-atomx-orbital-tol",
        type=float,
        default=1e-9,
        help="Independent Atomx L2 update threshold for the DBSR comparison",
    )
    parser.add_argument(
        "--dbsr-atomx-n-grid",
        type=int,
        default=2000,
        help="Atomx shooting points for the DBSR comparison",
    )
    args = parser.parse_args()
    if args.warm_repeats < 1:
        parser.error("--warm-repeats must be at least one")
    if args.out.exists():
        parser.error("output directory already exists; use a new run directory")
    args.out.mkdir(parents=True)
    manifest = dict(
        scope="structure panel; spectra/properties/scattering remain separate requirements",
        python=sys.version,
        platform=platform.platform(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        timing="Reference includes process startup/output; Atomx includes construction/solve, excludes imports. First is not guaranteed JIT-cold.",
        threads={
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        source_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
        runner_sha256=sha256(__file__),
        references={},
        rows=[],
    )
    for name in args.references:
        location = getattr(args, name)
        binary = location / "DHFS" if name == "radial" and location else location
        if binary is None or not binary.is_file():
            manifest["references"][name] = dict(status="unavailable", path=str(binary))
            continue
        location = location.resolve()
        binary = binary.resolve()
        manifest["references"][name] = dict(path=str(binary), sha256=sha256(binary))
        for z in args.panel:
            folder = args.out / f"{name}-{z:03d}"
            folder.mkdir()
            row = dict(reference_code=name, Z=z, artifacts=folder.name)
            try:
                if name == "dbsr":
                    payload = dbsr_case(
                        z,
                        binary,
                        folder,
                        args.engine,
                        args.warm_repeats,
                        grid=None
                        if args.dbsr_grid is None
                        else dict(zip(("hi", "he", "hmax"), args.dbsr_grid, strict=False)),
                        reference_c=args.dbsr_c,
                        reference_orbital_tol=args.dbsr_orbital_tol,
                        reference_tail_tol=args.dbsr_tail_tol,
                        atomx_orbital_tol=args.dbsr_atomx_orbital_tol,
                        atomx_n_grid=args.dbsr_atomx_n_grid,
                    )
                elif name == "radial":
                    payload = radial_case(z, location, folder, args.warm_repeats)
                else:
                    payload = fac_case(
                        z, binary, folder, args.engine, args.warm_repeats
                    )
                row.update(payload, status="measured")
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            manifest["rows"].append(row)
            (args.out / "results.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            print(
                name,
                z,
                row["status"],
                row.get("energy_difference_Ha", row.get("error")),
                flush=True,
            )
    (args.out / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if any(r["status"] != "measured" for r in manifest["rows"]) or any(
        "status" in r for r in manifest["references"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
