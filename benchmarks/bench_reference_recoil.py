"""Compare direct momentum and Breit-recoil radial integrals with live FAC.

The diagnostic executable only adds DumpVinti to sfac's command registry;
all integrals are evaluated by the supplied, unchanged libfac.a. The FAC
and Atomx orbitals solve a one-electron point-Coulomb Hamiltonian. These
matrix elements test recoil operators, not a many-electron isotope model.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from atomx.constants import chemical_symbols
from atomx.corrections.specific_recoil import relativistic_vinti_matrix, vinti_matrix
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.bench_reference_transitions import atomx_orbitals
from benchmarks.fac_runner import orbital_accuracy_commands, radial_grid_command

PAIRS = [
    ((1, -1), (2, 1)),
    ((1, -1), (2, -2)),
    ((2, 1), (3, 2)),
    ((2, -2), (3, 2)),
    ((2, -2), (3, -3)),
    ((3, 2), (4, 3)),
    ((3, -3), (4, 3)),
    ((3, -3), (4, -4)),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fac", type=Path, required=True)
    p.add_argument("--points", type=int, default=2000)
    p.add_argument("--fac-grid-density", type=float, default=1.0)
    p.add_argument("--fac-energy-tolerance", type=float)
    p.add_argument("--z", type=int, nargs="+", default=[1, 10, 26, 54, 82])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = dict(
        scope="point-Coulomb one-electron radial momentum and first-order Breit recoil",
        convention="compare absolute Vinti magnitudes (orbital phase invariant), and Breit/leading ratios",
        points=args.points,
        fac_grid_density=args.fac_grid_density,
        fac_energy_tolerance=args.fac_energy_tolerance,
        reference=dict(executable=str(args.fac), sha256=sha256(args.fac)),
        threads={
            k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        source_sha256={
            str(p.relative_to(root)): sha256(p) for p in (root / "src").rglob("*.py")
        },
        cases=[],
    )
    keys = sorted({key for pair in PAIRS for key in pair})
    for z in args.z:
        folder = args.out / f"Z{z:03d}"
        folder.mkdir()
        lines = orbital_accuracy_commands(args.fac_energy_tolerance) + [
            f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
            "SetBreit(0)",
            "SetSE(0)",
            "SetVP(0)",
            "SetMS(0,2)",
            radial_grid_command(args.points, density_scale=args.fac_grid_density),
            "Config('g','1s1')",
            "OptimizeRadial(['g'])",
        ]
        for a, b in PAIRS:
            for left, right in [(a, b), (b, a)]:
                lines.append(
                    f"DumpVinti('vinti.txt',{left[0]},{left[1]},{right[0]},{right[1]})"
                )
        (folder / "run.sf").write_text("\n".join(lines) + "\n")
        seconds = run_reference([str(args.fac), "run.sf"], folder)
        timings = []
        for _ in range(2):
            t = time.perf_counter()
            states = atomx_orbitals(z, keys, args.points)
            timings.append(time.perf_counter() - t)
        workspace = states[keys[0]].radial_state.correction_workspace()
        waves = [(states[k]._pa[0, 1:], states[k]._qa[0, 1:], k[1]) for k in keys]
        t = time.perf_counter()
        v = vinti_matrix(waves, workspace)
        vr = relativistic_vinti_matrix(waves, workspace, z)
        operator_seconds = time.perf_counter() - t
        results = []
        for n0, k0, n1, k1, v0, v1, v2 in np.loadtxt(folder / "vinti.txt"):
            a, b = (int(n0), int(k0)), (int(n1), int(k1))
            i, j = keys.index(a), keys.index(b)
            results.append(
                dict(
                    bra=a,
                    ket=b,
                    fac_terms=[v0, v1, v2],
                    atomx_leading=v[i, j],
                    atomx_breit=vr[i, j],
                    leading_magnitude_ratio=abs(v[i, j] / v0),
                    breit_magnitude_ratio=abs(vr[i, j] / (v0 + v1 + v2)),
                    correction_fraction_difference=(vr[i, j] - v[i, j]) / v[i, j]
                    - (v1 + v2) / v0,
                )
            )
        report["cases"].append(
            dict(
                z=z,
                fac_process_s=seconds,
                atomx_orbitals_s=timings,
                atomx_operators_s=operator_seconds,
                rows=results,
                atomx_momentum_antisymmetry_max=float(np.max(abs(v + v.T))),
            )
        )
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            z,
            "leading",
            max(abs(r["leading_magnitude_ratio"] - 1) for r in results),
            "Breit",
            max(abs(r["breit_magnitude_ratio"] - 1) for r in results),
            flush=True,
        )


if __name__ == "__main__":
    main()
