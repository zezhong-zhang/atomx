"""FAC photon operators at exact line frequencies, with shared/independent spinors.

The diagnostic frontend calls unmodified MultipoleRadialFR and exports the
radial orbitals. Each call clears the photon cache and installs one exact
frequency, isolating the operator from transition-table interpolation.
"""

import argparse
import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

from atomx.constants import chemical_symbols
from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import (
    radial_eL,
    radial_eL_velocity,
    radial_mL,
)
from atomx.properties.transitions.rates import einstein_A_eL
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.bench_reference_transitions import atomx_orbitals
from benchmarks.fac_runner import orbital_accuracy_commands, radial_grid_command


def channels():
    for L in range(1, 5):
        for kappa in (L, -L - 1):
            for gauge in (1, 2):
                yield L + 1, kappa, -L, gauge
        yield max(2, L), -L, L, 2


def read_common_spinors(folder, keys, order=5):
    """Integrate FAC spinors on their declared support, without a false tail.

    FAC Integrate stops at the smaller orbital ilast. Zero padding beyond
    ilast must not enter the interpolant or create an extra half-cell tail.
    """
    x, w = np.polynomial.legendre.leggauss(8)
    tables = {}
    ends = {}
    reference_grid = None
    for n, k in keys:
        filename = folder / f"wave-{n}-{k}.txt"
        data = np.loadtxt(filename)
        if reference_grid is None:
            reference_grid = data[:, 0]
        np.testing.assert_array_equal(data[:, 0], reference_grid)
        ends[n, k] = int(re.search(r"ilast=(\d+)", filename.read_text()).group(1)) + 1
        tables[n, k] = data
    grid = np.r_[0.0, reference_grid[: max(ends.values())]]
    half = np.diff(grid)[:, None] / 2
    nodes = (grid[:-1, None] + half * (1 + x)).ravel()
    weights = (half * w).ravel()
    waves = {}
    for key, data in tables.items():
        data = data[: ends[key]]
        r = np.r_[0.0, data[:, 0]]
        supported = nodes <= r[-1]
        components = []
        for j in (1, 2):
            values = np.zeros_like(nodes)
            values[supported] = make_interp_spline(r, np.r_[0.0, data[:, j]], k=order)(
                nodes[supported]
            )
            components.append(values)
        waves[key] = tuple(components)
    return waves, make_grid_workspace(nodes, weights)


def rate(waves, workspace, upper, multipole, gauge, omega):
    lower = (1, -1)
    L = abs(multipole)
    if multipole < 0:
        kernel = radial_eL_velocity if gauge == 1 else radial_eL
        radial = kernel(
            *waves[lower],
            *waves[upper],
            workspace,
            L,
            kappa_i=lower[1],
            kappa_f=upper[1],
            omega=omega,
        )
        angular = angular_eL(upper[1], lower[1], L)
    else:
        radial = radial_mL(*waves[lower], *waves[upper], workspace, L, omega=omega)
        angular = angular_mL(upper[1], lower[1], L)
    return einstein_A_eL(omega, (radial * angular) ** 2, L, 2 * abs(upper[1]) - 1)


def write_fixture(report, source, destination):
    """Retain raw reference spinors through ilast, never fitted Atomx rates."""
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    fixture = {
        key: report[key]
        for key in (
            "scope",
            "points",
            "fac_grid_density",
            "fac_energy_tolerance",
            "reference",
        )
    }
    fixture.update(source_artifact=str(source), cases=[])
    for case in report["cases"]:
        if case["z"] not in (1, 10, 82):
            continue
        z = case["z"]
        folder = source / f"Z{z:03d}"
        orbitals = []
        keys = sorted({(1, -1)} | {tuple(row["upper"]) for row in case["rows"]})
        for n, kappa in keys:
            path = folder / f"wave-{n}-{kappa}.txt"
            header = path.open().readline()
            end = int(re.search(r"ilast=(\d+)", header).group(1)) + 1
            key = f"Z{z}_{n}_{kappa}"
            arrays[key] = np.loadtxt(path)[:end, :3]
            orbitals.append(
                dict(
                    n=n,
                    kappa=kappa,
                    array=key,
                    raw_sha256=sha256(path),
                    header=header.strip(),
                )
            )
        rows = [
            {
                key: row[key]
                for key in (
                    "upper",
                    "multipole",
                    "gauge",
                    "omega_Ha",
                    "fac_amplitude",
                    "fac_gf",
                    "fac_A_per_s",
                )
            }
            for row in case["rows"]
        ]
        fixture["cases"].append(
            dict(
                z=z,
                orbitals=orbitals,
                rows=rows,
                photon_output_sha256=sha256(folder / "photon.txt"),
            )
        )
    spinors = destination / "fac_photon_spinors.npz"
    np.savez_compressed(spinors, **arrays)
    fixture["spinors_sha256"] = sha256(spinors)
    (destination / "fac_photon_reference.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fac", type=Path, default=Path("/tmp/atomx-fac-photon/sfac"))
    parser.add_argument("--z", type=int, nargs="+", default=[1, 10, 26, 54, 82])
    parser.add_argument("--points", type=int, default=8000)
    parser.add_argument("--fac-grid-density", type=float, default=4.0)
    parser.add_argument("--fac-energy-tolerance", type=float, default=1e-12)
    parser.add_argument("--interpolation-order", type=int, default=5, choices=[3, 5, 7])
    parser.add_argument("--fac-root", type=Path, default=Path.home() / "code/fac")
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="also export compact H/Ne/Pb reference fixtures",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    keys = sorted({(1, -1)} | {(n, k) for n, k, _, _ in channels()})
    report = dict(
        scope=__doc__,
        points=args.points,
        fac_grid_density=args.fac_grid_density,
        fac_energy_tolerance=args.fac_energy_tolerance,
        interpolation_order=args.interpolation_order,
        python=platform.python_version(),
        threads={
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        reference=dict(path=str(args.fac), sha256=sha256(args.fac)),
        cases=[],
    )
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        str(p.relative_to(root)): sha256(p) for p in (root / "src").rglob("*.py")
    }
    report["reference"]["observed_files_sha256"] = {
        str(p.relative_to(args.fac_root)): sha256(p)
        for p in [
            args.fac_root / name
            for name in (
                "libfac.a",
                "faclib/radial.c",
                "faclib/orbital.c",
                "faclib/transition.c",
                "faclib/dbase.c",
            )
        ]
    }
    frontend = args.fac.with_suffix(".c")
    shutil.copy2(frontend, args.out / "sfac.c")
    shutil.copy2(__file__, args.out / "bench_reference_photon_operator.py")
    report["reference"]["frontend_sha256"] = sha256(frontend)
    report["runner_sha256"] = sha256(__file__)
    for z in args.z:
        folder = args.out / f"Z{z:03d}"
        folder.mkdir()
        lines = orbital_accuracy_commands(args.fac_energy_tolerance) + [
            f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
            "SetBreit(0)",
            "SetSE(0)",
            "SetVP(0)",
            "SetMS(0,0)",
            radial_grid_command(args.points, density_scale=args.fac_grid_density),
            "Config('g','1s1')",
            "OptimizeRadial(['g'])",
        ]
        lines += [
            f"DumpPhoton('photon.txt',1,-1,{n},{k},{m},{g})"
            for n, k, m, g in channels()
        ]
        lines += [f"DumpSpinor('wave-{n}-{k}.txt',{n},{k})" for n, k in keys]
        (folder / "run.sf").write_text("\n".join(lines) + "\n")
        seconds = run_reference([str(args.fac), "run.sf"], folder)
        common, common_ws = read_common_spinors(folder, keys, args.interpolation_order)
        start = time.perf_counter()
        states = atomx_orbitals(z, keys, args.points)
        solve_seconds = time.perf_counter() - start
        native = {key: (s._pa[0, 1:], s._qa[0, 1:]) for key, s in states.items()}
        native_ws = states[keys[0]].radial_state.correction_workspace()
        rows = []
        for _n0, _k0, n, k, m, g, omega, amplitude, gf, A in np.loadtxt(
            folder / "photon.txt"
        ):
            upper = (int(n), int(k))
            m = int(m)
            g = int(g)
            shared_A = rate(common, common_ws, upper, m, g, omega)
            native_A = rate(native, native_ws, upper, m, g, omega)
            rows.append(
                dict(
                    upper=upper,
                    multipole=m,
                    gauge=g,
                    omega_Ha=omega,
                    fac_amplitude=amplitude,
                    fac_gf=gf,
                    fac_A_per_s=A,
                    common_A_per_s=shared_A,
                    native_A_per_s=native_A,
                    common_relative_error=shared_A / A - 1,
                    native_relative_error=native_A / A - 1,
                )
            )
        report["cases"].append(
            dict(z=z, fac_process_s=seconds, atomx_orbital_s=solve_seconds, rows=rows)
        )
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            z,
            "common",
            max(abs(r["common_relative_error"]) for r in rows),
            "native",
            max(abs(r["native_relative_error"]) for r in rows),
            flush=True,
        )
    if args.fixture_dir:
        write_fixture(report, args.out, args.fixture_dir)


if __name__ == "__main__":
    main()
