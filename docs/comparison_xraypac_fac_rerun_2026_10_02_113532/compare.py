"""Live, configuration-matched XATOM/FAC/atomx PI and K-hole Auger panel.

Run from any directory with atomx's virtual environment. External reference
sources are read-only; all input decks and raw outputs live beside this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ATOMX_COMPARE_SOURCE_ROOT", HERE.parents[1]))
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

from fac_runner import (  # noqa: E402
    parse_ai,
    parse_levels,
    radial_grid_command,
    run_sfac,
)

from atomx.angular.csf_hamiltonian import two_body_angular  # noqa: E402
from atomx.constants import AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV  # noqa: E402

PHOTONS = [1000.0, 2000.0, 4000.0, 8000.0, 12000.0]
CASES = {
    "Ne": {"Z": 10, "shells": {"1s": 2, "2s": 2, "2p": 6}},
    "Ar": {"Z": 18, "shells": {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6}},
}
A0_SQ_MB = (BOHR_RADIUS_M * 100.0) ** 2 / 1e-18
FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?"


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def config(occupations, separator=" "):
    return separator.join(f"{s}{q}" for s, q in occupations.items() if q)


def shell_key(label):
    return int(label[:-1]), "spdf".index(label[-1])


def kappas(l):
    return [-(l + 1)] + ([l] if l else [])


def parse_xatom(path):
    text = Path(path).read_text()
    if "SCF converged after" not in text:
        raise RuntimeError(f"XATOM did not converge: {path}")
    out = {"pi_mb": {}, "auger_channels": {}, "radial": {}, "orbital_eV": {}}
    for line in text.splitlines():
        m = re.match(
            rf"#\s+(\d[spdf])\s+\d+\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$", line
        )
        if m:
            out["orbital_eV"][m[1]] = float(m[3])
        m = re.match(rf"#\s+(\d[spdf])\s+:\s+({FLOAT})\s+({FLOAT})", line)
        if m:
            out["pi_mb"][m[1]] = float(m[2])
        m = re.match(
            rf"#\s+(\d[spdf])\s+-\s+(\d[spdf])\s+(\d[spdf])\s+:\s+({FLOAT})\s+({FLOAT})",
            line,
        )
        if m:
            out["auger_channels"][f"{m[2]}+{m[3]}"] = {
                "hole": m[1],
                "electron_eV": float(m[4]),
                "rate_au": float(m[5]),
                "rate_per_s": float(m[5]) * AU_RATE_PER_S,
            }
        m = re.match(
            rf"#\s+(\d+)\s+(\d\d)\s+(\d\d)\s+(\d\d)\s+(\d+)\s+({FLOAT})\s*$", line
        )
        if m:
            key = ",".join(m[i] for i in range(1, 6))
            value = float(m[6])
            if key in out["radial"]:
                assert abs(out["radial"][key] - value) < 1e-10
            out["radial"][key] = value
    for label, key in [
        ("Total P cs", "pi_total_mb"),
        ("Total A rate", "auger_total_au"),
        ("Total F rate", "fluorescence_total_au"),
    ]:
        m = re.search(rf"{label}\s*=\s*({FLOAT})", text)
        if m:
            out[key] = float(m[1])
    if "auger_total_au" in out:
        out["auger_total_per_s"] = out["auger_total_au"] * AU_RATE_PER_S
        assert (
            abs(
                sum(v["rate_au"] for v in out["auger_channels"].values())
                / out["auger_total_au"]
                - 1
            )
            < 2e-5
        )
    if "pi_total_mb" in out:
        assert abs(sum(out["pi_mb"].values()) / out["pi_total_mb"] - 1) < 2e-5
    return out


def reference_environment():
    env = os.environ.copy()
    lib = Path("/tmp/atomx-xraypac-toolchain/root/usr/lib/x86_64-linux-gnu")
    if lib.exists():
        env["LD_LIBRARY_PATH"] = (
            ":".join(map(str, [lib, lib / "blas", lib / "lapack"]))
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )
    return env


def run_xatom(binary, coarse=False):
    output = {}
    env = reference_environment()
    for element in CASES:
        tag = "xatom-coarse" if coarse else "xatom"
        directory = HERE / "raw" / tag / element
        directory.mkdir(parents=True, exist_ok=True)
        record = {"pi": {}, "auger": {}}
        jobs = [(f"pi-{int(e)}", ["-pcs", "-PE", str(e)], e) for e in PHOTONS]
        jobs += [("khole", ["-hole", "1s1", "-decay"], None)]
        for name, options, energy in jobs:
            path = directory / f"{name}.out"
            cmd = [
                str(binary),
                "-s",
                element,
                "-N",
                "200" if coarse else "300",
                "-rmax",
                "50" if coarse else "80",
                "-dr",
                "0.005" if coarse else "0.0025",
                *options,
                "-o",
                str(path),
            ]
            start = time.perf_counter()
            p = subprocess.run(
                cmd, cwd=directory, env=env, text=True, capture_output=True, timeout=300
            )
            (directory / f"{name}.stdout").write_text(p.stdout)
            (directory / f"{name}.stderr").write_text(p.stderr)
            if p.returncode:
                raise RuntimeError(f"XATOM failed: {p.stderr}")
            data = parse_xatom(path)
            dump(
                directory / f"{name}.command.json",
                {"argv": cmd, "seconds": time.perf_counter() - start},
            )
            if energy is not None:
                record["pi"][str(int(energy))] = data
            else:
                record["auger"] = data
        output[element] = record
        print(element, "XATOM", record["auger"]["auger_total_per_s"], flush=True)
        dump(HERE / f"{tag}.json", output)


def fac_header(element, grid=None):
    grid = (
        int(os.environ.get("ATOMX_COMPARE_FAC_POINTS", 3200)) if grid is None else grid
    )
    density = float(os.environ.get("ATOMX_COMPARE_FAC_DENSITY", 4.0))
    return [
        f"SetAtom('{element}')",
        "SetSE(0)",
        "SetVP(0)",
        "SetMS(0,0)",
        "SetBreit(0)",
        radial_grid_command(grid, density_scale=density),
        "SetOption('radial:scfeps',1e-9)",
    ]


def run_fac(coarse=False):
    os.environ.update(reference_environment())
    if coarse:
        os.environ["ATOMX_COMPARE_FAC_POINTS"] = "1600"
        os.environ["ATOMX_COMPARE_FAC_DENSITY"] = "2"
    tag = "fac-coarse" if coarse else "fac"
    os.environ["FAC_SFAC_PATH"] = os.environ.get(
        "FAC_SFAC_PATH", "/home/zzhang/code/fac/bin/sfac"
    )
    output = {}
    for element, case in CASES.items():
        shells = case["shells"]
        record = {"pi": {}, "auger_channels": {}}
        record["pi_levels"] = {}
        record["pi_seconds"] = 0.0
        for e in PHOTONS:
            record["pi"][str(int(e))] = {"pi_mb": {}, "pi_total_mb": 0.0}
        for shell in shells:
            directory = HERE / "raw" / tag / element / "pi-isolated" / shell
            directory.mkdir(parents=True, exist_ok=True)
            for stale in directory.glob("*.b"):
                stale.unlink()
            final = dict(shells)
            final[shell] -= 1
            lines = fac_header(element) + [
                f"Config('{config(shells)}',group='initial')",
                f"Config('{config(final)}',group='final')",
                "ConfigEnergy(0)",
                "OptimizeRadial(['initial'])",
                "ConfigEnergy(1)",
                "Structure('levels.b',['initial'])",
                "Structure('levels.b',['final'])",
                "MemENTable('levels.b')",
                "PrintTable('levels.b','levels.txt',1)",
                "SetRecQkMode(0)",
                "SetPEGrid(40,1.0,15000.0)",
                "RRTable('pi.b',['initial'],['final'],-1)",
                "PrintTable('pi.b','pi.txt',1)",
                f"TotalPICross('pi.b','pi-total.txt',0,{PHOTONS!r})",
            ]
            path = directory / "run.sf"
            path.write_text("\n".join(lines) + "\n")
            stdout, elapsed, _ = run_sfac(path, workdir=directory)
            (directory / "stdout.txt").write_text(stdout)
            record["pi_levels"][shell] = parse_levels(directory / "levels.txt")
            record["pi_seconds"] += elapsed
            data = np.loadtxt(directory / "pi-total.txt", ndmin=2)
            assert np.allclose(data[:, 0], PHOTONS) and np.isfinite(data).all()
            for e, sig in data[:, :2]:
                row = record["pi"][str(int(e))]
                row["pi_mb"][shell] = float(sig) * 1e-2
                row["pi_total_mb"] += float(sig) * 1e-2

        directory = HERE / "raw" / tag / element / "auger"
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.b"):
            stale.unlink()
        initial = dict(shells)
        initial["1s"] = 1
        channels = list(
            combinations_with_replacement([s for s in shells if s != "1s"], 2)
        )
        lines = fac_header(element) + [f"Config('{config(initial)}',group='initial')"]
        for a, b in channels:
            final = dict(shells)
            final[a] -= 1
            final[b] -= 1
            lines.append(f"Config('{config(final)}',group='final_{a}_{b}')")
        lines += [
            "ConfigEnergy(0)",
            "OptimizeRadial(['initial'])",
            "ConfigEnergy(1)",
            "Structure('levels.b',['initial'])",
        ]
        for a, b in channels:
            lines.append(f"Structure('levels.b',['final_{a}_{b}'])")
        lines += ["MemENTable('levels.b')", "PrintTable('levels.b','levels.txt',1)"]
        groups = [f"final_{a}_{b}" for a, b in channels]
        lines += [
            f"AITable('ai.b',['initial'],{groups!r})",
            "PrintTable('ai.b','ai.txt',1)",
        ]
        lines += ["Reinit(0)"]
        path = directory / "run.sf"
        path.write_text("\n".join(lines) + "\n")
        stdout, elapsed, _ = run_sfac(path, workdir=directory)
        (directory / "stdout.txt").write_text(stdout)
        record["auger_levels"] = parse_levels(directory / "levels.txt")
        all_rows = parse_ai(directory / "ai.txt")
        # Final levels are appended by configuration. The shell-hole
        # inventories have independent exact jj CSF counts.
        from atomx.config.csf import enumerate_jj_csfs_many

        start_level = 1
        for a, b in channels:
            final = dict(shells)
            final[a] -= 1
            final[b] -= 1
            count = len(enumerate_jj_csfs_many([config(final)]))
            end_level = start_level + count
            rows = [r for r in all_rows if start_level <= r["lower"] < end_level]
            assert rows and {r["upper"] for r in rows} == {0}
            record["auger_channels"][f"{a}+{b}"] = {
                "rate_per_s": sum(r["rate_per_s"] for r in rows),
                "levels": rows,
            }
            start_level = end_level
        assert start_level == len(record["auger_levels"])
        record["auger_total_per_s"] = sum(
            r["rate_per_s"] for r in record["auger_channels"].values()
        )
        record["auger_seconds"] = elapsed
        output[element] = record
        print(element, "FAC", record["auger_total_per_s"], flush=True)
        dump(HERE / f"{tag}.json", output)


def spinors(shell):
    n, l = shell_key(shell)
    return [
        (n, k, m) for k in kappas(l) for m in range(-(2 * abs(k) - 1), 2 * abs(k), 2)
    ]


def magnetic_average(a, b, radial):
    """Complete initial-hole average and final two-hole spinor sum.

    All source subshells are filled. Each unordered occupied spinor pair
    specifies an orthogonal final determinant. Average the two magnetic
    members of the single 1s vacancy; include exchange coherently.
    ``radial((na,ka),(nb,kb),k_eps,rank)`` returns R^rank(a,b,1s,epsilon).
    """
    sa, sb = spinors(a), spinors(b)
    pairs = combinations(sa, 2) if a == b else product(sa, sb)
    total = 0.0
    for x, y in pairs:
        for mh in (-1, 1):
            me = x[2] + y[2] - mh
            for l in range(shell_key(a)[1] + shell_key(b)[1] + 1):
                if (l + shell_key(a)[1] + shell_key(b)[1]) % 2:
                    continue
                for ke in kappas(l):
                    if abs(me) > 2 * abs(ke) - 1:
                        continue
                    direct = two_body_angular(-1, mh, ke, me, x[1], x[2], y[1], y[2])
                    exchange = two_body_angular(-1, mh, ke, me, y[1], y[2], x[1], x[2])
                    amp = sum(co * radial(x[:2], y[:2], ke, k) for k, co in direct)
                    amp -= sum(co * radial(y[:2], x[:2], ke, k) for k, co in exchange)
                    total += amp * amp
    return np.pi * total * AU_RATE_PER_S


def check_xatom_angular():
    refs = json.loads((HERE / "xatom.json").read_text())
    output = {}
    for element, record in refs.items():
        auger = record["auger"]
        output[element] = {}
        for channel, row in auger["auger_channels"].items():
            a, b = channel.split("+")

            def radial(ak, bk, ke, k, auger=auger):
                l1 = ak[1] if ak[1] > 0 else -ak[1] - 1
                l2 = bk[1] if bk[1] > 0 else -bk[1] - 1
                ne = ke if ke > 0 else -ke - 1
                key = f"{k},{ak[0]}{l1},{bk[0]}{l2},10,{ne}"
                return auger["radial"].get(key, 0.0)

            value = magnetic_average(a, b, radial)
            ratio = value / row["rate_per_s"]
            if abs(ratio - 1) > 3e-5:
                raise AssertionError(
                    f"XATOM angular audit {element} {channel}: {ratio}"
                )
            output[element][channel] = {
                "magnetic_rate_per_s": value,
                "xatom_rate_per_s": row["rate_per_s"],
                "ratio": ratio,
            }
    dump(HERE / "same_radial_angular_audit.json", output)


def solve_atomx(element, hole, ns, exchange="hf"):
    from atomx.methods.dirac_fock import DFSolver, Subshell

    shells = dict(CASES[element]["shells"])
    if hole:
        shells["1s"] = 1
    subs = []
    for label, q in shells.items():
        n, l = shell_key(label)
        for k in kappas(l):
            subs.append(Subshell(n, k, q * (2 * abs(k)) / (2 * (2 * l + 1))))
    solver = DFSolver(
        CASES[element]["Z"],
        subs,
        ns=ns,
        ks=9,
        r_max=80.0,
        nuclear="point",
        exchange=exchange,
        starting_potential="tfda",
    )
    result = solver.solve(max_iter=200, tol=1e-9, damping=0.5, mode="orbital")
    if not result.converged:
        raise RuntimeError(f"{element} hole={hole} ns={ns}: SCF did not converge")
    return result


def atomx_pi(result, photons, n_grid, thresholds=None):
    if thresholds is not None:
        import copy

        result = copy.copy(result)
        result.orbital_energies = dict(result.orbital_energies)
        for sh in result.subshells:
            result.orbital_energies[sh.label] = (
                -thresholds[(sh.n, sh.kappa)] / HARTREE_EV
            )
    from atomx.scattering.models import ContinuumModel
    from atomx.scattering.radiative_recombination import (
        _shooting_photoionization_cross_section,
    )

    model = ContinuumModel.fac_residual_core(n_grid=n_grid, r_max=80.0)
    output = {str(int(e)): {"pi_mb": {}, "pi_total_mb": 0.0} for e in photons}
    orbitals = result.orbitals_by_subshell()
    for sh in result.subshells:
        P, Q, _ = orbitals[(sh.n, sh.kappa)]
        values = _shooting_photoionization_cross_section(
            result, sh, P, Q, np.array(photons) / HARTREE_EV, model
        )
        label = f"{sh.n}{'spdf'[sh.kappa if sh.kappa > 0 else -sh.kappa - 1]}"
        for e, value in zip(photons, values, strict=True):
            row = output[str(int(e))]
            value = float(value) * sh.occupation * A0_SQ_MB
            row["pi_mb"][label] = row["pi_mb"].get(label, 0.0) + value
            row["pi_total_mb"] += value
    return output


def frozen_pi(result, photons, thresholds, n_grid=16000, refinement=2):
    """Matched FAC field; reuse atomx's angular and finite-frequency operator."""
    from atomx.constants import ALPHA
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.numerics.quadrature import make_fine_grid, p_rel
    from atomx.numerics.radial_metric import RadialMetric
    from atomx.numerics.radial_workspace import make_grid_workspace
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_many
    from atomx.potentials.static_field import static_hartree_potential
    from atomx.properties.transitions.angular import (
        allowed_final_kappas,
        line_strength_factor,
    )
    from atomx.properties.transitions.multipole import radial_eL

    requests = []
    for sh in result.subshells:
        for e in photons:
            kinetic = (e - thresholds[sh.n, sh.kappa]) / HARTREE_EV
            if kinetic > 0:
                requests.extend(
                    (ke, kinetic) for ke in allowed_final_kappas(sh.kappa, 1, "E")
                )
    field = static_hartree_potential(
        result, include_exchange=True, exchange="fac", n_radial=8000
    )
    r = make_fine_grid(
        80.0,
        p_rel(max(e for _, e in requests)),
        z_inner=result.Z,
        refinement=refinement,
    )
    r, waves = solve_dirac_continuum_many(
        requests,
        field,
        r_max=80.0,
        n_grid=n_grid,
        Z_residual=field.residual_charge,
        workers=1,
        interp_grid=r,
    )
    workspace = make_grid_workspace(r, RadialMetric.trapezoid(r).weights)
    out = {str(int(e)): {"pi_mb": {}, "pi_total_mb": 0.0} for e in photons}
    for sh in result.subshells:
        P, Q, _ = result.orbitals_by_subshell()[sh.n, sh.kappa]
        P, Q = eval_P(P, result._bsp_data, r), eval_Q(Q, result._bsp_data, r)
        label = f"{sh.n}{'spdf'[sh.kappa if sh.kappa > 0 else -sh.kappa - 1]}"
        for e in photons:
            kinetic = (e - thresholds[sh.n, sh.kappa]) / HARTREE_EV
            value = 0.0
            if kinetic > 0:
                omega = e / HARTREE_EV
                for ke in allowed_final_kappas(sh.kappa, 1, "E"):
                    angular = line_strength_factor(sh.kappa, ke, 1, "E") / (
                        2 * abs(sh.kappa)
                    )
                    pe, qe = waves[ke, kinetic]
                    d = radial_eL(
                        P,
                        Q,
                        pe,
                        qe,
                        workspace,
                        kappa_i=sh.kappa,
                        kappa_f=ke,
                        omega=omega,
                    )
                    value += 4 * np.pi**2 * ALPHA * omega / 3 * d * d * angular
            value *= sh.occupation * A0_SQ_MB
            row = out[str(int(e))]
            row["pi_mb"][label] = row["pi_mb"].get(label, 0.0) + value
            row["pi_total_mb"] += value
    return out


def atomx_auger(result, xatom_auger, n_grid, refinement=2, reference_label="xatom"):
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.numerics.quadrature import grid_slater_rk, make_fine_grid, p_rel
    from atomx.numerics.radial_metric import RadialMetric
    from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_many
    from atomx.potentials.static_field import static_hartree_potential

    data = result._bsp_data
    orbitals = result.orbitals_by_subshell()
    eps = {
        (sh.n, sh.kappa): float(result.orbital_energies[sh.label])
        for sh in result.subshells
    }
    channel_sources = {}
    requests = set()
    energies = {}
    for channel, xrow in xatom_auger["auger_channels"].items():
        a, b = channel.split("+")
        an, al = shell_key(a)
        bn, bl = shell_key(b)
        ap = [(an, k) for k in kappas(al)]
        bp = [(bn, k) for k in kappas(bl)]
        pairs = combinations_with_replacement(ap, 2) if a == b else product(ap, bp)
        channel_sources[channel] = list(pairs)
        for ak, bk in channel_sources[channel]:
            key = tuple(sorted((ak, bk)))
            e = eps[ak] + eps[bk] - eps[(1, -1)]
            if e <= 0:
                raise ValueError(f"closed Auger channel: {channel}")
            energies[key] = e
            for l in range(al + bl + 1):
                if (al + bl + l) % 2:
                    continue
                for ke in kappas(l):
                    requests.add((ke, e))
                    requests.add((ke, xrow["electron_eV"] / HARTREE_EV))
    energy_max = max(e for _, e in requests)
    r = make_fine_grid(80.0, p_rel(energy_max), z_inner=result.Z, refinement=refinement)
    field = static_hartree_potential(
        result,
        include_exchange=True,
        exchange="fac",
        screening="full",
        n_radial=8000,
        r_max=80.0,
    )
    r, waves = solve_dirac_continuum_many(
        sorted(requests),
        field,
        n_grid=n_grid,
        r_max=80.0,
        Z_residual=field.residual_charge,
        workers=1,
        interp_grid=r,
    )
    bound = {
        key: (eval_P(P, data, r), eval_Q(Q, data, r))
        for key, (P, Q, _) in orbitals.items()
    }
    metric = RadialMetric.trapezoid(r)
    norms = {
        str(k): float(metric.integrate(P * P + Q * Q)) for k, (P, Q) in bound.items()
    }
    assert max(abs(v - 1) for v in norms.values()) < 1e-3
    hP, hQ = bound[(1, -1)]

    @lru_cache(maxsize=None)
    def radial(ak, bk, ke, rank, energy):
        aP, aQ = bound[ak]
        bP, bQ = bound[bk]
        eP, eQ = waves[(ke, energy)]
        return grid_slater_rk(aP, aQ, bP, bQ, hP, hQ, eP, eQ, r, rank)

    output = {
        "auger_channels": {},
        "norms": norms,
        "continuum_tail_charge": float(field.residual_charge),
        "wave_requests": len(requests),
        "scattering_grid_points": len(r),
        "model": "coherent configuration-average magnetic sum; HF bound orbitals; frozen initial Gu/Latter continuum",
    }
    for channel, xrow in xatom_auger["auger_channels"].items():
        a, b = channel.split("+")

        def native(ak, bk, ke, rank):
            return radial(ak, bk, ke, rank, energies[tuple(sorted((ak, bk)))])

        def matched(ak, bk, ke, rank, xrow=xrow):
            return radial(ak, bk, ke, rank, xrow["electron_eV"] / HARTREE_EV)

        rates = [magnetic_average(a, b, f) for f in (native, matched)]
        output["auger_channels"][channel] = {
            "rate_per_s": float(rates[0]),
            f"at_{reference_label}_energy_per_s": float(rates[1]),
            "electron_energies_eV": [
                energies[tuple(sorted(pair))] * HARTREE_EV
                for pair in channel_sources[channel]
            ],
        }
    output["auger_total_per_s"] = sum(
        row["rate_per_s"] for row in output["auger_channels"].values()
    )
    output[f"auger_at_{reference_label}_energy_per_s"] = sum(
        row[f"at_{reference_label}_energy_per_s"]
        for row in output["auger_channels"].values()
    )
    return output


def run_atomx():
    references = json.loads((HERE / "xatom.json").read_text())
    output = {}
    for element in CASES:
        record = {}
        for ns, n_grid, refinement, label in [
            (80, 8000, 1, "coarse"),
            (120, 16000, 2, "fine"),
        ]:
            start = time.perf_counter()
            neutral = solve_atomx(element, False, ns)
            khole = solve_atomx(element, True, ns)
            auger = atomx_auger(khole, references[element]["auger"], n_grid, refinement)
            record[label] = {
                "pi": atomx_pi(neutral, PHOTONS, n_grid),
                "auger": auger,
                "ns": ns,
                "ks": 9,
                "continuum_n_grid": n_grid,
                "seconds": time.perf_counter() - start,
                "neutral_orbital_eV": {
                    k: float(v) * HARTREE_EV
                    for k, v in neutral.orbital_energies.items()
                },
                "khole_orbital_eV": {
                    k: float(v) * HARTREE_EV for k, v in khole.orbital_energies.items()
                },
                "neutral_total_Ha": float(neutral.total_energy),
                "khole_total_Ha": float(khole.total_energy),
            }
            print(element, label, "atomx", auger["auger_total_per_s"], flush=True)
            output[element] = record
            dump(HERE / "atomx.json", output)


def run_atomx_matched():
    fac = json.loads((HERE / "fac.json").read_text())
    output = {}
    for element in CASES:
        neutral = solve_atomx(element, False, 120, exchange="gu")
        khole = solve_atomx(element, True, 120, exchange="gu")
        thresholds = {}
        for shell, levels in fac[element]["pi_levels"].items():
            n, l = shell_key(shell)
            for level in levels[1:]:
                k = -(l + 1) if level["TWOJ"] == 2 * l + 1 else l
                thresholds[n, k] = level["ENERGY_eV"] - levels[0]["ENERGY_eV"]
        reference = {"auger_channels": {}}
        for channel, row in fac[element]["auger_channels"].items():
            energy = (
                sum(r["DELTA_E_eV"] * r["rate_per_s"] for r in row["levels"])
                / row["rate_per_s"]
            )
            reference["auger_channels"][channel] = {"electron_eV": energy}
        auger = atomx_auger(khole, reference, 16000, reference_label="fac")
        auger["model"] = (
            "coherent magnetic configuration average; Gu DFS bound orbitals; frozen initial Gu/Latter continuum"
        )
        output[element] = {
            "pi": atomx_pi(neutral, PHOTONS, 16000),
            "pi_at_fac_thresholds": atomx_pi(neutral, PHOTONS, 16000, thresholds),
            "pi_frozen_gu_at_fac_thresholds": frozen_pi(neutral, PHOTONS, thresholds),
            "auger": auger,
            "neutral_orbital_eV": {
                k: float(v) * HARTREE_EV for k, v in neutral.orbital_energies.items()
            },
            "khole_orbital_eV": {
                k: float(v) * HARTREE_EV for k, v in khole.orbital_energies.items()
            },
            "fac_auger_energy_reference": "rate-weighted mean per nonrelativistic shell pair, not a level-by-level identity",
        }
        dump(HERE / "atomx-gu.json", output)
        print(element, "Gu model", auger["auger_total_per_s"], flush=True)


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refine_pi():
    """Resolve weak valence-channel mesh sensitivity found by the first panel."""
    fac = json.loads((HERE / "fac.json").read_text())
    output = {}
    for element in CASES:
        thresholds = {}
        for shell, levels in fac[element]["pi_levels"].items():
            n, l = shell_key(shell)
            for level in levels[1:]:
                k = -(l + 1) if level["TWOJ"] == 2 * l + 1 else l
                thresholds[n, k] = level["ENERGY_eV"] - levels[0]["ENERGY_eV"]
        hf = solve_atomx(element, False, 160)
        gu = solve_atomx(element, False, 160, exchange="gu")
        output[element] = {
            "ns": 160,
            "shooting_grid": 32000,
            "scattering_refinement": 4,
            "HF": atomx_pi(hf, PHOTONS, 32000),
            "Gu_frozen_FAC_thresholds": frozen_pi(
                gu, PHOTONS, thresholds, n_grid=32000, refinement=4
            ),
        }
        dump(HERE / "pi-refinement.json", output)
        print(element, "PI refinement complete", flush=True)


def relative(value, reference):
    return value / reference - 1 if reference else None


def max_change(coarse, fine):
    def maximum(values):
        return max(abs(v) for v in values if v is not None)

    return {
        "pi_total_max_relative_change": maximum(
            relative(row["pi_total_mb"], fine["pi"][e]["pi_total_mb"])
            for e, row in coarse["pi"].items()
        ),
        "pi_subshell_max_relative_change": maximum(
            relative(value, fine["pi"][e]["pi_mb"][shell])
            for e, row in coarse["pi"].items()
            for shell, value in row["pi_mb"].items()
        ),
        "auger_total_relative_change": relative(
            coarse["auger"]["auger_total_per_s"], fine["auger"]["auger_total_per_s"]
        ),
        "auger_channel_max_relative_change": maximum(
            relative(
                row["rate_per_s"],
                fine["auger"]["auger_channels"][channel]["rate_per_s"],
            )
            for channel, row in coarse["auger"]["auger_channels"].items()
        ),
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pi_change(coarse, fine):
    return {
        "total_max_relative_change": max(
            abs(relative(row["pi_total_mb"], fine[e]["pi_total_mb"]))
            for e, row in coarse.items()
        ),
        "subshell_max_relative_change": max(
            abs(relative(v, fine[e]["pi_mb"][s]))
            for e, row in coarse.items()
            for s, v in row["pi_mb"].items()
            if fine[e]["pi_mb"][s]
        ),
    }


def provenance(xatom_binary):
    import platform
    from importlib.metadata import version

    def git(path):
        head = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).splitlines()
        return {"path": str(path), "HEAD": head, "status_at_summary": status}

    xsource = Path.home() / "code/xraypac/xatom"
    manifest = {
        str(p.relative_to(xsource)): sha256(p)
        for p in sorted(xsource.rglob("*"))
        if p.is_file()
        and (p.suffix.lower() in {".f", ".f90", ".inc"} or p.name == "CMakeLists.txt")
    }
    fac_binary = Path(
        os.environ.get("FAC_SFAC_PATH", str(Path.home() / "code/fac/bin/sfac"))
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "scipy", "numba", "matplotlib")},
        "atomx": {
            "source_root": str(ROOT),
            "repository_at_summary": git(HERE.parents[1]),
        },
        "fac": git(Path.home() / "code/fac"),
        "xatom_source": {
            "path": str(xsource),
            "git_repository": False,
            "sha256": manifest,
        },
        "binaries": {
            str(p): sha256(p) for p in (xatom_binary, fac_binary) if p.exists()
        },
        "driver_sha256": sha256(__file__),
        "threads": {
            k: os.environ.get(k)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "units": {
            "PI": "Mb = 1e-18 cm^2",
            "Auger": "s^-1 per one 1s vacancy",
            "Hartree_eV": HARTREE_EV,
            "atomic_rate_per_s": AU_RATE_PER_S,
            "a0_squared_Mb": A0_SQ_MB,
        },
        "grid": {
            "xatom_coarse": "N=200, rmax=50, dr=.005",
            "xatom_fine": "N=300, rmax=80, dr=.0025",
            "fac_coarse": "1600 points, radial_grid_command density_scale=2",
            "fac_fine": "3200 points, radial_grid_command density_scale=4",
            "atomx_coarse": "ns=80, ks=9, rmax=80, shooting=8000, scattering refinement=1",
            "atomx_fine": "ns=120, ks=9, rmax=80, shooting=16000, scattering refinement=2",
            "atomx_gu_controls": "Gu SCF tol=1e-9; frozen local-field sampling=8000; Auger on fine grid",
            "atomx_PI_final_refinement": "ns=160, ks=9, rmax=80, shooting=32000; matched frozen-field scattering refinement=4",
        },
        "models": {
            "xatom": "nonrelativistic Hartree-Fock-Slater, frozen initial potential, configuration average",
            "fac": "Dirac local Gu central field; initial configuration optimized; E1; Breit, SE, VP, MS off; per-configuration structure without interconfiguration CI",
            "atomx_HF_PI": "Dirac-Fock bound orbitals; production RR/PI shooting kernel; shell-specific residual-core Gu continuum; orbital thresholds",
            "atomx_HF_Auger": "benchmark magnetic configuration average using production angular/Slater/continuum kernels; HF bound orbitals; frozen initial Gu/Latter field; orbital-sum kinetics",
            "atomx_matched_PI": "Gu bound orbitals; frozen initial Gu/Latter field; FAC per-jj ionization thresholds; production finite-frequency E1 operator",
            "atomx_matched_Auger": "Gu bound orbitals; frozen initial Gu/Latter field; FAC rate-weighted mean electron energy per shell pair; not level-resolved identity",
        },
        "limitations": [
            "Ne and Ar only; PI at five photon energies, not a threshold scan",
            "Agreement is a model comparison, not experimental validation",
            "Auger configuration-average driver does not validate the public coarse Auger API",
            "Gu Auger controls have no independent bound-basis refinement run",
            "Git status records summary time; xraypac has no Git revision",
        ],
    }


def figures(xatom, fac, atomx, gu):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = [
        ("XATOM HFS", xatom, "#d97706", "s", "-"),
        ("FAC Gu", fac, "#18181b", "o", "-"),
        ("atomx HF + residual field", atomx, "#2563eb", "^", "-"),
        ("atomx Gu + frozen field + FAC thresholds", gu, "#059669", "x", "--"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.8), sharex="col", gridspec_kw={"height_ratios": [2.2, 1]}
    )
    for col, element in enumerate(CASES):
        reference = np.array(
            [fac[element]["pi"][str(int(e))]["pi_total_mb"] for e in PHOTONS]
        )
        for label, source, color, marker, ls in styles:
            record = source[element]
            if source is atomx:
                record = record["fine"]
            key = "pi_frozen_gu_at_fac_thresholds" if source is gu else "pi"
            values = np.array(
                [record[key][str(int(e))]["pi_total_mb"] for e in PHOTONS]
            )
            # Discrete points: avoid suggesting a resolved Ar K edge at 2--4 keV.
            axes[0, col].loglog(
                np.array(PHOTONS) / 1000,
                values,
                color=color,
                marker=marker,
                ls="none",
                label=label,
            )
            axes[1, col].semilogx(
                np.array(PHOTONS) / 1000,
                100 * (values / reference - 1),
                color=color,
                marker=marker,
                ls=ls,
            )
        axes[0, col].set_title(f"Neutral {element}: total photoionization")
        axes[1, col].axhline(0, color="#71717a", lw=0.7)
        axes[1, col].set_xlabel("Photon energy (keV)")
        axes[1, col].set_xticks(np.array(PHOTONS) / 1000, ["1", "2", "4", "8", "12"])
        for row in range(2):
            axes[row, col].grid(True, which="both", alpha=0.2)
    axes[0, 0].set_ylabel("Cross section (Mb)")
    axes[1, 0].set_ylabel("Difference from FAC (%)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.text(
        0.5,
        0.015,
        "Five sampled energies; Ar K-shell onset lies between 2 and 4 keV. Gu control uses FAC thresholds.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"photoionization.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 2.4]}
    )
    for ax, element in zip(axes, CASES, strict=True):
        channels = list(fac[element]["auger_channels"]) + ["TOTAL"]
        for label, source, color, marker, _ in styles:
            if source is fac:
                continue
            record = source[element]["fine"] if source is atomx else source[element]
            record = record["auger"]
            rates = [
                record["auger_channels"][ch][
                    "at_fac_energy_per_s" if source is gu else "rate_per_s"
                ]
                for ch in channels[:-1]
            ]
            rates.append(
                record[
                    "auger_at_fac_energy_per_s" if source is gu else "auger_total_per_s"
                ]
            )
            refs = [
                fac[element]["auger_channels"][ch]["rate_per_s"] for ch in channels[:-1]
            ]
            refs.append(fac[element]["auger_total_per_s"])
            if source is gu:
                label = "atomx Gu + FAC mean energies"
            elif source is atomx:
                label = "atomx HF (configuration average)"
            ax.plot(
                np.arange(len(channels)),
                100 * (np.array(rates) / refs - 1),
                color=color,
                marker=marker,
                ls="none",
                label=label,
            )
        ax.axhline(0, color="#18181b", lw=0.9, label="FAC Gu")
        ax.set_xticks(np.arange(len(channels)), channels, rotation=55, ha="right")
        ax.set_title(f"{element}+ single 1s vacancy: Auger")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylim(-29, 33)
    axes[0].set_ylabel("Difference from FAC (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"auger.{ext}", dpi=180)
    plt.close(fig)


def summarise(xatom_binary):
    xatom, fac, atomx, gu, xc, fc, refined = [
        json.loads((HERE / f"{s}.json").read_text())
        for s in (
            "xatom",
            "fac",
            "atomx",
            "atomx-gu",
            "xatom-coarse",
            "fac-coarse",
            "pi-refinement",
        )
    ]
    pi_rows, auger_rows, convergence = [], [], {}
    for element, case in CASES.items():
        for e in PHOTONS:
            key = str(int(e))
            for shell in [*case["shells"], "TOTAL"]:

                def value(record, shell=shell):
                    return (
                        record["pi_total_mb"]
                        if shell == "TOTAL"
                        else record["pi_mb"][shell]
                    )

                reference = value(fac[element]["pi"][key])
                row = {"element": element, "photon_eV": e, "shell": shell}
                for label, record in [
                    ("xatom", xatom[element]["pi"][key]),
                    ("fac", fac[element]["pi"][key]),
                    ("atomx_HF", refined[element]["HF"][key]),
                    ("atomx_HF_120ns", atomx[element]["fine"]["pi"][key]),
                    ("atomx_Gu_residual", gu[element]["pi"][key]),
                    (
                        "atomx_Gu_residual_FAC_thresholds",
                        gu[element]["pi_at_fac_thresholds"][key],
                    ),
                    (
                        "atomx_Gu_frozen_FAC_thresholds",
                        refined[element]["Gu_frozen_FAC_thresholds"][key],
                    ),
                    (
                        "atomx_Gu_frozen_FAC_thresholds_120ns",
                        gu[element]["pi_frozen_gu_at_fac_thresholds"][key],
                    ),
                ]:
                    v = value(record)
                    assert np.isfinite(v) and v >= 0
                    row[f"{label}_Mb"] = v
                    row[f"{label}_relative_to_fac"] = relative(v, reference)
                pi_rows.append(row)
        channels = fac[element]["auger_channels"]
        for channel in [*channels, "TOTAL"]:

            def value(
                record, rate="rate_per_s", total="auger_total_per_s", channel=channel
            ):
                return (
                    record[total]
                    if channel == "TOTAL"
                    else record["auger_channels"][channel][rate]
                )

            reference = value(fac[element])
            row = {"element": element, "hole": "1s", "source_pair": channel}
            for label, record, rate, total in [
                ("xatom", xatom[element]["auger"], "rate_per_s", "auger_total_per_s"),
                ("fac", fac[element], "rate_per_s", "auger_total_per_s"),
                (
                    "atomx_HF",
                    atomx[element]["fine"]["auger"],
                    "rate_per_s",
                    "auger_total_per_s",
                ),
                (
                    "atomx_HF_XATOM_energy",
                    atomx[element]["fine"]["auger"],
                    "at_xatom_energy_per_s",
                    "auger_at_xatom_energy_per_s",
                ),
                ("atomx_Gu", gu[element]["auger"], "rate_per_s", "auger_total_per_s"),
                (
                    "atomx_Gu_FAC_mean_energy",
                    gu[element]["auger"],
                    "at_fac_energy_per_s",
                    "auger_at_fac_energy_per_s",
                ),
            ]:
                v = value(record, rate, total)
                assert np.isfinite(v) and v > 0
                row[f"{label}_per_s"] = v
                row[f"{label}_relative_to_fac"] = relative(v, reference)
            auger_rows.append(row)
        fac_coarse = {"pi": fc[element]["pi"], "auger": fc[element]}
        fac_fine = {"pi": fac[element]["pi"], "auger": fac[element]}
        convergence[element] = {
            "xatom": max_change(xc[element], xatom[element]),
            "fac": max_change(fac_coarse, fac_fine),
            "atomx_HF": max_change(atomx[element]["coarse"], atomx[element]["fine"]),
            "atomx_HF_PI_fine_to_refined": pi_change(
                atomx[element]["fine"]["pi"], refined[element]["HF"]
            ),
            "atomx_Gu_PI_fine_to_refined": pi_change(
                gu[element]["pi_frozen_gu_at_fac_thresholds"],
                refined[element]["Gu_frozen_FAC_thresholds"],
            ),
        }
    write_csv("photoionization.csv", pi_rows)
    write_csv("auger.csv", auger_rows)
    dump(HERE / "mesh_convergence.json", convergence)
    dump(HERE / "metadata.json", provenance(xatom_binary))
    plot_atomx = {
        el: {"fine": {"pi": refined[el]["HF"], "auger": atomx[el]["fine"]["auger"]}}
        for el in CASES
    }
    plot_gu = {
        el: {
            **gu[el],
            "pi_frozen_gu_at_fac_thresholds": refined[el]["Gu_frozen_FAC_thresholds"],
        }
        for el in CASES
    }
    figures(xatom, fac, plot_atomx, plot_gu)
    print(
        f"Wrote {len(pi_rows)} PI rows and {len(auger_rows)} Auger rows, provenance, mesh checks, PNG/PDF plots."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "xatom",
            "xatom-coarse",
            "fac",
            "fac-coarse",
            "angular",
            "atomx",
            "atomx-matched",
            "pi-refine",
            "summary",
        ],
    )
    parser.add_argument(
        "--xatom", type=Path, default=Path("/tmp/atomx-xraypac-build/build/src/xatom")
    )
    args = parser.parse_args()
    if args.stage == "xatom":
        run_xatom(args.xatom)
    elif args.stage == "xatom-coarse":
        run_xatom(args.xatom, coarse=True)
    elif args.stage == "fac":
        run_fac()
    elif args.stage == "fac-coarse":
        run_fac(coarse=True)
    elif args.stage == "angular":
        check_xatom_angular()
    elif args.stage == "atomx":
        run_atomx()
    elif args.stage == "atomx-matched":
        run_atomx_matched()
    elif args.stage == "pi-refine":
        refine_pi()
    elif args.stage == "summary":
        summarise(args.xatom)


if __name__ == "__main__":
    main()
