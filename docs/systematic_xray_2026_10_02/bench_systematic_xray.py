"""Fresh Z-grid PI, fluorescence, K-yield and scattering benchmark.

References: XATOM native HFS output, FAC resolved RR/TR/AI tables, and
independent nonrelativistic LS postprocessing of FAC bound wavefunctions.
FAC has no native photon Rayleigh/Compton table in this panel.
Run with --help. Each reference job has an isolated directory and is cached
only when its command/script matches and its completion marker exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import spherical_jn

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "pyproject.toml").exists() and (p / "src/atomx").exists()
)
sys.path[:0] = [
    os.environ.get("ATOMX_SYSTEMATIC_SOURCE", str(ROOT / "src")),
    str(Path(os.environ.get("ATOMX_SYSTEMATIC_BENCHMARKS", str(ROOT / "benchmarks")))),
]
from fac_runner import (  # noqa: E402
    parse_ai,
    parse_levels,
    parse_transitions,
    radial_grid_command,
    run_sfac,
)

from atomx.angular.wigner import wigner_3j  # noqa: E402
from atomx.constants import (  # noqa: E402
    AU_RATE_PER_S,
    BOHR_RADIUS_M,
    HARTREE_EV,
)
from atomx.methods.dirac_fock import (  # noqa: E402
    DFSolver,
    Subshell,
    _eval_bspline_on_grid,
)
from atomx.properties.transitions.angular import angular_eL  # noqa: E402
from atomx.properties.transitions.rates import einstein_A_eL  # noqa: E402
from atomx.scattering.autoionization import auger_rate_csf_average  # noqa: E402
from atomx.scattering.xray import (  # noqa: E402
    AtomicScattering,
    ScatteringOrbital,
    klein_nishina,
    momentum_transfer,
    thomson,
)

CASES = {
    "C": (6, {"1s": 2, "2s": 2, "2p": 2}),
    "Ne": (10, {"1s": 2, "2s": 2, "2p": 6}),
    "Ar": (18, {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6}),
    "Fe": (26, {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 6, "4s": 2}),
    "Kr": (
        36,
        {"1s": 2, "2s": 2, "2p": 6, "3s": 2, "3p": 6, "3d": 10, "4s": 2, "4p": 6},
    ),
    "Xe": (
        54,
        {
            "1s": 2,
            "2s": 2,
            "2p": 6,
            "3s": 2,
            "3p": 6,
            "3d": 10,
            "4s": 2,
            "4p": 6,
            "4d": 10,
            "5s": 2,
            "5p": 6,
        },
    ),
}
FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?"
MB = BOHR_RADIUS_M**2 / 1e-22


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config(shells):
    return " ".join(f"{s}{q}" for s, q in shells.items() if q)


def nl(s):
    return int(s[:-1]), "spdf".index(s[-1])


def label(s):
    n, l = nl(s)
    return f"{n}{'spdf'[l]}{2 * l + 1}/2"


def environment():
    env = os.environ.copy()
    lib = Path("/tmp/atomx-xraypac-toolchain/root/usr/lib/x86_64-linux-gnu")
    if lib.exists():
        env["LD_LIBRARY_PATH"] = (
            ":".join(str(p) for p in (lib, lib / "blas", lib / "lapack"))
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )
    return env


def xatom(args, directory, element, options):
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "result.out"
    cmd = [
        str(args.xatom),
        "-s",
        element,
        "-config",
        config(CASES[element][1]),
        "-N",
        "300",
        "-rmax",
        "80",
        "-dr",
        "0.0025",
        *options,
        "-o",
        str(out),
    ]
    marker = directory / "complete.json"
    if marker.exists() and json.loads(marker.read_text())["argv"] == cmd:
        return out
    for p in directory.glob("*.ixs*"):
        p.unlink()
    for p in directory.glob("*.scd"):
        p.unlink()
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=directory,
        env=environment(),
        text=True,
        capture_output=True,
        timeout=600,
    )
    (directory / "stdout.txt").write_text(proc.stdout)
    (directory / "stderr.txt").write_text(proc.stderr)
    if (
        proc.returncode
        or not out.exists()
        or "SCF converged after" not in out.read_text()
    ):
        raise RuntimeError(
            f"XATOM failed or unconverged: {directory}: {proc.stderr[-1000:]}"
        )
    dump(marker, {"argv": cmd, "seconds": time.perf_counter() - start})
    return out


def parse_xatom(path):
    out = {"orbitals_eV": {}, "pi_mb": {}, "fluorescence": {}, "auger": {}}
    text = path.read_text()
    for line in text.splitlines():
        m = re.match(
            rf"#\s+(\d[spdf])\s+\d+\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$", line
        )
        if m:
            out["orbitals_eV"][m[1]] = float(m[3])
        m = re.match(rf"#\s+(\d[spdf])\s+:\s+({FLOAT})\s+({FLOAT})", line)
        if m:
            out["pi_mb"][m[1]] = float(m[2])
        m = re.match(
            rf"#\s+(\d[spdf])\s+-\s+(\d[spdf])\s*:\s*({FLOAT})\s+({FLOAT})", line
        )
        if m:
            out["fluorescence"][f"{m[2]}->{m[1]}"] = {
                "energy_eV": float(m[3]),
                "rate_per_s": float(m[4]) * AU_RATE_PER_S,
            }
        m = re.match(
            rf"#\s+(\d[spdf])\s+-\s+(\d[spdf])\s+(\d[spdf])\s*:\s*({FLOAT})\s+({FLOAT})",
            line,
        )
        if m:
            out["auger"][f"{m[2]}+{m[3]}"] = {
                "energy_eV": float(m[4]),
                "rate_per_s": float(m[5]) * AU_RATE_PER_S,
            }
    for field, tag in [
        ("radiative_per_s", "Total F rate"),
        ("auger_per_s", "Total A rate"),
    ]:
        m = re.search(rf"{tag}\s*=\s*({FLOAT})", text)
        if m:
            out[field] = float(m[1]) * AU_RATE_PER_S
    if "radiative_per_s" not in out and re.search(
        r"# Fluorescence.*?\n#\s+NONE", text, re.S
    ):
        out["radiative_per_s"] = 0.0
    return out


def fac_header(element):
    return [
        f"SetAtom('{element}')",
        "SetSE(0)",
        "SetVP(0)",
        "SetMS(0,0)",
        "SetBreit(0)",
        radial_grid_command(3200, density_scale=4),
        "SetOption('radial:scfeps',1e-9)",
    ]


def fac_job(directory, lines):
    directory.mkdir(parents=True, exist_ok=True)
    script = "\n".join(lines) + "\n"
    key = hashlib.sha256(script.encode()).hexdigest()
    marker = directory / "complete.json"
    if marker.exists() and json.loads(marker.read_text())["script_sha256"] == key:
        return
    for p in directory.glob("*.b"):
        p.unlink()
    path = directory / "run.sf"
    path.write_text(script)
    os.environ.update(environment())
    stdout, seconds, _ = run_sfac(path, workdir=directory, timeout=900)
    (directory / "stdout.txt").write_text(stdout)
    # sfac sometimes prints an error yet exits 0; all consumers also validate outputs.
    if "syntax error" in stdout.lower() or "cannot open" in stdout.lower():
        raise RuntimeError(f"FAC error: {directory}")
    dump(marker, {"script_sha256": key, "seconds": seconds})


def csf_count(shells):
    from atomx.config.csf import enumerate_jj_csfs_many

    return len(enumerate_jj_csfs_many([config(shells)]))


def average_rates(rows, levels, n_initial, field):
    weights = {r["ILEV"]: r["TWOJ"] + 1 for r in levels[:n_initial]}
    assert set(weights) == set(range(n_initial))
    total = sum(weights.values())
    return sum(weights[r["upper"]] * r[field] for r in rows) / total


def fac_decay(element, hole, directory):
    shells = CASES[element][1]
    initial = dict(shells)
    initial[hole] -= 1
    n_initial = csf_count(initial)
    sources = [s for s in shells if s != hole]
    groups = []
    lines = fac_header(element) + [f"Config('{config(initial)}',group='initial')"]
    for s in sources:
        final = dict(shells)
        final[s] -= 1
        group = "r_" + s
        groups.append(group)
        lines.append(f"Config('{config(final)}',group='{group}')")
    pairs = (
        list(itertools.combinations_with_replacement(sources, 2))
        if hole == "1s"
        else []
    )
    agroups = []
    for a, b in pairs:
        final = dict(shells)
        final[a] -= 1
        final[b] -= 1
        if min(final.values()) < 0:
            continue
        group = f"a_{a}_{b}"
        agroups.append(group)
        lines.append(f"Config('{config(final)}',group='{group}')")
    lines += [
        "ConfigEnergy(0)",
        "OptimizeRadial(['initial'])",
        "ConfigEnergy(1)",
        "Structure('levels.b',['initial'])",
    ]
    for group in groups + agroups:
        lines.append(f"Structure('levels.b',['{group}'])")
    lines += [
        "MemENTable('levels.b')",
        "PrintTable('levels.b','levels.txt',1)",
        f"TRTable('tr.b',{groups!r},['initial'],-1)",
        "PrintTable('tr.b','tr.txt',1)",
    ]
    if agroups:
        lines += [
            f"AITable('ai.b',['initial'],{agroups!r})",
            "PrintTable('ai.b','ai.txt',1)",
        ]
    fac_job(directory, lines)
    levels = parse_levels(directory / "levels.txt")
    tr = [r for r in parse_transitions(directory / "tr.txt") if r["upper"] < n_initial]
    assert all(r["upper"] < n_initial and r["lower"] >= n_initial for r in tr)
    record = {
        "configuration": config(initial),
        "initial_levels": n_initial,
        "initial_degeneracy": sum(r["TWOJ"] + 1 for r in levels[:n_initial]),
        "radiative_per_s": average_rates(tr, levels, n_initial, "A_per_s"),
        "fluorescence": {},
    }
    offset = n_initial
    for s in sources:
        final = dict(shells)
        final[s] -= 1
        end = offset + csf_count(final)
        row = [r for r in tr if offset <= r["lower"] < end]
        record["fluorescence"][f"{s}->{hole}"] = average_rates(
            row, levels, n_initial, "A_per_s"
        )
        offset = end
    if agroups:
        ai = parse_ai(directory / "ai.txt")
        assert ai and all(r["upper"] < n_initial and r["lower"] >= offset for r in ai)
        record["auger_per_s"] = average_rates(ai, levels, n_initial, "rate_per_s")
        record["auger"] = {}
        for a, b in pairs:
            final = dict(shells)
            final[a] -= 1
            final[b] -= 1
            if min(final.values()) < 0:
                continue
            end = offset + csf_count(final)
            row = [r for r in ai if offset <= r["lower"] < end]
            record["auger"][f"{a}+{b}"] = average_rates(
                row, levels, n_initial, "rate_per_s"
            )
            offset = end
    assert offset == len(levels)
    return record


def fac_pi(element, shell, photons, directory):
    shells = CASES[element][1]
    final = dict(shells)
    final[shell] -= 1
    ni = csf_count(shells)
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
        f"SetPEGrid(60,1.0,{max(photons) * 1.2})",
        "RRTable('pi.b',['initial'],['final'],-1)",
    ]
    for i in range(ni):
        lines.append(f"TotalPICross('pi.b','pi-{i}.txt',{i},{photons!r})")
    fac_job(directory, lines)
    levels = parse_levels(directory / "levels.txt")
    total = np.zeros(len(photons))
    degeneracy = sum(r["TWOJ"] + 1 for r in levels[:ni])
    for i in range(ni):
        table = np.loadtxt(directory / f"pi-{i}.txt", ndmin=2)
        assert np.isfinite(table).all() and np.allclose(table[:, 0], photons)
        total += table[:, 1] * (levels[i]["TWOJ"] + 1) / degeneracy * 0.01
    assert np.all(total > 0)
    return total.tolist()


def solve(element, hole, ns):
    Z, shells = CASES[element]
    occupations = dict(shells)
    if hole:
        occupations[hole] -= 1
    subs = []
    for s, q in occupations.items():
        n, l = nl(s)
        for k in [-(l + 1), l] if l else [-1]:
            subs.append(Subshell(n, k, q * abs(k) / (2 * l + 1)))
    solver = DFSolver(
        Z,
        subs,
        ns=ns,
        ks=9,
        r_max=80,
        nuclear="point",
        exchange="hf",
        starting_potential="tfda",
    )
    result = solver.solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    if not result.converged:
        raise RuntimeError(f"{element} hole={hole} SCF unconverged {result.residual}")
    return result


def grid_orbitals(result):
    bd = result._bsp_data
    r, w = bd["gr"].ravel(), bd["grw"].ravel()
    orbitals = [
        ScatteringOrbital(
            n,
            k,
            q,
            _eval_bspline_on_grid(P, bd, "p").ravel(),
            _eval_bspline_on_grid(Q, bd, "q").ravel(),
        )
        for (n, k), (P, Q, q) in result.orbitals_by_subshell().items()
    ]
    return r, w, orbitals


def radiative(result, hole, photon_energies=None):
    r, w, orbs = grid_orbitals(result)
    h_n, h_l = nl(hole)
    out = {}
    for a in orbs:
        if (a.n, a.l) == (h_n, h_l):
            continue
        for h in orbs:
            if (h.n, h.l) != (h_n, h_l):
                continue
            a_label = f"{a.n}{'spdf'[a.l]}{2 * abs(a.kappa) - 1}/2"
            h_label = f"{h.n}{'spdf'[h.l]}{2 * abs(h.kappa) - 1}/2"
            omega = result.orbital_energies[a_label] - result.orbital_energies[h_label]
            if photon_energies is not None:
                channel = f"{a.n}{'spdf'[a.l]}->{hole}"
                if channel not in photon_energies:
                    continue
                omega = photon_energies[channel] / HARTREE_EV
            angular = angular_eL(a.kappa, h.kappa, 1)
            if omega <= 0 or angular == 0:
                continue
            radial = np.sum(w * r * (a.P * h.P + a.Q * h.Q))
            rate = einstein_A_eL(
                omega, (angular * radial) ** 2, 1, 2 * abs(a.kappa) - 1
            )
            # One n-l vacancy spread isotropically over both jj partners.
            rate *= a.occupation / (2 * (2 * h_l + 1))
            channel = f"{a.n}{'spdf'[a.l]}->{hole}"
            out[channel] = out.get(channel, 0) + float(rate)
    return {"radiative_per_s": sum(out.values()), "fluorescence": out}


def auger(result, reference, n_grid):
    # Adapt DFSolver coefficient storage to the public CSF-average API.
    adapter = SimpleNamespace(
        Z=result.Z,
        bsp_data=result._bsp_data,
        orbitals_dict={
            key: pair[:2] for key, pair in result.orbitals_by_subshell().items()
        },
        occupations={
            key: pair[2] for key, pair in result.orbitals_by_subshell().items()
        },
        orbital_energies=result.orbital_energies,
    )
    out = {}
    for channel, x in reference["auger"].items():
        a, b = channel.split("+")
        energy = (
            result.orbital_energies[label(a)]
            + result.orbital_energies[label(b)]
            - result.orbital_energies["1s1/2"]
        ) * HARTREE_EV
        row = {"orbital_sum_eV": float(energy), "xatom_eV": x["energy_eV"]}
        for tag, E in [("native", energy), ("xatom_energy", x["energy_eV"])]:
            if E <= 0:
                row[tag + "_per_s"] = 0.0
            else:
                row[tag + "_per_s"] = float(
                    auger_rate_csf_average(
                        adapter,
                        hole_label="1s1/2",
                        source_label=label(a),
                        source2_label=label(b),
                        resonance_energy_eV=E,
                        shooting_n_grid=n_grid,
                        angular_mode="ls",
                    )["Gamma_per_s"]
                )
        out[channel] = row
    assert all(
        np.isfinite(v) and v >= 0
        for row in out.values()
        for key, v in row.items()
        if key.endswith("_per_s")
    )
    return {
        "auger": out,
        "auger_per_s": sum(row["native_per_s"] for row in out.values()),
        "auger_xatom_energy_per_s": sum(
            row["xatom_energy_per_s"] for row in out.values()
        ),
    }


def pi(result, photons, n_grid):
    from atomx.scattering.models import ContinuumModel
    from atomx.scattering.radiative_recombination import (
        _shooting_photoionization_cross_section,
    )

    output = {s: np.zeros(len(photons)) for s in ("1s", "2s", "2p")}
    model = ContinuumModel.fac_residual_core(n_grid=n_grid, r_max=80)
    orbitals = result.orbitals_by_subshell()
    for sh in result.subshells:
        s = f"{sh.n}{'spdf'[sh.kappa if sh.kappa > 0 else -sh.kappa - 1]}"
        if s not in output:
            continue
        P, Q, _ = orbitals[sh.n, sh.kappa]
        values = _shooting_photoionization_cross_section(
            result, sh, P, Q, np.array(photons) / HARTREE_EV, model
        )
        output[s] += values * sh.occupation * MB
    return {s: v.tolist() for s, v in output.items()}


def references(args):
    output = {}
    path = args.output / "references.json"
    if path.exists():
        output = json.loads(path.read_text())
    for el in args.elements:
        record = {}
        directory = args.output / "raw" / el
        neutral = parse_xatom(xatom(args, directory / "xatom-neutral", el, []))
        # Three energies above K threshold for every Z, rounded to integral eV.
        K = abs(neutral["orbitals_eV"]["1s"])
        photons = [float(round(K * f)) for f in (1.2, 2.0, 4.0)]
        record["photons_eV"] = photons
        record["configuration"] = config(CASES[el][1])
        record["xatom_pi"] = {}
        for energy in photons:
            record["xatom_pi"][str(int(energy))] = parse_xatom(
                xatom(
                    args,
                    directory / f"xatom-pi-{int(energy)}",
                    el,
                    ["-pcs", "-PE", str(energy)],
                )
            )["pi_mb"]
        record["fac_pi"] = {
            s: fac_pi(el, s, photons, directory / f"fac-pi-{s}")
            for s in ("1s", "2s", "2p")
        }
        record["decay"] = {}
        for hole in ("1s", "2s", "2p"):
            xr = parse_xatom(
                xatom(
                    args,
                    directory / f"xatom-{hole}-hole",
                    el,
                    ["-hole", hole + "1", "-decay" if hole == "1s" else "-fluo"],
                )
            )
            fr = fac_decay(el, hole, directory / f"fac-{hole}-hole")
            record["decay"][hole] = {"xatom": xr, "fac": fr}
            print(
                el,
                hole,
                "reference widths",
                fr["radiative_per_s"],
                fr.get("auger_per_s"),
                flush=True,
            )
        dump(path.with_name(path.stem + "-" + el + ".json"), record)
        if path.exists():
            output.update(json.loads(path.read_text()))
        output[el] = record
        dump(path, output)


def atomic(args):
    refs = json.loads((args.output / "references.json").read_text())
    path = args.output / "atomx.json"
    output = json.loads(path.read_text()) if path.exists() else {}
    for el in args.elements:
        record = {"ns": args.ns, "shooting_grid": args.shooting_grid, "decay": {}}
        start = time.perf_counter()
        neutral = solve(el, None, args.ns)
        record["pi_mb"] = pi(neutral, refs[el]["photons_eV"], args.shooting_grid)
        for hole in ("1s", "2s", "2p"):
            result = solve(el, hole, args.ns)
            row = radiative(result, hole)
            row["orbital_eV"] = {
                k: float(v) * HARTREE_EV for k, v in result.orbital_energies.items()
            }
            if hole == "1s":
                row.update(
                    auger(result, refs[el]["decay"][hole]["xatom"], args.shooting_grid)
                )
            record["decay"][hole] = row
            print(
                el,
                hole,
                "atomx widths",
                row["radiative_per_s"],
                row.get("auger_per_s"),
                flush=True,
            )
        record["seconds"] = time.perf_counter() - start
        dump(path.with_name(path.stem + "-" + el + ".json"), record)
        if path.exists():
            output.update(json.loads(path.read_text()))
        output[el] = record
        dump(path, output)


def independent_fac_scattering(element, directory, q):
    """FAC spinors, integrated independently via explicit magnetic amplitudes.

    This does not call atomx's AtomicScattering or its reduced_C coefficient.
    It evaluates exp(i q z) between individual magnetic states, sums the
    coherent rank amplitudes, then traces the occupied projector. Radial
    integrals use Simpson on FAC's printed logarithmic grid.
    """
    from scipy.integrate import simpson

    shells = CASES[element][1]
    lines = fac_header(element) + [
        f"Config('{config(shells)}',group='neutral')",
        "ConfigEnergy(0)",
        "OptimizeRadial(['neutral'])",
        "ConfigEnergy(1)",
    ]
    for s in shells:
        n, l = nl(s)
        for k in [-(l + 1), l] if l else [-1]:
            lines.append(f"WaveFuncTable('wave-{n}-{k}.txt',{n},{k})")
    fac_job(directory, lines)
    orbs, norms = [], {}
    for s, occupation in shells.items():
        n, l = nl(s)
        for k in [-(l + 1), l] if l else [-1]:
            data = np.loadtxt(directory / f"wave-{n}-{k}.txt")
            r, P, Q = data[:, 1], data[:, 4], data[:, 5]
            norms[f"{n},{k}"] = float(simpson(P * P + Q * Q, x=r))
            orbs.append((s, k, occupation / (2 * (2 * l + 1)), r, P, Q))
    F = np.zeros(len(q))
    for _, k, fraction, r, P, Q in orbs:
        F += (
            2
            * abs(k)
            * fraction
            * simpson(spherical_jn(0, q[:, None] * r) * (P * P + Q * Q), x=r, axis=1)
        )
    S = np.full(len(q), float(sum(shells.values())))
    for a, b in itertools.product(orbs, repeat=2):
        sa, ka, fa, ra, Pa, Qa = a
        sb, kb, fb, rb, Pb, Qb = b
        la, lb = nl(sa)[1], nl(sb)[1]
        ja, jb = 2 * abs(ka) - 1, 2 * abs(kb) - 1
        # Union of original nodes retains both orbital tails; no constant
        # extrapolation past FAC's finite printed bound-wave support.
        r = np.unique(np.concatenate((ra, rb)))
        overlap = np.interp(r, ra, Pa, left=0, right=0) * np.interp(
            r, rb, Pb, left=0, right=0
        ) + np.interp(r, ra, Qa, left=0, right=0) * np.interp(
            r, rb, Qb, left=0, right=0
        )
        ranks = {}
        for L in range(abs(la - lb), la + lb + 1):
            if (la + lb + L) % 2 or abs(ja - jb) > 2 * L or ja + jb < 2 * L:
                continue
            R = simpson(spherical_jn(L, q[:, None] * r) * overlap, x=r, axis=1)
            reduced = (
                (-1) ** ((ja + 1) // 2)
                * np.sqrt((ja + 1) * (jb + 1))
                * wigner_3j(ja, 2 * L, jb, 1, 0, -1)
            )
            ranks[L] = (R, reduced)
        projection = np.zeros(len(q))
        for m in range(-min(ja, jb), min(ja, jb) + 1, 2):
            amplitude = np.zeros(len(q), dtype=complex)
            for L, (R, reduced) in ranks.items():
                amplitude += (
                    (1j) ** L
                    * (2 * L + 1)
                    * (-1) ** ((ja - m) // 2)
                    * wigner_3j(ja, 2 * L, jb, -m, 0, m)
                    * reduced
                    * R
                )
            projection += np.abs(amplitude) ** 2
        weight = fa if sa == sb else fa * fb
        S -= weight * projection
    return F, S, norms


def scatter(args):
    from scipy.special import roots_legendre

    output = {}
    path = args.output / "scattering.json"
    if path.exists():
        output = json.loads(path.read_text())
    for el in [e for e in args.elements if e in ("Ne", "Ar", "Fe")]:
        directory = args.output / "raw" / el
        out = xatom(
            args,
            directory / "xatom-scattering",
            el,
            ["-scattering", "-Q", "16", "-N_scd", "1600"],
        )
        # .scd contains Q, f0 at native precision.
        scd = np.loadtxt(next(out.parent.glob("*.scd")))
        q, xf = scd[:, 0], scd[:, 1]
        result = solve(el, None, args.ns)
        r, w, orbitals = grid_orbitals(result)
        model = AtomicScattering(r, w, orbitals)
        af = model.form_factor(q)
        ff, _, norms = independent_fac_scattering(el, directory / "fac-scattering", q)
        record = {
            "q_a0_inverse": q.tolist(),
            "xatom_f0": xf.tolist(),
            "atomx_f0": af.tolist(),
            "fac_jj_postprocessed_f0": ff.tolist(),
            "fac_radial_norms": norms,
            "atomx_radial_norms": {
                f"{a.n},{a.kappa}": float(w @ (a.P * a.P + a.Q * a.Q)) for a in orbitals
            },
            "energies": {},
        }
        for energy in (1000.0, 8000.0, 20000.0):
            out = xatom(
                args,
                directory / f"xatom-compton-{int(energy)}",
                el,
                [
                    "-compton",
                    "WH",
                    "-PE",
                    str(energy),
                    "-comp_n_theta",
                    "181",
                    "-comp_d_theta",
                    "1",
                    "-comp_min_theta",
                    "0",
                    "-comp_lmax",
                    "8",
                ],
            )
            table = np.loadtxt(next(out.parent.glob("*.ixs")))
            theta, xs = table[:, 0], table[:, 1]
            qs = momentum_transfer(energy, theta)
            fs, facs, norms = independent_fac_scattering(
                el, directory / "fac-scattering", qs
            )
            atom_s = model.incoherent_function(qs)
            x_form = np.interp(
                qs, q, xf
            )  # native XATOM q grid, linear interpolation control
            row = {
                "theta_rad": theta.tolist(),
                "xatom_S": xs.tolist(),
                "atomx_S": atom_s.tolist(),
                "fac_jj_postprocessed_S": facs.tolist(),
                "atomx_rayleigh_barn_sr": model.rayleigh(energy, theta).tolist(),
                "fac_rayleigh_barn_sr": (thomson(theta) * fs**2).tolist(),
                "xatom_rayleigh_barn_sr": (thomson(theta) * x_form**2).tolist(),
                "atomx_compton_KN_barn_sr": model.compton(energy, theta).tolist(),
                "fac_compton_KN_barn_sr": (
                    klein_nishina(energy, theta) * facs
                ).tolist(),
            }
            # Integrated cross sections: fixed 181-angle reference grid (with
            # sin(theta) measure); atomx also independent 256-node Gauss check.
            sint = np.sin(theta)
            for prefix, S in [
                ("xatom", xs),
                ("atomx", atom_s),
                ("fac_jj_postprocessed", facs),
            ]:
                row[prefix + "_compton_WH_barn"] = float(
                    2 * np.pi * trapezoid(thomson(theta) * S * sint, theta)
                )
                row[prefix + "_compton_KN_barn"] = float(
                    2
                    * np.pi
                    * trapezoid(klein_nishina(energy, theta) * S * sint, theta)
                )
            row["atomx_rayleigh_barn"] = float(
                2 * np.pi * trapezoid(model.rayleigh(energy, theta) * sint, theta)
            )
            row["xatom_rayleigh_barn"] = float(
                2 * np.pi * trapezoid(thomson(theta) * x_form**2 * sint, theta)
            )
            row["fac_rayleigh_barn"] = float(
                2 * np.pi * trapezoid(thomson(theta) * fs**2 * sint, theta)
            )
            mu, gw = roots_legendre(256)
            gt = np.arccos(mu)
            row["atomx_rayleigh_gauss_barn"] = float(
                2 * np.pi * (gw @ model.rayleigh(energy, gt))
            )
            row["atomx_compton_KN_gauss_barn"] = float(
                2 * np.pi * (gw @ model.compton(energy, gt))
            )
            record["energies"][str(int(energy))] = row
        dump(path.with_name(path.stem + "-" + el + ".json"), record)
        if path.exists():
            output.update(json.loads(path.read_text()))
        output[el] = record
        dump(path, output)
        print(el, "scattering complete", flush=True)


def radiative_controls(args):
    from scipy.integrate import simpson

    ref = json.loads((args.output / "references.json").read_text())
    rows = []
    for el in args.elements:
        shells = dict(CASES[el][1])
        shells["1s"] -= 1
        directory = args.output / "raw" / el / "fac-k-radial-control"
        lines = fac_header(el) + [
            f"Config('{config(shells)}',group='initial')",
            "ConfigEnergy(0)",
            "OptimizeRadial(['initial'])",
            "ConfigEnergy(1)",
        ]
        for s in shells:
            n, l = nl(s)
            for k in [-(l + 1), l] if l else [-1]:
                lines.append(f"WaveFuncTable('wave-{n}-{k}.txt',{n},{k})")
        fac_job(directory, lines)
        orbs = {}
        for s in shells:
            n, l = nl(s)
            for k in [-(l + 1), l] if l else [-1]:
                path = directory / f"wave-{n}-{k}.txt"
                data = np.loadtxt(path)
                energy = float(
                    re.search(r"# energy =\s*(" + FLOAT + ")", path.read_text())[1]
                )
                orbs[n, k] = (data[:, 1], data[:, 4], data[:, 5], energy)
        h = orbs[1, -1]
        fac_photons = {}
        fac_reference = ref[el]["decay"]["1s"]["fac"]
        initial_levels = parse_levels(
            args.output / "raw" / el / "fac-1s-hole" / "levels.txt"
        )[: fac_reference["initial_levels"]]
        weights = {r["ILEV"]: r["TWOJ"] + 1 for r in initial_levels}
        tr = [
            r
            for r in parse_transitions(
                args.output / "raw" / el / "fac-1s-hole" / "tr.txt"
            )
            if r["upper"] in weights
        ]
        offset = fac_reference["initial_levels"]
        for s in [s for s in shells if s != "1s"]:
            final = dict(CASES[el][1])
            final[s] -= 1
            end = offset + csf_count(final)
            lines_r = [r for r in tr if offset <= r["lower"] < end]
            denom = sum(weights[r["upper"]] * r["A_per_s"] for r in lines_r)
            if denom:
                fac_photons[s + "->1s"] = (
                    sum(
                        weights[r["upper"]] * r["A_per_s"] * r["DELTA_E_eV"]
                        for r in lines_r
                    )
                    / denom
                )
            offset = end
        fac_orbital_rate, fac_table_energy_rate = 0.0, 0.0
        for (n, k), (r, P, Q, E) in orbs.items():
            l = k if k > 0 else -k - 1
            ang = angular_eL(k, -1, 1)
            if not ang or h[3] >= E:
                continue
            radial = simpson(
                r
                * (
                    P * np.interp(r, h[0], h[1], left=0, right=0)
                    + Q * np.interp(r, h[0], h[2], left=0, right=0)
                ),
                x=r,
            )
            source_q = shells[f"{n}{'spdf'[l]}"] * abs(k) / (2 * l + 1)
            strength = (ang * radial) ** 2
            fac_orbital_rate += (
                einstein_A_eL((E - h[3]) / HARTREE_EV, strength, 1, 2 * abs(k) - 1)
                * source_q
                / 2
            )
            photon = fac_photons.get(f"{n}{'spdf'[l]}->1s")
            if photon:
                fac_table_energy_rate += (
                    einstein_A_eL(photon / HARTREE_EV, strength, 1, 2 * abs(k) - 1)
                    * source_q
                    / 2
                )
        result = solve(el, "1s", args.ns)
        native = radiative(result, "1s")["radiative_per_s"]
        matching = radiative(result, "1s", fac_photons)["radiative_per_s"]
        row = dict(
            element=el,
            native_atomx_per_s=native,
            atomx_FAC_mean_photon_per_s=matching,
            fac_spinors_native_orbital_energy_per_s=float(fac_orbital_rate),
            fac_spinors_FAC_mean_photon_per_s=float(fac_table_energy_rate),
            fac_TR_per_s=fac_reference["radiative_per_s"],
        )
        rows.append(row)
        csv_write(args.output / "radiative_controls.csv", rows)
        print(el, "radiative controls", row, flush=True)


def csv_write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(args):
    ref = json.loads((args.output / "references.json").read_text())
    atom = json.loads((args.output / "atomx.json").read_text())
    for p in args.output.glob("atomx-*.json"):
        el = p.stem.removeprefix("atomx-")
        if el in CASES:
            atom[el] = json.loads(p.read_text())
    atom = {el: atom[el] for el in CASES if el in atom}
    dump(args.output / "atomx.json", atom)
    pirows, rr, yr, totals = [], [], [], []
    for el, a in atom.items():
        r = ref[el]
        for s in ("1s", "2s", "2p"):
            for i, E in enumerate(r["photons_eV"]):
                av, fv, xv = (
                    a["pi_mb"][s][i],
                    r["fac_pi"][s][i],
                    r["xatom_pi"][str(int(E))][s],
                )
                pirows.append(
                    dict(
                        element=el,
                        Z=CASES[el][0],
                        shell=s,
                        photon_eV=E,
                        atomx_Mb=av,
                        fac_Mb=fv,
                        xatom_Mb=xv,
                        atomx_over_fac=av / fv,
                        atomx_over_xatom=av / xv,
                    )
                )
        for hole, d in a["decay"].items():
            rd = r["decay"][hole]
            av, fv, xv = (
                d["radiative_per_s"],
                rd["fac"]["radiative_per_s"],
                rd["xatom"]["radiative_per_s"],
            )
            totals.append(
                dict(
                    element=el,
                    Z=CASES[el][0],
                    hole=hole,
                    atomx_per_s=av,
                    fac_per_s=fv,
                    xatom_per_s=xv,
                    atomx_over_fac=av / fv if fv else None,
                    atomx_over_xatom=av / xv if xv else None,
                )
            )
            channels = set(d["fluorescence"]) | set(rd["xatom"]["fluorescence"])
            for c in sorted(channels):
                av = d["fluorescence"].get(c, 0.0)
                fv = rd["fac"]["fluorescence"].get(c, 0.0)
                xv = rd["xatom"]["fluorescence"].get(c, {}).get("rate_per_s", 0.0)
                rr.append(
                    dict(
                        element=el,
                        Z=CASES[el][0],
                        hole=hole,
                        channel=c,
                        atomx_per_s=av,
                        fac_per_s=fv,
                        xatom_per_s=xv,
                        atomx_over_fac=av / fv if fv else None,
                        atomx_over_xatom=av / xv if xv else None,
                    )
                )
        d = a["decay"]["1s"]
        fr, xr = r["decay"]["1s"]["fac"], r["decay"]["1s"]["xatom"]

        def omega(row):
            return row["radiative_per_s"] / (
                row["radiative_per_s"] + row["auger_per_s"]
            )

        yr.append(
            dict(
                element=el,
                Z=CASES[el][0],
                atomx_radiative_per_s=d["radiative_per_s"],
                atomx_auger_per_s=d["auger_per_s"],
                fac_radiative_per_s=fr["radiative_per_s"],
                fac_auger_per_s=fr["auger_per_s"],
                xatom_radiative_per_s=xr["radiative_per_s"],
                xatom_auger_per_s=xr["auger_per_s"],
                atomx_omega_K=omega(d),
                fac_omega_K=omega(fr),
                xatom_omega_K=omega(xr),
                atomx_over_fac=omega(d) / omega(fr),
                atomx_over_xatom=omega(d) / omega(xr),
                atomx_xatom_energy_omega_K=d["radiative_per_s"]
                / (d["radiative_per_s"] + d["auger_xatom_energy_per_s"]),
            )
        )
    csv_write(args.output / "pi.csv", pirows)
    csv_write(args.output / "fluorescence.csv", rr)
    csv_write(args.output / "fluorescence_totals.csv", totals)
    csv_write(args.output / "yield.csv", yr)
    metrics = {
        "pi": {
            "points": len(pirows),
            "ratio_fac_range": [
                min(row["atomx_over_fac"] for row in pirows),
                max(row["atomx_over_fac"] for row in pirows),
            ],
        },
        "fluorescence": {"channels": len(rr), "hole_widths": len(totals)},
        "yield": {"Z_points": len(yr), "rows": yr},
    }
    if (args.output / "scattering.json").exists():
        sc = json.loads((args.output / "scattering.json").read_text())
        metrics["scattering"] = {
            "elements": list(sc),
            "q_points": sum(len(row["q_a0_inverse"]) for row in sc.values()),
        }
        sr = []
        for el, data in sc.items():
            for E, row in data["energies"].items():
                theta = np.array(row["theta_rad"])
                x_form = np.interp(
                    momentum_transfer(float(E), theta),
                    data["q_a0_inverse"],
                    data["xatom_f0"],
                )
                row["xatom_rayleigh_barn"] = float(
                    2
                    * np.pi
                    * trapezoid(thomson(theta) * x_form**2 * np.sin(theta), theta)
                )
                sr.append(
                    dict(
                        element=el,
                        photon_eV=int(E),
                        atomx_rayleigh_barn=row["atomx_rayleigh_barn"],
                        fac_rayleigh_barn=row["fac_rayleigh_barn"],
                        xatom_rayleigh_barn=row["xatom_rayleigh_barn"],
                        atomx_compton_KN_barn=row["atomx_compton_KN_barn"],
                        fac_compton_KN_barn=row["fac_jj_postprocessed_compton_KN_barn"],
                        xatom_compton_WH_barn=row["xatom_compton_WH_barn"],
                        atomx_compton_WH_barn=row["atomx_compton_WH_barn"],
                        atomx_rayleigh_angle_grid_error=row["atomx_rayleigh_barn"]
                        / row["atomx_rayleigh_gauss_barn"]
                        - 1,
                        atomx_compton_angle_grid_error=row["atomx_compton_KN_barn"]
                        / row["atomx_compton_KN_gauss_barn"]
                        - 1,
                    )
                )
        csv_write(args.output / "scattering.csv", sr)
    dump(args.output / "summary.json", metrics)
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=["references", "atomx", "scatter", "summary", "radiative-controls"],
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "docs/systematic_xray_2026_10_02"
    )
    p.add_argument("--elements", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument(
        "--xatom",
        type=Path,
        default=Path(
            os.environ.get("XATOM_BIN", "/tmp/atomx-xraypac-build/build/src/xatom")
        ),
    )
    p.add_argument("--ns", type=int, default=180)
    p.add_argument("--shooting-grid", type=int, default=16000)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    globals()[
        {
            "references": "references",
            "atomx": "atomic",
            "scatter": "scatter",
            "summary": "summarize",
            "radiative-controls": "radiative_controls",
        }[args.stage]
    ](args)


if __name__ == "__main__":
    main()
