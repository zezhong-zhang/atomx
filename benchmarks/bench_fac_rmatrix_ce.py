#!/usr/bin/env python3
"""FAC parity driver for R-matrix close-coupling electron-impact excitation.

AtomX uses the non-relativistic LS-coupling R-matrix solver with a true
He-like target basis (``1s^2 ^1S`` and ``1s n l ^1L / ^3L`` terms up to
n=2).  FAC provides the ``CETable`` distorted-wave reference for the
same He-like ions.

.. important::

    This benchmark currently does **not** meet the 15 % FAC RMS gate.  The
    AtomX calculation is a low-partial-wave close-coupling expansion without
    relativistic/top-up corrections, so it is expected to approximate the
    FAC distorted-wave reference only after the composite-target exchange
    kernel and convergence parameters are tuned.  The script is kept as a
    regression harness for the ``r_matrix_excitation_cross_section``
    production hook and will fail with a non-zero exit code until the RMS
    gate is achieved.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_runner import SFAC_BIN, run_sfac  # noqa: E402

from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.io.fac_writer import read_ce  # noqa: E402
from atomx.scattering.r_matrix import r_matrix_excitation_cross_section  # noqa: E402
from atomx.scattering.r_matrix.target_states import helike_target_states  # noqa: E402

Z_GRID = [6, 10]
ENERGY_FACTORS = [1.1, 1.5, 2.0, 3.0]
OUT = ROOT / "benchmarks" / "results_fac_rmatrix_ce.csv"

SYMBOLS = {
    6: "C",
    10: "Ne",
    18: "Ar",
}

FAC_TEMPLATE = """SetAtom('{symbol}')
Closed('')
Config('g', '1s2')
Config('e', '1s1 2p1')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
SetSE(0)
SetVP(0)
SetMS(0,0)
SetBreit(0)
Structure('{stem}.lev.b', ['g', 'e'])
MemENTable('{stem}.lev.b')
PrintTable('{stem}.lev.b', '{stem}.lev', 1)
SetCEGrid({nusr}, {emin}, {emax}, 0.0)
SetUsrCEGrid({nusr}, {emin}, {emax}, 0.0)
CETable('{stem}.ce.b', ['g'], ['e'])
PrintTable('{stem}.ce.b', '{stem}.ce', 1)
"""


class _MockAtom:
    """Lightweight Atom-like object carrying only the nuclear charge."""

    def __init__(self, Z: int):
        self.Z = Z


def threshold_eV(Z: int) -> float:
    """Hydrogenic 1s -> 2p threshold for a He-like ion."""
    return (3.0 / 8.0) * Z * Z * HARTREE_EV


def run_atomx(Z: int, energies_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atom = _MockAtom(Z)
    target_states = helike_target_states(Z=Z, n_max=2)
    sigma, omega = r_matrix_excitation_cross_section(
        atom,
        atom,
        energies_eV,
        target_states=target_states,
        initial_label="1s^2 ^1S",
        final_label="1s2p ^1P",
        target_spin=0.0,
        S_values=(0.5,),
        l_c_max=4,
        a_au=20.0,
        n_basis=90,
        k_spline_order=6,
        n_gauss=14,
    )
    return np.asarray(sigma, dtype=float), np.asarray(omega, dtype=float)


def run_fac(
    Z: int,
    energies_eV: np.ndarray,
    timeout: float,
    keep_dir: Path | None,
) -> tuple[np.ndarray, str]:
    if not SFAC_BIN.exists():
        return np.zeros_like(energies_eV), f"FAC unavailable: {SFAC_BIN}"

    stem = f"rmatrix_ce_z{Z}"
    script_text = FAC_TEMPLATE.format(
        symbol=SYMBOLS[Z],
        stem=stem,
        nusr=len(energies_eV),
        emin=f"{energies_eV[0]:.8e}",
        emax=f"{energies_eV[-1]:.8e}",
    )

    if keep_dir is None:
        ctx = tempfile.TemporaryDirectory()
        workdir = Path(ctx.name)
    else:
        ctx = None
        workdir = keep_dir
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        script = workdir / f"{stem}.sf"
        script.write_text(script_text)
        run_sfac(script, workdir=workdir, timeout=timeout)
        rows = read_ce(workdir / f"{stem}.ce")
        if not rows:
            return np.zeros_like(energies_eV), "FAC produced no CE rows"
        # He-like 1s2 -> 1s2p has several fine-structure rows.  The
        # dipole-allowed 1P1 row dominates the LS 1s -> 2p channel.
        row = max(rows, key=lambda r: float(np.nanmax(r["collision_strength"])))
        return np.asarray(row["collision_strength"], dtype=float), ""
    except Exception as exc:  # noqa: BLE001
        return np.zeros_like(energies_eV), "FAC failed: " + " ".join(str(exc).split())
    finally:
        if ctx is not None:
            ctx.cleanup()


def rms_relative(rows: list[dict[str, object]]) -> float:
    rels = [float(r["rel_diff"]) for r in rows if r["rel_diff"] != ""]
    if not rels:
        return math.nan
    return math.sqrt(sum(r * r for r in rels) / len(rels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-fac", action="store_true")
    parser.add_argument("--keep-fac-dir", default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for Z in Z_GRID:
        thr = threshold_eV(Z)
        energies = np.asarray([thr * f for f in ENERGY_FACTORS], dtype=float)
        sigma_atomx, omega_atomx = run_atomx(Z, energies)
        note = ""
        if args.skip_fac:
            omega_fac = np.zeros_like(energies)
            note = "FAC skipped (--skip-fac)"
        else:
            keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
            omega_fac, note = run_fac(Z, energies, args.timeout, keep)

        for factor, energy, atomx, fac in zip(
            ENERGY_FACTORS, energies, omega_atomx, omega_fac, strict=False
        ):
            rel = ((float(atomx) - float(fac)) / float(fac)) if fac > 0.0 else ""
            rows.append({
                "Z": Z,
                "transition": "1s -> 2p",
                "E_over_threshold": factor,
                "energy_eV": energy,
                "threshold_eV": thr,
                "omega_atomx": float(atomx),
                "omega_fac": float(fac),
                "rel_diff": rel,
                "note": note,
            })

    rms = rms_relative(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = [
            "Z",
            "transition",
            "E_over_threshold",
            "energy_eV",
            "threshold_eV",
            "omega_atomx",
            "omega_fac",
            "rel_diff",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "Z": "rms",
            "transition": "",
            "E_over_threshold": "",
            "energy_eV": "",
            "threshold_eV": "",
            "omega_atomx": "",
            "omega_fac": "",
            "rel_diff": "" if math.isnan(rms) else f"{rms:.6g}",
            "note": "target rms <= 0.15",
        })
    print(f"Wrote {out}")
    if not args.skip_fac and not math.isnan(rms) and rms > 0.15:
        print(
            f"ERROR: RMS relative difference {rms:.3f} exceeds the 0.15 gate; "
            "see the benchmark docstring for the known physics gap.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
