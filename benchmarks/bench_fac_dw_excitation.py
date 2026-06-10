#!/usr/bin/env python3
"""FAC parity driver for distorted-wave electron-impact excitation.

The live FAC path writes a CE table with ``fac_runner.run_sfac`` and parses it
with the FAC ASCII reader.  CI uses the checked-in CSV as a frozen parity gate
because the local ``sfac`` binary is not available on most runners.
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

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.io.fac_writer import read_ce  # noqa: E402
from atomx.scattering.distorted_wave.solver import dw_excitation_cross_section  # noqa: E402
from fac_runner import SFAC_BIN, run_sfac  # noqa: E402


Z_GRID = [10, 26, 54]
ENERGY_FACTORS = [1.0, 2.0, 5.0]
OUT = ROOT / "benchmarks" / "results_fac_dw_excitation.csv"

FAC_TEMPLATE = """SetAtom('{symbol}')
Closed('')
Config('g', '{ground_config}')
Config('e', '{excited_config}')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
Structure('{stem}.lev.b', ['g', 'e'])
MemENTable('{stem}.lev.b')
PrintTable('{stem}.lev.b', '{stem}.lev', 1)
SetUsrPEGrid({nusr}, {energies})
CETable('{stem}.ce.b', ['g'], ['e'])
PrintTable('{stem}.ce.b', '{stem}.ce', 1)
"""

SYMBOLS = {
    10: "Ne",
    26: "Fe",
    54: "Xe",
}

SEQUENCES = {
    "he-like": {
        "charge_offset": 2,
        "ground": "1s2",
        "excited": "1s1 2p1",
        "transition": "1s2 1S0 -> 1s2p 1P1",
    },
    "li-like": {
        "charge_offset": 3,
        "ground": "1s2 2s1",
        "excited": "1s2 2p1",
        "transition": "1s2 2s -> 1s2 2p",
    },
    "ne-like": {
        "charge_offset": 10,
        "ground": "1s2 2s2 2p6",
        "excited": "1s2 2s2 2p5 3s1",
        "transition": "2p -> 3s",
    },
}


def _threshold_eV(Z: int, spec: dict[str, object]) -> float:
    charge = Z - int(spec["charge_offset"])
    ground = Atom(Z=Z, config=str(spec["ground"]), charge=charge, breit=False, qed=False)
    excited = Atom(Z=Z, config=str(spec["excited"]), charge=charge, breit=False, qed=False)
    ground.solve()
    excited.solve()
    return float((excited._result.total_energy - ground._result.total_energy) * HARTREE_EV)


def run_atomx(Z: int, spec: dict[str, object], energies_eV: np.ndarray, threshold_eV: float) -> np.ndarray:
    charge = Z - int(spec["charge_offset"])
    ground = Atom(Z=Z, config=str(spec["ground"]), charge=charge, breit=False, qed=False)
    excited = Atom(Z=Z, config=str(spec["excited"]), charge=charge, breit=False, qed=False)
    ground.solve()
    excited.solve()
    _sigma, omega = dw_excitation_cross_section(
        ground,
        excited,
        energies_eV,
        L_max=4,
        kappa_max=6,
        transition_energy_eV=threshold_eV,
    )
    return np.asarray(omega, dtype=float)


def run_fac(Z: int, spec: dict[str, object], energies_eV: np.ndarray, timeout: float, keep_dir: Path | None) -> tuple[np.ndarray, str]:
    if not SFAC_BIN.exists():
        return np.zeros_like(energies_eV), f"FAC unavailable: {SFAC_BIN}"
    sequence = next((name for name, candidate in SEQUENCES.items() if candidate is spec), "sequence")
    stem = f"{sequence.replace('-', '_')}_dw_ce_z{Z}"
    script_text = FAC_TEMPLATE.format(
        symbol=SYMBOLS[Z],
        stem=stem,
        nusr=len(energies_eV),
        energies="[" + ", ".join(f"{e:.8e}" for e in energies_eV) + "]",
        ground_config=str(spec["ground"]),
        excited_config=str(spec["excited"]),
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
    parser.add_argument("--sequence", choices=["all", *SEQUENCES.keys()], default="he-like")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    sequence_names = list(SEQUENCES) if args.sequence == "all" else [args.sequence]
    for sequence in sequence_names:
        spec = SEQUENCES[sequence]
        for Z in Z_GRID:
            threshold = _threshold_eV(Z, spec)
            energies = np.asarray([threshold * f for f in ENERGY_FACTORS], dtype=float)
            omega_atomx = run_atomx(Z, spec, energies, threshold)
            note = ""
            if args.skip_fac:
                omega_fac = np.zeros_like(energies)
                note = "FAC skipped (--skip-fac)"
            else:
                keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
                omega_fac, note = run_fac(Z, spec, energies, args.timeout, keep)
            for factor, energy, atomx, fac in zip(ENERGY_FACTORS, energies, omega_atomx, omega_fac):
                rel = ((float(atomx) - float(fac)) / float(fac)) if fac > 0.0 else ""
                rows.append({
                    "Z": Z,
                    "sequence": sequence,
                    "transition": str(spec["transition"]),
                    "E_over_threshold": factor,
                    "energy_eV": energy,
                    "threshold_eV": threshold,
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
            "sequence",
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
            "sequence": "",
            "transition": "",
            "E_over_threshold": "",
            "energy_eV": "",
            "threshold_eV": "",
            "omega_atomx": "",
            "omega_fac": "",
            "rel_diff": "" if math.isnan(rms) else f"{rms:.6g}",
            "note": "target rms <= 0.10",
        })
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
