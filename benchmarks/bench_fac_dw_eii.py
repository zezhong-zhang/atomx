#!/usr/bin/env python3
"""FAC parity driver for distorted-wave electron-impact ionisation."""

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
from atomx.scattering.electron_ionization.coulomb_born import coulomb_born_ci_atom  # noqa: E402
from atomx.scattering.electron_ionization.dw import dw_eii_cross_section  # noqa: E402
from fac_runner import SFAC_BIN, parse_ci, run_sfac  # noqa: E402


Z_GRID = [10, 26, 54]
ENERGY_FACTORS = [1.0, 2.0, 5.0]
OUT = ROOT / "benchmarks" / "results_fac_dw_eii.csv"

SYMBOLS = {
    10: "Ne",
    26: "Fe",
    54: "Xe",
}

SEQUENCES = {
    "he-like": {"charge_offset": 2, "ground": "1s2", "ion": "1s1"},
    "li-like": {"charge_offset": 3, "ground": "1s2 2s1", "ion": "1s1 2s1"},
    "ne-like": {"charge_offset": 10, "ground": "1s2 2s2 2p6", "ion": "1s1 2s2 2p6"},
}

FAC_TEMPLATE = """SetAtom('{symbol}')
Closed('')
Config('g', '{ground_config}')
Config('ion', '{ion_config}')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
Structure('{stem}.lev.b', ['g', 'ion'])
MemENTable('{stem}.lev.b')
PrintTable('{stem}.lev.b', '{stem}.lev', 1)
SetUsrCIEGrid({energies})
CITable('{stem}.ci.b', ['g'], ['ion'])
PrintTable('{stem}.ci.b', '{stem}.ci', 1)
"""


def _k_threshold_eV(atom: Atom) -> float:
    eps = atom._result.orbital_energies["1s1/2"]
    return float(-eps * HARTREE_EV)


def run_atomx(Z: int, spec: dict[str, object], energies_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    charge = Z - int(spec["charge_offset"])
    atom = Atom(Z=Z, config=str(spec["ground"]), charge=charge, breit=False, qed=False)
    atom.solve()
    dw = dw_eii_cross_section(
        atom._result,
        energies_eV,
        kappa_max=4,
        n_outgoing_energies=6,
    )
    cb = coulomb_born_ci_atom(atom._result, energies_eV)["sigma_total_cm2"]
    return np.asarray(dw.sigma_total_cm2, dtype=float), np.asarray(cb, dtype=float)


def run_fac(Z: int, spec: dict[str, object], energies_eV: np.ndarray, timeout: float, keep_dir: Path | None) -> tuple[np.ndarray, str]:
    if not SFAC_BIN.exists():
        return np.zeros_like(energies_eV), f"FAC unavailable: {SFAC_BIN}"
    sequence = next((name for name, candidate in SEQUENCES.items() if candidate is spec), "sequence")
    stem = f"{sequence.replace('-', '_')}_dw_ci_z{Z}"
    script_text = FAC_TEMPLATE.format(
        symbol=SYMBOLS[Z],
        stem=stem,
        nusr=len(energies_eV),
        energies="[" + ", ".join(f"{e:.8e}" for e in energies_eV) + "]",
        ground_config=str(spec["ground"]),
        ion_config=str(spec["ion"]),
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
        rows = parse_ci(workdir / f"{stem}.ci")
        if not rows:
            return np.zeros_like(energies_eV), "FAC produced no CI rows"
        row = max(rows, key=lambda r: float(np.nanmax(r["sigma_1e20cm2"])))
        return np.asarray(row["sigma_1e20cm2"], dtype=float) * 1.0e-20, ""
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
            charge = Z - int(spec["charge_offset"])
            atom = Atom(Z=Z, config=str(spec["ground"]), charge=charge, breit=False, qed=False)
            atom.solve()
            threshold = _k_threshold_eV(atom)
            energies = np.asarray([threshold * f for f in ENERGY_FACTORS], dtype=float)
            sigma_atomx, sigma_cb = run_atomx(Z, spec, energies)
            if args.skip_fac:
                sigma_fac = np.zeros_like(energies)
                note = "FAC skipped (--skip-fac)"
            else:
                keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
                sigma_fac, note = run_fac(Z, spec, energies, args.timeout, keep)
            for factor, energy, atomx, fac, cb in zip(ENERGY_FACTORS, energies, sigma_atomx, sigma_fac, sigma_cb):
                rel = ((float(atomx) - float(fac)) / float(fac)) if fac > 0.0 else ""
                rows.append({
                    "Z": Z,
                    "sequence": sequence,
                    "shell": "K",
                    "E_over_threshold": factor,
                    "energy_eV": energy,
                    "threshold_eV": threshold,
                    "sigma_atomx_cm2": float(atomx),
                    "sigma_fac_cm2": float(fac),
                    "sigma_cb_cm2": float(cb),
                    "cb_over_dw": float(cb / atomx) if atomx > 0.0 else "",
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
            "shell",
            "E_over_threshold",
            "energy_eV",
            "threshold_eV",
            "sigma_atomx_cm2",
            "sigma_fac_cm2",
            "sigma_cb_cm2",
            "cb_over_dw",
            "rel_diff",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "Z": "rms",
            "sequence": "",
            "shell": "",
            "E_over_threshold": "",
            "energy_eV": "",
            "threshold_eV": "",
            "sigma_atomx_cm2": "",
            "sigma_fac_cm2": "",
            "sigma_cb_cm2": "",
            "cb_over_dw": "",
            "rel_diff": "" if math.isnan(rms) else f"{rms:.6g}",
            "note": "target rms <= 0.15",
        })
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
