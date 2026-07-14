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
# 1.05 (not 1.0): at exact threshold the scattered energy is 0 and the two
# codes would be compared at different clamped grid points; 5% above threshold
# both are well-defined and FAC's grid covers it without extrapolation.
ENERGY_FACTORS = [1.05, 2.0, 5.0]
OUT = ROOT / "benchmarks" / "results_fac_dw_excitation.csv"

# FAC computes CE collision strengths on its own automatically-constructed
# energy grid (ETYPE=1 => scattered-electron energy).  We read that grid and
# interpolate Omega(E_incident/threshold) to the requested E/threshold points,
# so the comparison is at matched collision dynamics even though atomx and FAC
# have slightly different absolute thresholds.  (The old SetUsrPEGrid(n, [list])
# call is not valid in FAC 1.1.5 -- the correct list form is SetUsrPEGrid([list])
# -- and forcing a custom grid is unnecessary here.)
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


def run_atomx(
    Z: int,
    spec: dict[str, object],
    energies_eV: np.ndarray,
    threshold_eV: float,
) -> tuple[np.ndarray, bool]:
    charge = Z - int(spec["charge_offset"])
    ground = Atom(Z=Z, config=str(spec["ground"]), charge=charge, breit=False, qed=False)
    excited = Atom(Z=Z, config=str(spec["excited"]), charge=charge, breit=False, qed=False)
    ground.solve()
    excited.solve()
    _sigma, omega = dw_excitation_cross_section(
        ground,
        excited,
        energies_eV,
        L_max=6,
        kappa_max=10,
        transition_energy_eV=threshold_eV,
    )
    converged = (
        bool(getattr(ground._result, "converged", True))
        and bool(getattr(excited._result, "converged", True))
    )
    return np.asarray(omega, dtype=float), converged


def run_fac(Z: int, spec: dict[str, object], factors, timeout: float, keep_dir: Path | None) -> tuple[np.ndarray, float, str]:
    """Return FAC multiplet-summed collision strengths at E_inc/threshold = ``factors``.

    atomx's DW driver returns the configuration (multiplet-summed) collision
    strength from the initial level, so the FAC reference sums Omega over ALL
    fine-structure lines of the g -> e transition array, each line evaluated
    at the same reduced incident energy E_inc = f * E_th(line).  FAC's CE grid
    is the scattered-electron energy (ETYPE=1): E_inc = E_s + E_th.

    Also returns FAC's strongest-line threshold (eV) for reporting.
    """
    factors = np.asarray(factors, dtype=float)
    if not SFAC_BIN.exists():
        return np.zeros_like(factors), 0.0, f"FAC unavailable: {SFAC_BIN}"
    sequence = next((name for name, candidate in SEQUENCES.items() if candidate is spec), "sequence")
    stem = f"{sequence.replace('-', '_')}_dw_ce_z{Z}"
    script_text = FAC_TEMPLATE.format(
        symbol=SYMBOLS[Z],
        stem=stem,
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
            return np.zeros_like(factors), 0.0, "FAC produced no CE rows"
        # Lines from the ground level (lower == 0) only.
        ground_rows = [r for r in rows if int(r["lower"]) == 0] or rows
        strongest = max(ground_rows,
                        key=lambda r: float(np.nanmax(r["collision_strength"])))
        fac_th = float(strongest["DELTA_E_eV"])
        omega_sum = np.zeros_like(factors)
        for row in ground_rows:
            th = float(row["DELTA_E_eV"])
            if th <= 0:
                continue
            E_scat = np.asarray(row["energy_eV"], dtype=float)
            omega = np.asarray(row["collision_strength"], dtype=float)
            keep = E_scat > 1.0e-3 * th  # drop FAC's sub-threshold sentinel row
            if not np.any(keep):
                continue
            x = (E_scat[keep] + th) / th
            order = np.argsort(x)
            omega_sum += np.interp(factors, x[order], omega[keep][order])
        return omega_sum, fac_th, ""
    except Exception as exc:  # noqa: BLE001
        return np.zeros_like(factors), 0.0, "FAC failed: " + " ".join(str(exc).split())
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
            omega_atomx, atomx_converged = run_atomx(Z, spec, energies, threshold)
            note = ""
            fac_th = 0.0
            if args.skip_fac:
                omega_fac = np.zeros_like(energies)
                note = "FAC skipped (--skip-fac)"
            else:
                keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
                omega_fac, fac_th, note = run_fac(Z, spec, ENERGY_FACTORS, args.timeout, keep)
            for factor, energy, atomx, fac in zip(ENERGY_FACTORS, energies, omega_atomx, omega_fac):
                rel = ((float(atomx) - float(fac)) / float(fac)) if fac > 0.0 else ""
                rows.append({
                    "Z": Z,
                    "sequence": sequence,
                    "transition": str(spec["transition"]),
                    "E_over_threshold": factor,
                    "energy_eV": energy,
                    "threshold_eV": threshold,
                    "fac_threshold_eV": fac_th,
                    "atomx_converged": int(atomx_converged),
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
            "fac_threshold_eV",
            "atomx_converged",
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
            "rel_diff": "" if math.isnan(rms) else f"{rms:.6g}",
            "note": "LIVE FAC 1.1.5; atomx DW vs FAC CE, matched E/threshold",
        })
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
