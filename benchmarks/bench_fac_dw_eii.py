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
# sigma(E) -> 0 at exact threshold for ionization; start the panel above it
# so both codes are compared at well-defined matched reduced energies.
ENERGY_FACTORS = [1.5, 2.0, 5.0]
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

# SetCIQkMode('dw'): pure distorted-wave, matching atomx's method exactly.
# FAC's default CI mode is QK_BED (binary-encounter-dipole scaled), a
# different model that suppresses the DW result by ~10-30% at E/th <= 5.
# SetCIPWOptions(tol, max_kl, max_eject, qr) with qr=36 forces EXACT
# (mode-1) Slater integrals for every partial wave: FAC's default
# phase-amplitude mode (IONLQR=0) is 4-14% low (Z=26..54), while atomx
# evaluates all integrals exactly — exact-to-exact is the honest gate.
# SetCIEGrid pins the INTERNAL grid too: FAC computes DW only on its
# internal egrid and evaluates user points through the 4-parameter
# CIRadialQkFromFit form — near threshold that fit mis-stated sigma by
# up to -44% at Z=54 (f=1.5: fit 8.77e-24 vs direct 1.578e-23).  With
# both grids pinned FAC evaluates DW directly at the requested points.
# FAC's CI user grid is the energy ABOVE threshold (incident - I).  The
# user grid is pinned explicitly at (f-1)*I so sigma is read at exactly the
# requested factors: letting FAC pick its default sparse grid and
# interpolating across the steep threshold rise mis-stated sigma(f=1.5) by
# -40% for He-like Fe (FAC-true 2.626e-22 vs 1.879e-22 interpolated).
FAC_TEMPLATE = """SetAtom('{symbol}')
Closed('')
Config('g', '{ground_config}')
Config('ion', '{ion_config}')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
Structure('{stem}.lev.b', ['g', 'ion'])
MemENTable('{stem}.lev.b')
SetCIQkMode('dw')
SetCIPWOptions(1e-3, 36, 4, 36)
SetCIEGrid({usr_grid})
SetUsrCIEGrid({usr_grid})
CITable('{stem}.ci.b', ['g'], ['ion'])
PrintTable('{stem}.ci.b', '{stem}.ci', 1)
"""


def _k_threshold_eV(atom: Atom) -> float:
    eps = atom._result.orbital_energies["1s1/2"]
    return float(-eps * HARTREE_EV)


def run_atomx(Z: int, spec: dict[str, object], energies_eV: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    charge = Z - int(spec["charge_offset"])
    atom = Atom(Z=Z, config=str(spec["ground"]), charge=charge, numerics="bspline", breit=False, qed=False)
    atom.solve()
    dw = dw_eii_cross_section(
        atom._result,
        energies_eV,
        kappa_max=10,
        n_outgoing_energies=25,
    )
    cb = coulomb_born_ci_atom(atom._result, energies_eV)["sigma_total_cm2"]
    return np.asarray(dw.sigma_total_cm2, dtype=float), np.asarray(cb, dtype=float)


def run_fac(Z: int, spec: dict[str, object], factors, timeout: float, keep_dir: Path | None,
            threshold_eV: float | None = None) -> tuple[np.ndarray, float, str]:
    """FAC pure-DW CI cross section at matched E_incident/threshold ``factors``.

    FAC's CI user grid is the energy above threshold; interpolate sigma at
    E_grid/I = f - 1 using FAC's own threshold I (the row's DELTA_E_eV).
    Also returns FAC's threshold (eV) for reporting.
    """
    factors = np.asarray(factors, dtype=float)
    if not SFAC_BIN.exists():
        return np.zeros_like(factors), 0.0, f"FAC unavailable: {SFAC_BIN}"
    sequence = next((name for name, candidate in SEQUENCES.items() if candidate is spec), "sequence")
    stem = f"{sequence.replace('-', '_')}_dw_ci_z{Z}"
    if threshold_eV:
        usr = [float((f - 1.0) * threshold_eV) for f in factors]
    else:
        usr = []
    script_text = FAC_TEMPLATE.format(
        symbol=SYMBOLS[Z],
        stem=stem,
        ground_config=str(spec["ground"]),
        ion_config=str(spec["ion"]),
        usr_grid="[" + ", ".join(f"{u:.4f}" for u in usr) + "]",
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
            return np.zeros_like(factors), 0.0, "FAC produced no CI rows"
        # Sum over final ion levels reachable from the ground level (for the
        # He-like K-shell panel there is a single 1s channel).
        ground_rows = [r for r in rows if int(r["bound"]) == 0] or rows
        fac_th = float(ground_rows[0]["DELTA_E_eV"])
        sigma_at = np.zeros_like(factors)
        for row in ground_rows:
            th = float(row["DELTA_E_eV"])
            if th <= 0:
                continue
            E_grid = np.asarray(row["E_eV"], dtype=float)
            sig = np.asarray(row["sigma_1e20cm2"], dtype=float) * 1.0e-20
            x = E_grid / th  # = E_inc/I - 1
            order = np.argsort(x)
            sigma_at += np.interp(factors - 1.0, x[order], sig[order])
        return sigma_at, fac_th, ""
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
            charge = Z - int(spec["charge_offset"])
            atom = Atom(Z=Z, config=str(spec["ground"]), charge=charge, numerics="bspline", breit=False, qed=False)
            atom.solve()
            threshold = _k_threshold_eV(atom)
            energies = np.asarray([threshold * f for f in ENERGY_FACTORS], dtype=float)
            sigma_atomx, sigma_cb = run_atomx(Z, spec, energies)
            fac_th = 0.0
            if args.skip_fac:
                sigma_fac = np.zeros_like(energies)
                note = "FAC skipped (--skip-fac)"
            else:
                keep = Path(args.keep_fac_dir) if args.keep_fac_dir else None
                sigma_fac, fac_th, note = run_fac(Z, spec, ENERGY_FACTORS, args.timeout, keep,
                                                  threshold_eV=threshold)
            for factor, energy, atomx, fac, cb in zip(ENERGY_FACTORS, energies, sigma_atomx, sigma_fac, sigma_cb):
                rel = ((float(atomx) - float(fac)) / float(fac)) if fac > 0.0 else ""
                rows.append({
                    "Z": Z,
                    "sequence": sequence,
                    "shell": "K",
                    "E_over_threshold": factor,
                    "energy_eV": energy,
                    "threshold_eV": threshold,
                    "fac_threshold_eV": fac_th,
                    "atomx_converged": int(bool(getattr(atom._result, "converged", True))),
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
            "fac_threshold_eV",
            "atomx_converged",
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
            "atomx_converged": "",
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
