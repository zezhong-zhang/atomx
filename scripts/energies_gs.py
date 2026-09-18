#!/usr/bin/env python3
"""Ground-state DHF total energy: atomx vs MCDHF (Table S12) and Radial DHFS.

For each requested Z, runs a single-config B-spline Dirac-Hartree-Fock
solve (defaults: HF exchange, Fermi nucleus, Breit+QED off — to match the
single-config DHF reference baseline) and prints a comparison table.

References
----------
* "Ref MCDHF": ``/Users/zhangzz/code/ASAP/energy_level.csv``, "This Work"
  column from Table S12 (Visscher-quality MCDHF, includes correlation).
* "Radial DHFS": hard-coded Z=2..18 values from running Salvat's DHFS
  code (Dirac-Hartree-Fock-Slater, local exchange). Kept inline because
  there is no upstream file to point at.

Notes
-----
Both atomx-DHF and Radial-DHFS miss correlation energy, so they sit
systematically above MCDHF. Differences from MCDHF are reported in mHa
and as a relative percent so they're comparable across the periodic table.

Examples
--------
    python scripts/energies_gs.py 2-18
    python scripts/energies_gs.py He-Ar -v
    python scripts/energies_gs.py --all
    python scripts/energies_gs.py 2-36 --csv energies.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from atomx import Atom
from atomx.constants import atomic_numbers, element_symbol

LOG = logging.getLogger("energies_gs")

DEFAULT_REF_CSV = Path(os.environ.get("ATOMX_MCDHF_REF_CSV", ""))

# Radial DHFS (Salvat) — kept inline; no upstream source file.
RADIAL_DHFS: dict[int, float] = {
    2: -2.861813,  3: -7.425690,  4: -14.566380,  5: -24.520540,
    6: -37.633770, 7: -54.284200, 8: -74.778220,  9: -99.447240,
    10: -128.623300, 11: -161.999800, 12: -199.853100,
    13: -242.242400, 14: -289.354200, 15: -341.385700,
    16: -398.485600, 17: -460.819600, 18: -528.554100,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_single(tok: str) -> int:
    try:
        Z = int(tok)
    except ValueError:
        if tok not in atomic_numbers:
            raise SystemExit(f"Unknown element token: {tok!r}") from None
        Z = atomic_numbers[tok]
    if not 1 <= Z <= 118:
        raise SystemExit(f"Z={Z} out of range 1..118")
    return Z


def parse_elements(tokens: list[str]) -> list[int]:
    """Accept 'Ag', '47', '1-10', '1..86', 'H-Ne' (mix freely)."""
    out: list[int] = []
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        sep = next((c for c in ("..", ":", "-") if c in tok and
                    len(tok.split(c)) == 2 and all(tok.split(c))), None)
        if sep:
            lo, hi = (_resolve_single(s.strip()) for s in tok.split(sep))
            step = 1 if lo <= hi else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(_resolve_single(tok))
    return out


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_ref_mcdhf(csv_path: Path) -> dict[int, float]:
    """Parse the 'This Work' column from Table S12."""
    refs: dict[int, float] = {}
    with csv_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                Z = int(row[0].strip())
                refs[Z] = float(row[2].strip())
            except (ValueError, IndexError):
                continue
    return refs


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_one(Z: int, args) -> tuple[float | None, str, float]:
    """Returns (total_energy, status, dt_seconds)."""
    t0 = time.time()
    try:
        atom = Atom(
            Z=Z,
            exchange=args.exchange,
            nuclear=args.nuclear,
            breit=args.breit,
            qed=args.qed,
        )
        atom.solve(tol=args.tol, max_iter=args.max_iter)
    except Exception as e:
        LOG.error("Z=%d (%s) solve failed: %s", Z, element_symbol(Z), e)
        return None, "FAIL", time.time() - t0
    status = "OK" if atom.converged else "NOCONV"
    return float(atom.total_energy), status, time.time() - t0


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("elements", nargs="*",
                    help="Symbols, Z ints, or ranges (e.g. 'Ag 47 2-86 H-Ne').")
    ap.add_argument("--all", action="store_true",
                    help="Iterate all Z present in the MCDHF reference CSV.")
    ap.add_argument("--ref-csv", default=DEFAULT_REF_CSV, type=Path,
                    help=f"MCDHF reference CSV (default: {DEFAULT_REF_CSV}).")
    ap.add_argument("--csv", default=None, type=Path,
                    help="Also write a machine-readable CSV.")
    ap.add_argument("--exchange", default="hf",
                    help="Atom(exchange=...) (default: 'hf').")
    ap.add_argument("--nuclear", default="fermi",
                    help="Atom(nuclear=...) (default: 'fermi').")
    ap.add_argument("--breit", action=argparse.BooleanOptionalAction, default=False,
                    help="Include Breit interaction (default: off, "
                         "matches single-config DHF baseline).")
    ap.add_argument("--qed", action=argparse.BooleanOptionalAction, default=False,
                    help="Include QED corrections (default: off, "
                         "matches single-config DHF baseline).")
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)

    level = (logging.DEBUG if args.verbose >= 2
             else logging.INFO if args.verbose == 1
             else logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    refs = load_ref_mcdhf(args.ref_csv)
    if not refs:
        ap.error(f"No MCDHF references parsed from {args.ref_csv}")

    if args.all:
        Zs = sorted(refs.keys())
    elif args.elements:
        Zs = parse_elements(args.elements)
    else:
        ap.error("Provide element tokens or --all")

    header = (f"{'Z':>3}  {'Elem':>4}  {'E_atomx':>14}  {'E_MCDHF':>14}  "
              f"{'ΔMCDHF[mHa]':>12}  {'err%':>7}  {'E_DHFS':>14}  "
              f"{'ΔDHFS[mHa]':>11}  {'Status':>6}  {'Time':>6}")
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    mcdhf_diffs: list[float] = []
    dhfs_diffs: list[float] = []

    for Z in Zs:
        symbol = element_symbol(Z)
        e_ref = refs.get(Z)
        e_dhfs = RADIAL_DHFS.get(Z)
        e_atomx, status, dt = solve_one(Z, args)

        if e_atomx is None or e_ref is None:
            d_ref_mha = err_pct = None
        else:
            d_ref_mha = (e_atomx - e_ref) * 1e3
            err_pct = abs(d_ref_mha) * 1e-3 / abs(e_ref) * 100
            mcdhf_diffs.append(d_ref_mha)

        if e_atomx is not None and e_dhfs is not None:
            d_dhfs_mha = (e_atomx - e_dhfs) * 1e3
            dhfs_diffs.append(d_dhfs_mha)
        else:
            d_dhfs_mha = None

        def fmt(x, spec: str, w: int):
            if x is None:
                return f"{'N/A':>{w}}"
            return format(x, spec)

        print(
            f"{Z:3d}  {symbol:>4}  "
            f"{fmt(e_atomx, '14.6f', 14)}  {fmt(e_ref, '14.6f', 14)}  "
            f"{fmt(d_ref_mha, '+12.3f', 12)}  "
            f"{fmt(err_pct, '7.3f', 7)}  "
            f"{fmt(e_dhfs, '14.6f', 14)}  "
            f"{fmt(d_dhfs_mha, '+11.3f', 11)}  "
            f"{status:>6}  {dt:5.1f}s"
        )
        rows.append({
            "Z": Z, "element": symbol,
            "E_atomx": e_atomx, "E_MCDHF": e_ref, "E_DHFS": e_dhfs,
            "diff_MCDHF_mHa": d_ref_mha, "diff_DHFS_mHa": d_dhfs_mha,
            "err_pct": err_pct, "status": status,
        })

    print("-" * len(header))
    if mcdhf_diffs:
        a = np.asarray(mcdhf_diffs)
        print(f"     atomx vs MCDHF  |  N={a.size}  "
              f"mean Δ = {a.mean():+.3f} mHa  "
              f"mean |Δ| = {np.abs(a).mean():.3f} mHa  "
              f"max |Δ| = {np.abs(a).max():.3f} mHa  "
              f"RMS = {np.sqrt((a**2).mean()):.3f} mHa")
    if dhfs_diffs:
        a = np.asarray(dhfs_diffs)
        print(f"     atomx vs DHFS   |  N={a.size}  "
              f"mean Δ = {a.mean():+.3f} mHa  "
              f"mean |Δ| = {np.abs(a).mean():.3f} mHa  "
              f"max |Δ| = {np.abs(a).max():.3f} mHa")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            fields = ["Z", "element", "E_atomx", "E_MCDHF", "E_DHFS",
                      "diff_MCDHF_mHa", "diff_DHFS_mHa", "err_pct", "status"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")

    fails = [r for r in rows if r["status"] == "FAIL"]
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
