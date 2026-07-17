#!/usr/bin/env python3
"""Benchmark atomx vs FAC — ion ground-state binding energies.

For each (Z, N_el) pair we run:
    atomx:  Atom(Z=Z, charge=Z-N_el).solve()
    FAC:    SetAtom; Closed(...); Config(...); OptimizeRadial; Structure; PrintTable

Both use non-relativistic configurations filled in jj-coupling order; both
include Breit/QED off by default.  Compares:
    - ground-level energy (FAC's "E0" absolute binding energy, eV)
    - atomx DiracFock total energy (Hartree, converted to eV)
    - wall time
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from fac_runner import parse_e0, parse_levels, run_sfac  # noqa: E402
from fac_matrix import print_matrix_summary, run_matrix, write_matrix_csv  # noqa: E402

EV_PER_HARTREE = HARTREE_EV


ISOSEQUENCE_CONFIGS = {
    "H-like": ("1s1", 1),       # 1 electron
    "He-like": ("1s2", 2),      # 2 electrons
    "Li-like": ("1s2 2s1", 3),
    "Be-like": ("1s2 2s2", 4),
    "B-like": ("1s2 2s2 2p1", 5),
    "C-like": ("1s2 2s2 2p2", 6),
    "N-like": ("1s2 2s2 2p3", 7),
    "O-like": ("1s2 2s2 2p4", 8),
    "F-like": ("1s2 2s2 2p5", 9),
    "Ne-like": ("1s2 2s2 2p6", 10),
    "Na-like": ("1s2 2s2 2p6 3s1", 11),
    "Mg-like": ("1s2 2s2 2p6 3s2", 12),
    "Ar-like": ("1s2 2s2 2p6 3s2 3p6", 18),
}

DEFAULT_Z_PANEL = [10, 18, 26, 36, 54, 74, 82]
DEFAULT_ISO_PANEL = ["He-like", "Ne-like", "Ar-like"]

FAC_TEMPLATE = """# ground-state structure for {name} Z={Z}
SetAtom('{symbol}', {Z})
{closed_cmds}
Config('{label}', '{open_config}')
ConfigEnergy(0)
OptimizeRadial(['{label}'])
ConfigEnergy(1)
Structure('out.lev.b', ['{label}'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
"""


ELEMENT_SYMBOLS = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
    18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As",
    34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb",
    42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
    50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb",
    66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta",
    74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac",
    90: "Th", 91: "Pa", 92: "U",
}


CLOSED_SHELLS_FOR_ISO = {
    "He-like": [],
    "Ne-like": ["1s"],
    "Na-like": ["1s", "2s", "2p"],
    "Mg-like": ["1s", "2s", "2p"],
    "Ar-like": ["1s", "2s", "2p", "3s"],
}
OPEN_CONFIG_FOR_ISO = {
    "He-like": "1s2",
    "Ne-like": "2*8",       # "2s²2p⁶" in FAC notation
    "Na-like": "3s1",
    "Mg-like": "3s2",
    "Ar-like": "3p6",
}


def build_fac_script(Z: int, iso: str) -> str:
    symbol = ELEMENT_SYMBOLS[Z]
    closed = CLOSED_SHELLS_FOR_ISO[iso]
    closed_cmds = "\n".join(f"Closed('{sh}')" for sh in closed) if closed else "# (no closed shells)"
    return FAC_TEMPLATE.format(
        name=iso, Z=Z, symbol=symbol, closed_cmds=closed_cmds,
        label="gnd", open_config=OPEN_CONFIG_FOR_ISO[iso],
    )


def run_fac(Z: int, iso: str):
    """Returns (E0_in_eV, wall_time_s). E0 is negative (total binding)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "run.sf"
        sf.write_text(build_fac_script(Z, iso))
        t0 = time.perf_counter()
        try:
            _stdout, elapsed, wd = run_sfac(sf, workdir=tmp_path)
        except Exception:
            return None, time.perf_counter() - t0
        lev_file = wd / "out.lev"
        if not lev_file.exists():
            return None, elapsed
        e0 = parse_e0(lev_file)
        levels = parse_levels(lev_file)
        _ = levels  # (unused but validates parse)
        return e0, elapsed


def run_atomx(Z: int, iso: str, breit: bool = False, qed: bool = False):
    cfg, nel = ISOSEQUENCE_CONFIGS[iso]
    charge = Z - nel
    t0 = time.perf_counter()
    # Engine pinned to the B-spline DHF: the FAC-parity CSVs/gates were
    # produced on it (the bare default now resolves to shooting).
    atom = Atom(Z=Z, charge=charge, config=cfg, breit=breit, qed=qed,
                numerics="bspline")
    atom.solve()
    t1 = time.perf_counter()
    return atom.total_energy, t1 - t0, atom.converged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", nargs="+", type=int, default=DEFAULT_Z_PANEL)
    p.add_argument("--iso", nargs="+", default=DEFAULT_ISO_PANEL)
    p.add_argument("--breit", action="store_true")
    p.add_argument("--qed", action="store_true")
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_structure.csv"))
    p.add_argument("--matrix", action="store_true", help="run the full Z x iso FAC-parity matrix")
    p.add_argument("--quick", action="store_true", help="with --matrix, skip Z=74 and Z=82")
    p.add_argument("--no-fac", action="store_true", help="with --matrix, omit sfac columns")
    p.add_argument("--timeout", type=float, default=300.0, help="sfac timeout per matrix cell")
    args = p.parse_args()

    if args.matrix:
        rows = run_matrix(quick=args.quick, include_fac=not args.no_fac, timeout=args.timeout)
        write_matrix_csv(rows, Path(args.out))
        print_matrix_summary(rows)
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("Z,iso,config,nel,E_atomx_Ha,E_atomx_eV,E_fac_eV,"
                "diff_eV,rel_diff,t_atomx_s,t_fac_s\n")
        hdr = (f"{'Z':>3}  {'iso':>8}  {'E_atomx_eV':>15}  {'E_fac_eV':>15}  "
               f"{'ΔE_eV':>11}  {'rel':>9}  {'t_ax':>7}  {'t_fac':>7}")
        print(hdr)
        print("-" * len(hdr))
        for Z in args.z:
            for iso in args.iso:
                if iso not in ISOSEQUENCE_CONFIGS:
                    continue
                cfg, nel = ISOSEQUENCE_CONFIGS[iso]
                if Z < nel:
                    continue  # unphysical (can't have more electrons than Z)
                try:
                    e_a_ha, t_a, conv = run_atomx(Z, iso, breit=args.breit, qed=args.qed)
                except Exception as e:
                    print(f"{Z:>3}  {iso:>8}  atomx FAILED: {e}")
                    f.write(f"{Z},{iso},{cfg},{nel},,,,,,,\n")
                    continue
                e_a_ev = e_a_ha * EV_PER_HARTREE
                try:
                    e_f_ev, t_f = run_fac(Z, iso)
                except Exception as e:
                    print(f"{Z:>3}  {iso:>8}  FAC FAILED: {e}")
                    f.write(f"{Z},{iso},{cfg},{nel},{e_a_ha:.6f},{e_a_ev:.3f},,,,{t_a:.3f},\n")
                    continue
                if e_f_ev is None:
                    print(f"{Z:>3}  {iso:>8}  {e_a_ev:15.3f}  {'—':>15}  {'—':>11}  {'—':>9}  {t_a:7.3f}  {t_f:7.3f}")
                    f.write(f"{Z},{iso},{cfg},{nel},{e_a_ha:.6f},{e_a_ev:.3f},,,,{t_a:.3f},{t_f:.3f}\n")
                    continue
                diff = e_a_ev - e_f_ev
                rel = abs(diff / e_f_ev) if e_f_ev else 0.0
                line = (f"{Z:>3}  {iso:>8}  {e_a_ev:15.3f}  {e_f_ev:15.3f}  "
                        f"{diff:11.3f}  {rel:9.2e}  {t_a:7.3f}  {t_f:7.3f}")
                print(line)
                f.write(
                    f"{Z},{iso},{cfg},{nel},{e_a_ha:.6f},{e_a_ev:.3f},{e_f_ev:.3f},"
                    f"{diff:.3f},{rel:.3e},{t_a:.3f},{t_f:.3f}\n"
                )


if __name__ == "__main__":
    main()
