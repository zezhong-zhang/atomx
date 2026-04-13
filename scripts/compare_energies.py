#!/usr/bin/env python3
"""Compare AtomX DHF total energies with radial DHFS and published MCDHF references.

Reference: Table S12 from energy_level.csv (high-accuracy MCDHF, "This Work" column).
"""

import numpy as np
import subprocess
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from atomx import Atom
from atomx.constants import HARTREE_EV

# Reference MCDHF values from energy_level.csv ("This Work" column)
REF_MCDHF = {
    2: -2.86181332, 3: -7.43353221, 4: -14.57589157, 5: -24.53655419,
    6: -37.67604065, 7: -54.32772182, 8: -74.82498547, 9: -99.50161293,
    10: -128.69192487, 11: -162.07808418, 12: -199.93506018,
    13: -242.33074750, 14: -289.46132790, 15: -341.49465335,
    16: -398.59789686, 17: -460.93833948, 18: -528.68376272,
    19: -601.525926, 20: -679.710115, 21: -763.378718,
    22: -852.819817, 23: -948.188716, 24: -1049.596071,
    25: -1157.321956, 26: -1271.392005, 27: -1392.001871,
    28: -1519.305544, 29: -1653.455051, 30: -1794.612741,
    31: -1942.563669, 32: -2097.470274, 33: -2259.441762,
    34: -2428.588375, 35: -2605.023243, 36: -2788.860943,
}

# Radial DHFS values (from running Salvat's DHFS code)
RADIAL_DHFS = {
    2: -2.861813, 3: -7.425690, 4: -14.566380, 5: -24.520540,
    6: -37.633770, 7: -54.284200, 8: -74.778220, 9: -99.447240,
    10: -128.623300, 11: -161.999800, 12: -199.853100,
    13: -242.242400, 14: -289.354200, 15: -341.385700,
    16: -398.485600, 17: -460.819600, 18: -528.554100,
}

SYMS = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
        'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']


def run_atomx(Z_list):
    """Run AtomX DHF for each Z (physics grid, tight tolerance)."""
    results = {}
    for Z in Z_list:
        try:
            a = Atom(Z=Z)  # physics grid (ns=113, ks=9)
            a.solve(tol=1e-7)
            results[Z] = a.total_energy
        except Exception as e:
            print(f"  Z={Z}: FAILED — {e}")
            results[Z] = None
    return results


def main():
    Z_list = list(range(2, 19))

    print("Running AtomX DHF for Z=2..18...")
    atomx_results = run_atomx(Z_list)

    # Print comparison table
    print()
    print("=" * 110)
    print(f"{'Z':>3}  {'Elem':4}  {'Ref MCDHF':>14}  {'AtomX DHF':>14}  {'Radial DHFS':>14}  "
          f"{'AtomX err%':>10}  {'Radial err%':>11}  {'AtomX-Rad':>10}")
    print("=" * 110)

    atomx_errs, radial_errs = [], []

    for Z in Z_list:
        sym = SYMS[Z] if Z < len(SYMS) else f"Z{Z}"
        E_ref = REF_MCDHF.get(Z)
        E_atomx = atomx_results.get(Z)
        E_radial = RADIAL_DHFS.get(Z)

        ref_str = f"{E_ref:14.6f}" if E_ref else "  N/A"
        atomx_str = f"{E_atomx:14.6f}" if E_atomx else "  N/A"
        radial_str = f"{E_radial:14.6f}" if E_radial else "  N/A"

        err_atomx = abs((E_atomx - E_ref) / E_ref) * 100 if E_atomx and E_ref else None
        err_radial = abs((E_radial - E_ref) / E_ref) * 100 if E_radial and E_ref else None
        diff_ar = (E_atomx - E_radial) if E_atomx and E_radial else None

        ea_str = f"{err_atomx:10.2f}%" if err_atomx is not None else "      N/A"
        er_str = f"{err_radial:11.4f}%" if err_radial is not None else "       N/A"
        d_str = f"{diff_ar:+10.4f}" if diff_ar is not None else "      N/A"

        if err_atomx is not None:
            atomx_errs.append(err_atomx)
        if err_radial is not None:
            radial_errs.append(err_radial)

        print(f"{Z:3d}  {sym:4s}  {ref_str}  {atomx_str}  {radial_str}  {ea_str}  {er_str}  {d_str}")

    print("=" * 110)
    if atomx_errs:
        print(f"     AtomX mean error: {np.mean(atomx_errs):.2f}%, max: {np.max(atomx_errs):.2f}%")
    if radial_errs:
        print(f"     Radial mean error: {np.mean(radial_errs):.4f}%, max: {np.max(radial_errs):.4f}%")

    print()
    print("Notes:")
    print("  - Ref MCDHF: Table S12 'This Work' (Visscher-quality, includes correlation)")
    print("  - AtomX DHF: Single-config Dirac-Hartree-Fock (no correlation)")
    print("  - Radial DHFS: Salvat's DHFS code (Dirac-Hartree-Fock-Slater, local exchange)")
    print("  - AtomX uses non-local HF exchange; radial uses local Slater exchange")
    print("  - Both miss correlation energy → systematically higher than MCDHF")


if __name__ == "__main__":
    main()
