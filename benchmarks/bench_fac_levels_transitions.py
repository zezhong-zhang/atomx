#!/usr/bin/env python3
"""Benchmark atomx vs FAC — excited levels + E1 transitions.

Uses the canonical Fe XVII test (FAC demo/structure/fe17_structure):
    ground: 1s² 2s² 2p⁶    (Ne-like Fe)
    excited: 1s² 2s² 2p⁵ 3l   (l=s,p,d)

Compares:
    1. energy separations of the 37 L-shell states  (FAC .lev)
       vs the corresponding atomx MCDHF eigenvalues.
    2. E1 A-coefficients for the 2p→3l lines.
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
from fac_runner import parse_levels, parse_transitions, run_sfac  # noqa: E402
from fac_matrix import print_matrix_summary, run_matrix, write_matrix_csv  # noqa: E402


# Canonical FAC demo script
FAC_SCRIPT_FE17 = """SetAtom('Fe')
Closed('1s')
Config('2*8', group='n2')
Config('n3', '2*7 3*1')
ConfigEnergy(0)
OptimizeRadial(['n2'])
ConfigEnergy(1)
Structure('fe17.lev.b', ['n2', 'n3'])
MemENTable('fe17.lev.b')
PrintTable('fe17.lev.b', 'fe17.lev', 1)

TransitionTable('fe17.tr.b', ['n2'], ['n3'])
PrintTable('fe17.tr.b', 'fe17.tr', 1)
"""


def run_fac_fe17(timeout: float = 300.0):
    """Run FAC Fe XVII demo and return (levels, transitions, elapsed_s)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "fe17.sf"
        sf.write_text(FAC_SCRIPT_FE17)
        _out, elapsed, wd = run_sfac(sf, workdir=tmp_path, timeout=timeout)
        lev = parse_levels(wd / "fe17.lev")
        trans = parse_transitions(wd / "fe17.tr")
        return lev, trans, elapsed


def run_atomx_fe17():
    """Fe XVII 2p⁶ → 2p⁵ 3l MCDHF run. Returns (levels, transitions, elapsed_s).

    levels: list of (energy_ha, twoJ, config)
    transitions: list of dicts with initial_idx, final_idx, multipole, A_per_s, Δ eV
    """
    t0 = time.perf_counter()
    # Build multi-config list: ground ⊕ 2p⁵3l (l=s,p,d)
    configs = [
        "1s2 2s2 2p6",
        # 2p5 3s
        "1s2 2s2 2p5 3s1",
        # 2p5 3p
        "1s2 2s2 2p5 3p1",
        # 2p5 3d
        "1s2 2s2 2p5 3d1",
    ]
    atom = Atom(Z=26, charge=16, configs=configs, breit=False, qed=False)
    atom.solve()
    trs = atom.transitions(multipoles=["E1"])
    t1 = time.perf_counter()

    # Pull raw levels from MCDHFResult
    r = atom._result
    levels = []
    if hasattr(r, "csf_energies"):
        for i, e in enumerate(r.csf_energies):
            levels.append({"idx": i, "E_ha": e})
    elif hasattr(r, "energies"):
        for i, e in enumerate(r.energies):
            levels.append({"idx": i, "E_ha": e})
    return levels, trs, t1 - t0, atom


def _ground_energy(levels_atomx):
    """Lowest-energy level in Hartree."""
    if not levels_atomx:
        return 0.0
    return min(lv["E_ha"] for lv in levels_atomx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default=str(ROOT / "benchmarks" / "results_fac_levels"))
    p.add_argument("--matrix", action="store_true", help="run the full Z x iso FAC-parity matrix")
    p.add_argument("--quick", action="store_true", help="with --matrix, skip Z=74 and Z=82")
    p.add_argument(
        "--out",
        default=str(ROOT / "benchmarks" / "results_fac_matrix.csv"),
        help="CSV path for --matrix",
    )
    p.add_argument("--no-fac", action="store_true", help="with --matrix, omit sfac columns")
    p.add_argument("--timeout", type=float, default=300.0, help="sfac timeout per matrix cell")
    args = p.parse_args()

    if args.matrix:
        rows = run_matrix(quick=args.quick, include_fac=not args.no_fac, timeout=args.timeout)
        write_matrix_csv(rows, Path(args.out))
        print_matrix_summary(rows)
        return

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run both
    print("Running FAC Fe XVII …", flush=True)
    try:
        fac_lev, fac_trans, t_fac = run_fac_fe17()
        print(f"  FAC: {len(fac_lev)} levels, {len(fac_trans)} transitions, {t_fac:.3f} s")
    except Exception as e:
        print(f"  FAC failed: {e}")
        fac_lev, fac_trans, t_fac = [], [], 0.0

    print("Running atomx Fe XVII MCDHF …", flush=True)
    try:
        ax_lev, ax_trans, t_ax, atom_obj = run_atomx_fe17()
        print(f"  atomx: {len(ax_lev)} levels, {len(ax_trans)} transitions, {t_ax:.3f} s")
    except Exception as e:
        print(f"  atomx failed: {e}")
        ax_lev, ax_trans, t_ax = [], [], 0.0

    # Convert atomx energies to eV relative to ground (matching FAC convention)
    if ax_lev:
        e_g = _ground_energy(ax_lev)
        for lv in ax_lev:
            lv["rel_eV"] = (lv["E_ha"] - e_g) * HARTREE_EV

    # ===== Level comparison =====
    print("\n--- Level Energies (eV above ground) ---")
    print(f"{'idx':>4}  {'atomx':>12}  {'fac':>12}  {'Δ':>10}  {'rel':>9}  {'fac_2J':>6}  {'fac_cfg':>20}")
    hdr_printed = True

    lev_csv = Path(f"{args.out_prefix}_levels.csv")
    with lev_csv.open("w") as f:
        f.write("idx,E_atomx_eV,E_fac_eV,diff_eV,rel_diff,fac_2J,fac_cfg\n")
        n_compare = min(len(fac_lev), len(ax_lev))
        for i in range(n_compare):
            e_fac = fac_lev[i]["ENERGY_eV"]
            cfg_fac = fac_lev[i]["config"][:20]
            twoj_fac = fac_lev[i]["TWOJ"]
            # Match by rank (after sorting) — imperfect without angular matching
            ax_sorted = sorted(ax_lev, key=lambda d: d["rel_eV"])
            if i >= len(ax_sorted):
                break
            e_ax = ax_sorted[i]["rel_eV"]
            diff = e_ax - e_fac
            rel = abs(diff / e_fac) if e_fac else 0.0
            if hdr_printed and i < 12:
                print(f"{i:>4}  {e_ax:12.3f}  {e_fac:12.3f}  {diff:10.3f}  {rel:9.2e}  {twoj_fac:>6}  {cfg_fac:>20}")
            f.write(f"{i},{e_ax:.4f},{e_fac:.4f},{diff:.4f},{rel:.3e},{twoj_fac},{cfg_fac}\n")
        if n_compare > 12:
            print(f"  ... ({n_compare - 12} more rows in {lev_csv})")

    # ===== Transition comparison =====
    print("\n--- E1 Transitions (sorted by FAC Δ E) ---")
    tr_csv = Path(f"{args.out_prefix}_transitions.csv")
    with tr_csv.open("w") as f:
        f.write("upper,lower,DeltaE_fac_eV,DeltaE_atomx_eV,gA_fac,A_atomx,"
                "ratio_A,mult\n")
        fac_e1 = [t for t in fac_trans if t["mult"] == 0 or t["mult"] is None]
        fac_e1_sorted = sorted(fac_e1, key=lambda d: d["DELTA_E_eV"])
        ax_e1 = [t for t in ax_trans if t.multipole == "E1"]
        ax_e1_sorted = sorted(ax_e1, key=lambda t: t.energy_diff_hartree)
        n = min(len(fac_e1_sorted), len(ax_e1_sorted))
        print(f"{'#':>3}  {'ΔE_fac':>9}  {'ΔE_ax':>9}  {'gA_fac':>12}  {'A_atomx':>12}  {'ratio':>8}")
        for i in range(n):
            f_t = fac_e1_sorted[i]
            a_t = ax_e1_sorted[i]
            dE_fac = f_t["DELTA_E_eV"]
            dE_ax = a_t.energy_diff_hartree * HARTREE_EV
            gA_fac = f_t["gA_per_s"]
            A_ax = a_t.A_rate_per_s
            ratio = A_ax / gA_fac if gA_fac else 0.0
            if i < 10:
                print(f"{i:>3}  {dE_fac:9.2f}  {dE_ax:9.2f}  {gA_fac:12.3e}  {A_ax:12.3e}  {ratio:8.2f}")
            f.write(f"{f_t['upper']},{f_t['lower']},{dE_fac:.3f},{dE_ax:.3f},"
                    f"{gA_fac:.4e},{A_ax:.4e},{ratio:.3f},{f_t['mult']}\n")
        if n > 10:
            print(f"  ... ({n - 10} more rows in {tr_csv})")

    print(f"\nTotals: atomx t = {t_ax:.3f} s, FAC t = {t_fac:.3f} s, atomx/fac = {t_ax/t_fac:.2f}x slower" if t_fac else "")


if __name__ == "__main__":
    main()
