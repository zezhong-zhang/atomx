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
import csv
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_matrix import print_matrix_summary, run_matrix, write_matrix_csv  # noqa: E402
from fac_runner import parse_levels, parse_transitions, run_sfac  # noqa: E402

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402

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
    # FAC's `Config('n3', '2*7 3*1')` expands to all 7 NR configurations
    # consistent with "n=2 has 7 electrons (one hole), n=3 has 1
    # electron" — that's the 2p-hole branch AND the 2s-hole branch.
    # Listing all 7 explicitly is what closes the 27 → 37 J-resolved
    # level gap vs the FAC reference (HYP-165, REPORT §4.2).
    configs = [
        # ground (closed-shell)
        "1s2 2s2 2p6",
        # 2p hole + 3l
        "1s2 2s2 2p5 3s1",
        "1s2 2s2 2p5 3p1",
        "1s2 2s2 2p5 3d1",
        # 2s hole + 3l
        "1s2 2s1 2p6 3s1",
        "1s2 2s1 2p6 3p1",
        "1s2 2s1 2p6 3d1",
    ]
    atom = Atom(
        Z=26,
        charge=16,
        configs=configs,
        breit=False,
        qed=False,
        j_resolved_csfs=True,
    )
    atom.solve()
    trs = atom.transitions(multipoles=["E1"])
    t1 = time.perf_counter()

    # Pull raw J-resolved levels from MCDHFResult.state_energies. With
    # the M3 wiring the multi-config Atom path block-diagonalises CI
    # per (J, parity), so each eigenvalue corresponds to one level in
    # the FAC sense (one row in `fe17.lev`).
    r = atom._result
    levels = []
    twoJ = getattr(r, "state_J2", None)
    par = getattr(r, "state_parity", None)
    for i, e in enumerate(r.state_energies):
        row = {"idx": i, "E_ha": float(e)}
        if twoJ is not None and i < len(twoJ):
            row["twoJ"] = int(twoJ[i])
        if par is not None and i < len(par):
            row["parity"] = int(par[i])
        levels.append(row)
    return levels, trs, t1 - t0, atom


def _ground_energy(levels_atomx):
    """Lowest-energy level in Hartree."""
    if not levels_atomx:
        return 0.0
    return min(lv["E_ha"] for lv in levels_atomx)


def _fac_parity(row):
    return 1 if int(row.get("P", 0)) == 0 else -1


def _match_levels_by_jpi(fac_levels, atomx_levels):
    """Greedy same-J/parity level matcher minimizing energy difference."""
    unmatched = set(range(len(atomx_levels)))
    matches = []
    for i, fac in enumerate(fac_levels):
        candidates = []
        for j in unmatched:
            ax = atomx_levels[j]
            if ax.get("twoJ") != int(fac.get("TWOJ", -999)):
                continue
            if ax.get("parity") != _fac_parity(fac):
                continue
            candidates.append((abs(ax["rel_eV"] - fac["ENERGY_eV"]), j))
        if not candidates:
            matches.append((i, None))
            continue
        _, best = min(candidates)
        unmatched.remove(best)
        matches.append((i, best))
    return matches


def _write_metric_csv(path: Path, metrics: dict[str, float | int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


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
        matches = _match_levels_by_jpi(fac_lev, ax_lev)
        n_compare = sum(1 for _, j in matches if j is not None)
        for row_i, (i, j) in enumerate(matches):
            if j is None:
                continue
            e_fac = fac_lev[i]["ENERGY_eV"]
            cfg_fac = fac_lev[i]["config"][:20]
            twoj_fac = fac_lev[i]["TWOJ"]
            ax = ax_lev[j]
            e_ax = ax["rel_eV"]
            diff = e_ax - e_fac
            rel = abs(diff / e_fac) if e_fac else 0.0
            if hdr_printed and row_i < 12:
                print(f"{i:>4}  {e_ax:12.3f}  {e_fac:12.3f}  {diff:10.3f}  {rel:9.2e}  {twoj_fac:>6}  {cfg_fac:>20}")
            f.write(f"{i},{e_ax:.4f},{e_fac:.4f},{diff:.4f},{rel:.3e},{twoj_fac},{cfg_fac}\n")
        if n_compare > 12:
            print(f"  ... ({n_compare - 12} more rows in {lev_csv})")

    # ===== Transition comparison =====
    print("\n--- E1 Transitions (sorted by FAC Δ E) ---")
    tr_csv = Path(f"{args.out_prefix}_transitions.csv")
    transition_metrics = {
        "strong_count": 0,
        "strong_mapped": 0,
        "strong_missing": 0,
        "strong_max_factor": 0.0,
        "strong_median_factor": 0.0,
    }
    with tr_csv.open("w") as f:
        f.write("upper,lower,DeltaE_fac_eV,DeltaE_atomx_eV,gA_fac,A_fac,A_atomx,"
                "ratio_A,mult\n")
        fac_e1 = [t for t in fac_trans if t["mult"] == 0 or t["mult"] is None]
        fac_e1_sorted = sorted(fac_e1, key=lambda d: d["DELTA_E_eV"])
        fac_levels_by_ilev = {int(lv["ILEV"]): lv for lv in fac_lev}
        fac_to_atomx = {
            int(fac_lev[i]["ILEV"]): j
            for i, j in _match_levels_by_jpi(fac_lev, ax_lev)
            if j is not None
        }
        ax_e1 = [
            t for t in ax_trans
            if t.multipole == "E1" and t.initial_label == "State_0"
        ]
        ax_by_final = {}
        for t in ax_e1:
            try:
                final_idx = int(str(t.final_label).split("_")[-1])
            except ValueError:
                continue
            ax_by_final[final_idx] = t
        rows = []
        for f_t in fac_e1_sorted:
            if int(f_t["lower"]) != 0:
                continue
            ax_idx = fac_to_atomx.get(int(f_t["upper"]))
            if ax_idx is None or ax_idx not in ax_by_final:
                continue
            rows.append((f_t, ax_by_final[ax_idx]))
        all_ground_fac = [t for t in fac_e1_sorted if int(t["lower"]) == 0]
        factors = []
        print(f"{'#':>3}  {'ΔE_fac':>9}  {'ΔE_ax':>9}  {'A_fac':>12}  {'A_atomx':>12}  {'ratio':>8}")
        for i, (f_t, a_t) in enumerate(rows):
            dE_fac = f_t["DELTA_E_eV"]
            dE_ax = a_t.energy_diff_hartree * HARTREE_EV
            gA_fac = f_t["gA_per_s"]
            upper_level = fac_levels_by_ilev.get(int(f_t["upper"]))
            g_upper = int(upper_level["TWOJ"]) + 1 if upper_level is not None else 1
            A_fac = gA_fac / g_upper if g_upper > 0 else gA_fac
            A_ax = a_t.A_rate_per_s
            ratio = A_ax / A_fac if A_fac else 0.0
            if ratio > 0.0:
                factors.append(max(ratio, 1.0 / ratio))
            if i < 10:
                print(f"{i:>3}  {dE_fac:9.2f}  {dE_ax:9.2f}  {A_fac:12.3e}  {A_ax:12.3e}  {ratio:8.2f}")
            f.write(f"{f_t['upper']},{f_t['lower']},{dE_fac:.3f},{dE_ax:.3f},"
                    f"{gA_fac:.4e},{A_fac:.4e},{A_ax:.4e},{ratio:.3f},{f_t['mult']}\n")
        if len(rows) > 10:
            print(f"  ... ({len(rows) - 10} more rows in {tr_csv})")
        factors.sort()
        if factors:
            mid = len(factors) // 2
            median = factors[mid] if len(factors) % 2 else 0.5 * (factors[mid - 1] + factors[mid])
            transition_metrics["strong_max_factor"] = f"{max(factors):.6g}"
            transition_metrics["strong_median_factor"] = f"{median:.6g}"
        transition_metrics["strong_count"] = len(all_ground_fac)
        transition_metrics["strong_mapped"] = len(rows)
        transition_metrics["strong_missing"] = max(0, len(all_ground_fac) - len(rows))

    _write_metric_csv(
        Path(f"{args.out_prefix}_transition_summary.csv"),
        transition_metrics,
    )

    print(f"\nTotals: atomx t = {t_ax:.3f} s, FAC t = {t_fac:.3f} s, atomx/fac = {t_ax/t_fac:.2f}x slower" if t_fac else "")


if __name__ == "__main__":
    main()
