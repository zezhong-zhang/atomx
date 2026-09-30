#!/usr/bin/env python3
"""Benchmark atomx vs FAC — Li-like 2s-2p splitting across Z.

Li-like ions (3 electrons) are a clean test of fine-structure: the only
open shell is a single n=2 electron, and the 2s₁/₂ ↔ 2p₁/₂, 2p₃/₂
E1 resonance lines dominate the spectrum.

For each Z we compare:
    - ionization/total binding energy of 1s²2s and 1s²2p states
    - 2s-2p₁/₂ and 2s-2p₃/₂ excitation energies
    - A-rates of the two E1 transitions
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

from fac_matrix import print_matrix_summary, run_matrix, write_matrix_csv  # noqa: E402
from fac_runner import parse_e0, parse_levels, parse_transitions, run_sfac  # noqa: E402

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402

DEFAULT_Z = [10, 18, 26, 36, 54, 74, 92]

SYMBOLS = {
    10: "Ne", 18: "Ar", 26: "Fe", 36: "Kr", 54: "Xe", 74: "W", 82: "Pb", 92: "U",
}

FAC_SCRIPT = """SetAtom('{symbol}', {Z})
Closed('1s')
Config('gnd', '2s1')
Config('exc', '2p1')
ConfigEnergy(0)
OptimizeRadial(['gnd'])
ConfigEnergy(1)
Structure('lilike.lev.b', ['gnd', 'exc'])
MemENTable('lilike.lev.b')
PrintTable('lilike.lev.b', 'lilike.lev', 1)
TransitionTable('lilike.tr.b', ['gnd'], ['exc'], -1)
PrintTable('lilike.tr.b', 'lilike.tr', 1)
"""


def run_fac(Z: int):
    """Run FAC on Li-like Z. Returns:
        (E_gnd_abs_eV, de_2p12_eV, de_2p32_eV, A_2p12, A_2p32, elapsed_s, err)
    All Δ E in eV above ground; ``A`` in s⁻¹.
    """
    symbol = SYMBOLS[Z]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sf = tmp_path / "run.sf"
        sf.write_text(FAC_SCRIPT.format(symbol=symbol, Z=Z))
        t0 = time.perf_counter()
        try:
            _out, elapsed, wd = run_sfac(sf, workdir=tmp_path)
        except Exception as e:
            return None, None, None, None, None, time.perf_counter() - t0, str(e)
        lev = parse_levels(wd / "lilike.lev")
        trans = parse_transitions(wd / "lilike.tr")
        e0 = parse_e0(wd / "lilike.lev")
        e_gnd_abs = e0
        # ``ENERGY`` in .lev is eV above ground.
        de_2p12 = None
        de_2p32 = None
        for L in lev:
            if "2p" in L["config"] and L["TWOJ"] == 1:
                de_2p12 = L["ENERGY_eV"]
            if "2p" in L["config"] and L["TWOJ"] == 3:
                de_2p32 = L["ENERGY_eV"]
        # FAC prints the unweighted spontaneous A coefficient.
        A_2p12 = A_2p32 = None
        for T in trans:
            up_lev = next((L for L in lev if L["ILEV"] == T["upper"]), None)
            if up_lev is None:
                continue
            if up_lev["TWOJ"] == 1 and "2p" in up_lev["config"]:
                A_2p12 = T["A_per_s"]
            elif up_lev["TWOJ"] == 3 and "2p" in up_lev["config"]:
                A_2p32 = T["A_per_s"]
        return e_gnd_abs, de_2p12, de_2p32, A_2p12, A_2p32, elapsed, None


def run_atomx(Z: int):
    t0 = time.perf_counter()
    atom = Atom(Z=Z, charge=Z - 3, config=["1s2 2s1", "1s2 2p1"], corrections="none")
    atom.solve()
    trs = atom.transitions(multipoles=["E1"])
    t1 = time.perf_counter()
    e_gnd_eV = atom.total_energy * HARTREE_EV
    r = atom._result
    de_ev = None
    if r is not None and hasattr(r, "state_energies"):
        se = r.state_energies
        if se is not None and len(se) >= 2:
            de_ev = (se - se[0]) * HARTREE_EV
    A_list = [t.A_rate_per_s for t in trs if t.multipole == "E1"]
    return e_gnd_eV, de_ev, A_list, t1 - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", nargs="+", type=int, default=DEFAULT_Z)
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "results_fac_lilike.csv"))
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("Z,E_gnd_atomx_eV,E_gnd_fac_eV,diff_gnd_eV,"
                "DeltaE_ax_eV,DeltaE_2p12_fac_eV,DeltaE_2p32_fac_eV,"
                "A_atomx,A_2p12_fac,A_2p32_fac,t_atomx_s,t_fac_s\n")
        hdr = (f"{'Z':>3}  {'E_gnd_ax':>14}  {'E_gnd_fac':>14}  {'Δgnd':>8}  "
               f"{'ΔE_ax':>9}  {'ΔE_2p12':>9}  {'ΔE_2p32':>9}  "
               f"{'A_ax':>10}  {'A_2p12':>10}  {'A_2p32':>10}  {'t_ax':>6}  {'t_fac':>6}")
        print(hdr)
        print("-" * len(hdr))
        for Z in args.z:
            try:
                # e_gnd_abs is the FAC E0 (absolute binding, eV);
                # de_2p12, de_2p32 are already ΔE above ground in eV.
                e_gnd_fac, de_2p12, de_2p32, A12, A32, t_f, err = run_fac(Z)
            except Exception as e:
                print(f"  Z={Z} FAC failed: {e}")
                continue
            if err:
                print(f"  Z={Z} FAC error: {err}")
                continue
            if e_gnd_fac is None:
                print(f"  Z={Z} FAC: no E0 parsed")
                continue
            try:
                e_gnd_ax, de_ax, A_ax, t_a = run_atomx(Z)
            except Exception as e:
                print(f"  Z={Z} atomx failed: {e}")
                continue
            diff_gnd = e_gnd_ax - e_gnd_fac
            de_ax_1 = de_ax[1] if (de_ax is not None and len(de_ax) >= 2) else 0.0
            de_2p12 = de_2p12 if de_2p12 is not None else 0.0
            de_2p32 = de_2p32 if de_2p32 is not None else 0.0
            A_a = A_ax[0] if A_ax else 0.0
            line = (f"{Z:>3}  {e_gnd_ax:14.3f}  {e_gnd_fac:14.3f}  {diff_gnd:8.2f}  "
                    f"{de_ax_1:9.2f}  {de_2p12:9.2f}  {de_2p32:9.2f}  "
                    f"{A_a:10.3e}  {A12 or 0:10.3e}  {A32 or 0:10.3e}  "
                    f"{t_a:6.2f}  {t_f:6.2f}")
            print(line)
            f.write(f"{Z},{e_gnd_ax:.4f},{e_gnd_fac:.4f},{diff_gnd:.4f},"
                    f"{de_ax_1:.4f},{de_2p12:.4f},{de_2p32:.4f},"
                    f"{A_a:.4e},{A12 or 0:.4e},{A32 or 0:.4e},{t_a:.3f},{t_f:.3f}\n")


if __name__ == "__main__":
    main()
