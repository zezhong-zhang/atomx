#!/usr/bin/env python3
"""Benchmark Cowan (rcnucd) vs atomx for neutral atoms.

Cowan: non-relativistic Hartree-Fock-Slater (rcnucd)
atomx: relativistic Dirac-Fock

Compares:
- Total energy (configuration average)
- Wall time
- Convergence behavior
"""

import argparse
import contextlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom
from atomx.constants import HARTREE_EV

COWAN_DIR = Path("/Users/zhangzz/code/ASAP/Cowan/Code")
LANL_DIR = Path("/Users/zhangzz/code/ASAP/Cowan/Code/Lanl")

# Conversion: Cowan outputs in Rydbergs (confirmed from documentation)
RYD_PER_HARTREE = 2.0
EV_PER_RYDBERG = 13.605698  # eV per Rydberg
# Neutral atom configs
NEUTRAL_CONFIGS = {
    10: ("Ne", "1s2 2s2 2p6", 1),
    18: ("Ar", "1s2 2s2 2p6 3s2 3p6", 1),
    26: ("Fe", "1s2 2s2 2p6 3s2 3p6 3d6 4s2", 9),
    36: ("Kr", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6", 1),
    54: ("Xe", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6", 1),
    74: ("W", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d4 6s2", 5),
    82: ("Pb", "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 5p6 5d10 6s2 6p2", 1),
}


def make_in36(Z, symbol, config, twojplus1):
    """Create in36 file for Cowan rcnucd."""
    # Control line - ihf=0 for normal HFS run (not HF-only)
    control = "2  -9    0    0  5.0    5.e-08    1.e-11-2   090   1.0 0.65  0.0 0.00   -6"
    
    # Config line - match working format from Lanl/in36
    label = symbol.lower().ljust(4)
    config_padded = config.ljust(15)
    config_line = f"   {Z:2d}    {twojplus1}{label}  {config_padded}       {config_padded}"
    
    lines = [
        control,
        config_line[:80],
        "   -1",
        "",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890"
    ]
    return "\n".join(lines)


def run_cowan(Z, symbol, config, twojplus1):
    """Run Cowan rcnucd. Returns (E_eV, time_s, success)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(LANL_DIR / "in36k", tmp_path / "in36k")
        
        in36 = tmp_path / "in36"
        in36.write_text(make_in36(Z, symbol, config, twojplus1))
        
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [str(COWAN_DIR / "rcnucd")],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmp_path,
            )
            t1 = time.perf_counter()
            
            # Parse Eav from output
            # Format: "Eav & pars  N  -XXXX.XXXX 0  YYYY.YY 1 ..."
            eav_ryd = None
            for line in result.stdout.split("\n"):
                if "Eav & pars" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        with contextlib.suppress(ValueError):
                            eav_ryd = float(parts[4])  # The value after the count
                    if eav_ryd is None and len(parts) >= 4:
                        with contextlib.suppress(ValueError):
                            eav_ryd = float(parts[3])
            if eav_ryd is not None:
                # Convert Rydberg to eV
                eav_ev = eav_ryd * EV_PER_RYDBERG
                return eav_ev, t1 - t0, result.returncode == 0
            else:
                return None, t1 - t0, False
                
        except Exception:
            return None, time.perf_counter() - t0, False


def run_atomx(Z, config):
    """Run atomx. Returns (E_eV, time_s, converged)."""
    charge = 0
    t0 = time.perf_counter()
    try:
        atom = Atom(Z=Z, charge=charge, config=config)
        atom.solve()
        t1 = time.perf_counter()
        # atomx total_energy is in Hartree
        e_ev = atom.total_energy * HARTREE_EV
        return e_ev, t1 - t0, atom.converged
    except Exception:
        return None, time.perf_counter() - t0, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--z", nargs="+", type=int, default=[10, 18, 26, 36])
    p.add_argument("--out", default="/tmp/results_cowan_vs_atomx.csv")
    args = p.parse_args()

    out = Path(args.out)
    with out.open("w") as f:
        f.write("Z,symbol,config,E_cowan_eV,E_atomx_eV,diff_eV,rel_diff,t_cowan_s,t_atomx_s,cowan_ok,atomx_ok\n")
        
        hdr = (f"{'Z':>3}  {'sym':>3}  {'E_cowan_eV':>15}  {'E_atomx_eV':>15}  "
               f"{'ΔE_eV':>11}  {'rel':>9}  {'t_cw':>7}  {'t_ax':>7}  {'ok':>4}")
        print(hdr)
        print("-" * len(hdr))
        
        for Z in args.z:
            if Z not in NEUTRAL_CONFIGS:
                print(f"{Z:>3}  {'?':>3}  {'SKIP':>15}  {'SKIP':>15}  "
                      f"{'—':>11}  {'—':>9}  {'—':>7}  {'—':>7}  {'NN':>4}")
                continue
            symbol, config, twojplus1 = NEUTRAL_CONFIGS[Z]
            
            e_c_ev, t_c, ok_c = run_cowan(Z, symbol, config, twojplus1)
            e_a_ev, t_a, ok_a = run_atomx(Z, config)
            
            if e_c_ev is not None and e_a_ev is not None:
                diff = e_c_ev - e_a_ev
                rel = abs(diff / e_a_ev) if e_a_ev != 0 else 0.0
                line = (f"{Z:>3}  {symbol:>3}  {e_c_ev:15.3f}  {e_a_ev:15.3f}  "
                        f"{diff:11.3f}  {rel:9.2e}  {t_c:7.3f}  {t_a:7.3f}  {'YY' if ok_c and ok_a else 'NN':>4}")
            else:
                c_str = f"{e_c_ev:15.3f}" if e_c_ev is not None else "FAIL"
                a_str = f"{e_a_ev:15.3f}" if e_a_ev is not None else "FAIL"
                line = (f"{Z:>3}  {symbol:>3}  {c_str:>15}  {a_str:>15}  "
                        f"{'—':>11}  {'—':>9}  {t_c:7.3f}  {t_a:7.3f}  {'NN':>4}")
            
            print(line)
            f.write(f"{Z},{symbol},{config},{e_c_ev or ''},{e_a_ev or ''},"
                    f"{diff if e_c_ev is not None and e_a_ev is not None else ''},"
                    f"{rel if e_c_ev is not None and e_a_ev is not None else ''},"
                    f"{t_c},{t_a},{ok_c},{ok_a}\n")


if __name__ == "__main__":
    main()
