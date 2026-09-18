#!/usr/bin/env python3
"""Benchmark atomx shooting DHFS vs the Fortran radial/DHFS program.

For each Z in a chosen panel:
  1. parse the Z{XXX}.in DHFS input file (atomic weight, shell list,
     alpha_x, RN, n_grid) — the single source of truth for the numerical
     setup.
  2. run the Fortran ``radial/DHFS`` program on that input, parse
     ``dhfs.dat`` for its higher-precision total-binding energy.
  3. run atomx's ``ShootingSolver`` **with the same numerical setup**
     (aw, alpha_x, r_max, n_grid, subshells) — pure apples-to-apples.
  4. compare total-binding energy and wall time (cold + warm).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.methods.dirac_fock import Subshell  # noqa: E402
from atomx.numerics.shooting.solver import ShootingSolver  # noqa: E402

# Reference code location comes from the environment (no machine-specific
# default): RADIAL_DHFS_DIR must contain the DHFS executable and the Z{NNN}.in
# inputs.
RADIAL_DIR = Path(os.environ.get("RADIAL_DHFS_DIR", ""))
DHFS_BIN = RADIAL_DIR / "DHFS"

DEFAULT_PANEL = [2, 6, 10, 18, 26, 36, 54, 74, 82, 92]


@dataclass
class DhfsInput:
    Z: int
    aw: float
    alpha_x: float
    r_max: float
    n_grid: int
    shells: list  # list[(n, l, twoj, occ)]
    raw: str


def parse_dhfs_input(path: Path) -> DhfsInput:
    lines = path.read_text().splitlines()
    shells = []
    aw = 0.0
    alpha_x = 1.5
    r_max = 75.0
    n_grid = 1000
    Z = 0
    for line in lines:
        m_shell = re.match(r"C3\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)", line)
        if m_shell:
            n, l, twoj, occ = int(m_shell[1]), int(m_shell[2]), int(m_shell[3]), float(m_shell[4])
            shells.append((n, l, twoj, occ))
            continue
        m_head = re.match(r"C2\s*\(\s*\S+\s*,\s*(\d+)", line)
        if m_head:
            Z = int(m_head[1])
            continue
        m_aw = re.match(r"C4\s*\(\s*(-?[0-9.]+)", line)
        if m_aw:
            aw = float(m_aw[1])
            continue
        m_alpha = re.match(r"C5\s*\(\s*([0-9.]+)", line)
        if m_alpha:
            alpha_x = float(m_alpha[1])
            continue
        m_rn = re.match(r"C6\s*\(\s*(-?[0-9.]+)", line)
        if m_rn:
            r_max = float(m_rn[1])
            continue
        m_ng = re.match(r"C7\s*\(\s*(\d+)", line)
        if m_ng:
            n_grid = int(m_ng[1])
            continue
    return DhfsInput(Z=Z, aw=aw, alpha_x=alpha_x, r_max=r_max, n_grid=n_grid,
                     shells=shells, raw=path.read_text())


def run_dhfs_binary(inp: DhfsInput) -> tuple[Optional[float], float]:
    """Run the compiled DHFS binary, parse dhfs.dat for high-precision E_tot."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(DHFS_BIN, tmp)
        (Path(tmp) / "input.in").write_text(inp.raw)
        t0 = time.perf_counter()
        try:
            subprocess.run(
                ["./DHFS"],
                cwd=tmp,
                input=inp.raw,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            return None, 600.0
        t1 = time.perf_counter()
        dhfs_dat = Path(tmp) / "dhfs.dat"
        if not dhfs_dat.exists():
            return None, t1 - t0
        for line in dhfs_dat.read_text().splitlines():
            if "Total binding energy" in line:
                # "Total binding energy = -1.2711952E+03 Eh = -3.4590984E+04 eV"
                m = re.search(r"=\s*(-?[0-9.E+-]+)\s*Eh", line)
                if m:
                    return float(m[1]), t1 - t0
        return None, t1 - t0


def run_atomx_shooting(inp: DhfsInput, n_reps: int = 1):
    """Run atomx ShootingSolver with the SAME numerical setup as DHFS.

    First call is timed (cold, JIT compile).  Extra reps with fresh solver
    instances measure warm (steady-state) time.

    Returns: (E_total, t_cold, converged, t_warm)
    """
    # Map (n, l, 2j) → kappa
    def kappa(l: int, twoj: int) -> int:
        # For j = l+1/2: kappa = -(l+1); for j = l-1/2: kappa = +l
        if twoj == 2 * l + 1:
            return -(l + 1)
        return l

    subshells = []
    for n, l, twoj, occ in inp.shells:
        subshells.append(Subshell(n=n, kappa=kappa(l, twoj), occupation=occ))

    t_cold = None
    t_warm = None
    r = None
    for rep in range(1 + n_reps):
        solver = ShootingSolver(
            Z=inp.Z, subshells=subshells,
            n_grid=inp.n_grid, r_max=inp.r_max,
            alpha_x=inp.alpha_x, aw=inp.aw, equation="dirac",
        )
        t0 = time.perf_counter()
        r = solver.solve(max_iter=300, tol=1e-9)
        t1 = time.perf_counter()
        if rep == 0:
            t_cold = t1 - t0
        else:
            t_warm = t1 - t0 if t_warm is None else min(t_warm, t1 - t0)
    assert r is not None and t_cold is not None
    return r.total_energy, t_cold, r.converged, (t_warm if t_warm is not None else t_cold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", nargs="+", type=int, default=DEFAULT_PANEL)
    p.add_argument("--out", default=str(ROOT / "benchmarks" / "results_shooting.csv"))
    p.add_argument("--warm-reps", type=int, default=2,
                   help="Number of warm reps after the cold JIT run.")
    p.add_argument("--nuclear", choices=["point", "fermi", "both"], default="both",
                   help="Nuclear model: point (aw=0), fermi (default from .in), or both.")
    args = p.parse_args()

    if not DHFS_BIN.is_file():
        p.error(
            "reference DHFS binary not found. Set RADIAL_DHFS_DIR to the "
            "directory containing the DHFS executable and the Z{NNN}.in "
            f"inputs (looked in {RADIAL_DIR or '<unset>'}).")

    models = ["point", "fermi"] if args.nuclear == "both" else [args.nuclear]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("Z,nuclear,config,E_atomx_Eh,E_radial_Eh,abs_diff_Eh,rel_diff,"
                "t_atomx_cold_s,t_atomx_warm_s,t_radial_s,speedup_cold,speedup_warm,converged\n")
        hdr = (f"{'Z':>3} {'nuc':>5}  {'E_atomx':>18}  {'E_radial':>18}  "
               f"{'abs_diff':>11}  {'rel':>10}  "
               f"{'t_cold':>7}  {'t_warm':>7}  {'t_rad':>7}  {'spdC':>6}  {'spdW':>6}")
        print(hdr)
        print("-" * len(hdr))

        for Z in args.panel:
            src = RADIAL_DIR / f"Z{Z:03d}.in"
            if not src.exists():
                print(f"  skip Z={Z}: no input file {src}")
                continue
            for model in models:
                inp = parse_dhfs_input(src)
                if model == "point":
                    inp.aw = 0.0
                    inp.raw = re.sub(
                        r"^C4\s*\([^)]*\)",
                        "C4 (  0.000000)",
                        inp.raw,
                        flags=re.MULTILINE,
                    )
                e_r, t_r = run_dhfs_binary(inp)
                e_a, t_cold, conv, t_warm = run_atomx_shooting(inp, n_reps=args.warm_reps)
                if e_r is None:
                    line = (f"{Z:>3} {model:>5}  {e_a:18.9f}  {'—':>18}  "
                            f"{'—':>11}  {'—':>10}  "
                            f"{t_cold:7.3f}  {t_warm:7.3f}  {'—':>7}  {'—':>6}  {'—':>6}")
                    print(line)
                    f.write(f"{Z},{model},\"{inp.shells}\",{e_a:.9f},,,,"
                            f"{t_cold:.4f},{t_warm:.4f},,,,{conv}\n")
                    continue
                diff = e_a - e_r
                rel = abs(diff / e_r)
                spd_c = t_r / t_cold if t_cold > 0 else float('inf')
                spd_w = t_r / t_warm if t_warm > 0 else float('inf')
                line = (f"{Z:>3} {model:>5}  {e_a:18.9f}  {e_r:18.9f}  "
                        f"{diff:11.2e}  {rel:10.2e}  "
                        f"{t_cold:7.3f}  {t_warm:7.3f}  {t_r:7.3f}  {spd_c:5.1f}x  {spd_w:5.1f}x")
                print(line)
                f.write(
                    f"{Z},{model},\"{inp.shells}\",{e_a:.9f},{e_r:.9f},{diff:.3e},{rel:.3e},"
                    f"{t_cold:.4f},{t_warm:.4f},{t_r:.4f},{spd_c:.3f},{spd_w:.3f},{conv}\n"
                )


if __name__ == "__main__":
    main()
