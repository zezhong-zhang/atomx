#!/usr/bin/env python3
"""Benchmark local-exchange flavours in atomx against sfac.

Runs each of ``exchange={'hf', 'slater', 'fac', 'shooting'}`` on the
calibration panel (Task 0) and reports total binding energies +
wall times side-by-side. ``sfac`` is invoked with the exact same
flags as the calibration (``SetSE(0); SetVP(0); SetMS(0,0); SetBreit(0)``)
so the comparison isolates the local-exchange functional.

Outputs:
  * stdout: human-readable table.
  * ``benchmarks/results_local_exchange.csv``: machine-readable.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402

# Configure the reference code from the environment (no machine-specific
# defaults): FAC_SFAC_PATH points at the sfac executable, FAC_DYLD_LIB at the
# gcc runtime dir it needs on macOS.
SFAC = os.environ.get("FAC_SFAC_PATH", "")
DYLD = os.environ.get("FAC_DYLD_LIB", "")

PANEL = [
    (2,  "He", "1s2"),
    (6,  "C",  "1s2 2s2 2p2"),
    (10, "Ne", "1s2 2s2 2p6"),
    (18, "Ar", "1s2 2s2 2p6 3s2 3p6"),
    (26, "Fe", "1s2 2s2 2p6"),       # Ne-like
    (36, "Kr", "1s2 2s2 2p6 3s2 3p6"),  # Ar-like
    (54, "Xe", "1s2 2s2 2p6"),       # Ne-like
    (74, "W",  "1s2 2s2 2p6"),       # Ne-like — re-enabled after the
    (82, "Pb", "1s2 2s2 2p6 3s2 3p6"),   # _slater_integral nsteps cap
    (92, "U",  "1s2 2s2 2p6"),       # fixed the heavy-Z stall.
]

FLAVOURS = ["shooting", "fac"]   # "slater", "hf" omitted: this
                                 # benchmark focuses on the local-
                                 # exchange flavours routed through
                                 # shooting.  The B-spline Slater path
                                 # (``DFSolver(exchange='slater')``) is
                                 # now correct to 10⁻³ vs shooting for
                                 # Ne/Ar; see
                                 # ``tests/test_dfsolver_slater.py``.


_SF_TEMPLATE = """# benchmark reference run
SetAtom('{sym}', {Z})
SetSE(0)
SetVP(0)
SetMS(0, 0)
SetBreit(0)
Config('g0', '{cfg}')
ConfigEnergy(0)
OptimizeRadial(['g0'])
ConfigEnergy(1)
Structure('out.lev.b', ['g0'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
"""


def run_sfac(Z: int, sym: str, cfg: str) -> tuple[float | None, float]:
    """Run sfac, return (E_total_eV, wall_s)."""
    if not SFAC:
        raise SystemExit(
            "sfac not configured: set FAC_SFAC_PATH to the sfac executable "
            "(and FAC_DYLD_LIB to the gcc runtime dir on macOS).")
    with tempfile.TemporaryDirectory(prefix="atomx_bench_") as tmp:
        sf = Path(tmp) / "run.sf"
        sf.write_text(_SF_TEMPLATE.format(sym=sym, Z=Z, cfg=cfg))
        env = {**os.environ, "DYLD_LIBRARY_PATH": DYLD}
        t0 = time.perf_counter()
        try:
            subprocess.run(
                [SFAC, "run.sf"], cwd=tmp, env=env, check=True,
                capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, 0.0
        t = time.perf_counter() - t0
        lev = Path(tmp) / "out.lev"
        if not lev.exists():
            return None, t
        for line in lev.read_text().splitlines():
            if line.startswith("E0"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    items = [s.strip() for s in parts[1].split(",")]
                    if len(items) >= 2:
                        try:
                            return float(items[1]), t
                        except ValueError:
                            pass
        return None, t


def run_atomx(Z: int, cfg: str, flavour: str) -> tuple[float | None, float]:
    nel = sum(int(tok[2:]) for tok in cfg.split())
    charge = max(Z - nel, 0)
    t0 = time.perf_counter()
    try:
        # Clean-room parity: the sfac deck runs SetSE(0)/SetVP(0)/
        # SetMS(0,0)/SetBreit(0), so the atomx side must be
        # correction-free too (breit/qed default True in Atom).
        atom = Atom(Z=Z, charge=charge, config=cfg, exchange=flavour,
                    breit=False, qed=False)
        atom.solve()
        return atom.total_energy * HARTREE_EV, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  atomx {flavour} failed for Z={Z} {cfg}: {e}", file=sys.stderr)
        return None, time.perf_counter() - t0


def main() -> None:
    out_csv = ROOT / "benchmarks" / "results_local_exchange.csv"
    header = (
        ["Z", "symbol", "config", "E_sfac_eV", "t_sfac_s"]
        + [f"E_{f}_eV" for f in FLAVOURS]
        + [f"t_{f}_s" for f in FLAVOURS]
        + [f"rel_{f}" for f in FLAVOURS]
    )
    rows: list[dict] = []

    title = f"{'Z':>3}  {'cfg':>22}  {'E_sfac':>14}"
    for f in FLAVOURS:
        title += f"  {'E_' + f:>14}  {'Δ_' + f:>8}"
    title += f"  {'t_sfac':>7}  " + "  ".join(f"t_{f:<8}" for f in FLAVOURS)
    print(title)
    print("-" * len(title))

    for Z, sym, cfg in PANEL:
        # atomx first to amortise numba compile across rows
        atomx_E: dict[str, float | None] = {}
        atomx_t: dict[str, float] = {}
        for f in FLAVOURS:
            e, t = run_atomx(Z, cfg, f)
            atomx_E[f] = e
            atomx_t[f] = t
        e_ref, t_ref = run_sfac(Z, sym, cfg)

        row = {"Z": Z, "symbol": sym, "config": cfg,
               "E_sfac_eV": e_ref, "t_sfac_s": t_ref}
        line = f"{Z:>3}  {cfg:>22}  "
        line += (f"{e_ref:>14.3f}" if e_ref is not None else f"{'--':>14}")
        for f in FLAVOURS:
            e = atomx_E[f]
            if e is None or e_ref is None:
                line += f"  {'--':>14}  {'--':>8}"
                row[f"E_{f}_eV"] = None
                row[f"rel_{f}"] = None
            else:
                rel = abs(e - e_ref) / abs(e_ref)
                line += f"  {e:>14.3f}  {rel:>8.1e}"
                row[f"E_{f}_eV"] = e
                row[f"rel_{f}"] = rel
            row[f"t_{f}_s"] = atomx_t[f]

        line += f"  {t_ref:>7.2f}  "
        line += "  ".join(f"{atomx_t[f]:>8.2f}" for f in FLAVOURS)
        print(line)
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
