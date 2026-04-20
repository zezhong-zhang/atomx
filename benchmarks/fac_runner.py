"""Thin wrapper to run sfac scripts with the right DYLD_LIBRARY_PATH.

Usage:
    from benchmarks.fac_runner import run_sfac
    stdout, elapsed = run_sfac("my_script.sf", workdir=Path("tmp/"))
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

SFAC_BIN = Path("/Users/zhangzz/code/ASAP/fac/bin/bin/sfac")
DYLD_LIB = "/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current"


def run_sfac(script_path: Path, workdir: Path | None = None, timeout: float = 300.0) -> tuple[str, float, Path]:
    """Run sfac on the given script. Returns (stdout, elapsed_s, workdir)."""
    script_path = Path(script_path).resolve()
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="fac_bench_"))
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / script_path.name
    if dest.resolve() != script_path.resolve():
        shutil.copy(script_path, dest)

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD_LIB + ":" + env.get("DYLD_LIBRARY_PATH", "")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(SFAC_BIN), script_path.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    t1 = time.perf_counter()
    if proc.returncode != 0:
        raise RuntimeError(f"sfac failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return proc.stdout, t1 - t0, workdir


def parse_levels(lev_file: Path) -> list[dict]:
    """Parse a FAC .lev file (ASCII form, as written by PrintTable).

    Returns list of dicts with keys: ILEV, IBASE, ENERGY_eV, P, VNL, TWOJ, config, state.
    """
    rows = []
    in_data = False
    for line in Path(lev_file).read_text().splitlines():
        if line.strip().startswith("ILEV") and "IBASE" in line:
            in_data = True
            continue
        if not in_data:
            continue
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        try:
            ilev = int(parts[0])
            ibase = int(parts[1])
            energy = float(parts[2])
            p = int(parts[3])
            vnl = parts[4]
            twoj = int(parts[5])
        except (ValueError, IndexError):
            continue
        config = " ".join(parts[6:])[:64].strip() if len(parts) > 6 else ""
        rows.append({
            "ILEV": ilev, "IBASE": ibase, "ENERGY_eV": energy,
            "P": p, "VNL": vnl, "TWOJ": twoj, "config": config,
        })
    return rows


def parse_e0(lev_file: Path) -> float | None:
    """Parse the E0 reference from the .lev header (eV)."""
    for line in Path(lev_file).read_text().splitlines():
        if line.startswith("E0"):
            # "E0\t= 0, -3.12294355E+04"
            parts = line.split("=")
            if len(parts) >= 2:
                tail = parts[1].strip()
                # "0, -3.12294355E+04"
                items = [x.strip() for x in tail.split(",")]
                if len(items) >= 2:
                    try:
                        return float(items[1])
                    except ValueError:
                        return None
    return None


def parse_transitions(tr_file: Path) -> list[dict]:
    """Parse a FAC .tr file written by PrintTable(..., 1).

    Yields dicts with upper/lower level indices, Δ E in eV, gA rate in s⁻¹, gf.

    File format (after headers):
      upper  p_up  lower  p_lo   Delta_E_eV   gf   gA_per_s   gf_alt
    """
    rows = []
    current = None
    for line in Path(tr_file).read_text().splitlines():
        ls = line.strip()
        if ls.startswith("MULTIP"):
            try:
                current = int(ls.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                current = None
            continue
        parts = ls.split()
        if len(parts) < 7:
            continue
        try:
            up = int(parts[0])
            p_up = int(parts[1])
            lo = int(parts[2])
            p_lo = int(parts[3])
            delta_e = float(parts[4])
            gf = float(parts[5])
            ga = float(parts[6])
        except (ValueError, IndexError):
            continue
        rows.append({
            "mult": current, "upper": up, "lower": lo,
            "DELTA_E_eV": delta_e, "gf": gf, "gA_per_s": ga,
            "p_up": p_up, "p_lo": p_lo,
        })
    return rows
