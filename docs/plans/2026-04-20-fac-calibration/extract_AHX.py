#!/usr/bin/env python3
"""Run sfac across a (Z, config) panel and collect FAC's converged AHX.

For each panel row:
  1. Write a minimal .sf script (no SE, VP, MS, Breit — pure local
     exchange, exactly the physics we're porting).
  2. Invoke sfac with DYLD_LIBRARY_PATH.
  3. Parse the resulting out.pot header for AHX / HXS / IHX / N / Z.
  4. Parse out.lev for E0 (eV).

Writes the collected rows to calibration.csv.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "fac_data"
# Reference code location from the environment (no machine-specific default).
SFAC = Path(os.environ.get("FAC_SFAC_PATH", ""))
DYLD = os.environ.get("FAC_DYLD_LIB", "")

# Panel: one row per (Z, config). Closed shells are not compressed
# so we probe a range of N (electron counts) for each Z.
# Symbols must match FAC's internal ElementSymbol() table.
SYMBOLS = {
    2: "He", 3: "Li", 4: "Be", 6: "C", 10: "Ne", 18: "Ar",
    26: "Fe", 36: "Kr", 54: "Xe", 74: "W", 82: "Pb", 92: "U",
}

PANEL = [
    # (Z, config, N electrons)
    (2,  "1s2",                       2),
    (3,  "1s2 2s1",                   3),
    (4,  "1s2 2s2",                   4),
    (6,  "1s2 2s2 2p2",               6),
    (10, "1s2 2s2 2p6",              10),
    (18, "1s2 2s2 2p6 3s2 3p6",      18),
    (26, "1s2 2s2 2p6",              10),   # Fe^{16+} Ne-like
    (26, "1s2 2s2 2p6 3s2",          12),   # Fe^{14+} Mg-like
    (36, "1s2 2s2 2p6 3s2 3p6",      18),
    (54, "1s2 2s2 2p6",              10),   # Xe^{44+} Ne-like
    (74, "1s2 2s2 2p6",              10),   # W^{64+} Ne-like
    (82, "1s2 2s2 2p6 3s2 3p6",      18),   # Pb^{64+} Ar-like
    (92, "1s2 2s2 2p6",              10),   # U^{82+} Ne-like
]


_SF = """# extract AHX for Z={Z} config='{cfg}'
SetAtom('{sym}', {Z})
SetSE(0)
SetVP(0)
SetMS(0, 0)
SetBreit(0)
Config('g0', '{cfg}')
ConfigEnergy(0)
OptimizeRadial(['g0'])
ConfigEnergy(1)
GetPotential('out.pot')
Structure('out.lev.b', ['g0'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
"""


def run_sfac(Z: int, sym: str, cfg: str, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    sf = workdir / "run.sf"
    sf.write_text(_SF.format(Z=Z, sym=sym, cfg=cfg))
    env = {**os.environ, "DYLD_LIBRARY_PATH": DYLD}
    subprocess.run(
        [str(SFAC), sf.name], cwd=workdir, env=env, check=True,
        capture_output=True, text=True,
    )


_HDR_RE = re.compile(r"#\s*([A-Za-z]+)\s*=\s*(-?[0-9.Ee+-]+)")


def parse_pot_header(pot_file: Path) -> dict:
    """Return {key: float} for every "# KEY = value" line."""
    out: dict = {}
    for line in pot_file.read_text().splitlines():
        if not line.startswith("#"):
            break
        m = _HDR_RE.match(line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def parse_e0(lev_file: Path) -> float | None:
    for line in lev_file.read_text().splitlines():
        if line.startswith("E0"):
            # "E0\t= 0, -3.50028100E+03"
            parts = line.split("=", 1)
            if len(parts) == 2:
                items = [s.strip() for s in parts[1].split(",")]
                if len(items) >= 2:
                    try:
                        return float(items[1])
                    except ValueError:
                        return None
    return None


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rows: list[dict] = []
    for Z, cfg, N in PANEL:
        sym = SYMBOLS[Z]
        tag = f"Z{Z:03d}_N{N:02d}_{cfg.replace(' ', '_')}"
        workdir = DATA / tag
        print(f"  running {tag} ...", flush=True)
        run_sfac(Z, sym, cfg, workdir)
        hdr = parse_pot_header(workdir / "out.pot")
        e0 = parse_e0(workdir / "out.lev")
        row = {
            "Z": Z,
            "N_expected": N,
            "config": cfg,
            "AHX_fac": hdr.get("AHX"),
            "HXS_fac": hdr.get("HXS"),
            "IHX_fac": hdr.get("IHX"),
            "N_fac":   hdr.get("N"),
            "HX0_fac": hdr.get("HX0"),
            "HX1_fac": hdr.get("HX1"),
            "E0_eV":   e0,
            "tag":     tag,
        }
        rows.append(row)
        print(
            f"     Z={Z:>3}  N_fac={row['N_fac']:>5}  "
            f"AHX={row['AHX_fac']:.8f}  "
            f"HXS={row['HXS_fac']:.4f}  IHX={row['IHX_fac']:.4f}",
            flush=True,
        )

    csv_path = HERE / "calibration.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Z", "N_expected", "N_fac", "config",
                "AHX_fac", "HXS_fac", "IHX_fac",
                "HX0_fac", "HX1_fac",
                "E0_eV", "tag",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
