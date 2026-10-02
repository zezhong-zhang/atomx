"""XATOM uniform continuum mesh check at unchanged HFS/GPS physics."""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b

HERE = Path(__file__).resolve().parent
refs = json.loads(
    (HERE.parent / "systematic_xray_2026_10_02/references.json").read_text()
)


def job(case):
    el, photon, dr = case
    args = SimpleNamespace(xatom=Path("/tmp/atomx-xraypac-build/build/src/xatom"))
    label = "drhalf" if dr == 0.00125 else "drquarter"
    raw = HERE / "raw" / el / f"xatom-pi-{int(photon)}-{label}"
    out = b.xatom(args, raw, el, ["-dr", str(dr), "-pcs", "-PE", str(photon)])
    values = b.parse_xatom(out)["pi_mb"]
    rows = [
        dict(
            element=el,
            shell=shell,
            photon_eV=photon,
            xatom_original_Mb=refs[el]["xatom_pi"][str(int(photon))][shell],
            xatom_fine_Mb=values[shell],
            uniform_dr_a0=dr,
            fine_over_original=values[shell]
            / refs[el]["xatom_pi"][str(int(photon))][shell],
        )
        for shell in ("1s", "2s", "2p")
    ]
    print(el, photon, [r["fine_over_original"] for r in rows], flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dr", type=float, choices=[0.00125, 0.000625], default=0.00125
    )
    dr = parser.parse_args().dr
    cases = [(el, E, dr) for el in b.CASES for E in refs[el]["photons_eV"]]
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = [r for group in executor.map(job, cases) for r in group]
    filename = "xatom_pi_grid.csv" if dr == 0.00125 else "xatom_pi_grid_quarter.csv"
    with (HERE / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
