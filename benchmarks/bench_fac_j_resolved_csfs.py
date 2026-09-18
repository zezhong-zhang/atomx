#!/usr/bin/env python3
"""Benchmark AtomX J-resolved CSF counts against FAC level counts.

The comparison is deliberately structural: for a single configuration, FAC's
printed level table contains one row per fine-structure CSF/level before any
cross-configuration matching problem appears.  Counting rows by TWOJ should
therefore match AtomX's antisymmetric jj-coupled J-resolved generator.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_runner import parse_levels, run_sfac  # noqa: E402

from atomx.methods.mcdhf import count_j_resolved_csfs  # noqa: E402

CASES = [
    ("C", "2p1"),
    ("C", "2p2"),
    ("N", "2p3"),
    ("O", "2p4"),
    ("F", "2p5"),
    ("Ne", "2p6"),
    ("Ti", "3d1"),
    ("Ti", "3d2"),
]


def _fac_script(atom: str, config: str) -> str:
    return f"""SetAtom('{atom}')
Config('g', '{config}')
ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)
Structure('levels.lev.b', ['g'])
PrintTable('levels.lev.b', 'levels.lev', 1)
"""


def _fac_counts(atom: str, config: str, timeout: float) -> tuple[dict[int, int], float]:
    with tempfile.TemporaryDirectory(prefix="fac_jcsf_") as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "case.sf"
        script.write_text(_fac_script(atom, config))
        _stdout, elapsed, wd = run_sfac(script, workdir=tmp_path, timeout=timeout)
        levels = parse_levels(wd / "levels.lev")

    counts: dict[int, int] = {}
    for row in levels:
        j2 = int(row["TWOJ"])
        counts[j2] = counts.get(j2, 0) + 1
    return dict(sorted(counts.items())), elapsed


def _format_counts(counts: dict[int, int]) -> str:
    return " ".join(f"{j2}:{n}" for j2, n in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "benchmarks" / "results_fac_j_resolved_csfs.csv"),
        help="CSV output path",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "atom",
                "config",
                "atomx_total",
                "fac_total",
                "match",
                "fac_elapsed_s",
                "atomx_counts_by_2J",
                "fac_counts_by_2J",
            ],
        )
        writer.writeheader()

        print(f"{'case':>8}  {'AtomX':>18}  {'FAC':>18}  {'match':>5}")
        for atom, config in CASES:
            atomx_counts = count_j_resolved_csfs(config)
            try:
                fac_counts, elapsed = _fac_counts(atom, config, args.timeout)
            except Exception as exc:
                failures += 1
                fac_counts = {}
                elapsed = 0.0
                match = False
                print(f"{config:>8}  {_format_counts(atomx_counts):>18}  FAC failed: {exc}")
            else:
                match = atomx_counts == fac_counts
                failures += 0 if match else 1
                print(
                    f"{config:>8}  {_format_counts(atomx_counts):>18}  "
                    f"{_format_counts(fac_counts):>18}  {str(match):>5}"
                )

            writer.writerow({
                "atom": atom,
                "config": config,
                "atomx_total": sum(atomx_counts.values()),
                "fac_total": sum(fac_counts.values()),
                "match": match,
                "fac_elapsed_s": f"{elapsed:.6f}",
                "atomx_counts_by_2J": _format_counts(atomx_counts),
                "fac_counts_by_2J": _format_counts(fac_counts),
            })

    print(f"\nWrote {out_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
