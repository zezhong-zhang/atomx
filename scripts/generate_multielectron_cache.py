#!/usr/bin/env python
"""Calculate atomx DHF and MCDHF state tables for selected NIST ions.

Results are checkpointed after each rung. Re-run compare_theory_nist.py to
rebuild the site from the updated cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from verify_nist_levels import (  # noqa: E402
    CASES, CACHE_PATH, Cache, choose_model, compare_ci_rung, compare_config_rung)
import nist_asd as nist  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", metavar="'Li I'",
                        help="NIST ion label; may be repeated")
    parser.add_argument("--all-panel", action="store_true",
                        help="run every multielectron system in the benchmark panel")
    parser.add_argument("--dhf-only", action="store_true",
                        help="calculate only the two DHF configuration rungs")
    parser.add_argument("--ci-only", action="store_true",
                        help="calculate only the two MCDHF state rungs")
    args = parser.parse_args()
    if args.dhf_only and args.ci_only:
        parser.error("--dhf-only and --ci-only are mutually exclusive")
    available = {case.label: case for case in CASES}
    if args.all_panel:
        cases = [case for case in CASES if
                 nist.load(case.label, with_lines=False).n_electrons > 1]
    elif args.system:
        unknown = set(args.system) - set(available)
        if unknown:
            parser.error("unknown benchmark system: " + ", ".join(sorted(unknown)))
        cases = [available[label] for label in args.system]
    else:
        parser.error("choose at least one --system or use --all-panel")
    cache = Cache(CACHE_PATH)
    for case in cases:
        ion = nist.load(case.label, with_lines=False)
        model, _, _ = choose_model(case, ion)
        if not args.ci_only:
            for breit, qed, name in [(False, False, "DHF"),
                                      (True, True, "DHF+B+QED")]:
                try:
                    energies = {}
                    for config in model:
                        energies[config] = cache.scf(case.z, config, breit, qed)
                        cache.save()
                    result = compare_config_rung(case, ion, name, model, energies, 0)
                    print(f"{case.label} {name}: {result.n_levels} configuration-averaged "
                          f"levels, RMS {result.stats.get('rms_cm1', float('nan')):.5g} cm^-1",
                          flush=True)
                except Exception as exc:
                    print(f"{case.label} {name}: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
        if args.dhf_only:
            continue
        for breit, qed, name in [(False, False, "MCDHF"),
                                  (True, True, "MCDHF+B+QED")]:
            try:
                payload = cache.ci(case.z, model, breit, qed)
                result = compare_ci_rung(case, ion, name, model, payload, 0)
                cache.save()
                print(f"{case.label} {name}: {result.n_levels} paired levels, "
                      f"{result.n_excluded_blocks} excluded blocks, "
                      f"RMS {result.stats.get('rms_cm1', float('nan')):.5g} cm^-1",
                      flush=True)
            except Exception as exc:
                print(f"{case.label} {name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
    print(f"cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
