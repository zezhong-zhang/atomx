#!/usr/bin/env python3
"""Resume same-grid audits of stored heavy-ion low-p-hole GSZ branches."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_dhf_core_p_roots import audit_one, candidate_jobs  # noqa: E402
from scripts.generate_full_dhf_nist import OUTPUT  # noqa: E402
from scripts.summarize_dhf_heavy_core_p_probe import (  # noqa: E402
    PROBE, has_low_p_hole,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=PROBE)
    parser.add_argument("--min-z", type=int, default=41)
    parser.add_argument("--max-z", type=int, default=74)
    parser.add_argument("--max-jobs", type=int, default=20)
    parser.add_argument("--max-per-element", type=int, default=2)
    parser.add_argument("--max-original-seconds", type=float, default=30)
    args = parser.parse_args()
    if (args.max_jobs < 1 or args.max_per_element < 1
            or args.max_original_seconds <= 0
            or not 21 <= args.min_z <= args.max_z <= 118):
        parser.error("positive limits required")
    seen = set()
    if args.report.exists():
        for line in args.report.read_text().splitlines():
            row = json.loads(line)
            seen.add((row["z"], row["charge"], row["configuration"]))
    candidates = [row for row in candidate_jobs(
        args.ledger, args.min_z, args.max_z, args.max_original_seconds)
        if has_low_p_hole(row[2]) and row[:3] not in seen]
    by_z = defaultdict(list)
    for row in candidates:
        by_z[row[0]].append(row)
    selected = []
    for index in range(args.max_per_element):
        for z in sorted(by_z):
            if index < len(by_z[z]):
                selected.append(by_z[z][index])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.report.open("a") as stream:
        for row in selected[:args.max_jobs]:
            result = audit_one(row)
            result["probe_scope"] = (f"historical_heavy_low_p_hole_"
                                     f"z{args.min_z}_{args.max_z}")
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            counts[result["status"]] += 1
            print(result["z"], result["charge"],
                  result["configuration"], result["status"],
                  f"{result['elapsed_seconds']:.1f}s", flush=True)
    print(json.dumps({"new_rows": sum(counts.values()),
                      "statuses": dict(counts),
                      "eligible_not_yet_audited": len(candidates)}), flush=True)


if __name__ == "__main__":
    main()
