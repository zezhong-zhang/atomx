"""Separate FAC angular-sum and line-output cutoffs for neutral Fe holes."""

import csv
import sys
from pathlib import Path

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b
from fac_runner import parse_levels, parse_transitions

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "systematic_xray_2026_10_02"


def run():
    rows = []
    archived_channels = list(
        csv.DictReader((HERE / "fluorescence_channels.csv").open())
    )
    shells = b.CASES["Fe"][1]
    for hole in ("1s", "2s", "2p"):
        original = REF / "raw/Fe" / f"fac-{hole}-hole"
        initial = dict(shells)
        initial[hole] -= 1
        n = b.csf_count(initial)
        # sfac PSetTransitionCut incorrectly declares c0/c as int; a decimal
        # 1e-4 argument is truncated to zero. -1 resets the documented default
        # in faclib SetTransitionCut, preserving the requested other cutoff.
        for tag, cut in [
            ("angular_full", (0, -1)),
            ("lines_full", (-1, 0)),
            ("all_full", (0, 0)),
        ]:
            script = [
                x
                for x in (original / "run.sf").read_text().splitlines()
                if not x.startswith(("AITable(", "PrintTable('ai.b'"))
            ]
            index = next(i for i, x in enumerate(script) if x.startswith("TRTable("))
            script.insert(index, f"SetTransitionCut({cut[0]},{cut[1]})")
            raw = HERE / "raw/Fe" / f"fac-{hole}-{tag}"
            b.fac_job(raw, script)
            levels = parse_levels(raw / "levels.txt")
            lines = [r for r in parse_transitions(raw / "tr.txt") if r["upper"] < n]
            weights = {r["ILEV"]: r["TWOJ"] + 1 for r in levels[:n]}
            g = sum(weights.values())
            offset = n
            for source in shells:
                if source == hole:
                    continue
                final = dict(shells)
                final[source] -= 1
                end = offset + b.csf_count(final)
                selected = [r for r in lines if offset <= r["lower"] < end]
                offset = end
                if not selected:
                    continue
                old = next(
                    r
                    for r in archived_channels
                    if (r["element"], r["exchange"], r["hole"], r["source"])
                    == ("Fe", "gu", hole, source)
                )
                width = sum(weights[t["upper"]] * t["A_per_s"] for t in selected) / g
                strength = (
                    sum(
                        weights[t["upper"]] * t["A_per_s"] / t["DELTA_E_eV"] ** 3
                        for t in selected
                    )
                    / g
                )
                collapsed = strength * float(old["native_photon_eV"]) ** 3
                rows.append(
                    dict(
                        element="Fe",
                        hole=hole,
                        source=source,
                        cut_mode=tag,
                        angular_cut=(cut[0] if cut[0] >= 0 else 1e-4),
                        line_cut=(cut[1] if cut[1] >= 0 else 1e-4),
                        line_count=len(selected),
                        resolved_per_s=width,
                        same_native_photon_per_s=collapsed,
                        resolved_over_default=width / float(old["fac_resolved_per_s"]),
                        strength_over_default=collapsed
                        / float(old["fac_same_native_photon_per_s"]),
                    )
                )
            print(hole, tag, "lines", len(lines), flush=True)
    with (HERE / "fac_transition_cuts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run()
