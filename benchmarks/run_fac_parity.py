"""Run the FAC parity benchmark suite.

This is the stable entry point for tracking AtomX against FAC while the
physics backends are brought to FAC parity.  Individual benchmark scripts keep
their process-specific details; this runner records command status and elapsed
time in one machine-readable summary.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FULL_SUITE = [
    ("rcfp_primitives", ["benchmarks/bench_rcfp_primitives.py"]),
    ("j_resolved_csfs", ["benchmarks/bench_fac_j_resolved_csfs.py"]),
    ("local_exchange", ["benchmarks/bench_local_exchange.py"]),
    ("structure", [
        "benchmarks/bench_fac_structure.py",
        "--z", "10", "18", "26", "36", "54", "74", "82",
        "--iso", "He-like", "Ne-like", "Ar-like",
        "--qed",
    ]),
    ("lithium_like", [
        "benchmarks/bench_fac_lithium_like.py",
        "--z", "10", "18", "26", "36", "54", "74", "82", "92",
    ]),
    ("levels_transitions", ["benchmarks/bench_fac_levels_transitions.py"]),
    ("pi_rr", ["benchmarks/bench_fac_pi_rr.py"]),
    ("ce_ci_ai", ["benchmarks/bench_fac_ce_ci_ai.py"]),
]


QUICK_SUITE = [
    ("rcfp_primitives", ["benchmarks/bench_rcfp_primitives.py"]),
    ("j_resolved_csfs", ["benchmarks/bench_fac_j_resolved_csfs.py"]),
    ("local_exchange", ["benchmarks/bench_local_exchange.py"]),
    ("levels_transitions", ["benchmarks/bench_fac_levels_transitions.py"]),
    ("pi_rr", ["benchmarks/bench_fac_pi_rr.py"]),
    ("ce_ci_ai", ["benchmarks/bench_fac_ce_ci_ai.py"]),
]


def run_command(command: list[str]) -> tuple[int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, *command], cwd=ROOT)
    return proc.returncode, time.perf_counter() - t0


def _read_metric_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {
            row["metric"]: row["value"]
            for row in csv.DictReader(f)
            if row.get("metric")
        }


def benchmark_metrics(name: str) -> str:
    """Compact process accuracy metrics emitted by benchmark scripts."""
    if name == "levels_transitions":
        metrics = _read_metric_csv(
            ROOT / "benchmarks" / "results_fac_levels_transition_summary.csv"
        )
        if not metrics:
            return ""
        parts = [
            f"strong_mapped={metrics.get('strong_mapped', '')}/"
            f"{metrics.get('strong_count', '')}",
            f"strong_missing={metrics.get('strong_missing', '')}",
            f"strong_max_factor={metrics.get('strong_max_factor', '')}",
            f"strong_median_factor={metrics.get('strong_median_factor', '')}",
        ]
        optional = (
            "strong_recoup_dominant_sign_flips",
            "strong_ci_recoup_hybrid_max_factor",
            "strong_ci_recoup_abs_hybrid_max_factor",
            "strong_ci_recoup_gauge_hybrid_max_factor",
            "strong_ci_recoup_create_first_max_factor",
            "strong_phase_gauge_max_conflicts",
        )
        parts.extend(f"{key}={metrics[key]}" for key in optional if metrics.get(key))
        return "; ".join(parts)
    if name == "ce_ci_ai":
        return scattering_metrics()
    return ""


def _rel_diff_stats(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    values = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                values.append(abs(float(row["rel_diff"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
    return max(values), median


def scattering_metrics() -> str:
    """Compact CE/CI/AI relative-difference metrics."""
    metrics = _read_metric_csv(
        ROOT / "benchmarks" / "results_fac_ce_ci_ai_summary.csv"
    )
    if metrics:
        return "; ".join(
            f"{key}={metrics[key]}"
            for key in (
                "ce_max_abs_rel",
                "ce_median_abs_rel",
                "ce_omega_max_abs_rel",
                "ce_omega_median_abs_rel",
                "ci_max_abs_rel",
                "ci_median_abs_rel",
                "ai_max_abs_rel",
                "ai_median_abs_rel",
                "ai_exchange_plus_max_abs_rel",
                "ai_exchange_plus_median_abs_rel",
            )
            if key in metrics
        )

    parts = []
    for label, filename in (
        ("ce", "results_fac_ce.csv"),
        ("ci", "results_fac_ci.csv"),
        ("ai", "results_fac_ai.csv"),
    ):
        stats = _rel_diff_stats(ROOT / "benchmarks" / filename)
        if stats is None:
            continue
        max_abs, median_abs = stats
        parts.append(f"{label}_max_abs_rel={max_abs:.3g}")
        parts.append(f"{label}_median_abs_rel={median_abs:.3g}")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run the smaller FAC suite used during iterative development.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Run remaining benchmarks even if one command fails.",
    )
    parser.add_argument(
        "--summary",
        default="benchmarks/results_fac_parity_summary.csv",
        help="CSV path for command status and timing.",
    )
    args = parser.parse_args()

    suite = QUICK_SUITE if args.quick else FULL_SUITE
    rows = []
    for name, command in suite:
        print(f"\n== {name} ==")
        print(" ".join([sys.executable, *command]))
        rc, elapsed = run_command(command)
        rows.append({
            "benchmark": name,
            "returncode": rc,
            "elapsed_s": f"{elapsed:.3f}",
            "metrics": benchmark_metrics(name),
            "command": " ".join(command),
        })
        print(f"{name}: rc={rc}, elapsed={elapsed:.3f}s")
        if rc != 0 and not args.continue_on_error:
            break

    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark", "returncode", "elapsed_s", "metrics", "command",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary_path}")
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
