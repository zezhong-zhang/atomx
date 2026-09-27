#!/usr/bin/env python3
"""Compare DHF configuration excitations with complete NIST level centroids.

This is a configuration-average check, distinct from a resolved spectral-line
comparison.  Every J level in a configuration must be present and evaluated;
otherwise its centroid remains unscored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomx.config.parsing import parse_nr_shells
from atomx.constants import HARTREE_CM1

from scripts.audit_nist_level_weights import weight_from_j
from scripts.generate_full_dhf_nist import expand_nist_config

ARCHIVE = ROOT / "output/nist_spectra/spectra.sqlite"
LEDGER = ROOT / "output/nist_spectra/full_dhf.sqlite"
CSV_OUT = ROOT / "output/nist_spectra/theory_comparison/configuration_centroids_dhf_nist.csv"
JSON_OUT = ROOT / "output/nist_spectra/theory_comparison/configuration_centroids_summary.json"


def expected_weight(configuration: str) -> int:
    weight = 1
    for _n, ell, occupation in parse_nr_shells(configuration):
        weight *= comb(4 * ell + 2, occupation)
    return weight


def level_centroids(source: sqlite3.Connection, jobs: set[tuple[int, int, str]],
                    max_z: int) -> dict[tuple[int, int, str], dict]:
    groups: dict[tuple[int, int, str], dict] = defaultdict(lambda: dict(
        level_rows=0, weighted_energy=0.0, weight_sum=0,
        missing_energy_or_j=0, non_evaluated=0, g_disagreements=0))
    for z, charge, config, j_text, energy, kind, g_raw in source.execute(
            """SELECT z,charge,configuration,j_text,energy_cm1,energy_kind,
                      statistical_weight_raw FROM levels WHERE z<=?""",
            (max_z,)):
        if not config:
            continue
        full = expand_nist_config(config, z - charge)
        key = (z, charge, full)
        if full is None or key not in jobs:
            continue
        group = groups[key]
        group["level_rows"] += 1
        g = weight_from_j(j_text)
        if g is None or energy is None or not math.isfinite(energy):
            group["missing_energy_or_j"] += 1
            continue
        group["weight_sum"] += g
        group["weighted_energy"] += g * energy
        if kind != "evaluated":
            group["non_evaluated"] += 1
        try:
            if g_raw and Fraction(g_raw.strip()) != g:
                group["g_disagreements"] += 1
        except (ValueError, ZeroDivisionError):
            pass
    result = {}
    for key in jobs:
        group = groups.get(key)
        expected = expected_weight(key[2])
        complete = (group is not None and group["level_rows"] > 0
                    and group["missing_energy_or_j"] == 0
                    and group["weight_sum"] == expected)
        status = ("missing_nist_levels" if group is None else
                  "incomplete_manifold" if not complete else
                  "nonexperimental_complete" if group["non_evaluated"] else
                  "evaluated_complete")
        result[key] = {
            "status": status,
            "centroid_cm1": group["weighted_energy"] / expected if complete else None,
            "expected_weight": expected,
            "listed_weight": group["weight_sum"] if group else 0,
            "level_rows": group["level_rows"] if group else 0,
            "missing_energy_or_j": group["missing_energy_or_j"] if group else 0,
            "non_evaluated_levels": group["non_evaluated"] if group else 0,
            "nist_g_disagreements": group["g_disagreements"] if group else 0,
        }
    return result


def collect(archive: Path = ARCHIVE, ledger: Path = LEDGER,
            max_z: int = 118) -> tuple[list[dict], dict]:
    with sqlite3.connect(archive) as source, sqlite3.connect(ledger) as db:
        species = {(z, charge): label for z, charge, label in source.execute(
            "SELECT z,charge,label FROM species WHERE z<=?", (max_z,))}
        jobs = {(z, charge, config): dict(status=status, energy_ha=energy,
                                          is_ground=bool(is_ground), isotope_a=isotope)
                for z, charge, config, status, energy, is_ground, isotope in db.execute(
                    """SELECT z,charge,configuration,status,energy_ha,
                              is_ground,isotope_a FROM config_jobs WHERE z<=?""",
                    (max_z,))}
        grounds = {}
        for key, job in jobs.items():
            if job["is_ground"]:
                ion = key[:2]
                if ion in grounds:
                    raise ValueError(f"multiple ground jobs for {ion}")
                grounds[ion] = key
        centroids = level_centroids(source, set(jobs), max_z)
    rows = []
    for key in sorted(jobs):
        z, charge, config = key
        job = jobs[key]
        ground_key = grounds.get((z, charge))
        ground_job = jobs.get(ground_key) if ground_key else None
        target = centroids[key]
        ground = centroids.get(ground_key) if ground_key else None
        theory_gap = (None if job["energy_ha"] is None or ground_job is None
                      or ground_job["energy_ha"] is None else
                      (job["energy_ha"] - ground_job["energy_ha"]) * HARTREE_CM1)
        nist_gap = (None if target["centroid_cm1"] is None or ground is None
                    or ground["centroid_cm1"] is None else
                    target["centroid_cm1"] - ground["centroid_cm1"])
        if key == ground_key:
            status = "ground_reference"
        elif job["status"] not in ("done", "basis_limited", "untrusted"):
            status = "waiting_for_theory" if job["status"] in (
                "pending", "running", "retry_pending", "repair_pending",
                "candidate_pending") else "theory_failed"
        elif job["status"] != "done":
            status = "theory_failed"
        elif ground_job is None or ground_job["status"] != "done":
            status = "missing_theory_ground"
        elif ground is None or ground["status"] != "evaluated_complete":
            status = "missing_evaluated_nist_ground"
        elif target["status"] != "evaluated_complete":
            status = target["status"]
        elif nist_gap is None or nist_gap <= 0:
            status = "invalid_nist_excitation"
        elif theory_gap is None or theory_gap <= 0:
            status = "inverted_theory_order"
        else:
            status = "compared"
        residual = theory_gap - nist_gap if status == "compared" else None
        rows.append({
            "z": z, "charge": charge,
            "spectrum": species.get((z, charge), f"Z={z} q={charge}"),
            "configuration": config, "ground_configuration": ground_key[2] if ground_key else None,
            "is_ground": int(job["is_ground"]), "isotope_a": job["isotope_a"],
            "theory_status": job["status"], "theory_energy_ha": job["energy_ha"],
            "theory_excitation_cm1": theory_gap,
            "nist_manifold_status": target["status"],
            "nist_centroid_cm1": target["centroid_cm1"],
            "nist_ground_centroid_cm1": ground["centroid_cm1"] if ground else None,
            "nist_excitation_cm1": nist_gap,
            "nist_level_rows": target["level_rows"],
            "nist_expected_weight": target["expected_weight"],
            "nist_listed_weight": target["listed_weight"],
            "nist_missing_energy_or_j": target["missing_energy_or_j"],
            "nist_non_evaluated_levels": target["non_evaluated_levels"],
            "nist_g_disagreements": target["nist_g_disagreements"],
            "comparison_status": status,
            "residual_cm1": residual,
            "energy_error_percent": (100 * residual / nist_gap
                                     if residual is not None else None),
        })
    counts = Counter(r["comparison_status"] for r in rows)
    by_element = []
    for z in range(1, max_z + 1):
        scored = [r for r in rows if r["z"] == z and
                  r["comparison_status"] == "compared"]
        errors = [abs(r["energy_error_percent"]) for r in scored]
        by_element.append({"z": z, "symbol": species.get((z, 0), "").split(" ")[0],
                           "n_configurations": len(scored),
                           "median_abs_error_percent": median(errors) if errors else None,
                           "rms_error_percent": (math.sqrt(sum(e * e for e in errors)
                                                           / len(errors)) if errors else None)})
    return rows, {"model": "DHF+B+QED+normal mass shift",
                  "comparison_basis": "complete evaluated 2J+1 configuration centroids",
                  "configuration_rows": len(rows), "statuses": dict(sorted(counts.items())),
                  "elements": by_element}


def write_outputs(rows: list[dict], summary: dict,
                  csv_path: Path = CSV_OUT, summary_path: Path = JSON_OUT) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_part = csv_path.with_name(f"{csv_path.name}.{os.getpid()}.part")
    with csv_part.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    csv_part.replace(csv_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_part = summary_path.with_name(
        f"{summary_path.name}.{os.getpid()}.part")
    summary_part.write_text(json.dumps(summary, indent=2) + "\n")
    summary_part.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--csv", type=Path, default=CSV_OUT)
    parser.add_argument("--summary", type=Path, default=JSON_OUT)
    parser.add_argument("--max-z", type=int, default=118)
    args = parser.parse_args()
    if not 1 <= args.max_z <= 118:
        parser.error("--max-z must be within 1..118")
    rows, summary = collect(args.archive, args.ledger, args.max_z)
    write_outputs(rows, summary, args.csv, args.summary)
    print(json.dumps({k: summary[k] for k in ("configuration_rows", "statuses")},
                     indent=2))


if __name__ == "__main__":
    main()
