"""Re-read historical native Eu calibration and the exact cached NIST rows.

This bounded read-only scientific audit performs no solver work and never
changes historical logs, scripts or the SQLite database. The default writes
only this audit's evidence and query snapshot; --check writes nothing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OWNER = Path(__file__).resolve().parent
TIER_SCRIPT = ROOT / "scripts/eu3_tierA_anchor.py"
TIER_LOG = ROOT / "docs/review_2026_10_02/codex_fix_validation/eu-tier-a.log"
CF_LOG = ROOT / "docs/review_2026_10_02/codex_fix_validation/eu-cf.log"
DB = ROOT / "output/nist_spectra/spectra.sqlite"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def binding(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes())}


def required(pattern, value, description):
    match = re.search(pattern, value)
    if match is None:
        raise ValueError(f"Missing {description}: {pattern}")
    return match


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def read_script_contract():
    text = TIER_SCRIPT.read_text()
    tree = ast.parse(text)
    wanted = {"REFS", "NIST_7F", "S2_PRIOR"}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != wanted:
        raise ValueError("Historical script no longer declares all required constants")
    if [float(row[0]) for row in found["REFS"]] != [0.0, 1.0, 2.0]:
        raise ValueError("Historical target J identities changed")
    if len(found["NIST_7F"]) != 7:
        raise ValueError("Historical 7F target list changed")
    # The script's own explicit source annotation outranks its printed heading.
    if "AQUO-ION (Carnall IV)" not in text:
        raise ValueError("Historical anchor source annotation changed; re-audit provenance")
    found["module_docstring"] = ast.get_docstring(tree)
    found["reference_annotation"] = required(
        r"# References are AQUO-ION \(Carnall IV\) values used for the free\s*\n"
        r"\s*# ion within their uncertainty — flagged, not hidden\.",
        text,
        "explicit historical reference annotation",
    ).group(0)
    return found


def read_nist_rows():
    query = (
        "SELECT * FROM levels WHERE z=63 AND charge=3 AND configuration='4f6' "
        "AND ((term='7F' AND j_text IN ('0','1','2','3','4','5','6')) "
        "OR (term='5D3' AND j_text IN ('0','1','2'))) ORDER BY row_index"
    )
    source_query = "SELECT * FROM nist_level_detail_sources WHERE z=63 AND charge=3"
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(query)]
        sources = [dict(row) for row in conn.execute(source_query)]
    expected = [f"063004.{number:06d}" for number in range(1, 11)]
    if [row["level_id"] for row in rows] != expected:
        raise ValueError("Exact cached Eu IV level identities changed")
    if len(sources) != 1 or sources[0]["status"] != "imported":
        raise ValueError("Exact cached Eu IV source identity missing or not imported")
    return {
        "database": str(DB.relative_to(ROOT)),
        "database_open_mode": "ro",
        "queries": {"levels": query, "detail_source": source_query},
        "level_rows": rows,
        "detail_source_rows": sources,
        "query_payload_sha256": digest(canonical({"levels": rows, "sources": sources}).encode()),
        "binding_scope": "Exact query records and official raw source identity; not a whole-database hash",
    }


def build_evidence(snapshot):
    script = read_script_contract()
    text = TIER_LOG.read_text()
    cf_text = CF_LOG.read_text()
    scales = [float(value) for value in required(
        r"Eu Tier-A .*?s = \(([+-]?[\d.]+), ([+-]?[\d.]+), ([+-]?[\d.]+)\)",
        text, "fitted scales").groups()]
    if scales[0] != float(script["S2_PRIOR"]):
        raise ValueError("Saved fitted s2 disagrees with the source prior")
    logged_rms = float(required(r"residual RMS = ([\d.]+) cm\^-1", text, "logged RMS").group(1))
    rows_by_id = {row["level_id"]: row for row in snapshot["level_rows"]}
    fitted = []
    matches = re.findall(
        r"5D(\d):\s+([\d.]+) vs\s+([\d.]+)\s+\(\s*([+-]?[\d.]+)\)", text)
    if [int(row[0]) for row in matches] != [0, 1, 2]:
        raise ValueError("Saved 5D labels or number of targets changed")
    for j_string, calculated, target, logged_residual in matches:
        j = int(j_string)
        calculated, target, logged_residual = map(float, (calculated, target, logged_residual))
        if target != float(script["REFS"][j][1]):
            raise ValueError(f"Saved 5D{j} fit target disagrees with source AST")
        if abs((calculated - target) - logged_residual) > 0.051:
            raise ValueError(f"Saved 5D{j} residual disagrees beyond printed precision")
        nist = rows_by_id[f"063004.{8 + j:06d}"]
        if nist["term"] != "5D3" or nist["j_text"] != str(j):
            raise ValueError(f"NIST 5D{j} identity disagrees")
        fitted.append({
            "historical_label": f"5D{j}", "J": j,
            "calculated_cm1_rounded_log": calculated,
            "historical_fit_target_cm1": target,
            "historical_logged_residual_cm1": logged_residual,
            "residual_vs_historical_fit_target_cm1_rounded_log": calculated - target,
            "nist_level_id": nist["level_id"], "nist_term": nist["term"],
            "nist_cm1": nist["energy_cm1"], "nist_energy_raw": nist["energy_raw"],
            "nist_energy_kind": nist["energy_kind"], "nist_bracketed": bool(nist["bracketed"]),
            "nist_uncertainty_cm1": nist["uncertainty_cm1"], "nist_reference": nist["reference"],
            "residual_vs_cached_nist_cm1_rounded_log": calculated - nist["energy_cm1"],
            "independent_experimental_validation": False,
        })

    holdout = []
    matches = re.findall(
        r"7F(\d):\s+([\d.]+) vs\s+([\d.]+)\s+\(\s*([+-]?[\d.]+)%\)", text)
    if [int(row[0]) for row in matches] != list(range(1, 7)):
        raise ValueError("Saved unfitted 7F labels changed")
    for j_string, calculated, target, percentage in matches:
        j = int(j_string)
        calculated, target, percentage = map(float, (calculated, target, percentage))
        if target != float(script["NIST_7F"][j]):
            raise ValueError(f"Saved 7F{j} historical target disagrees with source AST")
        nist = rows_by_id[f"063004.{1 + j:06d}"]
        holdout.append({
            "historical_label": f"7F{j}", "J": j,
            "calculated_cm1_rounded_log": calculated,
            "historical_unfitted_target_cm1": target,
            "historical_logged_error_percent": percentage,
            "historical_reference_kind": "legacy driver constants labeled NIST; not this live NIST query",
            "residual_vs_historical_target_cm1_rounded_log": calculated - target,
            "nist_level_id": nist["level_id"], "nist_cm1": nist["energy_cm1"],
            "nist_energy_raw": nist["energy_raw"], "nist_energy_kind": nist["energy_kind"],
            "nist_bracketed": bool(nist["bracketed"]),
            "nist_uncertainty_cm1": nist["uncertainty_cm1"], "nist_reference": nist["reference"],
            "historical_target_minus_cached_nist_cm1": target - nist["energy_cm1"],
            "residual_vs_cached_nist_cm1_rounded_log": calculated - nist["energy_cm1"],
            "error_vs_cached_nist_percent_rounded_log": (calculated / nist["energy_cm1"] - 1) * 100,
            "fit_on_this_observable": False, "independent_experimental_validation": False,
        })

    cf_match = required(r"full M space: (\d+) dets; 5D0 anchor = ([\d.]+) cm\^-1", cf_text,
                        "full-M saved CF control")
    if int(cf_match.group(1)) != 3003:
        raise ValueError("Historical complete 4f6 determinant count changed")
    if float(cf_match.group(2)) != fitted[0]["calculated_cm1_rounded_log"]:
        raise ValueError("Saved CF anchor disagrees with saved fitted ladder")
    cf_trace = float(required(r"CF trace over full M space = ([+-]?[\d.eE+-]+) cm\^-1",
                              cf_text, "saved CF trace").group(1))
    tier_rms = rms([row["residual_vs_historical_fit_target_cm1_rounded_log"] for row in fitted])
    nist_rms = rms([row["residual_vs_cached_nist_cm1_rounded_log"] for row in fitted])
    return {
        "schema_version": 1,
        "status": "completed_historical_evidence_audit",
        "solver_rerun": False,
        "atom": {"Z": 63, "charge": 3, "spectrum": "Eu IV", "configuration": "[Xe]4f6"},
        "native_execution": {
            "description": "Atomx DFSolver -> jj-CSF/full-M sparse CI -> per-k Slater delta -> native crystal-field operator",
            "external_physics_backend": False,
            "historical_full_M_determinants": 3003,
            "scope": "Frozen statistical Xe-core 4f6 manifold; success is calibration, not an all-core ab-initio closure certificate",
        },
        "calibration": {
            "s2_fixed_prior": script["S2_PRIOR"], "scales_rounded_log": scales,
            "fitted_parameters": ["s4", "s6"], "target_count": 3,
            "reference_kind": "Historical Carnall aquo-ion anchors as declared by the driver; not independently verified free-ion experimental targets",
            "reference_annotation": script["reference_annotation"],
            "script_docstring": script["module_docstring"],
            "logged_rms_cm1": logged_rms,
            "rms_vs_historical_fit_targets_cm1_rounded_log": tier_rms,
            "rms_vs_cached_nist_cm1_rounded_log": nist_rms,
            "nist_comparison_kind": "comparison to cached NIST semiempirical bracketed levels; not independent experimental validation",
            "fitted_levels": fitted,
            "independent_experimental_validation": False,
        },
        "unfitted_7F_holdout": holdout,
        "cf_saved_control": {
            "full_M_determinants": 3003,
            "anchor_cm1_rounded_log": float(cf_match.group(2)),
            "assembled_trace_shift_cm1_logged": cf_trace,
            "interpretation": "historical full-M native assembly and calibrated anchor control; not an experimental crystal-spectrum validation",
        },
        "precision_limit": {
            "calculated_energy_display_decimals": 1,
            "calculated_energy_rounding_half_width_cm1": 0.05,
            "derived_rms_from_displayed_energies": True,
            "uncertainty_rule": "NIST unknown uncertainty remains null; printed RMS is not propagated physical uncertainty",
            "state_assignment_rule": "Historical 5D J labels selected lowest level above 15000 cm-1 in each J; no fresh wavefunction/parentage certification",
        },
        "inputs": {
            "tier_a_script": binding(TIER_SCRIPT),
            "tier_a_log": binding(TIER_LOG),
            "cf_log": binding(CF_LOG),
            "query_snapshot": binding(OWNER / "query_snapshot.json"),
        },
        "nist_query_payload_sha256": snapshot["query_payload_sha256"],
        "audit_identity": binding(Path(__file__).resolve()),
        "card_summary": (
            f"Eu³⁺ Atomx原生4f⁶/3003行列式：历史三目标标定RMS {logged_rms:.1f} cm⁻¹；"
            f"对本地NIST括号半经验⁵D₀/₁/₂ RMS {nist_rms:.2f} cm⁻¹。"
            "s₂先验固定、s₄/s₆拟合；此列为标定成功与参考对比，独立实验认证未建立；未拟合⁷F单列保留。"
        ),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only verify saved evidence against live exact sources")
    args = parser.parse_args()
    snapshot = read_nist_rows()
    if args.check:
        saved_snapshot = json.loads((OWNER / "query_snapshot.json").read_text())
        if snapshot != saved_snapshot:
            raise ValueError("Live exact Eu IV query differs from saved snapshot")
    else:
        write_json(OWNER / "query_snapshot.json", snapshot)
    evidence = build_evidence(snapshot)
    if args.check:
        saved = json.loads((OWNER / "evidence.json").read_text())
        if evidence != saved:
            raise ValueError("Re-read Eu evidence differs from saved evidence or source bindings")
    else:
        write_json(OWNER / "evidence.json", evidence)
    print(json.dumps({
        "status": "verified" if args.check else "written_and_re_read",
        "inputs": len(evidence["inputs"]),
        "nist_rows": len(snapshot["level_rows"]),
        "fitted_levels": len(evidence["calibration"]["fitted_levels"]),
        "unfitted_holdout_levels": len(evidence["unfitted_7F_holdout"]),
        "card_summary": evidence["card_summary"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
