"""Replay source-table arithmetic. This does not calculate or fit a new spectrum."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import subprocess

TOPIC = Path(__file__).resolve().parent
REPO = TOPIC.parents[1]
TAIL = TOPIC.parent / "yb_correlation_tail_2026_10_08"
LABELS = ["5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, data):
    (TOPIC / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    contract = json.loads((TOPIC / "contract.json").read_text())
    assert contract["resource_caps"]["many_body_runs"] == 0
    archive_checks = {}
    for topic in ("yb_correlation_tail_2026_10_08", "yb_energy_dependent_sigma_2026_10_08",
                  "yb_diagnosis_repair_2026_10_08"):
        manifest = json.loads((TOPIC.parent / topic / "artifact_manifest.json").read_text())
        for key, record in manifest.items():
            path = REPO / key if key.startswith("docs/") else TOPIC.parent / topic / key
            expected = record if isinstance(record, str) else record["sha256"]
            assert sha(path) == expected, key
        archive_checks[topic] = {"count": len(manifest), "unchanged": True}

    old = json.loads((TAIL / "capture_source_sha256.json").read_text())
    files = subprocess.check_output(["git", "ls-files", "src"], cwd=REPO, text=True).splitlines()
    current = {p: sha(REPO / p) for p in files if p.endswith(".py")}
    assert current == old, "production source changed since measured result"
    write_json("source_identity.json", {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "actual_dirty_production_sha256": current,
        "source_matches_measured_tail": True,
        "archive_checks": archive_checks,
        "input_sha256": {str(p.relative_to(REPO)): sha(p) for p in (
            TAIL / "analysis/results.json", TAIL / "absolute_anchor/results.json",
            TAIL / "report_summary.json", TAIL / "literature_check/results.json")},
    })

    data = json.loads((TAIL / "analysis/results.json").read_text())
    native = data["primaries"]["fine_l6_e200"]
    actual = {r["label"]: r for r in native["records"]}
    experiment = {s: actual[s]["experimental_excitation_cm1"] for s in LABELS}
    # Facts transcribed from the printed primary tables, preserving their precision.
    printed = {
        "porsev2012_mbpt": {"values": [24450, 25952, 28636, 32242], "source": "porsev2012", "table": "I", "order": "second order"},
        "porsev2012_linearized_sd": {"values": [22820, 24261, 27945, 31481], "source": "porsev2012", "table": "I", "order": "linearized SD all order"},
        "pototschnig2021_eom_ea_extrap": {"values": [23951, 25257, 27913, 31380], "source": "pototschnig2021", "table": "2, atomic Yb+", "order": "EOM-EA-CCSD complete-basis extrapolation"},
        "pototschnig2021_ihfs_extrap": {"values": [23938, 25246, 27907, 31371], "source": "pototschnig2021", "table": "3, atomic Yb+", "order": "IHFS-CCSD complete-basis extrapolation"},
    }
    # Dzuba 2011 Table I gives binding energies, not excitation energies.
    binding_2011 = {"6s_1/2": 99477, "5d_3/2": 75551, "6p_1/2": 70728, "6p_3/2": 67101}
    printed["dzuba2011_unscaled_brueckner"] = {
        "values": [binding_2011["6s_1/2"] - binding_2011[s] if s in binding_2011 else None for s in LABELS],
        "source": "dzuba2011", "table": "I binding energies, excitation differences reconstructed",
        "order": "full orbital Brueckner with second-order correlation potential, unscaled",
    }
    write_json("primary_table_data.json", {
        "label_order": LABELS, "printed": printed,
        "dzuba2011_printed_binding_cm1": binding_2011,
        "porsev2012_printed_experiment_cm1": [22961, 24333, 27062, 30392],
        "pototschnig2021_printed_experiment_cm1": [22961, 24333, 27062, 30392],
        "porsev2012_15e_ci_largest_cm1": [24615, 25464, 24289, 27324],
        "pototschnig2021_hole_F7_2_cm1": {"experiment_printed": 21419, "EOM_IP_extrap": 17966, "IHFS_extrap": 14514},
        "notes": "Own comparisons use the retained, independently verified four experimental excitations. Do not confuse the historical absolute thresholds or combine different methods into a prediction.",
    })

    rows = []
    for label in LABELS:
        r = actual[label]
        rows.append({"method": "atomx_fine_l6_e200_scalar_dyson", "source": "retained native evidence", "label": label,
                     "excitation_cm1": r["excitation_dyson_cm1"], "experiment_cm1": experiment[label],
                     "residual_cm1": r["experiment_residual_cm1"],
                     "relative_error_percent": 100 * r["relative_experimental_error"]})
    for method, table in printed.items():
        for label, value in zip(LABELS, table["values"]):
            if value is None:
                continue
            delta = value - experiment[label]
            rows.append({"method": method, "source": table["source"], "label": label,
                         "excitation_cm1": value, "experiment_cm1": experiment[label],
                         "residual_cm1": delta, "relative_error_percent": 100 * delta / experiment[label]})
    with (TOPIC / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # E3 and Coulomb-Breit B2 are different diagram classes, not substitutes.
    components = {
        "6s_1/2": {"E3": 1016, "B1": 128, "B2": -259, "ELS": 18},
        "5d_3/2": {"E3": -1027, "B1": 156, "B2": -546, "ELS": 0},
        "5d_5/2": {"E3": -1301, "B1": 114, "B2": -499, "ELS": 0},
        "6p_1/2": {"E3": 335, "B1": 91, "B2": -120, "ELS": -1},
        "6p_3/2": {"E3": 166, "B1": 63, "B2": -110, "ELS": 0},
    }
    increments = {s: {k: components[s][k] - components["6s_1/2"][k] for k in components[s]} for s in LABELS}
    retained = json.loads((TAIL / "literature_check/results.json").read_text())
    for rec in retained["records"]:
        if rec["label"] in increments:
            assert increments[rec["label"]]["E3"] == rec["paper_printed_E3_excitation_difference_cm1"]

    sd_values = printed["porsev2012_linearized_sd"]["values"]
    mbpt_values = printed["porsev2012_mbpt"]["values"]
    outputs = {
        "source": "printed primary tables and immutable retained Atomx results",
        "new_atomic_many_body_runs": 0,
        "corrections_injected": False,
        "experimental_fit": False,
        "cross_method_is_same_hamiltonian": False,
        "rows": rows,
        "porsev2012_SD_minus_MBPT_excitation_cm1": dict(zip(LABELS, [a - b for a, b in zip(sd_values, mbpt_values)])),
        "safronova2009_printed_attachment_components_cm1": components,
        "safronova2009_relative_components_cm1": increments,
        "max_abs_relative_percent_by_method": {m: max(abs(r["relative_error_percent"]) for r in rows if r["method"] == m)
                                             for m in sorted({r["method"] for r in rows})},
        "hole_sign_check": {
            "measured_NIST_P3_cm1": experiment["6p_3/2"], "measured_NIST_4f13_5d6s_J3_2_cm1": 28757.98,
            "observed_gap_cm1": experiment["6p_3/2"] - 28757.98,
            "conditional_2level_sign": "A bare hole state below a bare P state raises the P-like root through level repulsion. Experimental mixed levels only identify proximity and cannot supply a bare denominator or coupling.",
        },
        "limits": "No inferred correction is a controlled error budget; no published result is imported into Atomx. First-order Breit does not bound B2; scalar Dyson does not include full orbital or SD amplitudes.",
    }
    write_json("results.json", outputs)

    paths = ["src/atomx/methods/coupled_cluster.py", "src/atomx/methods/mbpt.py",
             "src/atomx/methods/mbpt_sigma2.py", "src/atomx/methods/ci_mbpt.py",
             "src/atomx/methods/mbpt_transitions.py", "src/atomx/methods/rpa.py"]
    audit = {p: {"sha256": sha(REPO / p), "symbols": {n.name: n.lineno for n in ast.parse((REPO / p).read_text()).body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))}} for p in paths}
    write_json("implementation_audit.json", audit)
    print(json.dumps({"archive_checks": archive_checks, "source_files": len(current),
                      "SD_minus_MBPT_cm1": outputs["porsev2012_SD_minus_MBPT_excitation_cm1"],
                      "B2_excitation_cm1": {s: increments[s]["B2"] for s in LABELS},
                      "max_percent": outputs["max_abs_relative_percent_by_method"]}, indent=2))


if __name__ == "__main__":
    main()
