"""Independent archive and complete-spectrum replay for the CV discriminator."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
from scipy.linalg import eigvalsh  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "same005_orbitals001"
CONTRACT = json.loads((HERE / "contract.json").read_text())
INPUT = ROOT / CONTRACT["fixed_conditions"]["input_run"]
FROZEN = ROOT / CONTRACT["fixed_conditions"]["frozen_source_root"]
sys.path.insert(0, str(FROZEN))

from atomx.constants import HARTREE_CM1  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    result = json.loads((OUT / "result.json").read_text())
    binding = json.loads((OUT / "target_binding.json").read_text())
    basis = json.loads((OUT / "csf_basis.json").read_text())
    source = json.loads((INPUT / "source_identity.json").read_text())
    native_result = json.loads((INPUT / "result.json").read_text())
    rows = basis["CSFs"]
    blocks = {}
    for row in rows:
        blocks.setdefault((row["twoJ"], row["parity"]), []).append(row["index"])
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    def numerical(name, error, tolerance):
        error = float(error)
        checks.append({"name": name, "error": error, "limit": tolerance,
                       "passed": bool(np.isfinite(error) and error <= tolerance)})

    check("frozen_353_file_source_identity", all(
        sha(FROZEN / Path(name).relative_to("src")) == expected
        for name, expected in source["all_atomx_python_sha256"].items()))
    check("fixed_seven_input_identities", all(sha(INPUT / name) == expected
          for name, expected in CONTRACT["inputs"].items()))
    check("prospective_contract_identity", sha(HERE / "contract.json") == result["contract_sha256"])
    check("execution_source_identity", sha(HERE / "run.py") == sha(OUT / "run_source.py"))
    check("archived_scientific_artifact_identities", all(sha(OUT / name) == expected
          for name, expected in result["artifacts_sha256"].items()))
    check("archived_numerical_gates", all(item["passed"] for item in result["checks"]))
    check("all_68_CSFs_preserved", len(rows) == 68)
    check("all_69_electron_patterns", all(sum(q for _, _, q in row["subshells"]) == 69 for row in rows))
    check("complete_expected_family_counts", [len(family["indices"]) for family in basis["families"]] == [5, 39, 24])
    check("preserved_005_bare_stationarity_pass", native_result["stationarity_passed"] is True)
    check("no_expanded_MCDHF_stationarity_claim", result["expanded68_MCDHF_stationarity_passed"] is None)
    check("diagnostic_only_no_production_or_experiment_claim", result["diagnostic_only"] is True
          and result["production_certified"] is False and result["experimental_gate_passed"] is False)

    arrays = np.load(OUT / "ci_arrays.npz")
    overlaps = np.load(OUT / "overlap_arrays.npz")
    native = np.load(INPUT / "ci_arrays.npz")
    audit = {"energy_window_Ha": CONTRACT["state_binding"]["near_degenerate_energy_window_Ha"],
             "complete_symmetry_clusters": {}, "targets": []}
    signed_reference = {}
    for model, suffix in (("DC", "DC"), ("static_Breit", "static")):
        h = arrays["matrix_" + suffix + "_Ha"]
        e, v = arrays["energies_" + suffix + "_Ha"], arrays["mixing_" + suffix]
        bare_key = "replay_dc_mixing" if model == "DC" else "rci_mixing"
        bare_energies = native["replay_dc_energies_Ha" if model == "DC" else "rci_energies_Ha"]
        archived_rows = json.loads((OUT / (model + "_all_roots.json")).read_text())["all_roots"]
        check(model + "/complete_saved_roots", len(archived_rows) == 68 and v.shape == (68, 68))
        numerical(model + "/independent_eigvalsh_full_spectrum_Ha",
                  max(abs(eigvalsh(h) - e)), 1e-8)
        numerical(model + "/complete_signed_matrix_reconstruction_Ha", max(abs((v * e) @ v.T - h).flat), 1e-8)
        numerical(model + "/complete_left_closure", max(abs(v @ v.T - np.eye(68)).flat), 1e-8)
        # Cauchy interlacing is independent of the tracked branch selection.
        interlacing = max(0., float(np.max(e[:5] - bare_energies)),
                          float(np.max(bare_energies - e[63:])))
        numerical(model + "/five_bare_principal_interlacing_Ha", interlacing, 1e-8)
        embedded = np.zeros((68, 5))
        embedded[:5] = native[bare_key]
        signed = embedded.T @ v
        signed_reference[model] = signed
        check(model + "/signed_bare_overlap_archive", np.array_equal(
            signed, overlaps["bare_DC_to_full_signed" if model == "DC" else "bare_static_to_full_signed"]))
        numerical(model + "/bare_full_spectral_closure", max(abs(signed @ signed.T - np.eye(5)).flat), 1e-8)
        clusters = []
        for symmetry, indices in blocks.items():
            columns = [row["global_column"] for row in archived_rows
                       if (row["twoJ"], row["parity"]) == symmetry]
            check(f"{model}/all_roots_block/{symmetry}", len(columns) == len(indices))
            current = []
            for col in columns:
                if current and e[col] - e[current[-1]] > audit["energy_window_Ha"]:
                    if len(current) > 1:
                        clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": current})
                    current = []
                current.append(col)
            if len(current) > 1:
                clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": current})
        audit["complete_symmetry_clusters"][model] = clusters
        for row in archived_rows:
            col = row["global_column"]
            numerical(f"{model}/energy_ledger/{col}", abs(row["energy_Ha"] - e[col]), 1e-14)
            numerical(f"{model}/global_ground_reference/{col}",
                      abs(row["excitation_from_finite_model_global_ground_cm1"] - (e[col] - e[0]) * HARTREE_CM1), 1e-10)
            for family in basis["families"]:
                numerical(f"{model}/family_weight/{col}/{family['name']}",
                          abs(row["family_weights"][family["name"]] - np.sum(v[family["indices"], col]**2)), 1e-14)
        for i, target in enumerate(binding["targets"]):
            symmetry = (target["twoJ"], target["parity"])
            columns = [row["global_column"] for row in archived_rows
                       if (row["twoJ"], row["parity"]) == symmetry]
            selected = target["DC_column" if model == "DC" else "static_column"]
            others = [col for col in columns if col != selected]
            gap = min((abs(float(e[col] - e[selected])) for col in others), default=None)
            clustered = gap is not None and gap <= audit["energy_window_Ha"]
            audit["targets"].append({"model": model, "label": target["label"], "selected_column": selected,
                                     "minimum_same_symmetry_gap_Ha": gap, "near_degenerate": clustered,
                                     "single_root_identity_allowed_by_energy_separation": not clustered})
            check(f"{model}/{target['label']}/no_near_degenerate_single_root_claim", not clustered)
            if model == "DC":
                best = max(columns, key=lambda col: signed[i, col]**2)
                check(f"DC/{target['label']}/predeclared_bare_overlap_choice", best == selected)
                numerical(f"DC/{target['label']}/bare_weight_ledger",
                          abs(signed[i, selected]**2 - target["DC_bare_squared_overlap"]), 1e-14)
            else:
                best = max(columns, key=lambda col: signed[i, col]**2)
                check(f"static/{target['label']}/bare_alternative_binding_retained", best == target["static_bare_best_column"])
            traced = result["tracked_levels"][model][i]
            ground = result["tracked_levels"][model][0]["column"]
            excitation = (e[selected] - e[ground]) * HARTREE_CM1
            numerical(f"{model}/{target['label']}/tracked_excitation_replay_cm1",
                      abs(excitation - traced["excitation_from_tracked_6s_cm1"]), 1e-10)
            old = native_result["rows_dc" if model == "DC" else "rows_rci"][i]
            numerical(f"{model}/{target['label']}/CV_increment_replay_cm1",
                      abs(excitation - old["computed_cm1"] - traced["delta_excitation_from_bare_cm1"]), 1e-10)
            numerical(f"{model}/{target['label']}/NIST_gap_replay_cm1",
                      abs(excitation - old["experimental_cm1"] - traced["gap_cm1"]), 1e-10)
        for label, lower, upper in (("5d", 1, 2), ("6p", 3, 4)):
            traced = result["tracked_levels"][model]
            g = [traced[i]["twoJ"] + 1 for i in (lower, upper)]
            centroid = sum(traced[i]["excitation_from_tracked_6s_cm1"] * weight
                           for i, weight in zip((lower, upper), g, strict=True)) / sum(g)
            split = traced[upper]["excitation_from_tracked_6s_cm1"] - traced[lower]["excitation_from_tracked_6s_cm1"]
            numerical(f"{model}/{label}/multiplet_centroid_cm1",
                      abs(centroid - result["multiplets"][model][label]["centroid_cm1"]), 1e-10)
            numerical(f"{model}/{label}/multiplet_splitting_cm1",
                      abs(split - result["multiplets"][model][label]["splitting_cm1"]), 1e-10)
    dc_to_static = arrays["mixing_DC"].T @ arrays["mixing_static"]
    check("complete_signed_DC_to_static_overlap_archive", np.array_equal(dc_to_static, overlaps["DC_to_static_signed"]))
    assigned = binding["complete_DC_to_static_assignment"]
    check("complete_assignment_is_bijection", sorted(row["DC_column"] for row in assigned) == list(range(68))
          and sorted(row["static_column"] for row in assigned) == list(range(68)))
    for row in assigned:
        numerical(f"all_root_assignment/{row['DC_column']}/overlap_squared",
                  abs(row["squared_overlap"] - dc_to_static[row["DC_column"], row["static_column"]]**2), 1e-14)
    write(OUT / "degeneracy_audit.json", audit)
    broken = []
    for target in re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text()):
        if "://" in target or target.startswith("#"):
            continue
        path = HERE / target.split("#", 1)[0]
        if path != HERE / "verification.json" and not path.exists():
            broken.append(target)
    check("documentation_links", not broken)
    verification = {"passed": all(row["passed"] for row in checks), "check_count": len(checks),
                    "checks": checks, "broken_documentation_links": broken,
                    "scope": "Independent complete-H eigvalsh, signed closure and matrix/ledger/identity/branch replay; not an independent atomic Hamiltonian implementation or 68-CSF MCDHF stationarity proof.",
                    "numeric_run_gate_count": len(result["checks"]), "diagnostic_only": True,
                    "expanded_MCDHF_certified": False,
                    "files_sha256": {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*"))
                                     if p.is_file() and p.name != "verification.json" and "__pycache__" not in p.parts}}
    write(HERE / "verification.json", verification)
    print(json.dumps({"passed": verification["passed"], "check_count": len(checks),
                      "numeric_run_gates": len(result["checks"]),
                      "near_degenerate_targets": [row for row in audit["targets"] if row["near_degenerate"]]}))
    if not verification["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
