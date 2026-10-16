"""Independent full-matrix, complete-state, and nested-space archive replay."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

import numpy as np  # noqa: E402
from scipy.linalg import eigvalsh  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = json.loads((HERE / "contract.json").read_text())
FIXED = CONTRACT["fixed_conditions"]
INPUT = ROOT / FIXED["native_run"]
FROZEN = ROOT / FIXED["frozen_source_root"]
sys.path.insert(0, str(FROZEN))

# Unit conversion only; replay does not call the CI builder or solver.
from atomx.constants import HARTREE_CM1  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    started = time.monotonic()
    result, basis = load(HERE / "result.json"), load(HERE / "basis_identity.json")
    source, native_result = load(INPUT / "source_identity.json"), load(INPUT / "result.json")
    native = np.load(INPUT / "ci_arrays.npz")
    checks, audit = [], {"energy_window_Ha": 1e-8, "stages": []}

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})

    def num(name, error, limit=1e-8):
        error = float(error)
        checks.append({"name": name, "error": error, "limit": limit,
                       "passed": bool(np.isfinite(error) and error <= limit)})

    check("source_353_files", len(source["all_atomx_python_sha256"]) == 353 and all(
        sha(FROZEN / Path(relative).relative_to("src")) == expected
        for relative, expected in source["all_atomx_python_sha256"].items()))
    check("source_map_identity", result["source_map_sha256"] == FIXED["source_map_sha256"]
          == source["source_map_sha256"])
    check("prospective_input_hashes", all(sha(ROOT / relative) == expected
          for relative, expected in CONTRACT["inputs_sha256"].items()))
    check("prospective_contract_identity", sha(HERE / "contract.json") == result["contract_sha256"])
    check("actual_execution_source", sha(HERE / "execution_source.py") == sha(HERE / "run.py")
          == result["execution_source_sha256"])
    check("frontier_basis_identity", sha(HERE / "basis_identity.json") ==
          CONTRACT["inputs_sha256"][FIXED["frontier_enumeration"]])
    check("recorded_scientific_gates", result["numerical_gates_passed"] and all(c["passed"] for c in result["checks"]))
    check("source_and_experimental_input_unchanged", result["inputs_sha256"] == CONTRACT["inputs_sha256"])
    check("bare005_stationarity_preserved", native_result["stationarity_passed"] is True)
    check("fixed_orbital_only_no_expanded_or_production_claim", result["diagnostic_only"] is True
          and result["expanded_MCDHF_stationarity_passed"] is None
          and not result["production_certified"] and not result["experimental_gate_passed"]
          and not result["complete_CV_claim"])
    check("bounded_resource_observation", result["resource"]["threads"] == 1
          and result["resource"]["peak_RSS_MiB"] < 4096 and result["resource"]["total_seconds_uncontrolled"] < 900
          and result["resource"]["controlled_performance"] is False)
    check("all_three_stages", [s["CSFs"] for s in result["stages"]] == [294, 468, 531])
    check("complete69electronCSFs", len(basis["CSFs"]) == 531
          and all(sum(q for _, _, q in b["subshells"]) == 69 for b in basis["CSFs"]))
    native_groups = {}
    for col, row in enumerate(native_result["rows_dc"]):
        native_groups.setdefault((row["twoJ"], row["parity"]), []).append(col)
    old = {"n": 5, "name": "bare005"}
    for model, key_e, key_v in (("DC", "replay_dc_energies_Ha", "replay_dc_mixing"),
                                ("static_Breit", "rci_energies_Ha", "rci_mixing")):
        old[model] = {"e": native[key_e], "v": native[key_v], "groups": native_groups,
                      "h": native["matrix_dc"] if model == "DC" else
                      (native[key_v] * native[key_e]) @ native[key_v].T,
                      "targets": list(range(5)),
                      "rows": native_result["rows_dc" if model == "DC" else "rows_rci"]}

    def assign(signed, old_groups, new_groups):
        mapping = {}
        for symmetry, left_columns in old_groups.items():
            right_columns = new_groups[symmetry]
            a, b = linear_sum_assignment(-signed[np.ix_(left_columns, right_columns)]**2)
            for i, j in zip(a, b, strict=True):
                mapping[left_columns[i]] = right_columns[j]
        return mapping

    for plan, stage in zip(FIXED["stages"], result["stages"], strict=True):
        name, n = stage["name"], stage["CSFs"]
        owner, current = HERE / name, {"name": name, "n": n}
        check(name + "/plan", name == plan["name"] and stage["families"] == plan["families"])
        check(name + "/stage_identity", load(owner / "result.json") == stage)
        check(name + "/scientific_arrays_hashes", all(sha(owner / filename) == expected
              for filename, expected in stage["files_sha256"].items()))
        binding, overlaps = load(owner / "target_binding.json"), np.load(owner / "overlaps.npz")
        check(name + "/previous_identity", binding["previous_stage"] == old["name"])
        physical_groups = {}
        for row in basis["CSFs"][:n]:
            physical_groups.setdefault((row["twoJ"], row["parity"]), []).append(row["index"])
        stage_audit = {"name": name, "near_degenerate_clusters": {}, "targets": []}
        for model in ("DC", "static_Breit"):
            arr = np.load(owner / (model + "_arrays.npz"))
            h, e, v = arr["matrix_Ha"], arr["energies_Ha"], arr["mixing"]
            ledger = load(owner / (model + "_all_roots.json"))
            prefix = name + "/" + model
            check(prefix + "/complete_finite_shapes", h.shape == (n, n) and v.shape == (n, n)
                  and e.shape == (n,) and len(ledger["all_roots"]) == n
                  and all(np.isfinite(x).all() for x in (h, e, v)))
            num(prefix + "/independent_full_eigvalsh_Ha", np.max(abs(eigvalsh(h) - e)))
            num(prefix + "/signed_full_matrix_reconstruction_Ha", np.max(abs((v * e) @ v.T - h)))
            num(prefix + "/left_spectral_closure", np.max(abs(v @ v.T - np.eye(n))))
            num(prefix + "/right_spectral_closure", np.max(abs(v.T @ v - np.eye(n))))
            num(prefix + "/residual_Ha", np.max(np.linalg.norm(h @ v - v * e, axis=0)))
            num(prefix + "/unchanged_previous_principal_Ha", np.max(abs(h[:old["n"], :old["n"]] - old[model]["h"])))
            root_groups, clusters = {}, []
            for row in ledger["all_roots"]:
                c, s = row["column"], (row["twoJ"], row["parity"])
                root_groups.setdefault(s, []).append(c)
                num(prefix + f"/root{c}/energy_Ha", abs(row["energy_Ha"] - e[c]), 1e-14)
                num(prefix + f"/root{c}/symmetry_support", abs(np.sum(v[physical_groups[s], c]**2) - 1))
                check(prefix + f"/root{c}/rank", row["local_root"] == len(root_groups[s]) - 1)
                for family in ("bare", "4f", "5p", "5s"):
                    ix = [b["index"] for b in basis["CSFs"][:n] if b["family"] == family]
                    num(prefix + f"/root{c}/family/{family}", abs(row["family_weights"][family] - np.sum(v[ix, c]**2)), 1e-12)
                num(prefix + f"/root{c}/bareDC_character", np.max(abs(np.array(row["bare_reference_weights"]) - (native["replay_dc_mixing"].T @ v[:5, c])**2)), 1e-12)
            check(prefix + "/all_complete_J_blocks", set(root_groups) == set(physical_groups)
                  and all(len(root_groups[s]) == len(indices) for s, indices in physical_groups.items()))
            for symmetry, indices in physical_groups.items():
                cols = root_groups[symmetry]
                num(prefix + f"/block{symmetry}/independent_eigenvalues_Ha", np.max(abs(eigvalsh(h[np.ix_(indices, indices)]) - e[cols])))
                if symmetry in old[model]["groups"]:
                    oe = old[model]["e"][old[model]["groups"][symmetry]]
                    ne = e[cols]
                    num(prefix + f"/block{symmetry}/nested_interlacing_Ha", max(0., float(np.max(ne[:len(oe)] - oe)), float(np.max(oe - ne[len(ne)-len(oe):]))))
                cluster = []
                for c in cols:
                    if cluster and e[c] - e[cluster[-1]] > 1e-8:
                        if len(cluster) > 1:
                            clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": cluster})
                        cluster = []
                    cluster.append(c)
                if len(cluster) > 1:
                    clusters.append({"twoJ": symmetry[0], "parity": symmetry[1], "columns": cluster})
            check(prefix + "/near_degenerate_cluster_ledger", clusters == ledger["same_symmetry_near_degenerate_clusters"])
            stage_audit["near_degenerate_clusters"][model] = clusters
            embedded = np.zeros((n, old["n"]))
            embedded[:old["n"]] = old[model]["v"]
            signed = embedded.T @ v
            saved_key = "previous_DC_to_current_signed" if model == "DC" else "previous_static_to_current_signed"
            num(prefix + "/complete_signed_adjacent_overlaps", np.max(abs(signed - overlaps[saved_key])), 1e-14)
            num(prefix + "/adjacent_spectral_closure", np.max(abs(signed @ signed.T - np.eye(old["n"]))))
            amap = assign(signed, old[model]["groups"], root_groups)
            check(prefix + "/complete_adjacent_assignment", {r["source_column"]: r["destination_column"]
                  for r in binding["complete_previous_to_current_assignment"][model]} == amap)
            current[model] = {"h": h, "e": e, "v": v, "groups": root_groups, "signed_adjacent": signed,
                              "assignment": amap, "rows": stage["tracked_levels"][model],
                              "targets": overlaps["target_columns_DC" if model == "DC" else "target_columns_static"].tolist()}
        signed_operator = current["DC"]["v"].T @ current["static_Breit"]["v"]
        num(name + "/complete_signed_DC_static_overlaps", np.max(abs(signed_operator - overlaps["current_DC_to_static_signed"])), 1e-14)
        operator_assignment = assign(signed_operator, current["DC"]["groups"], current["static_Breit"]["groups"])
        check(name + "/complete_DC_static_assignment", {r["source_column"]: r["destination_column"]
              for r in binding["complete_DC_to_static_assignment"]} == operator_assignment)
        for k, original in enumerate(native_result["rows_dc"]):
            for model, bare_v_key in (("DC", "replay_dc_mixing"), ("static_Breit", "rci_mixing")):
                state, row = current[model], current[model]["rows"][k]
                b = binding["targets"][k]["models"][model]
                c, old_c = state["targets"][k], old[model]["targets"][k]
                prefix = f"{name}/{model}/{original['label']}"
                s = (original["twoJ"], original["parity"])
                columns = state["groups"][s]
                ref = np.zeros(n)
                ref[:5] = native[bare_v_key][:, k]
                weights = (ref @ state["v"][:, columns])**2
                check(prefix + "/symmetry_and_label", row["label"] == original["label"]
                      and (row["twoJ"], row["parity"]) == s and c in columns)
                desired_c = state["assignment"][old_c] if model == "DC" else operator_assignment[current["DC"]["targets"][k]]
                check(prefix + "/primary_complete_state_binding", desired_c == c == b["column"] == row["column"])
                num(prefix + "/full_bare_weight_distribution", np.max(abs(weights - np.array([w["weight"] for w in b["complete_bare_distribution"]]))), 1e-12)
                num(prefix + "/full_previous_weight_distribution", np.max(abs(state["signed_adjacent"][old_c, columns]**2 - np.array([w["weight"] for w in b["complete_previous_distribution"]]))), 1e-12)
                num(prefix + "/bare_spectral_closure", abs(np.sum(weights) - 1))
                center = float(ref @ state["h"] @ ref)
                variance = float(np.linalg.norm(state["h"] @ ref - center * ref)**2)
                num(prefix + "/centered_spectral_moment_Ha", abs(weights @ (state["e"][columns] - center)))
                num(prefix + "/centered_variance_Ha2", abs(weights @ (state["e"][columns] - center)**2 - variance))
                gap = min((abs(state["e"][j] - state["e"][c]) for j in columns if j != c), default=None)
                op_weight = float(signed_operator[current["DC"]["targets"][k], current["static_Breit"]["targets"][k]]**2)
                pw = float(state["signed_adjacent"][old_c, c]**2)
                bw = float(weights[columns.index(c)])
                safe = bool(bw > .5 and pw > .5 and op_weight > .5 and columns[int(np.argmax(weights))] == c
                            and state["assignment"][old_c] == c and (gap is None or gap > 1e-8))
                num(prefix + "/bare_character_ledger", abs(bw - b["bare_weight"]), 1e-12)
                num(prefix + "/previous_character_ledger", abs(pw - b["previous_overlap_weight"]), 1e-12)
                num(prefix + "/operator_character_ledger", abs(op_weight - b["operator_overlap_weight"]), 1e-12)
                check(prefix + "/nearest_gap_ledger", b["minimum_same_symmetry_gap_Ha"] == gap)
                check(prefix + "/unchanged_ambiguity_gates", safe == row["binding_unambiguous"] == b["unambiguous"])
                stage_audit["targets"].append({"label": original["label"], "model": model, "column": c,
                                              "nearest_same_J_gap_Ha": float(gap) if gap is not None else None,
                                              "bare_weight": bw, "previous_weight": pw, "operator_weight": op_weight,
                                              "unambiguous": safe})
                e, old_e = state["e"], old[model]["e"]
                actual = float((e[c] - e[state["targets"][0]]) * HARTREE_CM1)
                preceding = float((old_e[old_c] - old_e[old[model]["targets"][0]]) * HARTREE_CM1)
                bare_row = native_result["rows_dc" if model == "DC" else "rows_rci"][k]
                expected_values = {"energy_Ha": e[c], "energy_delta_from_previous_Ha": e[c] - old_e[old_c],
                    "energy_delta_from_bare_Ha": e[c] - bare_row["energy_Ha"],
                    "excitation_from_tracked_6s_cm1": actual,
                    "excitation_from_actual_finite_ground_cm1": (e[c] - e[0]) * HARTREE_CM1,
                    "transition_delta_from_previous_cm1": actual - preceding,
                    "transition_delta_from_bare_cm1": actual - bare_row["computed_cm1"],
                    "NIST_nominal_cm1": bare_row["experimental_cm1"], "gap_cm1": actual - bare_row["experimental_cm1"]}
                for key, expected in expected_values.items():
                    num(prefix + "/ledger/" + key, abs(row[key] - expected), 1e-9 if key.endswith("cm1") else 1e-12)
                check(prefix + "/nominal_comparison_unchanged", row["nominal_1percent_comparison_pass"] ==
                      (abs((actual - bare_row["experimental_cm1"]) / bare_row["experimental_cm1"]) < .01 if k else None))
        for model in ("DC", "static_Breit"):
            check(name + "/" + model + "/actual_ground_character", stage["actual_finite_model_ground"][model]
                  == load(owner / (model + "_all_roots.json"))["ground"])
            check(name + "/" + model + "/real_ground_identity", stage["tracked_6s_is_actual_ground"][model]
                  == (current[model]["targets"][0] == 0))
            for multiplet, lo, hi in (("5d", 1, 2), ("6p", 3, 4)):
                rows = current[model]["rows"]
                expected_center = sum(rows[k]["excitation_from_tracked_6s_cm1"] * (rows[k]["twoJ"] + 1)
                                      for k in (lo, hi)) / sum(rows[k]["twoJ"] + 1 for k in (lo, hi))
                expected_split = rows[hi]["excitation_from_tracked_6s_cm1"] - rows[lo]["excitation_from_tracked_6s_cm1"]
                num(name + "/" + model + "/" + multiplet + "/centroid", abs(stage["multiplets"][model][multiplet]["centroid_cm1"] - expected_center), 1e-9)
                num(name + "/" + model + "/" + multiplet + "/splitting", abs(stage["multiplets"][model][multiplet]["splitting_cm1"] - expected_split), 1e-9)
                for reference, ref_rows, key in (
                    ("previous", old[model]["rows"], "computed_cm1" if old["n"] == 5 else "excitation_from_tracked_6s_cm1"),
                    ("bare", native_result["rows_dc" if model == "DC" else "rows_rci"], "computed_cm1"),
                ):
                    center = sum(ref_rows[k][key] * (rows[k]["twoJ"] + 1) for k in (lo, hi)) / sum(rows[k]["twoJ"] + 1 for k in (lo, hi))
                    splitting = ref_rows[hi][key] - ref_rows[lo][key]
                    num(name + "/" + model + "/" + multiplet + "/centroid_delta_" + reference,
                        abs(stage["multiplets"][model][multiplet]["centroid_delta_from_" + reference + "_cm1"] - (expected_center - center)), 1e-9)
                    num(name + "/" + model + "/" + multiplet + "/splitting_delta_" + reference,
                        abs(stage["multiplets"][model][multiplet]["splitting_delta_from_" + reference + "_cm1"] - (expected_split - splitting)), 1e-9)
        audit["stages"].append(stage_audit)
        old = current
    check("card_final_static_rows", result["card_summary"]["final_static_rows"] == result["stages"][-1]["tracked_levels"]["static_Breit"])
    check("card_stages_bind_same_science", all(c["static_rows"] == s["tracked_levels"]["static_Breit"]
          for c, s in zip(result["card_summary"]["stages"], result["stages"], strict=True)))
    write(HERE / "degeneracy_audit.json", audit)
    identity_files = ["contract.json", "run.py", "execution_source.py", "result.json", "preflight.json", "basis_identity.json", "degeneracy_audit.json", "verify.py"]
    verification = {"all_passed": all(c["passed"] for c in checks), "checks_count": len(checks), "checks": checks,
                    "diagnostic_only": True, "independent_scope": "Saved full H eigvalsh and complete signed spectrum replay without CI builder/solver; conditional on frozen Atomx H, not an independent external Hamiltonian.",
                    "expanded_MCDHF_stationarity_passed": None, "production_certified": False,
                    "artifacts_sha256": {name: sha(HERE / name) for name in identity_files},
                    "resource": {"threads": 1, "seconds_uncontrolled": time.monotonic() - started,
                                 "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}}
    write(HERE / "verification.json", verification)
    print(json.dumps({"all_passed": verification["all_passed"], "checks_count": len(checks),
                      "failed": [c for c in checks if not c["passed"]], "resource": verification["resource"]}), flush=True)
    if not verification["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
