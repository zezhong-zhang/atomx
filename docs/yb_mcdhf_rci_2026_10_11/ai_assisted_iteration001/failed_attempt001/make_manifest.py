"""Freeze explicit CLOSED JSON catalog references; never open numerical archives."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = "docs/yb_mcdhf_rci_2026_10_11/"


def reference(relative):
    path = REPO / CAMPAIGN / relative
    if path.suffix != ".json":
        raise ValueError("Metadata references must be JSON")
    return {"path": str(path.relative_to(REPO)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(relative):
    return json.loads((REPO / CAMPAIGN / relative).read_bytes())


def spec(name, adapter, physics_class, directory, proof, proof_field, anchor, scope):
    result_file = {"screened_f": "actual/result.json", "failed_nuclear_f": "actual/failure.json",
                   "kernel_expectation": "actual/result.json", "trajectory": "actual/result.json"}.get(adapter, "result.json")
    return {
        "id": name, "family": "physics_contribution", "physics_class": physics_class,
        "adapter": adapter, "point_group_anchor_sha256": anchor,
        "proof_pass_field": proof_field,
        "scope": scope, "cost_scope": "Whole recorded discriminator, including baseline/control/setup/archive; not isolated increment timing",
        "roles": {"result": reference(directory + "/" + result_file),
                  "contract": reference(directory + "/contract.json"),
                  "receipt": reference(directory + "/root_execution.json"), "proof": reference(proof)},
    }


def main():
    records = []
    for directory, model, physics_class in (
        ("rank_only_discriminator", "rank_5p_double", "core_5p_double_hole_space"),
        ("virtual_only_discriminator", "balanced_virtual", "virtual_sdp_layer_nuclear_seed"),
        ("screened_core_virtual_discriminator002", "balanced_virtual", "virtual_sdp_layer_closed68_seed"),
    ):
        result = read(directory + "/result.json")
        record = spec(directory, "legacy_stages", physics_class, directory,
            directory + "/saved_array_audit/verification.json", "all_independent_saved_array_checks_passed",
            result["checkpoint_arrays_sha256"], result["scope"])
        record["after_model"] = model
        records.append(record)
    directory = "public531_endpoint_static_rci"
    contract = read(directory + "/contract.json")
    anchor = contract["inputs_sha256"][contract["fixed"]["endpoint_arrays"]]
    records.append(spec("static_Breit_fixed160", "static531", "complete_static_Breit", directory,
        directory + "/saved_array_verification.json", "all_passed", anchor,
        "Same exact accepted160 full531 all16-block static Gaunt+zero-frequency retardation versus DC; not reoptimized"))
    directory = "screened_f_next_step208/execution001"
    result = read(directory + "/actual/result.json")
    record = spec("screened_f_fixed208", "screened_f", "virtual_f_layer_closed68_seed", directory,
        directory + "/completion.json", "all_actual_live_controls_passed", result["checkpoint_arrays_sha256"],
        "Literal208 unchanged old29 plus two screened5f orbitals, full1046 versus1595 target space; no31 orbital optimization")
    record["roles"]["result"] = reference(directory + "/actual/result.json")
    for role, filename in (("before_model", "baseline_same_point_evidence"), ("after_model", "complete5f_same_point_evidence"),
                           ("before_owner", "baseline_real_owner"), ("after_owner", "complete5f_real_owner")):
        record["roles"][role] = reference(directory + "/actual/" + filename + ".json")
    records.append(record)
    directory = "complete5f_fixed208_import_repair_preparation"
    contract = read(directory + "/contract.json")
    anchor = contract["input_roles"]["literal208_arrays"]["sha256"]
    proof = directory + "/completion_counter_audit/saved_array_qualification_preparation/output_serialization_repair/actual/verification.json"
    record = spec("nuclear_f_fixed208_failed_executor", "failed_nuclear_f", "virtual_f_layer_nuclear_seed", directory,
        proof, "saved_array_qualification_passed", anchor,
        "Literal208 old29 plus two unscreened nuclear5f orbitals; finite outputs retained but original final hS-count gate FAILED")
    record["roles"]["result"] = reference(directory + "/actual/failure.json")
    for role, filename in (("before_model", "baseline_same_point_evidence"), ("after_model", "complete5f_same_point_evidence"),
                           ("before_owner", "baseline_real_owner"), ("after_owner", "complete5f_real_owner")):
        record["roles"][role] = reference(directory + "/actual/" + filename + ".json")
    records.append(record)
    directory = "rank_aware_cell_discriminator_preparation"
    contract = read(directory + "/contract.json")
    record = spec("positive_inner_Gauss_order_fixed_CI", "kernel_expectation", "positive_inner_quadrature_operator_expectation", directory,
        directory + "/completion.json", "all_completion_metadata_checks_passed", contract["inputs"]["saved_point"]["sha256"],
        "Complete affected7398 radial keys, exact normalized five target signed tables at fixed recovered29 point; changed-operator expectation only, no new H/CI/g")
    record["roles"]["result"] = reference(directory + "/actual/result.json")
    record["roles"]["point_owner"] = dict(contract["inputs"]["saved_point_owner"])
    record["declared_source_change"] = True
    record["interpretation"] = "Changed radial operator expectation on the SAME archived CI; not eigenenergy or a reoptimized/red diagonalized spectrum"
    records.append(record)
    directory = "screened_f_next_step208/native31_f043_relaxation_preparation001"
    first = read(directory + "/actual/checkpoints/accepted_0000.json")
    record = spec("native31_16step_trajectory_aggregate", "trajectory", "orbital_relaxation_trajectory", directory,
        directory + "/closure001/completion.json", "all_scalar_completion_controls_passed", first["array_sha256"],
        "One whole correlated16-step optimization trajectory; checkpoints are not independent physics examples")
    record["family"] = "optimization_trajectory"
    record["roles"]["result"] = reference(directory + "/actual/result.json")
    record["roles"]["before_model"] = reference(directory + "/actual/checkpoints/accepted_0000.json")
    record["roles"]["after_model"] = reference(directory + "/actual/checkpoints/accepted_0016.json")
    records.append(record)
    manifest = {"schema_version": 1, "unit_conversion": reference("rank_aware_cell_discriminator_preparation/contract.json"),
                "experiments": records,
                "selection_policy": "Explicit closed paired metadata catalog; signed bad/worse outcomes retained; no NIST outcome selection",
                "excluded_future_or_incomplete_scopes": [
                    "Current mutable source and unclosed experiments",
                    "All individual correlated optimization checkpoints as independent training examples",
                    "Derivative/owner/performance qualification probes as independent physics increments",
                    "Historical CV ledger without a separately recorded foreground-closed receipt in its directory; no fabricated closure",
                ]}
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"experiments": len(records), "manifest_sha256": hashlib.sha256((HERE / "manifest.json").read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
