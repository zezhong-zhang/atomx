"""Explicit adapters for recorded native JSON schemas; unsupported schemas reject."""

from dataset_builder import identity

WORKSPACE_PATH = "docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol005_projected_increment/orbitals_workspace.npz"


def fixed(contract):
    for key in ("fixed_conditions", "fixed"):
        if key in contract:
            return contract[key]
    raise ValueError("Unsupported scientific contract schema")


def input_sha(contract, path):
    if path in contract.get("inputs_sha256", {}):
        return contract["inputs_sha256"][path]
    for group in ("input_roles", "inputs"):
        for ref in contract.get(group, {}).values():
            if isinstance(ref, dict) and ref.get("path") == path:
                return ref["sha256"]
    return None


def selected_stage(result, name):
    matches = [s for s in result["stages"] if s.get("name") == name and "rows" in s]
    if len(matches) != 1:
        raise ValueError("Ambiguous or missing whole saved model")
    return matches[0]


def archive(result, name):
    matches = [a["sha256"] for a in result.get("completed_archives", []) if a.get("name") == name]
    return matches[0] if len(matches) == 1 else None


def owner_summary(header):
    return {"representation": header.get("representation"), "units": header.get("units"),
            "whole_header_canonical_sha256": identity(header),
            "radial_geometry_sha256": header.get("radial_binding", {}).get("geometry"),
            "sampled_nucleus_sha256": header.get("radial_binding", {}).get("nuclear")}


def model_identity(source, f, contract, anchor, owner, nradial, ncsf, operator="no-pair Dirac-Coulomb", variant=None):
    return {
        "source_map_sha256": source, "ion_Z": f.get("Z"),
        "electron_count": f.get("electron_count", f.get("electrons")),
        "nucleus": {"declared_profile": f.get("nucleus"),
                    "native_workspace_archive_sha256": input_sha(contract, WORKSPACE_PATH)},
        "operator": operator,
        "radial_representation": {"kind": "native-DBSR-B-spline", "ns": f.get("ns"),
                                  "no_pair": "unchanged original nuclear electronic P"},
        "point": {"anchor_sha256": anchor, "radial_variant": variant},
        "owner": owner,
        "space": {"n_radial": nradial, "n_target_CSF": ncsf},
    }


def extract_pair(spec, loaded):
    result, contract, receipt = loaded["result"], loaded["contract"], loaded["receipt"]
    f = fixed(contract)
    reasons, failed = [], [c.get("name", "unnamed") for c in result.get("checks", []) if not c["passed"]]
    closed = (type(receipt.get("exit_code")) is int and
              (receipt.get("status") == "foreground_closed" or "log_sha256" in receipt))
    if not closed:
        reasons.append("missing_closed_foreground_receipt")
    elif receipt["exit_code"] != 0:
        reasons.append("original_executor_failed")
    if result.get("contract_sha256") != spec["roles"]["contract"]["sha256"]:
        reasons.append("result_contract_identity_mismatch")
    proof_passed = loaded["proof"].get(spec["proof_pass_field"]) is True
    if not proof_passed:
        reasons.append("saved_proof_not_passed")
    proof = loaded["proof"]
    for key in ("bound_input_sha256", "inputs_sha256"):
        bindings = proof.get(key, {})
        if not isinstance(bindings, dict):
            continue
        result_ref = spec["roles"]["result"]
        if result_ref["path"] in bindings and bindings[result_ref["path"]] != result_ref["sha256"]:
            raise ValueError("Saved proof binds another result")
    if failed:
        reasons.append("original_numerical_gate_failed")
    anchor = spec["point_group_anchor_sha256"]
    source = result.get("source_map_sha256", f.get("source_map_sha256"))
    mode = spec["adapter"]
    cost = {"scope": spec["cost_scope"], "controlled_timing": False,
            "incremental_cost_isolated": False, "elapsed_uncontrolled_s": result.get("elapsed_uncontrolled_s"),
            "peak_RSS_MiB": result.get("peak_RSS_MiB"),
            "primitive_calls": result.get("completed", result.get("calls"))}
    if mode == "legacy_stages":
        old = selected_stage(result, "baseline")
        new = selected_stage(result, spec["after_model"])
        nold = len(f["radial_keys"])
        nnew = nold + len(f.get("new_virtuals", {}).get("new_keys", []))

        def endpoint(stage, nr):
            owner = {"kind": "legacy_saved_complete_H_and_signed_CI",
                     "archive_sha256": stage.get("CI_sha256"), "H_archive_sha256": stage.get("H_sha256"),
                     "signed_basis_sha256": stage.get("signed_basis_sha256"),
                     "tracking_sha256": stage.get("tracking_sha256"), "units": "Hartree"}
            return {"rows": stage["rows"], "identity": model_identity(source, f, contract, anchor, owner,
                nr, stage.get("retained_target_CSFs"), variant=result.get("virtual_control_archive_sha256"))}

        before, after = endpoint(old, nold), endpoint(new, nnew)
        passed = result["numeric_identity_gates_passed"]
    elif mode in ("screened_f", "failed_nuclear_f"):
        old, new = loaded["before_model"], loaded["after_model"]

        def endpoint(model, role):
            owner = {"kind": "real_native_CI_owner_metadata", "header_sha256": spec["roles"][role]["sha256"],
                     "archive_sha256": archive(result, model["name"] + "_whole_CI.npz"),
                     "summary": owner_summary(loaded[role])}
            return {"rows": model["rows"], "identity": model_identity(source, f, contract, anchor, owner,
                model["radial_orbitals"], model["complete_target_CSFs"], variant=owner["header_sha256"])}

        before, after = endpoint(old, "before_owner"), endpoint(new, "after_owner")
        passed = result.get("all_original_numeric_identity_gates_passed", False)
    elif mode == "static531":
        old_owner = {"kind": "previous_same_point_DC_H_CI_archive", "archive_sha256": input_sha(contract, f["endpoint_arrays"])}
        new_owner = {"kind": "complete_static_H_CI_archive", "archive_sha256": result["arrays_sha256"]}
        before = {"rows": result["DC_rows"], "identity": model_identity(source, f, contract, anchor, old_owner,
            f["radial_orbitals"], f["CSFs"])}
        after = {"rows": result["static_rows"], "identity": model_identity(source, f, contract, anchor, new_owner,
            f["radial_orbitals"], f["CSFs"], operator="no-pair DC + full static Breit (Gaunt + zero-frequency retardation)")}
        passed = result["static_RCI_numeric_integrity_passed"]
        cost["elapsed_uncontrolled_s"] = result["timings_uncontrolled_s"]
    elif mode == "kernel_expectation":
        old_owner = {"kind": "frozen_CI_expectation_owner", "header_sha256": spec["roles"]["point_owner"]["sha256"],
                     "summary": owner_summary(loaded["point_owner"])}
        new_owner = {"kind": "same_frozen_CI_changed_radial_operator_expectation",
                     "header_sha256": spec["roles"]["point_owner"]["sha256"],
                     "expectation_result_sha256": spec["roles"]["result"]["sha256"]}
        old_source, new_source = result["source_maps"]["ordinary"], result["source_maps"]["candidate"]
        before = {"rows": result["targets"], "identity": model_identity(old_source, f, contract, anchor,
            old_owner, f["radial_keys"], f["complete_selected_CSFS"], operator=f["operator"])}
        after = {"rows": result["targets"], "identity": model_identity(new_source, f, contract, anchor,
            new_owner, f["radial_keys"], f["complete_selected_CSFS"], operator=f["operator"])}
        passed = result["all_declared_discriminator_controls_passed"]
        cost["elapsed_uncontrolled_s"] = receipt["elapsed_uncontrolled_seconds"]
        cost["primitive_calls"] = result["calls_completed"]
    elif mode == "trajectory":
        old, new = loaded["before_model"], loaded["after_model"]
        f = dict(f, Z=old["owner"]["radial_binding"]["Z"], ns=old["owner"]["radial_binding"]["ns"])

        def endpoint(model):
            owner = {"kind": "accepted_real_CI_checkpoint", "archive_sha256": model["array_sha256"],
                     "header_sha256": identity(model["owner"]), "summary": owner_summary(model["owner"])}
            return {"rows": model["rows"], "identity": model_identity(model["source_map_sha256"], f,
                contract, model["array_sha256"], owner, f["radial_orbitals"], f["target_CSFS"], variant=model["iteration"])}

        before, after = endpoint(old), endpoint(new)
        # Counts are explicit in the scientific model contract, not NIST or outcome-derived.
        for endpoint_value in (before, after):
            endpoint_value["identity"]["space"]["n_radial"] = f.get("radial_orbitals")
            endpoint_value["identity"]["space"]["n_target_CSF"] = f["target_CSFS"]
        passed = result["all_completed_controls_passed"]
        reasons.append("correlated_optimization_trajectory_not_independent_physics_example")
    else:
        raise ValueError("Unsupported explicitly registered evidence adapter")
    if not passed:
        reasons.append("original_numeric_integrity_not_passed")
    stationarity = {"before": result.get("original_stationarity_passed", result.get("DC_original_stationarity_passed")),
                    "after": "not_evaluated" if mode != "trajectory" else loaded["after_model"]["original_stationarity_passed"],
                    "scope": "Original metric/threshold status; not an energy or spectrum error bound"}
    return before, after, {
        "paired_provenance": closed and proof_passed,
        "numerical_gates_passed": passed and not failed,
        "failed_gates": failed,
        "stationarity": stationarity,
        "unsuitable_reasons": reasons,
    }, cost
