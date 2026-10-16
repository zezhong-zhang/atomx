"""SOURCE/JSON verification; never inspects a numerical archive."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location("source_preparation", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c, paths, _ = module.metadata()
    tree = ast.parse((HERE / "run.py").read_text())
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    check(
        "top_level_imports_stdlib_only",
        all(
            not any(
                (n.name if isinstance(n, ast.alias) else n).split(".")[0]
                in ("numpy", "scipy", "atomx", "numba")
                for n in (
                    node.names if isinstance(node, ast.Import) else [node.module or ""]
                )
            )
            for node in imports
        ),
    )
    check(
        "no_numerical_modules_imported",
        not any(
            k.split(".")[0] in ("numpy", "scipy", "atomx", "numba") for k in sys.modules
        ),
    )
    check("execution_authorization_initially_false", c["execution_authorized"] is False)
    check(
        "root_binding_and_once_actual_directory",
        "execution_authorized_by_root" in (HERE / "run.py").read_text()
        and "output.mkdir(exist_ok=False)" in (HERE / "run.py").read_text(),
    )
    check("six_bound_deferred_arrays", len(c["array_access"]) == 6)
    for name, plan in c["array_access"].items():
        check(
            "whole_declared_header_schema/" + name,
            plan["fields"]
            and all(
                row["dtype"] in ("<f8", "<f16", "<i8")
                and ("shape" in row or "shape_choices" in row)
                for row in dict(plan.get("extras", {}), **plan["fields"]).values()
            ),
        )
    check(
        "workspace18_selected_plus2_header_only",
        len(c["array_access"]["workspace"]["fields"]) == 18
        and set(c["array_access"]["workspace"]["extras"])
        == {"energies_dc_Ha", "mixing_dc"},
    )
    check(
        "reference2_selected_plus_completeH_extra",
        set(c["array_access"]["reference_arrays"]["fields"])
        == {"energies_Ha", "mixing"}
        and set(c["array_access"]["reference_arrays"]["extras"]) == {"matrix_Ha"},
    )
    check(
        "S7_selected_plus27_original_geometry_headers",
        len(c["array_access"]["saved_S"]["fields"]) == 7
        and len(c["array_access"]["saved_S"]["extras"]) == 27,
    )
    check(
        "checkpoint_complete35_header_fields",
        len(c["array_access"]["checkpoint_arrays"]["fields"])
        + len(c["array_access"]["checkpoint_arrays"]["extras"])
        == 35,
    )
    check(
        "all_seven_inverse_blocks_carried",
        all(
            "inverse_blocks_" + str(i)
            in c["array_access"]["checkpoint_arrays"]["fields"]
            for i in range(7)
        ),
    )
    check(
        "full100NR2574CSF31wave_model",
        c["fixed"]["NR_configs"] == 100
        and c["fixed"]["target_CSFs"] == 2574
        and c["fixed"]["radials"] == 31
        and c["fixed"]["coordinates"] == 3689,
    )
    check(
        "expanded_raw_support2659_is_resource_cap",
        c["resources"]["declared_source_work"]["raw_determinants_per_block"] == 2659,
    )
    check("three_complete_gradient_calls", c["resources"]["calls"]["gradient"] == 3)
    check(
        "five_actualnorm_bilinear_RDMs", c["resources"]["calls"]["unfiltered_RDM"] == 5
    )
    check(
        "one_literal_native_RDM",
        c["resources"]["calls"]["retained_RDM"] == 1
        and c["fixed"]["native_filter_preserved"] == 1e-15,
    )
    check("raw_h_constructor7_allowed", c["resources"]["calls"]["raw_hS"] == 7)
    check(
        "no_H_CI_eigh_Dirac_optimizer_state",
        all(c["resources"]["calls"][k] == 0 for k in c["forbidden_calls"]),
    )
    check(
        "raw_h_quotient_action_budget_separate",
        c["resources"]["calls"]["raw_h_correction_MV"] >= 5 * 57,
    )
    check(
        "whole_unfiltered_key_support_no_pruning",
        c["resources"]["declared_source_work"]["structural_key_union_upper"] == 70971
        and "exact zero only" in (HERE / "run.py").read_text(),
    )
    check(
        "same_original_metric_and_projectors",
        c["fixed"]["metric_sha256"]
        == "feda81d4cf3349a9a36c7ca4959a9990a70a9cbfb71145711f9f85800075c8c9"
        and c["fixed"]["geometry_sha256"]
        == "0abc1f536a971f6663daa7bb97f42b02ae0025633d9efd2adc6d24f58aa56f79",
    )
    check(
        "literal_native_tracker_order_is_preflighted",
        c["fixed"]["tracker_native_UID_order"]
        == [[1, 1, 0], [5, 1, 0], [3, 1, 0], [3, -1, 0], [1, -1, 0]],
    )
    check("archive_has_no_owned_snapshot", c["fixed"]["expanded_legacy_owner"] is None)
    check(
        "no_ordinary_FD_gate_upgrade",
        c["gates"]["ordinary_FD"]
        == {"absolute": 1e-8, "relative": 1e-5, "not_executed": True},
    )
    check(
        "source_seal_all356_files",
        len(module.read(paths["source_identity"])["all_atomx_python_sha256"]) == 356,
    )
    check(
        "bounded_resources_and_decoding",
        c["resources"]["AS_GiB"] == 8
        and c["resources"]["inner_seconds"] == 600
        and c["resources"]["outer_seconds"] == 615
        and c["resources"]["decoded_bytes_max"] == 256 * 1024**2,
    )
    check(
        "one_thread_literal_interpreter",
        c["resources"]["threads"] == 1
        and c["resources"]["interpreter_literal"]
        == "/data/code/atomx/.venv/bin/python",
    )
    check(
        "header_producer_sources_bound",
        all(
            k in c["input_roles"]
            for k in [
                "native_source_driver",
                "workspace_producer",
                "reference_producer",
                "basis_source",
                "S_producer",
            ]
        ),
    )
    headers = c["header_source_bindings"]
    for binding in headers:
        source = module.ROOT / c["input_roles"][binding["role"]]["path"]
        lines = source.read_text().splitlines(keepends=True)
        check(
            "bound_header_producer_excerpt/" + binding["role"],
            hashlib.sha256(
                "".join(lines[binding["start"] - 1 : binding["stop"]]).encode()
            ).hexdigest()
            == binding["sha256"],
        )
    check(
        "result_scope_no_line_prediction",
        "individual-line derivative" in c["result_scope"],
    )
    result = {
        "status": "SOURCE_only_complete_preparation",
        "all_SOURCE_checks_passed": all(r["passed"] for r in checks),
        "checks": checks,
        "run_sha256": module.sha(HERE / "run.py"),
        "contract_sha256": module.sha(HERE / "contract.json"),
        "NPZ_stat_hash_header_body_access": 0,
        "numerical_imports": 0,
        "science_calls": 0,
        "actual_headers_not_yet_observed": True,
        "execution_authorized": False,
    }
    (HERE / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(checks),
                "all_SOURCE_checks_passed": result["all_SOURCE_checks_passed"],
                "NPZ_access": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
