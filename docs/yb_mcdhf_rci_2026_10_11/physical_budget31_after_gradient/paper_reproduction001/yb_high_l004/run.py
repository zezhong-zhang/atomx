"""SOURCE-only preparation of frozen Yb68 complete-g block successor004.

Numerical admission is deliberately unavailable until the parent binds a
tested shared block kernel and the original-span residual-floor decision.
"""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/code/atomx").resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(contract):
    checks = []

    def require(name, value):
        checks.append({"name": name, "PASS": bool(value)})
        if not value:
            raise RuntimeError("preparation failed: " + name)

    previous = contract["frozen_predecessor"]
    for name, binding in previous.items():
        require("frozen003:" + name, sha(HERE / binding["path"]) == binding["sha256"])
    bound = contract["predecessor_header_metadata"]
    require("header_receipt_identity", sha(HERE / bound["path"]) == bound["sha256"])
    headers = json.loads((HERE / bound["path"]).read_text())
    require(
        "header_capsule_identity",
        headers["sha256"] == previous["failed_arrays"]["sha256"],
    )
    old = json.loads((HERE / previous["contract"]["path"]).read_text())
    failure = json.loads((HERE / previous["failure"]["path"]).read_text())
    for label, source in contract["physical_response_proposal_source_bindings"].items():
        require(
            "proposal_source:" + label, sha(ROOT / source["path"]) == source["sha256"]
        )
    for key in (
        "historical_model",
        "paper_difference",
        "inputs",
        "workspace_whole_field_plan",
        "workspace_decoded_bytes_cap",
        "g_partners",
        "energy_band_upper_Ha",
        "gates",
        "resources",
        "original_driver_archive_gap",
    ):
        require("unchanged_model_gate:" + key, contract[key] == old[key])
    require("source_no_execution", contract["numerical_execution_performed"] is False)
    require("no_automatic_dispatch", contract["automatic_dispatch"] is False)
    require("original_Gram_gate", contract["gates"]["full_S_Gram"] == 1e-10)
    require("original_dual_gate", contract["gates"]["band_dual_residual_Ha"] == 1e-8)
    require("complete_fixed_partners", contract["g_partners"] == [4, -5])
    require(
        "prior_fail_preserved",
        failure["status"] == "FAIL"
        and any(
            check["name"] == "full_electron_S_Gram:4" and not check["PASS"]
            for check in failure["checks"]
        ),
    )
    expected = {
        "occupied_keys": ([19, 2], "<i8"),
        "occupied_P": ([19, 123], "<f8"),
        "occupied_Q": ([19, 123], "<f8"),
        "Fock_raw_4": ([246, 246], "<f8"),
        "S_4": ([246, 246], "<f8"),
        "original_full_electron_E_4": ([119], "<f8"),
        "original_full_electron_V_4": ([246, 119], "<f8"),
        "fixed_band_indices_4": ([67], "<i8"),
    }
    require(
        "exact_deferred_eight_fields", set(headers["deferred_decode"]) == set(expected)
    )
    require("whole28_fields", len(headers["whole_field_plan"]) == 28)
    for field, (shape, dtype) in expected.items():
        plan = headers["whole_field_plan"][field]
        require(
            "saved_field:" + field,
            plan
            == dict(
                shape=shape,
                dtype=dtype,
                fortran_order=False,
            ),
        )
    require(
        "failed_refined_block_excluded",
        not set(headers["deferred_decode"])
        & {
            "full_electron_E_4",
            "full_electron_V_4",
        },
    )
    require("one_fresh_Fock", contract["entry_caps"]["Fock"] == 1)
    require("one_fresh_spectrum", contract["entry_caps"]["Dirac_spectrum"] == 1)
    require(
        "bounded_fresh_direct_exchange",
        (
            contract["entry_caps"]["direct_frontend"] == 19
            and contract["entry_caps"]["exchange_frontend_upper"] == 34
        ),
    )
    require(
        "same_8GiB_single_thread600s",
        (
            contract["resources"]["AS_bytes"] == 8 * 1024**3
            and contract["resources"]["threads"] == 1
            and contract["resources"]["inner_seconds"] == 600
        ),
    )
    return {
        "status": "SOURCE_PREPARATION_PASS_NUMERICAL_ADMISSION_PENDING",
        "contract_sha256": sha(HERE / "contract.json"),
        "checks": checks,
        "numerical_imports": 0,
        "array_body_decodes": 0,
        "execution_available": False,
        "unresolved": [
            "source-bound native joint block kernel API and independent tests",
            "source-derived actual kernel eigensolve/linear/peak-work caps",
            "same-original-span full-H outside-span residual-floor decision",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    report = prepare(contract)
    if args.execute:
        raise RuntimeError(
            "004 SOURCE preparation only; no sealed numerical driver/admission"
        )
    print(json.dumps(report, indent=2, allow_nan=False))
