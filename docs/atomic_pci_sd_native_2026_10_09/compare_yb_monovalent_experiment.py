"""Frozen nominal Yb II level comparison, only after all full-space gates pass.

This diagnostic uses the existing atomx unit constant and a single computed
ground reference. It does not supply large SD physical states or line strengths.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomx.constants import DEFAULT_CODATA_VERSION, HARTREE_CM1  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    run, out = OWNER / args.run_name, OWNER / args.reference_name
    if (out / "comparison.json").exists():
        raise ValueError("retain completed comparison; use a new reference increment")
    reference_file = out / "reference.json"
    reference = json.loads(reference_file.read_text())
    contract = reference["matching_contract"]
    if args.run_name != contract["computed_run"]:
        raise ValueError("predeclared reference/run matching differs")
    if (run / "failure.json").exists():
        raise ValueError("failed complete calculation cannot certify a comparison")
    if contract.get("require_numeric_capacity_audits"):
        if (run / "numeric64_failure.json").exists():
            raise ValueError(
                "failed numerical-capacity execution cannot certify comparison"
            )
        numeric = json.loads((run / "numeric64_result.json").read_text())
        if numeric["status"] != "pass" or not numeric["source_match"]:
            raise ValueError("complete numerical64 execution/source audit required")
    if contract.get("require_gpu_execution_audits"):
        if (run / "gpu_failure.json").exists():
            raise ValueError("failed GPU execution cannot certify a comparison")
        gpu_result = json.loads((run / "gpu_result.json").read_text())
        gpu_sources = json.loads((run / "gpu_source_postcheck.json").read_text())
        if gpu_result["status"] != "pass" or gpu_sources["status"] != "pass":
            raise ValueError("complete GPU wrapper and source audits required")
        for key in contract["keys"]:
            audit = json.loads(
                (run / f"cpu_root_recheck_{key[0]}_{key[1]}.json").read_text()
            )
            if (
                audit["status"] != "pass"
                or max(audit["raw_residual_Ha"], audit["energy_residual_Ha"]) > 1e-10
            ):
                raise ValueError("actual complete CPU root recheck required")
    summary = json.loads((run / "result.json").read_text())
    postcheck = json.loads((run / "source_postcheck.json").read_text())
    if (
        len(summary["roots"]) != 5
        or len(summary["controls"]) != 5
        or postcheck["status"] != "pass"
    ):
        raise ValueError("all five source-bound complete map/root gates required")
    if (
        digest(out / reference["sources"]["extraction_file"])
        != reference["sources"]["extraction_sha256"]
    ):
        raise ValueError("frozen experimental extraction changed")
    bindings, roots, payload_hashes = [], {}, {}
    omitted = {"method", "resources", "diagnostics", "restarts"}
    for level in reference["levels"]:
        key = tuple(level["key"])
        prefix = f"root_{key[0]}_{key[1]}"
        path = run / (prefix + ".json")
        root = json.loads(path.read_text())
        if tuple(root["column_key"]) != key:
            raise ValueError("nominal level/actual root key differs")
        residuals = np.array([root["residual_max_Ha"], root["energy_residual_Ha"]])
        if not np.all(np.isfinite(residuals)) or np.max(residuals) > 1e-10:
            raise ValueError("actual complete raw/energy gate failed")
        if len(root["continuation_path"]) != 8:
            raise ValueError("complete reference-connected continuation required")
        if (
            contract.get("require_numeric_capacity_audits")
            and root["provenance"]["resources"]["subspace"] != 64
        ):
            raise ValueError("actual complete numerical-capacity policy differs")
        bindings.append(
            {k: v for k, v in root["provenance"].items() if k not in omitted}
        )
        roots[key] = root
        for suffix in (".json", ".npz"):
            filename = prefix + suffix
            payload_hashes[filename] = digest(run / filename)
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise ValueError("upper/lower physical, source or solver identity differs")
    ground = roots[6, -1]
    gate = float(contract["relative_error_gate"])
    rows = []
    for level in reference["levels"]:
        key = tuple(level["key"])
        root = roots[key]
        excitation = (
            root["attachment_energy_Ha"] - ground["attachment_energy_Ha"]
        ) * HARTREE_CM1
        experiment = level["energy_cm1"]
        error = excitation - experiment
        relative = error / experiment if experiment else None
        rows.append(
            dict(
                key=list(key),
                configuration=level["configuration"],
                twoJ=level["twoJ"],
                parity=level["parity"],
                orbital_energy_Ha=root["orbital_energy_Ha"],
                correlation_energy_Ha=root["correlation_energy_Ha"],
                attachment_energy_Ha=root["attachment_energy_Ha"],
                computed_cm1=excitation,
                experimental_cm1=experiment,
                error_cm1=error,
                relative_error=relative,
                gate_pass=abs(relative) <= gate if relative is not None else None,
                computed_ground_line_vacuum_nm=1e7 / excitation
                if excitation > 0
                else None,
                experimental_ground_line_vacuum_nm=1e7 / experiment
                if experiment > 0
                else None,
                energy_only_multipole=("E1" if level["parity"] == "odd" else "E2")
                if key != (6, -1)
                else None,
            )
        )
    indexed = {tuple(row["key"]): row for row in rows}
    splittings = []
    for label, lower, upper in (
        ("6p", (6, 1), (6, -2)),
        ("5d", (5, 2), (5, -3)),
    ):
        a, b = indexed[lower], indexed[upper]
        actual = b["computed_cm1"] - a["computed_cm1"]
        expected = b["experimental_cm1"] - a["experimental_cm1"]
        splittings.append(
            dict(
                label=label,
                lower=list(lower),
                upper=list(upper),
                computed_cm1=actual,
                experimental_cm1=expected,
                error_cm1=actual - expected,
                relative_error=(actual - expected) / expected,
                gate_pass=abs((actual - expected) / expected) <= gate,
            )
        )
    result = dict(
        status="computed comparison completed; experimental gates recorded separately",
        computed_run=args.run_name,
        reference_sha256=digest(reference_file),
        source_postcheck_sha256=digest(run / "source_postcheck.json"),
        computed_payloads_sha256=payload_hashes,
        comparison_source_sha256=digest(__file__),
        common_physical_and_solver_binding=bindings[0],
        constants=dict(CODATA=DEFAULT_CODATA_VERSION, HARTREE_CM1=HARTREE_CM1),
        rows=rows,
        splittings=splittings,
        excitation_gate_pass=all(
            row["gate_pass"] for row in rows if row["relative_error"] is not None
        ),
        splitting_gate_pass=all(row["gate_pass"] for row in splittings),
        relative_error_gate=gate,
        scope=contract["scope"],
        limitations=contract["unmatched_physics"],
        highest_level_reached=False,
        no_fitted_shift=True,
        no_line_strength_claim=True,
    )
    with (out / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                rows=rows,
                splittings=splittings,
                excitation_gate_pass=result["excitation_gate_pass"],
                splitting_gate_pass=result["splitting_gate_pass"],
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="yb_full_monovalent004")
    parser.add_argument("--reference-name", default="yb_monovalent_experiment001")
    main(parser.parse_args())
