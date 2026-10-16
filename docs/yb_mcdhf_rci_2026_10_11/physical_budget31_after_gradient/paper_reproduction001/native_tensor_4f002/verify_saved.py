"""Persisted-source, paired-work and full4f projected-equation readback."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from angular import AngularCoordinates
from solve_and_attach import load_operator

from atomx.constants import HARTREE_CM1
from atomx.methods.tensor_coupled_cluster import TensorCoreResult

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / "actual003/result.json").read_text())
    contract = json.loads((HERE / "davidson_J_contract.json").read_text())
    checks = []

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        checks.append(dict(name=name, measured=measured, cap=cap, passed=bool(passed)))
        assert passed, checks[-1]

    for epoch in ("before", "after"):
        source = json.loads((HERE / (epoch + "_source.json")).read_text())
        with tarfile.open(HERE / (epoch + "_source.tar.gz")) as a:
            gate(
                epoch + ":all_archive_members",
                set(a.getnames()) == set(source["files"]),
            )
            gate(
                epoch + ":all_source_archive_bytes",
                all(
                    hashlib.sha256(a.extractfile(n).read()).hexdigest() == h
                    for n, h in source["files"].items()
                ),
            )
        if epoch == "after":
            gate(
                "all_current_source_bytes",
                all(sha(ROOT / n) == h for n, h in source["files"].items()),
            )
    gate(
        "solve_contract_binding",
        report["contract_sha256"] == sha(HERE / "davidson_J_contract.json"),
    )
    gate(
        "all_solve_inputs",
        all(sha(ROOT / n) == h for n, h in contract["inputs"].items()),
    )
    gate(
        "actual_completed",
        report["status"] == "PASS_FULL4F_TARGETED_DAVIDSON",
    )
    gate("all_live_controls", all(x["passed"] for x in report["controls"]))
    for name in ("solve_contract.json", "davidson_contract.json"):
        original = json.loads((HERE / name).read_text())
        gate(
            name + ":original_inputs_preserved",
            all(sha(ROOT / n) == h for n, h in original["inputs"].items()),
        )
    first = json.loads((HERE / "actual001/result.json").read_text())
    gate(
        "original_SR_cap_failure_retained",
        first["status"] == "FAILED"
        and "predeclared matrix-free EA action limit" in first["exception"],
    )
    second = json.loads((HERE / "actual002/result.json").read_text())
    gate(
        "unprojected_small_J_failure_retained",
        second["status"] == "FAILED" and "D3:basis_J2" in second["exception"],
    )
    reporting = json.loads((HERE / "reporting_reference_binding.json").read_text())
    gate(
        "historical_reporting_result",
        sha(ROOT / reporting["path"])
        == reporting["sha256"]
        == reporting["owner_declared_result_sha256"],
    )
    gate(
        "historical_reporting_receipt",
        sha(ROOT / reporting["owner_receipt"]) == reporting["owner_receipt_sha256"],
    )
    perf = json.loads((HERE / "performance_contract.json").read_text())
    gate(
        "paired_input_bindings",
        all(sha(HERE / n) == h for n, h in perf["inputs"].items()),
    )
    maximum = 0.0
    with np.load(HERE / "before_values.npz") as reference:
        for label in ("A1", "B1", "B2", "A2"):
            row = json.loads((HERE / (label + ".json")).read_text())
            gate(label + ":completed", row["status"] == "PASS_MATCHED_CELL")
            with np.load(HERE / (label + "_values.npz")) as z:
                for key in ("core_residual", "EA_action"):
                    error = float(np.max(abs(z[key] - reference[key])))
                    maximum = max(maximum, error)
                    gate(label + ":all_values:" + key, error, 1e-10)
    op, ledger = load_operator(HERE, 18)
    with np.load(HERE / "actual001/core.npz") as z:
        amplitudes = z["amplitudes"]
        new_channel = z["new_channel"]
    residual = op.core_residual(amplitudes)
    gate("all189834_SDT_equations", np.max(abs(residual)), 1e-10)
    gate(
        "core_energy_ledger",
        abs(op.core_energy(amplitudes) - report["core"]["energy_relative_Ha"]),
        1e-12,
    )
    gate("new4f_amplitudes_nonzero", np.linalg.norm(amplitudes[new_channel]) > 1e-8)
    active = op.scalar_core_indices()
    outside = np.setdiff1d(np.arange(op.core.size), active)
    gate("exact_zero_forbidden_core_amplitudes", np.all(amplitudes[outside] == 0))
    angular_core = AngularCoordinates(op, False)
    gate("core_signed_Jplus", np.linalg.norm(angular_core.ladder(amplitudes, 1)), 1e-7)
    core = TensorCoreResult(
        amplitudes,
        op.core_energy(amplitudes),
        op.core_energy(amplitudes),
        float(np.max(abs(residual))),
        0,
        op.operator_sha256,
    )
    angular = AngularCoordinates(op, True)
    root_errors = []
    for name, m2 in (("S", 1), ("D3", 3), ("D5", 5)):
        with np.load(HERE / "actual003" / (name + "_root.npz")) as z:
            indices, vector, energy = z["indices"], z["vector"], float(z["energy"])
        gate(
            name + ":complete_highest_M_coordinates",
            np.array_equal(indices, angular.sector(m2)),
        )
        gate(
            name + ":recorded_energy",
            abs(energy - report["targets"][name]["EA_Ha"]),
            1e-12,
        )
        error = float(np.max(abs(op.attachment_action(core, vector) - energy * vector)))
        root_errors.append(error)
        gate(name + ":all_original_R3_root_equations", error, 1e-10)
    with np.load(HERE / "actual003/all_target_partners.npz") as z:
        gate("all12_target_partner_arrays", len(z.files) == 12)
        for name in ("S", "D3", "D5"):
            j = {"S": 0.5, "D3": 1.5, "D5": 2.5}[name]
            energy = report["targets"][name]["EA_Ha"]
            keys = [key for key in z.files if key.startswith(name + "_M")]
            gate(name + ":all_magnetic_partners", len(keys) == int(2 * j + 1))
            for key in keys:
                m2 = int(key.split("_M")[1])
                vector = z[key]
                gate(key + ":canonical_R_norm", abs(np.linalg.norm(vector) - 1), 1e-8)
                gate(
                    key + ":exact_other_M_parity_zero",
                    np.all(vector[(angular.m2 != m2) | (angular.parity != 0)] == 0),
                )
                gate(
                    key + ":signed_J2",
                    np.linalg.norm(
                        angular.j_squared(vector, m2 / 2) - j * (j + 1) * vector
                    ),
                    1e-7,
                )
                gate(
                    key + ":all_R3_equations",
                    np.max(abs(op.attachment_action(core, vector) - energy * vector)),
                    1e-10,
                )
    lines = {
        name: (report["targets"][name]["EA_Ha"] - report["targets"]["S"]["EA_Ha"])
        * HARTREE_CM1
        for name in ("D3", "D5")
    }
    for name in lines:
        gate(
            "same_state_line:" + name,
            abs(lines[name] - report["lines_cm1"][name]),
            1e-9,
        )
    output = dict(
        status="PASS_SAVED_FULL4F_PROJECTED_REVIEW",
        controls=checks,
        paired_all_vector_difference_Ha=maximum,
        maximum_three_target_root_equation_Ha=max(root_errors),
        lines_cm1=lines,
        scope="ProjectedSDT/R3 equations, exact M/parity, independent signed angular ladders and stored provenance. CanonicalR norms are not physical exp(T) norms; full-H residual and radial/virtual/experimental convergence remain uncomputed.",
    )
    (HERE / "saved_data_review.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                status=output["status"],
                controls=len(checks),
                maximum_root_residual_Ha=max(root_errors),
            )
        )
    )


if __name__ == "__main__":
    main()
