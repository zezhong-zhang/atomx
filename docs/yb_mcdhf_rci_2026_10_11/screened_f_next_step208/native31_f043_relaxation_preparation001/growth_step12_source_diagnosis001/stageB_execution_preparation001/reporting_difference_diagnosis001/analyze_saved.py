"""Bounded saved-JSON scalar bookkeeping; no array or physical operator access."""

import hashlib
import json
import os
import time
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
START = time.monotonic()
EXPECTED = {
    "actual/result.json": "ff4ad89476cbe6136b3f2cea48b0c4dc5dc0912a61ca96d24d1ce32208469255",
    "actual/failure.json": "7d56149359905d5747308879cbea66e2be55572f6b986f20936ad91226026e6c",
    "run.py": "662248e7e0a3a2e6d39516533c9cd743adef825eedce4376a1ff884bd7cdd041",
    "contract.json": "52dba8b6423ed68f5a419dd6652ab0009e22aadc4623cc108df5fa2a89c167a8",
    "preparation.json": "959879ab79113dddffe475bff3766e4852e65e2836a8e3eceacc6160db47a43b",
    "handoff.json": "a6b3c3a4a1aef1f5a21f0089de9ccdf9d2e95a46c7127688c45ee81bb389b648",
}
JSON_NAMES = [
    "root_execution.json", "actual/result.json", "actual/failure.json",
    "actual/derivative_domains.json", "actual/five_point_reported_states.json",
    "actual/baseline_full_signed_Lambda.json",
] + [f"actual/h{i}_{sign}_{kind}.json" for i in range(2)
     for sign in ("minus", "plus")
     for kind in ("mixed_finite_change", "centered_diagnostic")]
assert all(not name.endswith(".npz") for name in JSON_NAMES + list(EXPECTED))
assert sum((BASE / name).stat().st_size for name in JSON_NAMES) <= 2 * 1024**2
if 31 in os.sched_getaffinity(0):
    os.sched_setaffinity(0, {31})
inputs = {}
data = {}
for name in dict.fromkeys(JSON_NAMES + list(EXPECTED)):
    raw = (BASE / name).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert name not in EXPECTED or sha == EXPECTED[name], name
    inputs[str((BASE / name).relative_to(BASE.parents[5]))] = sha
    if name in JSON_NAMES:
        data[name] = json.loads(raw)
receipt, result = data["root_execution.json"], data["actual/result.json"]
assert receipt["status"] == "foreground_closed" and receipt["exit_code"] == 1
assert receipt["result_sha256"] == EXPECTED["actual/result.json"]
assert receipt["failure_sha256"] == EXPECTED["actual/failure.json"]
assert receipt["source_and_inputs_unchanged"]
assert len(result["checks"]) == 745 and sum(c["passed"] for c in result["checks"]) == 743
assert not result["all_declared_controls_passed"]


def f(value):
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(value)


def display(value):
    return float(value)


def subtotal(rows, field, index):
    return sum((f(row["weight"]) * f(row[field][index]) for row in rows), Fraction())


states = data["actual/five_point_reported_states.json"]
uid_labels = {tuple(row["UID"]): row["label"] for row in states["baseline"]["rows"]}
assert len(uid_labels) == 5
endpoint = {}
baseline_q_offsets = None
for i in range(2):
    for sign in ("minus", "plus"):
        tag = f"h{i}_{sign}"
        mixed = data[f"actual/{tag}_mixed_finite_change.json"]
        centered = data[f"actual/{tag}_centered_diagnostic.json"]
        assert mixed == centered["mixed_change"]
        rows = mixed["targets"]
        assert {tuple(row["reference_level"]) for row in rows} == set(uid_labels)
        targets = {}
        for row in rows:
            uid = tuple(row["reference_level"])
            assert tuple(row["selected_reference"]) == uid == tuple(row["selected_trial"])
            assert row["weight"] == .2 and row["phase"] in (-1, 1)
            targets[uid_labels[uid]] = {
                "UID": list(uid), "weight": row["weight"], "phase": row["phase"],
                "norm0_minus_one": display(f(row["norm0"]) - 1),
                "norm1_minus_one": display(f(row["norm1"]) - 1),
                "baseline_q_minus_lambda_Ha": row["quotient_minus_reported"][0],
                "trial_q_minus_lambda_Ha": row["quotient_minus_reported"][1],
                "reported_response_terms_Ha": row["reported_terms"],
                "rayleigh_response_terms_Ha": row["rayleigh_terms"],
                "residual_norms_Ha": row["residual_norms"],
            }
        q0 = {label: row["baseline_q_minus_lambda_Ha"] for label, row in targets.items()}
        assert baseline_q_offsets is None or q0 == baseline_q_offsets
        baseline_q_offsets = q0
        per_kappa, totals = {}, {}
        for kappa, block in centered["samepoint_fixedLambda"].items():
            assert block["all_Gram_identities_exact"] and block["no_offdiagonal_or_quadratic_omission"]
            parts = {key: f(value) for key, value in block["exact_signed_Lambda_DeltaGram"].items()}
            assert parts["diagonal"] + parts["offdiagonal"] == parts["linear"] + parts["quadratic"]
            assert parts["linear"] + parts["quadratic"] == parts["explicit"] + parts["reconstruction"]
            per_kappa[kappa] = {key: display(value) for key, value in parts.items()}
            for key, value in parts.items():
                totals[key] = totals.get(key, Fraction()) + value
        normal = f(centered["fixedLambda_DeltaGram"])
        assert totals["diagonal"] + totals["offdiagonal"] == normal
        weighted_lambda = sum((f(row["weight"]) * f(row["energy_Ha"])
                               for row in states[tag]["rows"]), Fraction())
        eo_offset = f(states[tag]["EOL_reported_Ha"]) - weighted_lambda
        weighted_q_offset = subtotal(rows, "quotient_minus_reported", 1)
        endpoint[tag] = dict(
            scalar_ledger={key: mixed[key] for key in (
                "explicit_radial", "reconstruction_radial", "reported_response", "rayleigh_response",
                "reported_change", "rayleigh_change", "direct_reported_root_difference",
                "direct_retained_quotient_difference", "reported_closure_error", "quotient_closure_error",
                "floating_EO_difference", "floating_averaging_offset")},
            fixed_signed_Lambda_DeltaGram_Ha=display(normal),
            signed_Lambda_DeltaGram_by_kappa_Ha=per_kappa,
            signed_Lambda_DeltaGram_total_parts_Ha={key: display(value) for key, value in totals.items()},
            centered_rayleigh_change_Ha=display(f(centered["centered_rayleigh_change"])),
            EO_minus_exact_weighted_reported_roots_Ha=display(eo_offset),
            exact_weighted_exported_q_minus_reported_Ha=display(weighted_q_offset),
            exported_weighted_q_minus_EO_Ha=display(weighted_q_offset - eo_offset),
            targets=targets,
            weighted_reported_response_components_Ha=[display(subtotal(rows, "reported_terms", j)) for j in range(4)],
            weighted_rayleigh_response_components_Ha=[display(subtotal(rows, "rayleigh_terms", j)) for j in range(3)],
        )

base_rows = states["baseline"]["rows"]
base_weighted_lambda = sum((f(row["weight"]) * f(row["energy_Ha"]) for row in base_rows), Fraction())
base_eo_offset = f(states["baseline"]["EOL_reported_Ha"]) - base_weighted_lambda
base_q_offset = sum((f(row["weight"]) * f(baseline_q_offsets[row["label"]])
                     for row in base_rows), Fraction())
central = {}
for i, h in enumerate((1e-5, 5e-6)):
    minus, plus = endpoint[f"h{i}_minus"], endpoint[f"h{i}_plus"]
    denominator = 2 * f(h)

    def cd(key, plus=plus, minus=minus, denominator=denominator):
        return (f(plus["scalar_ledger"][key]) - f(minus["scalar_ledger"][key])) / denominator

    normal = (f(data[f"actual/h{i}_plus_centered_diagnostic.json"]["fixedLambda_DeltaGram"])
              - f(data[f"actual/h{i}_minus_centered_diagnostic.json"]["fixedLambda_DeltaGram"])) / denominator
    closure = -cd("quotient_closure_error")
    root_q = cd("direct_reported_root_difference") - cd("direct_retained_quotient_difference")
    averaging = cd("floating_averaging_offset")
    domains = data["actual/derivative_domains.json"][str(h)]
    ordinary = domains["ordinary_reported"]
    centered_derivative = f(domains["independent_centered"][0]["derivative"])
    split = normal + closure + root_q + averaging
    residual = f(ordinary["derivative"]) - centered_derivative - split
    assert abs(float(residual)) < 1e-16
    assert not ordinary["passed"] and all(d["passed"] for d in domains["independent_centered"])
    central[str(h)] = dict(
        ordinary_reported=ordinary,
        centered_derivative_Ha=display(centered_derivative),
        centered_errors_Ha=[display(f(d["error"])) for d in domains["independent_centered"]],
        ordinary_minus_centered_terms_Ha=dict(
            signed_Lambda_DeltaGram=display(normal), negative_mixed_Q_vs_stored_Q_closure=display(closure),
            reported_minus_retained_quotient=display(root_q), EO_averaging=display(averaging)),
        four_term_sum_Ha=display(split), exact_saved_scalar_identity_remainder_Ha=display(residual),
        components_Ha={key: display(cd(key)) for key in (
            "explicit_radial", "reconstruction_radial", "rayleigh_response", "reported_response",
            "rayleigh_change", "direct_retained_quotient_difference", "direct_reported_root_difference")},
        signed_Gram_parts_central_Ha={key: (plus["signed_Lambda_DeltaGram_total_parts_Ha"][key]
             - minus["signed_Lambda_DeltaGram_total_parts_Ha"][key]) / (2 * h)
             for key in plus["signed_Lambda_DeltaGram_total_parts_Ha"]},
        target_quotient_minus_reported_central_Ha={label:
            float((f(plus["targets"][label]["trial_q_minus_lambda_Ha"])
                   - f(minus["targets"][label]["trial_q_minus_lambda_Ha"])) / denominator)
            for label in uid_labels.values()},
        weighted_response_component_central_Ha={kind: [
            (plus[kind][j] - minus[kind][j]) / (2 * h) for j in range(len(plus[kind]))]
            for kind in ("weighted_reported_response_components_Ha", "weighted_rayleigh_response_components_Ha")},
    )

assert time.monotonic() - START < 30
output = dict(
    status="saved_JSON_reporting_difference_decomposed_original_FAIL_preserved",
    scope="four closed mixed finite ledgers and exact saved scalar/rational bookkeeping only",
    inputs_sha256=inputs, analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    bounds=dict(json_files=14, json_input_bytes=sum((BASE / n).stat().st_size for n in JSON_NAMES),
                maximum_json_bytes=2 * 1024**2, maximum_elapsed_s=30, actual_affinity=sorted(os.sched_getaffinity(0))),
    baseline=dict(reported_EO_Ha=states["baseline"]["EOL_reported_Ha"],
        EO_minus_exact_weighted_reported_roots_Ha=display(base_eo_offset),
        exact_weighted_exported_q_minus_reported_Ha=display(base_q_offset),
        exported_weighted_q_minus_EO_Ha=display(base_q_offset - base_eo_offset),
        q_minus_lambda_per_target_Ha=baseline_q_offsets),
    endpoint=endpoint, central=central,
    observed_largest_contribution="mixed radial polynomial versus retained-H quotient closure central difference",
    closure_producer_cause_resolved=False,
    ordinary_FD_passed=False, independent_centered_four_comparisons_passed=True,
    production_objective_or_callback_changed=False, global_bound=False,
    new_physical_or_array_calls=dict(NPZ_reads=0, NPZ_hashes=0, H=0, CI=0, gradient=0,
                                      Dirac=0, eigensolver=0, optimizer=0),
    limitations=["Exported longdouble quotient offsets are saved residual diagnostics, not a new exact stored-H replay.",
                 "The closure mismatch is measured between two saved operator-evaluation routes; its cause is not established.",
                 "Signed Gram contractions retain all offdiagonal/quadratic terms; no new radial Gram arrays are reconstructed.",
                 "No tolerance, h, model, weight, source, old failure, or optimization decision is changed."],
    elapsed_scalar_analysis_s=time.monotonic() - START,
)
(OUT / "result.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": output["status"], "result_sha256": hashlib.sha256((OUT / "result.json").read_bytes()).hexdigest(),
                  "elapsed_s": output["elapsed_scalar_analysis_s"], "scientific_calls": 0}))
