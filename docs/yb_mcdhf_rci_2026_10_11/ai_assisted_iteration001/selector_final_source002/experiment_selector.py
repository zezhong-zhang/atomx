"""Advisory, group-validated GP prototype for paired physical contributions.

Reads an evidence ledger; never dispatches jobs, chooses CSFs, changes physical
gates, or consumes NIST labels. NumPy is imported only after data admission.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

POLICY = {
    "version": 1,
    "JSON_bytes_max": 64 * 1024**2,
    "minimum_independent_groups": 6,
    "maximum_rows": 256,
    "maximum_groups": 32,
    "feature_names": (
        "n_radial_before",
        "n_radial_after",
        "n_target_CSF_before",
        "n_target_CSF_after",
    ),
    "kernel_length_scale": 1.0,
    "kernel_noise": 1e-5,
    "required_group_cv_MAE_ratio": 0.9,
    "scope": "Prototype admission policy, not physical tolerances or a precision certificate",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def verify_json_inputs(ledger, repo):
    """Verify explicit JSON provenance before a CLI decision; never follow NPZ refs."""
    root = Path(repo).resolve()
    bindings = ledger["inputs_sha256"]
    if not bindings or len(bindings) > 64:
        raise ValueError("complete bounded JSON provenance required")
    total = 0
    for name, expected in bindings.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".json":
            raise ValueError("provenance must be repository-local JSON only")
        total += path.stat().st_size
        if total > POLICY["JSON_bytes_max"]:
            raise ValueError("JSON provenance byte cap")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("bound JSON provenance changed")
    return total


def features(record):
    """Whitelist physical counts; experimental targets cannot become features."""
    values = record["predictor_features"]
    unexpected = (
        set(values) - set(POLICY["feature_names"]) - {"ion_Z", "electron_count"}
    )
    if unexpected:
        raise ValueError("unregistered predictor features: " + str(sorted(unexpected)))
    result = []
    for name in POLICY["feature_names"]:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("missing or nonnumeric physical feature: " + name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("positive finite physical feature required: " + name)
        result.append(math.log1p(value))
    return result


def context_key(record):
    context = record["comparable_context"]
    required = (
        "source_map_sha256",
        "nucleus",
        "operator",
        "radial_representation",
        "ordered_state_UIDs",
    )
    if not all(name in context and context[name] is not None for name in required):
        raise ValueError("incomplete comparable physical context")
    ion = {
        name: record["predictor_features"][name] for name in ("ion_Z", "electron_count")
    }
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in ion.values()
    ):
        raise ValueError("explicit positive integer ion/electron identity required")
    return canonical({"context": context, "ion": ion})


def admitted_rows(ledger, candidate):
    rows = ledger["rows"]
    if len(rows) > POLICY["maximum_rows"]:
        raise ValueError("prototype ledger row cap")
    key = context_key(candidate)
    accepted, rejected, seen = [], [], set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate paired experiment id")
        seen.add(row["id"])
        if row["physics_class"] != candidate["physics_class"]:
            continue
        if not row["validity"]["training_eligible"]:
            rejected.append({"id": row["id"], "reason": "ledger_training_ineligible"})
            continue
        if not row["validity"]["paired_provenance"]:
            raise ValueError("eligible row lacks paired provenance")
        try:
            if context_key(row) != key:
                rejected.append(
                    {"id": row["id"], "reason": "different_physical_context"}
                )
                continue
            x = features(row)
        except (ValueError, KeyError) as error:
            rejected.append({"id": row["id"], "reason": str(error)})
            continue
        y = row["contribution"]["transition_vector_cm1"]
        if len(y) != 4 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in y
        ):
            raise ValueError("four finite signed D3,D5,P1,P3 contributions required")
        group = row["independent_group_id"]
        if not isinstance(group, str) or not group:
            raise ValueError("explicit independent group required")
        accepted.append({"id": row["id"], "group": group, "x": x, "y": list(y)})
    return accepted, rejected


def _predict(train, query):
    """Fixed RBF posterior; group weights stop duplicate trajectories dominating."""
    import numpy as np

    x = np.asarray([r["x"] for r in train], dtype=float)
    y = np.asarray([r["y"] for r in train], dtype=float)
    multiplicities = Counter(r["group"] for r in train)
    weights = np.asarray([1 / multiplicities[r["group"]] for r in train])
    center = np.average(y, axis=0, weights=weights)
    scale = np.sqrt(np.average((y - center) ** 2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-12)
    length = POLICY["kernel_length_scale"]
    distance = np.sum((x[:, None] - x[None, :]) ** 2, axis=-1)
    kernel = np.exp(-distance / (2 * length**2))
    kernel += np.diag(POLICY["kernel_noise"] / weights)
    delta = x - np.asarray(query)
    cross = np.exp(-np.sum(delta**2, axis=1) / (2 * length**2))
    cholesky = np.linalg.cholesky(kernel)
    rhs = (y - center) / scale
    alpha = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, rhs))
    mean = center + (cross @ alpha) * scale
    solved = np.linalg.solve(cholesky, cross)
    variance = max(0.0, 1.0 - float(solved @ solved))
    return mean.tolist(), (np.sqrt(variance) * scale).tolist()


def group_validation(rows):
    """Whole experiments held out; all normalization/fitting is train-only."""
    groups = sorted({r["group"] for r in rows})
    if len(groups) > POLICY["maximum_groups"]:
        raise ValueError("prototype held-out group cap")
    errors, baseline_errors, folds = [], [], []
    for group in groups:
        train = [r for r in rows if r["group"] != group]
        test = [r for r in rows if r["group"] == group]
        # A group contributes one mean target vector to the constant baseline.
        independent = []
        for other in groups:
            if other == group:
                continue
            own = [r["y"] for r in train if r["group"] == other]
            independent.append([sum(a[i] for a in own) / len(own) for i in range(4)])
        constant = [sum(a[i] for a in independent) / len(independent) for i in range(4)]
        e, b = [], []
        for row in test:
            prediction, _ = _predict(train, row["x"])
            e.extend(abs(prediction[i] - row["y"][i]) for i in range(4))
            b.extend(abs(constant[i] - row["y"][i]) for i in range(4))
        errors.append(sum(e) / len(e))
        baseline_errors.append(sum(b) / len(b))
        folds.append(
            {
                "held_out_group": group,
                "train_groups": sorted({r["group"] for r in train}),
                "test_row_ids": [r["id"] for r in test],
                "MAE_cm1": errors[-1],
                "constant_baseline_MAE_cm1": baseline_errors[-1],
            }
        )
    error = sum(errors) / len(errors)
    baseline = sum(baseline_errors) / len(baseline_errors)
    passed = baseline > 0 and error <= POLICY["required_group_cv_MAE_ratio"] * baseline
    return {
        "folds": folds,
        "group_MAE_cm1": error,
        "constant_baseline_group_MAE_cm1": baseline,
        "admission_passed": passed,
        "validation_is_not_calibrated_uncertainty_or_physical_accuracy": True,
    }


def assess(ledger, candidate):
    # Reject experimental predictors even when too little evidence exists to fit.
    registered = set(POLICY["feature_names"]) | {"ion_Z", "electron_count"}
    if set(candidate["predictor_features"]) - registered:
        raise ValueError("unregistered candidate predictor features")
    rows, rejected = admitted_rows(ledger, candidate)
    groups = sorted({r["group"] for r in rows})
    result = {
        "candidate_id": candidate["id"],
        "status": "abstain",
        "matching_rows": len(rows),
        "independent_groups": len(groups),
        "rejected_rows": rejected,
        "learned_prediction": None,
        "automatic_execution": False,
        "physical_gates_changed": False,
    }
    if len(groups) < POLICY["minimum_independent_groups"]:
        result["reason"] = "insufficient_independent_comparable_experiments"
        return result
    query = features(candidate)
    if any(
        not min(r["x"][i] for r in rows) <= q <= max(r["x"][i] for r in rows)
        for i, q in enumerate(query)
    ):
        result["reason"] = "physical_descriptor_extrapolation"
        return result
    cv = group_validation(rows)
    result["group_validation"] = cv
    if not cv["admission_passed"]:
        result["reason"] = "held_out_groups_do_not_beat_constant_baseline"
        return result
    mean, spread = _predict(rows, query)
    result.update(
        status="advisory_prediction_only",
        reason="prototype_group_gate_passed",
        learned_prediction={
            "transition_vector_cm1": mean,
            "model_spread_cm1": spread,
            "spread_is_not_an_error_bound": True,
        },
    )
    return result


def select(ledger, candidates):
    """Keep physical readiness/failed predecessors ahead of learned advice."""
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate experiment candidate")
    assessments = []
    for candidate in candidates:
        if candidate.get("status") in {"failed", "interrupted", "completed"}:
            assessments.append(
                {
                    "candidate_id": candidate["id"],
                    "status": "ineligible",
                    "reason": "preserve_prior_outcome_no_automatic_retry",
                    "learned_prediction": None,
                    "automatic_execution": False,
                }
            )
        else:
            assessments.append(assess(ledger, candidate))
    admitted = [
        c
        for c in candidates
        if c.get("status") not in {"failed", "interrupted", "completed"}
    ]
    by_id = {r["candidate_id"]: r for r in assessments}
    scores = {}
    for priority in sorted({c["physical_priority"] for c in admitted}):
        tier = [c for c in admitted if c["physical_priority"] == priority]
        if not all(
            by_id[c["id"]]["status"] == "advisory_prediction_only" for c in tier
        ):
            continue
        for candidate in tier:
            budget = candidate["planned_wall_cap_s"]
            if (
                not isinstance(budget, (int, float))
                or not math.isfinite(budget)
                or budget <= 0
            ):
                raise ValueError(
                    "positive declared wall budget required for acquisition"
                )
            spread = by_id[candidate["id"]]["learned_prediction"]["model_spread_cm1"]
            scores[candidate["id"]] = sum(v * v for v in spread) / budget
            by_id[candidate["id"]]["acquisition_proxy"] = scores[candidate["id"]]
    admitted.sort(
        key=lambda c: (c["physical_priority"], -scores.get(c["id"], 0), c["id"])
    )
    return {
        "schema_version": 1,
        "policy": POLICY,
        "assessments": assessments,
        "advisory_order": [c["id"] for c in admitted],
        "order_basis": "declared physical priorities; within fully admitted tiers, model variance / declared wall cap; otherwise id order",
        "NIST_used_for_training_or_features": False,
        "automatic_execution": False,
        "cards_or_timers": False,
        "AI_acceleration_measured": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    candidates_raw = args.candidates.read_bytes()
    ledger = json.loads(raw)
    evidence_bytes = verify_json_inputs(ledger, args.repo)
    result = select(ledger, json.loads(candidates_raw)["candidates"])
    result["bound_JSON_evidence_bytes_checked"] = evidence_bytes
    result["input_bindings"] = {
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "candidates_sha256": hashlib.sha256(candidates_raw).hexdigest(),
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
