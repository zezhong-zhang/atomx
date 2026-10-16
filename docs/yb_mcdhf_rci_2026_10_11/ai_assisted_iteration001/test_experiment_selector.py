"""Independent provenance/group-leakage and advisory-only controls."""

import copy
import hashlib
import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "experiment_selector", Path(__file__).with_name("experiment_selector.py")
)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)

CONTEXT = {
    "source_map_sha256": "synthetic-source-not-an-Atomx-measurement",
    "nucleus": "synthetic-fixed-nucleus",
    "operator": "synthetic-DC",
    "radial_representation": "synthetic-cell",
    "ordered_state_UIDs": [[1, 1, 0], [3, 1, 0], [5, 1, 0], [1, -1, 0], [3, -1, 0]],
}


def fixture_row(i, group=None):
    count = 10 + i
    return {
        "id": "synthetic-" + str(i),
        "physics_class": "synthetic-complete-core-class",
        "comparable_context": dict(CONTEXT),
        "independent_group_id": group or str(i),
        "predictor_features": {
            "ion_Z": 70,
            "electron_count": 69,
            "n_radial_before": count,
            "n_radial_after": count,
            "n_target_CSF_before": count * 10,
            "n_target_CSF_after": count * 20,
        },
        "contribution": {
            "transition_vector_cm1": [
                100 * math.log1p(count),
                -30 * math.log1p(count),
                10 * math.log1p(count),
                50 * math.log1p(count),
            ]
        },
        "validity": {"paired_provenance": True, "training_eligible": True},
        "validation_only": {"NIST": "synthetic-not-training"},
    }


def candidate(i=3):
    row = fixture_row(i)
    row.update(
        id="candidate",
        physical_priority=1,
        planned_wall_cap_s=900,
        status="source_preparing",
    )
    return row


def test_many_correlated_rows_do_not_admit_learning():
    ledger = {"rows": [fixture_row(i, "one-trajectory") for i in range(100)]}
    result = selector.assess(ledger, candidate())
    assert result["independent_groups"] == 1
    assert result["status"] == "abstain" and result["learned_prediction"] is None


@pytest.mark.parametrize(
    "field",
    [
        "source_map_sha256",
        "nucleus",
        "operator",
        "radial_representation",
        "ordered_state_UIDs",
    ],
)
def test_different_physical_context_cannot_supply_training_labels(field):
    rows = [fixture_row(i) for i in range(6)]
    for row in rows:
        row["comparable_context"][field] = "different"
    result = selector.assess({"rows": rows}, candidate())
    assert result["matching_rows"] == 0 and result["status"] == "abstain"


def test_NIST_metadata_is_inert_and_cannot_become_a_predictor():
    ledger = {"rows": [fixture_row(i) for i in range(6)]}
    first = selector.assess(ledger, candidate())
    changed = copy.deepcopy(ledger)
    for row in changed["rows"]:
        row["validation_only"] = {"NIST": -1e100, "nominal_gap": 1e100}
    assert selector.assess(changed, candidate()) == first
    query = candidate()
    query["predictor_features"]["NIST_gap"] = 123
    with pytest.raises(ValueError, match="unregistered"):
        selector.assess({"rows": []}, query)


@pytest.mark.parametrize("field", ["ion_Z", "electron_count"])
def test_ion_identity_is_not_only_a_regression_feature(field):
    rows = [fixture_row(i) for i in range(6)]
    for row in rows:
        row["predictor_features"][field] += 1
    result = selector.assess({"rows": rows}, candidate())
    assert result["matching_rows"] == 0 and result["status"] == "abstain"


def test_whole_independent_group_is_held_out_and_inputs_are_unchanged():
    rows = [fixture_row(i, str(i // 2)) for i in range(12)]
    ledger = {"rows": rows}
    saved = copy.deepcopy(ledger)
    admitted, _ = selector.admitted_rows(ledger, candidate())
    result = selector.group_validation(admitted)
    assert len(result["folds"]) == 6
    for fold in result["folds"]:
        assert fold["held_out_group"] not in fold["train_groups"]
        assert len(fold["test_row_ids"]) == 2
    assert ledger == saved


def test_constant_data_does_not_demonstrate_learned_advantage():
    rows = [fixture_row(i) for i in range(6)]
    for row in rows:
        row["contribution"]["transition_vector_cm1"] = [1, 2, 3, 4]
    result = selector.assess({"rows": rows}, candidate())
    assert result["status"] == "abstain"
    assert result["reason"] == "held_out_groups_do_not_beat_constant_baseline"


def test_descriptor_extrapolation_is_not_accepted_as_prediction():
    result = selector.assess(
        {"rows": [fixture_row(i) for i in range(6)]}, candidate(100)
    )
    assert result["status"] == "abstain"
    assert result["reason"] == "physical_descriptor_extrapolation"


def test_eligible_nonstationary_measurement_is_conditional_not_convergence():
    rows = [fixture_row(i) for i in range(6)]
    for row in rows:
        row["validity"]["stationarity"] = False
    result = selector.assess({"rows": rows}, candidate())
    assert result["status"] == "advisory_prediction_only"
    assert result["learned_prediction"]["spread_is_not_an_error_bound"]
    assert not result["automatic_execution"] and not result["physical_gates_changed"]


def test_failed_and_interrupted_candidates_are_preserved_without_retry():
    queries = [candidate() for _ in range(3)]
    for row, state in zip(queries, ("failed", "interrupted", "completed"), strict=True):
        row.update(id=state, status=state)
    result = selector.select({"rows": []}, queries)
    assert result["advisory_order"] == []
    assert all(a["status"] == "ineligible" for a in result["assessments"])
    assert not result["automatic_execution"] and not result["cards_or_timers"]


def test_learning_cannot_override_physical_priority_or_dispatch_a_job():
    queries = [candidate(), candidate(2)]
    queries[0].update(id="core", physical_priority=1)
    queries[1].update(id="closure", physical_priority=0)
    result = selector.select({"rows": [fixture_row(i) for i in range(6)]}, queries)
    assert result["advisory_order"] == ["closure", "core"]
    assert not result["AI_acceleration_measured"]


def test_numerically_failed_labels_and_duplicate_experiments_are_rejected():
    row = fixture_row(0)
    row["validity"]["training_eligible"] = False
    result = selector.assess({"rows": [row]}, candidate())
    assert result["matching_rows"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        selector.assess({"rows": [fixture_row(0), fixture_row(0)]}, candidate())


def test_bound_provenance_rejects_modified_or_array_evidence(tmp_path):
    path = tmp_path / "proof.json"
    raw = b'{"closed":true}'
    path.write_bytes(raw)
    ledger = {"inputs_sha256": {"proof.json": hashlib.sha256(raw).hexdigest()}}
    assert selector.verify_json_inputs(ledger, tmp_path) == len(raw)
    path.write_bytes(b'{"closed":false}')
    with pytest.raises(ValueError, match="changed"):
        selector.verify_json_inputs(ledger, tmp_path)
    with pytest.raises(ValueError, match="JSON only"):
        selector.verify_json_inputs(
            {"inputs_sha256": {"waves.npz": "irrelevant"}}, tmp_path
        )
