"""Independent synthetic metadata invariants; no scientific qualification claim."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from catalog_adapters import extract_pair
from dataset_builder import TARGETS, EvidenceStore, digest, paired_row, summarise


def fixture():
    # Signed worsening/mixed response with distinct lower and upper shifts.
    before_energy = [-100., -99.9, -99.8, -99.7, -99.6]
    shifts = [-.01, -.03, -.02, .01, -.04]
    endpoints = []
    for energy in (before_energy, [a + b for a, b in zip(before_energy, shifts, strict=True)]):
        rows = [dict(label=label, UID=list(uid), selected_level=list(uid), energy_Ha=value,
                     excitation_cm1=(value - energy[0]) * 100., NIST_nominal_cm1=1.)
                for (label, uid), value in zip(TARGETS, energy, strict=True)]
        endpoints.append({"rows": rows, "identity": {
            "source_map_sha256": "a" * 64, "ion_Z": 70, "electron_count": 69,
            "nucleus": {"declared_profile": "Fermi A175", "native_workspace_archive_sha256": "b" * 64},
            "operator": "DC", "radial_representation": {"kind": "bspline", "ns": 123},
            "point": {"anchor_sha256": "c" * 64}, "owner": {"archive_sha256": "d" * 64},
            "space": {"n_radial": 29, "n_target_CSF": 1046}}})
    spec = {"id": "synthetic", "family": "physics_contribution", "physics_class": "core_rank",
            "scope": "synthetic mechanics only", "point_group_anchor_sha256": "c" * 64}
    admission = {"paired_provenance": True, "numerical_gates_passed": True, "failed_gates": [],
                 "stationarity": {"before": False, "after": "not_evaluated"}, "unsuitable_reasons": []}
    return spec, *endpoints, {}, admission, {}, 100.


class DatasetInvariantTests(unittest.TestCase):
    def test_signed_upper_lower_centroid_and_splitting(self):
        row = paired_row(*fixture())
        for value, expected in zip(row["contribution"]["transition_vector_cm1"], (-2., -1., 2., -3.), strict=True):
            self.assertAlmostEqual(value, expected, places=10)
        self.assertAlmostEqual(row["contribution"]["multiplet_deltas"]["5d"]["centroid_cm1"], -1.4)
        self.assertAlmostEqual(row["contribution"]["multiplet_deltas"]["5d"]["splitting_cm1"], 1.)
        self.assertAlmostEqual(row["contribution"]["multiplet_deltas"]["6p"]["centroid_cm1"], -4/3)
        self.assertAlmostEqual(row["contribution"]["multiplet_deltas"]["6p"]["splitting_cm1"], -5.)
        self.assertTrue(row["validity"]["training_eligible"])
        self.assertFalse(row["validity"]["stationarity"]["before"])

    def test_permutation_does_not_change_states_or_data(self):
        values = fixture()
        expected = paired_row(*copy.deepcopy(values))
        values[1]["rows"].reverse()
        values[2]["rows"] = values[2]["rows"][2:] + values[2]["rows"][:2]
        self.assertEqual(paired_row(*values), expected)

    def test_NIST_never_changes_predictors_contributions_or_eligibility(self):
        values = fixture()
        expected = paired_row(*copy.deepcopy(values))
        for side in values[1:3]:
            for row in side["rows"]:
                row["NIST_nominal_cm1"] = -999999.
                row["nominal_gap_cm1"] = 1e30
        changed = paired_row(*values)
        for key in ("predictor_features", "comparable_context", "contribution", "validity"):
            self.assertEqual(changed[key], expected[key])
        self.assertNotEqual(changed["validation_only"], expected["validation_only"])

    def test_changed_UID_rejects_even_if_label_column_agrees(self):
        values = fixture()
        values[2]["rows"][1]["UID"] = [5, 1, 0]
        with self.assertRaises(ValueError):
            paired_row(*values)

    def test_changed_selected_physical_root_rejects(self):
        values = fixture()
        values[2]["rows"][1]["selected_level"] = [3, 1, 1]
        with self.assertRaises(ValueError):
            paired_row(*values)

    def test_missing_or_duplicate_manifold_rejects(self):
        for rows in (fixture()[2]["rows"][:-1], fixture()[2]["rows"] + [fixture()[2]["rows"][0]]):
            values = fixture()
            values[2]["rows"] = rows
            with self.assertRaises(ValueError):
                paired_row(*values)

    def test_failed_gate_retains_signed_result_but_excludes_training(self):
        values = fixture()
        values[4]["failed_gates"] = ["reported_derivative"]
        values[4]["unsuitable_reasons"] = ["original_numerical_gate_failed"]
        row = paired_row(*values)
        self.assertFalse(row["validity"]["training_eligible"])
        self.assertLess(row["contribution"]["transition_vector_cm1"][0], 0.)

    def test_missing_native_count_is_null_and_ineligible(self):
        values = fixture()
        values[1]["identity"]["space"].pop("n_target_CSF")
        row = paired_row(*values)
        self.assertIsNone(row["predictor_features"]["n_target_CSF_before"])
        self.assertFalse(row["validity"]["predictor_eligible"])

    def test_missing_owner_and_nuclear_identity_ineligible(self):
        values = fixture()
        values[2]["identity"]["owner"] = {}
        values[1]["identity"]["nucleus"]["native_workspace_archive_sha256"] = None
        row = paired_row(*values)
        self.assertFalse(row["validity"]["training_eligible"])

    def test_source_change_requires_explicit_operator_experiment(self):
        values = fixture()
        values[2]["identity"]["source_map_sha256"] = "e" * 64
        row = paired_row(*values)
        self.assertIn("unqualified_source_mismatch", row["validity"]["unsuitable_reasons"])

    def test_unpaired_orbital_point_ineligible(self):
        values = fixture()
        values[2]["identity"]["point"]["anchor_sha256"] = "f" * 64
        self.assertFalse(paired_row(*values)["validity"]["training_eligible"])

    def test_correlated_channels_count_once_and_trajectory_never_trains(self):
        one = paired_row(*fixture())
        values = fixture()
        values[0]["id"], values[0]["physics_class"] = "other_channel", "Breit"
        two = paired_row(*values)
        values = fixture()
        values[0]["id"], values[0]["family"] = "optimizer", "optimization_trajectory"
        three = paired_row(*values)
        summary = summarise([one, two, three])
        self.assertEqual(summary["eligible_pairs"], 2)
        self.assertEqual(summary["effective_independent_point_groups_upper_bound"], 1)
        self.assertFalse(three["validity"]["training_eligible"])

    def test_inconsistent_energy_reference_rejects(self):
        values = fixture()
        values[2]["rows"][2]["excitation_cm1"] += 1.
        with self.assertRaises(ValueError):
            paired_row(*values)

    def test_closed_JSON_hash_mutation_and_archive_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b'{"closed":true}'
            (root / "evidence.json").write_bytes(raw)
            store = EvidenceStore(root)
            reference = {"path": "evidence.json", "sha256": digest(raw)}
            self.assertEqual(store.load(reference), {"closed": True})
            (root / "evidence.json").write_text('{"closed":false}')
            with self.assertRaises(ValueError):
                EvidenceStore(root).load(reference)
            with self.assertRaises(ValueError):
                store.load({"path": "never_read.npz", "sha256": "a" * 64})
            with self.assertRaises(ValueError):
                store.load({"path": "../outside.json", "sha256": "a" * 64})

    def test_IO_budget_checked_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = json.dumps({"large": "x" * 100}).encode()
            (root / "a.json").write_bytes(raw)
            with self.assertRaises(ValueError):
                EvidenceStore(root, maximum_bytes=10).load({"path": "a.json", "sha256": digest(raw)})

    def test_admission_cannot_be_overridden_by_empty_reason_list(self):
        for field in ("paired_provenance", "numerical_gates_passed"):
            values = fixture()
            values[4][field] = False
            self.assertFalse(paired_row(*values)["validity"]["training_eligible"])

    def test_receipt_and_proof_identity_are_actual_admission_controls(self):
        values = fixture()
        spec = dict(values[0], adapter="legacy_stages", after_model="new", cost_scope="fixture",
                    proof_pass_field="all_passed", roles={"contract": {"sha256": "a" * 64},
                    "result": {"path": "result.json", "sha256": "b" * 64}})
        stages = [dict(name=name, rows=model["rows"], CI_sha256="d" * 64,
                       H_sha256="e" * 64, retained_target_CSFs=1046)
                  for name, model in zip(("baseline", "new"), values[1:3], strict=True)]
        loaded = {"contract": {"fixed_conditions": {"Z": 70, "electron_count": 69,
                    "ns": 123, "nucleus": "Fermi175", "radial_keys": [[1, -1]]}},
                  "result": {"contract_sha256": "a" * 64, "source_map_sha256": "c" * 64,
                    "stages": stages, "numeric_identity_gates_passed": True},
                  "receipt": {"status": "running"}, "proof": {"all_passed": False}}
        before, after, admission, cost = extract_pair(spec, loaded)
        row = paired_row(spec, before, after, {}, admission, cost, 100.)
        self.assertIn("missing_closed_foreground_receipt", row["validity"]["unsuitable_reasons"])
        self.assertIn("saved_proof_not_passed", row["validity"]["unsuitable_reasons"])
        loaded["proof"] = {"all_passed": True, "bound_input_sha256": {"result.json": "f" * 64}}
        with self.assertRaises(ValueError):
            extract_pair(spec, loaded)


if __name__ == "__main__":
    unittest.main()
