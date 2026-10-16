"""Read-only JSON evidence ledger; no solver, dispatcher, learned model or array IO."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
TARGETS = (
    ("6s_1/2", (1, 1, 0)), ("5d_3/2", (3, 1, 0)),
    ("5d_5/2", (5, 1, 0)), ("6p_1/2", (1, -1, 0)), ("6p_3/2", (3, -1, 0)),
)
FEATURE_KEYS = (
    "ion_Z", "electron_count", "n_radial_before", "n_radial_after",
    "n_target_CSF_before", "n_target_CSF_after",
)
CONTEXT_KEYS = ("source_map_sha256", "nucleus", "operator", "radial_representation", "ordered_state_UIDs")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value):
    return digest(canonical(value).encode())


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class EvidenceStore:
    """A manifest is explicit; references in scientific JSON are never followed."""

    def __init__(self, root, maximum_files=64, maximum_bytes=64 * 1024**2):
        self.root = Path(root).resolve()
        self.maximum_files, self.maximum_bytes = maximum_files, maximum_bytes
        self.data, self.refs, self.bytes_read = {}, {}, 0

    def load(self, reference):
        path, expected = reference["path"], reference["sha256"]
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root) or full.suffix != ".json" or not valid_sha(expected):
            raise ValueError("Only repository-local explicitly bound JSON evidence is allowed")
        if path in self.data:
            if self.refs[path] != expected:
                raise ValueError("Conflicting expected evidence hash")
            return self.data[path]
        size = full.stat().st_size
        if len(self.data) >= self.maximum_files or self.bytes_read + size > self.maximum_bytes:
            raise ValueError("Prospective metadata resource cap exceeded")
        raw = full.read_bytes()
        if digest(raw) != expected:
            raise ValueError("Bound JSON evidence changed")
        value = json.loads(raw, parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
        self.data[path], self.refs[path] = value, expected
        self.bytes_read += len(raw)
        return value


def physical_rows(rows):
    """Match physical UID, never global column or insertion order."""
    result = {}
    for row in rows:
        if "UID" in row:
            uid = tuple(row["UID"])
        elif "reference_UID" in row:
            uid = tuple(row["reference_UID"])
        elif all(k in row for k in ("twoJ", "parity", "local_root")):
            uid = (row["twoJ"], row["parity"], row["local_root"])
        else:
            raise ValueError("Missing physical state UID")
        selected = tuple(row.get("selected_level", uid))
        if uid in result or selected != uid or (row["label"], uid) not in TARGETS:
            raise ValueError("Duplicate, relabeled or changed physical target")
        for field in ("energy_Ha", "excitation_cm1"):
            if field in row and (isinstance(row[field], bool) or not math.isfinite(row[field])):
                raise ValueError("Nonfinite observed energy")
        result[uid] = row
    if set(result) != {uid for _, uid in TARGETS}:
        raise ValueError("Incomplete paired target manifold")
    return [result[uid] for _, uid in TARGETS]


def multiplets(vector):
    return {
        "5d": {"centroid_cm1": (4 * vector[0] + 6 * vector[1]) / 10,
               "splitting_cm1": vector[1] - vector[0], "degeneracy_weights": [4, 6]},
        "6p": {"centroid_cm1": (2 * vector[2] + 4 * vector[3]) / 6,
               "splitting_cm1": vector[3] - vector[2], "degeneracy_weights": [2, 4]},
    }


def validate_row(row):
    if set(row["predictor_features"]) != set(FEATURE_KEYS):
        raise ValueError("Predictors must contain only pre-outcome native metadata")
    if any(value is not None and (type(value) is not int or value <= 0)
           for value in row["predictor_features"].values()):
        raise ValueError("Invalid native count predictor")
    if row["validation_only"]["used_for_prediction_or_target_selection"]:
        raise ValueError("NIST leakage into selection")
    if len(row["contribution"]["transition_vector_cm1"]) != 4:
        raise ValueError("Incomplete signed transition vector")
    if row["family"] == "optimization_trajectory" and row["validity"]["training_eligible"]:
        raise ValueError("Correlated optimizer trajectory cannot be independent physics training")
    if row["validity"]["training_eligible"] and row["validity"]["unsuitable_reasons"]:
        raise ValueError("Unsuitable evidence cannot be training eligible")
    return row


def paired_row(spec, before, after, provenance, admission, cost, conversion):
    """Normalize one whole five-target paired experiment, retaining negative results."""
    old, new = physical_rows(before["rows"]), physical_rows(after["rows"])
    reasons = list(admission["unsuitable_reasons"])
    if not admission["paired_provenance"]:
        reasons.append("missing_paired_provenance")
    if not admission["numerical_gates_passed"] or admission["failed_gates"]:
        reasons.append("original_numerical_gate_failed")
    identities = [before["identity"], after["identity"]]
    for side, item in zip(("before", "after"), identities, strict=True):
        for key in ("source_map_sha256", "nucleus", "operator", "radial_representation", "point", "owner", "space"):
            if not item.get(key):
                reasons.append(f"{side}_missing_{key}")
        if not valid_sha(item.get("source_map_sha256")):
            reasons.append(f"{side}_missing_source_hash")
        if not item.get("nucleus", {}).get("declared_profile") or not valid_sha(
                item.get("nucleus", {}).get("native_workspace_archive_sha256")):
            reasons.append(f"{side}_missing_nuclear_binding")
        if type(item.get("radial_representation", {}).get("ns")) is not int:
            reasons.append(f"{side}_missing_radial_representation")
        if not valid_sha(item.get("point", {}).get("anchor_sha256")):
            reasons.append(f"{side}_missing_point_binding")
        if not item.get("owner", {}).get("archive_sha256") and not item.get("owner", {}).get("header_sha256"):
            reasons.append(f"{side}_missing_owner_archive_binding")
    if before["identity"].get("nucleus") != after["identity"].get("nucleus"):
        reasons.append("unmatched_nucleus")
    if (identities[0].get("source_map_sha256") != identities[1].get("source_map_sha256")
            and not (spec.get("declared_source_change", False)
                     and spec["physics_class"] == "positive_inner_quadrature_operator_expectation")):
        reasons.append("unqualified_source_mismatch")
    anchor = spec.get("point_group_anchor_sha256")
    if not valid_sha(anchor):
        reasons.append("missing_independent_point_group_identity")
    if (spec["family"] != "optimization_trajectory"
            and before["identity"].get("point", {}).get("anchor_sha256") != after["identity"].get("point", {}).get("anchor_sha256")):
        reasons.append("unpaired_frozen_base_point")
    energies_available = all("energy_Ha" in r for r in old + new)
    shifts = [new[i]["energy_Ha"] - old[i]["energy_Ha"] for i in range(5)] if energies_available else None
    if all("delta_expectation_Ha" in r for r in new):
        shifts = [r["delta_expectation_Ha"] for r in new]
    if shifts is None:
        reasons.append("missing_absolute_upper_lower_contributions")
    if all("excitation_cm1" in r for r in old + new):
        vector = [new[i]["excitation_cm1"] - old[i]["excitation_cm1"] for i in range(1, 5)]
    elif all("delta_transition_cm1" in r for r in new):
        vector = [r["delta_transition_cm1"] for r in new[1:]]
    else:
        raise ValueError("Missing signed paired transitions; no fabricated zeros")
    if shifts is not None:
        expected = [(shift - shifts[0]) * conversion for shift in shifts[1:]]
        if max(abs(a - b) for a, b in zip(vector, expected, strict=True)) > 1e-5:
            raise ValueError("Transition shift disagrees with matched absolute upper-minus-lower shifts")
    features = {
        "ion_Z": before["identity"].get("ion_Z"),
        "electron_count": before["identity"].get("electron_count"),
        "n_radial_before": before["identity"].get("space", {}).get("n_radial"),
        "n_radial_after": after["identity"].get("space", {}).get("n_radial"),
        "n_target_CSF_before": before["identity"].get("space", {}).get("n_target_CSF"),
        "n_target_CSF_after": after["identity"].get("space", {}).get("n_target_CSF"),
    }
    if any(value is None for value in features.values()):
        reasons.append("missing_native_predictor_metadata")
    context = {key: (identities[0].get(key) if identities[0].get(key) == identities[1].get(key)
                     else {"before": identities[0].get(key), "after": identities[1].get(key)})
               for key in CONTEXT_KEYS[:-1]}
    context["ordered_state_UIDs"] = [list(uid) for _, uid in TARGETS]
    if any(value is None or value == {} for value in context.values()):
        reasons.append("missing_comparable_context")
    reasons = sorted(set(reasons))
    validation = {"used_for_prediction_or_target_selection": False, "NIST_role": "validation_only",
                  "before": [{k: v for k, v in r.items() if k.startswith("NIST") or k.startswith("nominal_")}
                             for r in old],
                  "after": [{k: v for k, v in r.items() if k.startswith("NIST") or k.startswith("nominal_")}
                            for r in new]}
    row = {
        "id": spec["id"], "family": spec["family"], "physics_class": spec["physics_class"],
        "scope": spec["scope"], "independent_group_id": "point_" + identity({"anchor": anchor})[:20],
        "group_anchor_sha256": anchor, "comparable_context": context,
        "predictor_features": features, "before": before["identity"], "after": after["identity"],
        "contribution": {
            "ordered_targets": [label for label, _ in TARGETS],
            "absolute_state_energy_shift_Ha": shifts,
            "lower_absolute_energy_shift_Ha": shifts[0] if shifts else None,
            "upper_absolute_energy_shift_Ha": shifts[1:] if shifts else None,
            "transition_vector_labels": [label for label, _ in TARGETS[1:]],
            "transition_vector_cm1": vector, "multiplet_deltas": multiplets(vector),
            "observed_absolute_energies_before_Ha": [r.get("energy_Ha") for r in old],
            "observed_absolute_energies_after_Ha": [r.get("energy_Ha") for r in new],
            "interpretation": spec.get("interpretation", "Finite paired conditional response; not an additive missing-physics bound"),
        },
        "validity": {"paired_provenance": admission["paired_provenance"],
                     "numerical_gates_passed": admission["numerical_gates_passed"],
                     "failed_gates": admission["failed_gates"],
                     "stationarity": admission["stationarity"],
                     "training_eligible": not reasons and spec["family"] == "physics_contribution",
                     "predictor_eligible": not reasons and spec["family"] == "physics_contribution",
                     "unsuitable_reasons": reasons,
                     "transfer_to_current31_qualified": False},
        "cost": cost, "provenance": provenance, "validation_only": validation,
    }
    return validate_row(row)


def summarise(rows):
    eligible = [r for r in rows if r["validity"]["training_eligible"]]
    groups = sorted({r["independent_group_id"] for r in eligible})
    by_class = {}
    for physics_class in sorted({r["physics_class"] for r in rows}):
        selected = [r for r in eligible if r["physics_class"] == physics_class]
        contexts = {}
        for row in selected:
            contexts.setdefault(identity(row["comparable_context"]), set()).add(row["independent_group_id"])
        by_class[physics_class] = {"eligible_pairs": len(selected), "point_groups": len({r["independent_group_id"] for r in selected}),
                                  "exact_context_group_counts": {k: len(v) for k, v in contexts.items()}}
    return {"rows": len(rows), "physics_pairs": sum(r["family"] == "physics_contribution" for r in rows),
            "trajectory_aggregates": sum(r["family"] == "optimization_trajectory" for r in rows),
            "eligible_pairs": len(eligible), "effective_independent_point_groups_upper_bound": len(groups),
            "groups_are_not_IID_certified": True, "per_physics_class": by_class,
            "unsuitable_reason_counts": dict(Counter(reason for r in rows for reason in r["validity"]["unsuitable_reasons"])),
            "learned_prediction_performed": False, "AI_speedup_or_uncertainty_certified": False}


def build_dataset(root, manifest):
    from catalog_adapters import extract_pair

    store = EvidenceStore(root)
    conversion = store.load(manifest["unit_conversion"])["Hartree_cm1"]
    rows = []
    for spec in manifest["experiments"]:
        loaded = {role: store.load(ref) for role, ref in spec["roles"].items()}
        before, after, admission, cost = extract_pair(spec, loaded)
        refs = {role: dict(ref) for role, ref in spec["roles"].items()}
        for role in refs:
            data = loaded[role]
            for field in ("result_sha256", "actual_result_sha256"):
                if field in data and data[field] != refs["result"]["sha256"]:
                    raise ValueError("Proof/receipt is bound to a different result")
        rows.append(paired_row(spec, before, after, refs, admission, cost, conversion))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate experiment ID")
    dataset = {"schema_version": SCHEMA_VERSION, "rows": rows,
               "independent_groups": {g: [r["id"] for r in rows if r["independent_group_id"] == g]
                                      for g in sorted({r["independent_group_id"] for r in rows})},
               "summary": summarise(rows), "inputs_sha256": dict(sorted(store.refs.items())),
               "metadata_IO": {"JSON_files": len(store.data), "JSON_bytes": store.bytes_read,
                               "NPZ_reads_or_hashes": 0, "new_scientific_calls": 0},
               "unit_conversion": manifest["unit_conversion"],
               "no_dispatch_or_fitting": True}
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    result = build_dataset(args.repo, manifest)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
