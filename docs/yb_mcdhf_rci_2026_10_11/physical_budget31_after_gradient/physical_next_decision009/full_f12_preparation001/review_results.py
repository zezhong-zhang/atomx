"""Offline saved-result review; no Atomx import, H build or eigensolve.

Default is source-only. --review requires all dispatched workers to have
terminated and writes a new aggregate directory exactly once. Incomplete or
failed sectors are preserved and cannot produce a complete spectral table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
SECTORS = ((1, 1), (3, 1), (5, 1), (1, -1), (3, -1))
LABELS = ("6s1/2", "5d3/2", "5d5/2", "6p1/2", "6p3/2")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def review():
    import numpy as np

    out = HERE / "aggregate001"
    out.mkdir()
    checks, inputs, rows = [], {}, []
    c = read(HERE / "worker_contract.json")

    def check(name, observed, limit=None):
        passed = bool(observed) if limit is None else bool(observed <= limit)
        checks.append(dict(name=name, observed=plain(observed), limit=limit, PASS=passed))
        if not passed:
            raise RuntimeError("Saved-result gate failed: " + name)

    result = dict(status="REVIEWING", checks=checks, inputs=inputs, rows=rows,
                  evaluator="Offline NumPy/algebra only; no Atomx import, H build or eigensolve.",
                  live_residual_scope="Full-operator residuals are source-bound live gates, not independently reconstructed here.",
                  source_map_sha256=c["source_map_sha256"], complete_spectral_comparison=False)
    try:
        for name, spec in c["metadata_inputs"].items():
            check("bound metadata " + name, sha(ROOT / spec["path"]) == spec["sha256"])
        for name, spec in c["array_inputs"].items():
            check("bound original arrays " + name, sha(ROOT / spec["path"]) == spec["sha256"])
        check("bound worker", sha(HERE / "worker.py") == c["driver_sha256"])
        source = read(HERE.parent / "matched001/current_source_identity.json")["all_atomx_python_sha256"]
        for path, expected in source.items():
            check("current source " + path, sha(ROOT / path) == expected)
        maps = read(ROOT / c["metadata_inputs"]["basis_maps"]["path"])
        prior = {}
        for arm in ("A", "B"):
            p = ROOT / c["array_inputs"][arm + "_R0"]["path"]
            with np.load(p, allow_pickle=False) as z:
                prior[arm] = z["mixing"].copy()
        completed = []
        for sector, label in zip(SECTORS, LABELS, strict=True):
            d = HERE / (f"actual_J{sector[0]}_" + ("even" if sector[1] == 1 else "odd"))
            if not (d / "result.json").exists():
                rows.append(dict(label=label, sector=sector, status="NOT_COMPLETED"))
                continue
            r = read(d / "result.json")
            inputs[str(d.relative_to(ROOT) / "result.json")] = sha(d / "result.json")
            if not r["status"].startswith("PASS"):
                rows.append(dict(label=label, sector=sector, status=r["status"],
                                 error=r.get("error"), stage=r["stage"],
                                 seconds=r["elapsed_seconds"], RSS_MiB=r["peak_RSS_MiB"]))
                continue
            check(label + " all live checks", bool(r["checks"]) and all(x["PASS"] for x in r["checks"]))
            check(label + " same source", r["source_map_sha256"] == c["source_map_sha256"])
            check(label + " zero angular tables", r["coefficient_table_calls"] == 0)
            check(label + " one full B H", r["counts"]["H_build"] == 1)
            check(label + " two native solves", r["counts"]["Davidson"] == 2)
            basis = read(d / "basis.json")
            sigs, a_indices = basis["B_signatures"], basis["A_nested_indices"]
            check(label + " unique B signatures", len(set(sigs)) == len(sigs))
            check(label + " unique A restriction", len(set(a_indices)) == len(a_indices))
            index = {s: i for i, s in enumerate(sigs)}
            record = dict(label=label, sector=sector, status=r["status"],
                          seconds=r["elapsed_seconds"], RSS_MiB=r["peak_RSS_MiB"],
                          structural_pairs=r["counts"]["structural_pairs"], Hdet=r["Hdet"], arms={})
            for arm, model in (("A", "baseline"), ("B", "expanded")):
                selected = r["selected"][arm]
                p = d / (arm + "_selected.npz")
                check(label + arm + " selected-file identity", sha(p) == selected["selected_arrays_sha256"])
                with np.load(p, allow_pickle=False) as z:
                    e, v, ref = z["energies"], z["mixing"], z["accepted13_reference"]
                    selection, residual, overlap = z["B_nested_indices"], z["residual_L2_Ha"], z["signed_overlap"]
                    check(label + arm + " finite arrays", all(np.isfinite(z[k]).all() for k in z.files))
                    check(label + arm + " complete rows", v.shape == (selected["full_CSF_rows"], 2))
                    expected_selection = np.array(a_indices if arm == "A" else range(len(sigs)))
                    check(label + arm + " exact signed restriction", np.array_equal(selection, expected_selection))
                    check(label + arm + " saved selected Gram", float(np.max(abs(v.T @ v - np.eye(2)))), 1e-10)
                    check(label + arm + " saved ref norm", abs(float(ref @ ref) - 1), 1e-10)
                    old_sigs = maps[model]["selected_signatures"]
                    old_rows = [i for i, s in enumerate(old_sigs) if tuple(json.loads(s)[2:]) == sector]
                    old_record = read(ROOT / c["metadata_inputs"][arm + "_R0_record"]["path"])
                    target = next(x for x in old_record["rows"] if tuple(x["UID"][:2]) == sector)
                    expected_ref = np.zeros(len(sigs))
                    expected_ref[[index[old_sigs[i]] for i in old_rows]] = prior[arm][old_rows, target["column"]]
                    expected_ref /= np.linalg.norm(expected_ref)
                    check(label + arm + " actual signed owned004 embedding", float(np.max(abs(ref - expected_ref[selection]))), 1e-14)
                    calculated_overlap = ref @ v
                    check(label + arm + " saved signed overlap", float(np.max(abs(overlap - calculated_overlap))), 1e-14)
                    chosen = int(np.argmax(calculated_overlap**2))
                    check(label + arm + " selected transport", chosen == selected["selected_local_root"])
                    check(label + arm + " strict majority", float(calculated_overlap[chosen]**2) > 0.5)
                    check(label + arm + " saved direct residual record", float(np.max(residual)), 1e-8)
                    check(label + arm + " energy record", float(e[chosen]) == selected["energy_Ha"])
                    check(label + arm + " original energy identity", selected["accepted13_energy_Ha"] == target["energy_Ha"])
                    f_counts = np.asarray([sum(q for n, k, q in json.loads(sigs[int(i)])[0]
                                              if n == 4 and k in (3, -4)) for i in selection])
                    f_weights = {str(q): float(np.sum(v[f_counts == q, chosen]**2))
                                 for q in sorted(set(f_counts.tolist()))}
                    check(label + arm + " complete 4f-occupation weights", abs(sum(f_weights.values()) - 1), 1e-10)
                    record["arms"][arm] = dict(energy_Ha=float(e[chosen]),
                        original_energy_Ha=target["energy_Ha"],
                        state_shift_Ha=float(e[chosen] - target["energy_Ha"]),
                        four_f_occupation_projector_weights=f_weights,
                        reference_weight=float(calculated_overlap[chosen]**2),
                        full_operator_residual_L2_Ha=float(np.max(residual)))
            rows.append(record)
            completed.append(record)
        if len(completed) == 5:
            factor = read(HERE.parent / "unit_identity.json")["HARTREE_CM1"]
            result["HARTREE_CM1"] = factor
            for arm in ("A", "B"):
                g = rows[0]["arms"][arm]
                for row in rows:
                    a = row["arms"][arm]
                    a["state_shift_cm1"] = a["state_shift_Ha"] * factor
                    a["excitation_cm1"] = float((np.longdouble(a["energy_Ha"]) - np.longdouble(g["energy_Ha"])) * factor)
                    a["original_excitation_cm1"] = float((np.longdouble(a["original_energy_Ha"]) - np.longdouble(g["original_energy_Ha"])) * factor)
                    a["transition_shift_cm1"] = a["excitation_cm1"] - a["original_excitation_cm1"]
                observables = {}
                for term, lo, hi, wl, wh in (("D", 1, 2, 4, 6), ("P", 3, 4, 2, 4)):
                    left, right = rows[lo]["arms"][arm], rows[hi]["arms"][arm]
                    observables[term] = dict(
                        centroid_shift_cm1=(wl * left["transition_shift_cm1"] + wh * right["transition_shift_cm1"]) / (wl + wh),
                        splitting_shift_cm1=right["transition_shift_cm1"] - left["transition_shift_cm1"])
                result[arm + "_term_shifts"] = observables
            for row in rows:
                a, b = row["arms"]["A"], row["arms"]["B"]
                row["B_minus_A_excitation_cm1"] = b["excitation_cm1"] - a["excitation_cm1"]
                row["correlation_interaction_cm1"] = b["transition_shift_cm1"] - a["transition_shift_cm1"]
            result.update(status="PASS_complete_fixed_R0_f12_saved_review", complete_spectral_comparison=True)
        else:
            result.update(status="INCOMPLETE_full_class_preserved_no_complete_spectral_table")
    except BaseException as error:
        result.update(status="FAILED_offline_review_preserved", error=repr(error))
    result.update(checks=checks, inputs=inputs, rows=rows,
                  checks_passed=sum(x["PASS"] for x in checks), checks_total=len(checks))
    (out / "result.json").write_text(json.dumps(plain(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], passed=result["checks_passed"], checks=len(checks))))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    if args.review:
        raise SystemExit(review())
    print(json.dumps(dict(status="SOURCE_ONLY_OFFLINE_REVIEWER", sha256=sha(Path(__file__)))))
