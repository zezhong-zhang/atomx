"""Independent frozen-consumer audit: no Atomx imports or physical evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "docs/yb_mcdhf_rci_2026_10_11/"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    prospective = load(HERE / "contract.json")
    caps = prospective["resources"]
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[-1]})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    cap = int(caps["address_space_cap_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Readonly audit cap")))
    signal.alarm(caps["wall_cap_s"])
    admission = {"CPU_affinity": sorted(os.sched_getaffinity(0)), "numerical_threads": 1,
                 "address_space_cap_bytes": cap, "wall_cap_s": caps["wall_cap_s"],
                 "physical_operator_calls_authorized": 0, "atomx_imports_authorized": 0}
    (HERE / "resource_admission.json").write_text(json.dumps(admission, indent=2) + "\n")
    import numpy as np

    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    consumer = ROOT / (PREFIX + "public_mixed_resume_run002")
    for relative, expected in prospective["input_sha256"].items():
        check("immutable_input:" + relative, sha(ROOT / relative) == expected)
    for relative, expected in prospective["frozen_successor_source_files"].items():
        check("actual_frozen_source:" + relative,
              sha(consumer / "source" / Path(relative).relative_to("src")) == expected)
    actual_map = {"src/" + str(p.relative_to(consumer / "source")): sha(p)
                  for p in sorted((consumer / "source/atomx").rglob("*.py"))}
    canonical = hashlib.sha256(json.dumps(actual_map, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    check("complete_source_map_and_required_SHA", actual_map == prospective["frozen_successor_source_files"]
          and canonical == prospective["fixed_conditions"]["successor_source_map_sha256"])
    contract = load(consumer / "contract.json")
    fixed, gates = contract["fixed_conditions"], contract["acceptance"]
    check("frozen_driver_identity", sha(consumer / "run.py")
          == prospective["fixed_conditions"]["consumer_driver_sha256"])
    check("frozen_verifier_identity", sha(consumer / "verify_endpoint.py")
          == prospective["fixed_conditions"]["verifier_sha256"])
    check("frozen_consumer_contract_identity", sha(consumer / "contract.json")
          == prospective["fixed_conditions"]["consumer_contract_sha256"])
    check("required_same_scientific_source_as_failed_predecessor",
          canonical == contract["scientific_source_map_required_sha256"])
    check("strict_original_four_gates", gates["bare_same_recovered_point_projected_gradient_max"] == 1e-9
          and gates["bare_same_recovered_point_preconditioned_gradient_max"] == 1e-10
          and gates["original_orbital_tolerance"] == 1e-7
          and gates["original_estimated_energy_tolerance_Ha"] == 1e-8)
    reference_info = contract["matched_recovered_point_reference"]
    check("reference_frozen_full_input_set", reference_info["status"] == "frozen_verified_same_recovered_point"
          and set(reference_info["inputs_sha256"]) == {
              str(Path(reference_info["directory"]) / name) for name in reference_info["files"]})
    check("reference_hashes_same_top_inputs", all(contract["inputs_sha256"].get(name) == value
                                                for name, value in reference_info["inputs_sha256"].items()))
    reference_dir = ROOT / reference_info["directory"]
    reference_result, reference_proof = load(reference_dir / "result.json"), load(reference_dir / "verification.json")
    check("same_point_reference_verified_and_original_source_bound", reference_proof["all_passed"]
          and reference_result["point_identity"] == "direct_original_projector_times_archived_recovered_x"
          and reference_result["original005_source_map_sha256"] == fixed["source_map_sha256"]
          and reference_result["original_metric_sha256"] == fixed["original_projector_and_preconditioner_metric_sha256"])
    check("reference_proof_does_not_claim_old_current_full_g", reference_result["comparisons"][
          "complete_old_current_elementwise_g_comparison_obtained"] is False)
    archive = np.load(reference_dir / "arrays.npz")
    recovered = np.load(ROOT / (PREFIX + "native_run/nuclear_eol005_projected_increment/stationarity_replay002/recovered_coordinates.npz"))
    metric_dir = ROOT / (PREFIX + "native_run/nuclear_eol004_stable_delta_trace")
    layout = load(metric_dir / "failed_coordinate_layout.json")
    metric = np.load(metric_dir / "failed_coordinates.npz")
    for name, shape in (("x", (2856,)), ("g", (2856,)), ("preconditioned_gradient", (2856,)),
                        ("P", (24, 123)), ("Q", (24, 123)), ("radial_gradient", (24, 246))):
        check("full_reference_array:" + name, archive[name].dtype == np.float64
              and archive[name].shape == shape and np.isfinite(archive[name]).all())
    check("authoritative_recovered_x_exact", np.array_equal(archive["x"], recovered["x"]))
    keys = [tuple(k) for k in archive["orbital_keys"]]
    key_index = {key: i for i, key in enumerate(keys)}
    check("complete_reference_keys", len(keys) == len(set(keys)) == 24
          and set(keys) == {tuple(k) for row in layout for k in row["keys"]})
    g, pg = np.zeros(2856), np.zeros(2856)
    metric_digest = hashlib.sha256()
    for i, row in enumerate(layout):
        group = [tuple(k) for k in row["keys"]]
        shape, where = tuple(row["shape"]), slice(*row["slice"])
        p, y = metric[f"projector_{i}"], archive["x"][where].reshape(shape)
        waves = p @ y
        for col, key in enumerate(group):
            n = key_index[key]
            check(f"exact_direct_original_P_x:{key}", np.array_equal(waves[:123, col], archive["P"][n])
                  and np.array_equal(waves[123:, col], archive["Q"][n]))
        raw = p.T @ archive["radial_gradient"][[key_index[k] for k in group]].T
        g_block = raw - .5 * y @ (y.T @ raw + raw.T @ y)
        g[where] = g_block.ravel()
        # Block-diagonal matrix action differs from producer's column loop.
        blocks = metric[f"inverse_blocks_{i}"]
        ncol, nrow = shape[1], shape[0]
        full = np.zeros((ncol * nrow, ncol * nrow))
        for col, block in enumerate(blocks):
            start = col * nrow
            full[start:start + nrow, start:start + nrow] = block
        z = (full @ g_block.T.ravel()).reshape(ncol, nrow).T
        pg[where] = (z - .5 * y @ (y.T @ z + z.T @ y)).ravel()
        metric_digest.update(repr((group, shape)).encode())
        metric_digest.update(np.ascontiguousarray(p).tobytes())
    metric_digest.update(metric["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for block in metric[f"inverse_blocks_{i}"]:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
    check("legacy_original_metric_SHA_exact", metric_digest.hexdigest()
          == prospective["fixed_conditions"]["original_metric_sha256"])
    g_error, pg_error = float(np.max(abs(g - archive["g"]))), float(np.max(abs(pg - archive["preconditioned_gradient"])))
    check("saved_full_radial_to_tangent_g_reconstruction", g_error <= 1e-12, g_error)
    check("saved_original_inverse_pg_independent_action", pg_error <= 1e-14, pg_error)
    target = load(ROOT / (PREFIX + "balanced_cv_replay/stage_4f5p5s/target_binding.json"))["targets"]
    selectors = [(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in target]
    check("same_five_reference_selectors_and_weights", set(selectors) == set(map(tuple, reference_result["physical_model"]["selectors"]))
          and len(selectors) == 5 and reference_result["physical_model"]["state_weights"] == [.2] * 5
          and np.array_equal(archive["state_weights"], np.repeat(.2, 5)))
    driver_text = (consumer / "run.py").read_text()
    verifier_text = (consumer / "verify_endpoint.py").read_text()
    driver_ast, verifier_ast = ast.parse(driver_text), ast.parse(verifier_text)
    optimizer_calls = [n for n in ast.walk(driver_ast) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == "optimize_orbitals"]
    check("exactly_one_public_optimizer_invocation", len(optimizer_calls) == 1)
    kwargs = {kw.arg: kw.value for kw in optimizer_calls[0].keywords}
    for name, value in (("allow_objective_change", True), ("return_state", True), ("tolerance", 1e-7),
                        ("energy_tolerance", 1e-8), ("projector_source", "nuclear"),
                        ("occupation_floor", .05), ("energy_difference", None), ("linear_energy_increments", False)):
        check("public_call_keyword:" + name, ast.literal_eval(kwargs[name]) == value)
    check("public_call_original_state_and_checkpoint", isinstance(kwargs["resume_state"], ast.Name)
          and kwargs["resume_state"].id == "initial" and kwargs["checkpoint_callback"].id == "checkpoint")
    check("no_optimizer_in_independent_verifier", not any(isinstance(n, ast.Call)
          and isinstance(n.func, ast.Name) and n.func.id == "optimize_orbitals" for n in ast.walk(verifier_ast)))
    control_dict = next(n for n in ast.walk(driver_ast) if isinstance(n, ast.Dict)
                        and any(isinstance(k, ast.Constant) and k.value == "baseline_gradient_calls" for k in n.keys if k))
    controls = {k.value: ast.get_source_segment(driver_text, v) for k, v in zip(control_dict.keys, control_dict.values) if isinstance(k, ast.Constant)}
    check("whole_vector_raw_g_same_point_comparison", controls["bare_same_recovered_point_gradient_replay_max"]
          == "float(np.max(abs(bare_g - reference['g'])))")
    check("whole_vector_pg_same_point_comparison", "bare_preconditioned" in controls["bare_same_recovered_point_preconditioned_gradient_replay_max"]
          and "reference['preconditioned_gradient']" in controls["bare_same_recovered_point_preconditioned_gradient_replay_max"]
          and "np.max(abs(" in controls["bare_same_recovered_point_preconditioned_gradient_replay_max"])
    check("matched_vector_gate_precedes_single_optimizer_call", driver_text.index("New-source matched baseline failed")
          < driver_text.index("final_waves, final_ci, diagnostics = optimize_orbitals"))
    for relative, expected in prospective["input_sha256"].items():
        check("unchanged_after_readonly_controls:" + relative, sha(ROOT / relative) == expected)
    check("no_atomx_imported", not any(name == "atomx" or name.startswith("atomx.") for name in __import__("sys").modules))
    result = {"all_passed": all(row["passed"] for row in checks), "checks_count": len(checks), "checks": checks,
              "audit_contract_sha256": sha(HERE / "contract.json"), "auditor_script_sha256": sha(Path(__file__)),
              "frozen_successor_source_map_sha256": canonical, "saved_array_g_error_max": g_error,
              "saved_array_pg_error_max": pg_error, "physical_H_evaluations": 0, "CI_evaluations": 0,
              "radial_gradient_evaluations": 0, "optimizer_invocations": 0,
              "scope": "source AST, immutable hashes and saved-array linear algebra only; not a science execution",
              "resource_admission": admission, "elapsed_s": time.monotonic() - started,
              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    signal.alarm(0)
    print(json.dumps({key: result[key] for key in ("all_passed", "checks_count", "saved_array_g_error_max",
                                                  "saved_array_pg_error_max", "elapsed_s", "peak_RSS_MiB")}))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
