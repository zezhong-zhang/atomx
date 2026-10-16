"""Metadata-only preparation proof; never run auxiliary field, radial, spectral or CI numerical consumers."""
from __future__ import annotations

import ast
import hashlib
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    rows = []

    def check(name, value):
        rows.append(dict(name=name, passed=bool(value)))
        if not value:
            raise ValueError(name)

    c = load(HERE / "contract.json")
    models = load(HERE / "models.json")
    check("finite_models_only_baseline_and_balanced_virtual", list(models["models"]) == ["baseline", "balanced_virtual"])
    for name, counts in (("baseline", (21, 531, 261, 24)), ("balanced_virtual", (69, 2163, 1046, 29))):
        m = models["models"][name]
        check(name + "/bound_model_counts", (len(m["configs"]), m["full_CSFs"], m["complete_target_CSFs"], m["radial_spinors"]) == counts)
        check(name + "/complete_sector_counts_sum", sum(b["CSFs"] for b in m["complete_target_blocks"]) == m["complete_target_CSFs"])
    check("old21_NR_configs_unmodified_prefix", models["models"]["balanced_virtual"]["configs"][:21] == models["models"]["baseline"]["configs"])
    previous = load(TOP / "rank_vs_virtual_discriminator/models.json")
    check("original_complete_virtual_and_baseline_definitions", all(models["models"][name] == previous["models"][name] for name in models["models"]))
    check("same_fixed_sectors", models["selected_sectors"] == c["fixed_conditions"]["sectors"])
    caps, gates = c["resources"], c["acceptance"]
    check("original_caps_not_relaxed", (caps["CPU_threads"], caps["address_space_cap_GiB"], caps["whole_execution_wall_cap_s"], caps["max_signed_overlap_determinant_pairs"], caps["max_raw_determinants_per_block"]) == (1, 4, 2040, 1000000, 2000))
    check("finite_native_call_budget", (caps["complete_CI_calls_budget"], caps["total_many_electron_H_builds_budget"], caps["screened_complement_SVD_calls_budget"], caps["screened_original_P_full_gvd_spectral_controls"], caps["initial_one_body_matrix_kappa_count"], caps["virtual_control_hS_consumption_count"], caps["additional_virtual_control_hS_builds"]) == (2, 4, 5, 5, 7, 5, 0))
    check("zero_forbidden_new_consumers", all(caps[k] == 0 for k in ("public_virtual_generator_calls_budget", "nuclear_Dirac_spectral_calls", "orbital_gradients_budget", "optimizer_calls_budget", "nuclear_complement_lowest_root_spectral_calls", "original_P_complement_lowest_two_spectral_controls")))
    check("original_minimum_leakage_and_stationarity_gates", (gates["screened_complement_minimum_energy_match_Ha"], gates["nuclear_projector_leakage_S_norm_squared"], gates["original_stationarity_residual"]) == (1e-8, 1e-12, 1e-7))
    virtual = c["fixed_conditions"]["new_virtuals"]
    core = virtual["closed_core"]
    check("core19_closed68", len(core) == 19 and sum(r["occupation"] for r in core) == 68
          and all(r["occupation"] == 2 * abs(r["key"][1]) for r in core))
    check("auxiliary_budget", (caps["screened_JK_builder_calls_budget"], caps["closed_core_one_body_calls_budget"],
          caps["synthetic_probe_count"], caps["synthetic_ordered_same_kappa_bilinear_count"],
          caps["scalar_radial_callback_calls_budget"], caps["scalar_Slater_calls_budget"]) == (10, 1, 20, 80, 5000, 5000))
    check("no_constructor_SCF_radial_snapshot", all(caps[k] == 0 for k in ("DFS_constructor_calls", "SCF_calls", "radial_snapshot_calls")))
    check("CI_internal_hS_count_declared_separate", caps["internal_CI_onebody_builds_budget"] == 28)
    scalar_count = 0
    for k in (-3, -2, -1, 1, 2):
        ell = k if k > 0 else -k - 1
        rank_count = 0
        for row in core:
            j = row["key"][1]
            ell_j = j if j > 0 else -j - 1
            rank_count += sum((r + ell + ell_j) % 2 == 0
                              for r in range(abs(abs(k) - abs(j)), abs(k) + abs(j)))
        scalar_count += 16 * (len(core) + rank_count)
    check("scalar_kinematic_callback_bound", scalar_count == 3584
          and scalar_count == caps["scalar_radial_callback_kinematic_upper_bound"]
          and scalar_count <= caps["scalar_radial_callback_calls_budget"])
    check("unchanged_projector_not_screened_boundary", virtual["nuclear_branch"].startswith("Original nuclear projector bytes"))
    helper = (HERE / "screened_operator.py").read_text()
    helper_tree = ast.parse(helper)
    check("helper_no_top_level_physics_import", not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in helper_tree.body))
    helper_calls = [n.func.id for n in ast.walk(helper_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    check("helper_finite_scalar_call_sites", helper_calls.count("compute_slater_rk_cell") == 1
          and helper_calls.count("closed_core_one_body") == 1)
    check("helper_no_DFS_constructor_or_spectral_or_CI", not set(helper_calls) & {"DFSolver", "ci_solve", "build_ci_matrix", "eigh", "null_space", "solve_dirac", "CIRadialSnapshot"})
    check("helper_native_full_four_quadrant_and_hN_once", 'skip_pq_exchange=False' in helper
          and 'np.block([[pp, pq], [qp, qq]])' in helper and 'full = h_nuclear + jk' in helper)
    check("helper_adapter_cache_and_order_guards", all(x in helper for x in
          ("DFSolver.__new__(DFSolver)", "geometry.cache.same_grid(data)", "auxiliary_all_core_waves_present",
           "auxiliary_complete_same_kappa_source_order", "auxiliary_source_shape/")))
    check("helper_independent_nonzero_PQ_QP_qualification", 'auxiliary_measured_cross_anchor/' in helper
          and 'auxiliary_scalar_cross_relative/' in helper and 'renormalized=False' in helper)
    source = (HERE / "run.py").read_text()
    tree = ast.parse(source)
    check("no_physical_top_level_import", all(not isinstance(node, (ast.Import, ast.ImportFrom)) or not (getattr(node, "module", "") or "").startswith(("numpy", "scipy", "atomx")) for node in tree.body))
    call_names = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    check("no_generator_Dirac_gradient_optimizer_call_sites", not set(call_names) & {"extend_nuclear_virtuals", "solve_dirac", "optimize_orbitals", "gradient"})
    check("exact_scientific_call_sites_in_finite_loops", all(call_names.count(k) == 1 for k in ("null_space", "eigh", "ci_solve", "build_ci_matrix")))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "eigh"]
    check("only_full_gvd_vectors_no_subset", len(calls) == 1 and {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {"eigvals_only": False, "driver": "gvd"})
    check("native_arrays_saved_before_gvd_and_gates", source.index('"_native_complement.npz"') < source.index('reported, coefficients = eigh') < source.index('"_full_gvd_vectors.npz"') < source.index('"virtual_full_vector_norms_positive/"'))
    namespace = runpy.run_path(str(HERE / "run.py"), run_name="prepared_metadata_only")
    _, _, checkpoint, proof, _ = namespace["metadata"]()
    check("metadata_all_inputs_full353_and_original_tracker_pass", checkpoint["iteration"] == 160 and proof["original_stationarity_gates_passed"] is False)
    check("metadata_no_Atomx_NumPy_SciPy_imports", not any(n == "atomx" or n.startswith(("atomx.", "numpy", "scipy")) for n in sys.modules))
    fixed = c["fixed_conditions"]
    proposal = dict(execution_authorized_by_root=False, pending_exact_root_execution_contract=True,
                    input_status="frozen_verified_public531_checkpoint", preparation_contract_sha256=sha(HERE / "contract.json"),
                    driver_sha256=sha(HERE / "run.py"), scientific_inputs_sha256=c["inputs_sha256"],
                    expected_checkpoint_manifest_sha256=fixed["checkpoint"]["manifest_sha256"],
                    expected_checkpoint_objective_sha256=fixed["checkpoint"]["objective_sha256"],
                    expected_source_map_sha256=fixed["source_map_sha256"],
                    expected_card_delivery_proof_sha256=c["iteration_trigger"]["delivery_proof_sha256"],
                    actual_iteration_card_message_id=c["iteration_trigger"]["message_id"],
                    tracker_inputs_attested_original_objective_by_root=False,
                    preparation_only_no_scientific_execution=True)
    proposal_path = HERE / "execution_binding_proposal.json"
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
    try:
        namespace["authorize"](proposal_path)
    except ValueError as error:
        check("pending_proposal_fails_before_science", "NEW root binding" in str(error))
    else:
        raise AssertionError("Pending proposal erroneously authorized")
    check("no_scientific_output_artifacts", not any((HERE / n).exists() for n in ("result.json", "failure.json", "progress.json", "native_hS_original_geometry.npz")))
    proof = dict(status="metadata_preparation_passed_pending_NEW_root_binding", all_preparation_checks_passed=True,
                 checks_count=len(rows), checks=rows, physical_imports=0, scientific_calls=0,
                 driver_sha256=sha(HERE / "run.py"), contract_sha256=sha(HERE / "contract.json"),
                 models_sha256=sha(HERE / "models.json"), auxiliary_helper_sha256=sha(HERE / "screened_operator.py"),
                 exact_scalar_kinematic_callback_bound=scalar_count, actual_card_message_id=c["iteration_trigger"]["message_id"], preparation_driver_sha256=sha(Path(__file__).resolve()),
                 execution_binding_proposal_sha256=sha(proposal_path), frozen_source_map_sha256=fixed["source_map_sha256"],
                 original_stationarity_passed=False, expanded_stationarity_evaluated=False,
                 numerical_virtual_minima_measured_in_this_successor=False, experimental_gate_passed=False)
    (HERE / "preparation_verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "checks"}))


if __name__ == "__main__":
    main()
