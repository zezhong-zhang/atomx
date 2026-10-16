"""Create prospective metadata only: no NumPy/Atomx or scientific evaluation."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
ROOT = TOP.parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    if path.exists():
        raise FileExistsError("Preserve prospective metadata: " + name)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    old = load(TOP / "target_sector_equivalence002/contract.json")
    binding = load(TOP / "target_sector_equivalence002/execution_binding.json")
    source = load(TOP / "public_mixed_resume_run002/source_identity.json")
    kin = load(TOP / "physical_frontier004/kinematic_preflight.json")
    basis = load(TOP / "balanced_cv_replay/basis_identity.json")
    static = load(TOP / "public531_endpoint_static_rci/result.json")
    staticproof = load(TOP / "public531_endpoint_static_rci/saved_array_verification.json")
    core = basis["configs"][0]["config"].rsplit(" ", 1)[0]
    base, extra = ["6s", "5d", "6p"], ["7s", "6d", "7p"]

    def configuration(hole, count, particles):
        occupations = {}
        for token in core.split():
            position = next(i for i, c in enumerate(token) if c.isalpha()) + 1
            occupations[token[:position]] = int(token[position:])
        if hole:
            occupations[hole] -= count
        ordering = list(occupations) + base + extra
        for shell in particles:
            occupations[shell] = occupations.get(shell, 0) + 1
        return " ".join(f"{key}{occupations[key]}" for key in ordering
                        if occupations.get(key, 0))

    baseline = [{"family": row["family"], "config": row["config"]}
                for row in basis["configs"]]
    rank = baseline + [{"family": "5p_double", "config": configuration("5p", 2, triple)}
                       for triple in itertools.combinations_with_replacement(base, 3)
                       if triple.count("6s") <= 2]
    virtual = baseline + [{"family": "bare_new_virtual", "config": configuration(None, 0, [shell])}
                          for shell in extra]
    old_configs = {row["config"] for row in baseline}
    for hole in ("4f", "5p", "5s"):
        for pair in itertools.combinations_with_replacement(base + extra, 2):
            config = configuration(hole, 1, pair)
            if config not in old_configs:
                virtual.append({"family": hole + "_new_virtual", "config": config})
    if len(rank) != 30 or len(virtual) != 69 or len({row["config"] for row in virtual}) != 69:
        raise ValueError("Complete NR configuration count changed")
    names = [("baseline", "baseline531", baseline),
             ("rank_5p_double", "baseline_plus_complete_5p_double_holes", rank),
             ("balanced_virtual", "baseline_plus_one_balanced_sdp_virtual_layer", virtual)]
    models = {"core_NR": core, "base_particle_shells": base, "new_particle_shells": extra,
              "selected_sectors": [[1, -1], [1, 1], [3, -1], [3, 1], [5, 1]], "models": {}}
    for name, archive, rows in names:
        counts = kin["models"][archive]
        target = counts["five_complete_target_sectors"]
        models["models"][name] = dict(
            configs=rows, kinematic_archive_key=archive, full_CSFs=counts["CSFs"],
            radial_spinors=counts["radial_spinor_count"], full_blocks=counts["symmetry_blocks"],
            complete_target_CSFs=target["CSFs"],
            complete_target_blocks=counts["target_symmetry_blocks"],
            raw_pair_work_upper=target["raw_pair_work_upper"],
            dense_CSF_block_bytes=target["dense_CSF_matrix_bytes"],
            dense_raw_block_bytes=target["dense_raw_matrix_bytes"],
            max_selected_raw_determinants=max(row["raw_M_equals_J_determinants"]
                                             for row in counts["target_symmetry_blocks"]),
            dense_selected_full_matrix_bytes=8 * target["CSFs"]**2)
    write("models.json", models)
    evidence = dict(
        point="accepted_0160 unchanged despite subsequent continuation",
        source_map_sha256=source["source_map_sha256"], checkpoint=old["fixed_conditions"]["checkpoint"],
        DC_rows=static["DC_rows"], static_rows=static["static_rows"], multiplets=static["multiplets"],
        static_result_sha256=sha(TOP / "public531_endpoint_static_rci/result.json"),
        static_independent_proof_sha256=sha(TOP / "public531_endpoint_static_rci/saved_array_verification.json"),
        static_independent_saved_array_checks=staticproof["checks_count"], no_new_physical_calls=True,
        missing_same_point_quantities={name: None for name in (
            "5p_double_response", "balanced_virtual_response", "combined_rank_virtual_response",
            "core4f_double_or_mixed_core_double_response", "same_state_dynamic_Breit_QED_recoil",
            "nuclear_vs_screened_no_pair_boundary_response", "expanded_model_orbital_stationarity")},
        reading="14.148Ha absolute static shift cancels strongly. Only measured upper-minus-lower "
        "2.384/-28.751/-20.697/-44.262cm^-1 belongs to the same-point spectrum. "
        "Nominal gaps are not missing-physics bounds. No003 corrections are consumed.")
    write("same_point_evidence.json", evidence)
    inputs = {row["path"]: row["sha256"] for row in binding["files"].values()}
    for relative in (
            "target_sector_equivalence002/run.py", "target_sector_equivalence002/contract.json",
            "target_sector_equivalence002/execution_binding.json", "target_sector_equivalence002/result.json",
            "target_sector_equivalence002/saved_array_audit/verification.json",
            "physical_frontier004/kinematic_preflight.json", "physical_frontier004/count_spaces.py",
            "physical_frontier004/verification.json", "public531_endpoint_static_rci/result.json",
            "public531_endpoint_static_rci/saved_array_verification.json",
            "public531_endpoint_static_rci/contract.json", "rank_vs_virtual_discriminator/models.json",
            "rank_vs_virtual_discriminator/same_point_evidence.json"):
        path = TOP / relative
        inputs[str(path.relative_to(ROOT))] = sha(path)
    critical = ["numerics/bspline/orbital_subspace.py", "numerics/bspline/dirac_assembly.py",
                "angular/csf_hamiltonian.py", "config/csf.py", "methods/ci.py", "methods/csf/variational.py",
                "physics/csf/matrix.py", "physics/csf/tracking.py", "physics/csf/overlap.py",
                "physics/csf/levels.py", "potentials/nuclear.py", "constants.py"]
    fixed = {key: old["fixed_conditions"][key] for key in (
        "Z", "electron_count", "ns", "ks", "radial_keys", "original_metric_sha256", "checkpoint")}
    fixed.update(
        frozen_source_root=binding["frozen_source_root"], source_map_sha256=source["source_map_sha256"],
        full_atomx_source_count=353, nucleus="Original native005 Fermi A175; byte-bound workspace",
        operator="Original scalar native no-pair Dirac-Coulomb; no Breit/QED/recoil/effective/empirical shifts",
        original_tracker="Original balanced531 DC CI and native005 radial reference; no endpoint rebase",
        old24_wave_policy="Keep exact accepted160 P/Q, authoritative x, original P/inverse bytes unchanged",
        point_selection="accepted0160 chosen prospectively; later accepted0320 must not substitute",
        stationarity="Inherited residual4.717047646600387e-5 FAIL1e-7; indicator6.8897239137767995e-9 "
        "PASS1e-8. Extensions are not evaluated for stationarity.",
        targets="Original five physical targets, equal0.2; unchanged signed-overlap/ambiguity/degeneracy gates",
        sectors=models["selected_sectors"],
        sector_scope="Complete261/1240/1046 target sectors of full531/2483/2163; no amplitude/configuration "
        "pruning or all-spectrum/globalground claim",
        virtual_generator=dict(
            public_function="atomx.numerics.bspline.orbital_subspace.extend_nuclear_virtuals",
            layers=1, kappas=[-3, -2, -1, 1, 2], new_keys=[[6, -3], [6, 2], [7, -2], [7, -1], [7, 1]],
            nuclear_branch="Same electronic branch above negative-continuum edge and original min-n/kappa "
            "shell exclusion as savedP; not energy>0",
            minimization="Lowest nuclear Rayleigh quotient in retained positive span orthogonal to all "
            "frozen same-kappa spinors; never adjust oldwaves",
            phase="Jointly flip new P/Q so largest-absolute original-P coordinate is positive; save raw sign",
            radial_label="Consecutive correlation-orbital bookkeeping only; no spectroscopic/frequency assignment",
            gap_control="Original-P complement lowest-two eigenvalues; refuse minimum cluster<=1e-10Ha"))
    resources = dict(
        CPU_threads=1, GPU=False, address_space_cap_GiB=4, max_raw_determinants_per_block=2000,
        max_signed_overlap_determinant_pairs=1_000_000, wall_cap_each_point_s=900,
        preflight_and_virtual_wall_cap_s=240, whole_execution_wall_cap_s=2940,
        complete_CI_calls_budget=3, implicit_public_CI_H_builds_budget=3,
        explicit_archive_H_builds_budget=3, total_many_electron_H_builds_budget=6,
        orbital_gradients_budget=0, optimizer_calls_budget=0, public_virtual_generator_calls_budget=1,
        nuclear_Dirac_spectral_calls=5, nuclear_complement_lowest_root_spectral_calls=5,
        original_P_complement_lowest_two_spectral_controls=5, initial_one_body_matrix_kappa_count=7,
        minimum_available_affinity_CPUs=3, minimum_host_available_GiB=6.0,
        max_load1_per_available_CPU=0.8,
        affinity_selection="Exact sorted original affinity[-2], fail below3; sequential one-process three points",
        performance_claim=False,
        cost_limit="Raw-pair counts/dense bytes are not FLOPs/RSS/runtime; unknown timings, no silent changes")
    acceptance = dict(
        old24_waves_bitwise_equal=True, original_metric_SHA_equal=True, actual_radial_Gram_error=1e-10,
        saved_P_metric_Gram_error=1e-10, nuclear_projector_leakage_S_norm_squared=1e-12,
        new_spinor_projected_nuclear_residual_relative=1e-8,
        nuclear_complement_minimum_energy_match_Ha=1e-8, nuclear_minimum_cluster_gap_Ha=1e-10,
        CI_residual_Ha=1e-8, CI_orthogonality=1e-10, matrix_hermiticity_Ha=1e-10,
        cross_symmetry_zero_Ha=1e-14, baseline_saved_H_replay_Ha=5e-9,
        baseline_saved_target_spectrum_replay_Ha=5e-9, nested_signed_old261_principal_H_error_Ha=5e-9,
        nested_ordered_Ritz_eigenvalue_increase_Ha=1e-8,
        public_tracking_squared_principal_strict_min=0.5, public_tracking_score_boundary_strict_min=1e-10,
        original_stationarity_residual=1e-7, original_energy_indicator_Ha=1e-8,
        nominal_relative_NIST_goal=0.01, numeric_success_is_not_experimental_or_stationarity_success=True)
    procedure = [
        "Before imports validate NEW independent root binding contract/driver/allinputs/full353 frozen10d481 source.",
        "Record host admission before exact affinity[-2] pin, install caps, count attempted calls, preserve failure.",
        "Public complete NR/jj enumeration/signatures/electrons/keys/counts; exact raw-support and grouped "
        "signed-overlap pair budgets before any H/CI/virtual generation. Never truncate to fit.",
        "Exact accepted160 x/P/inverse/oldwaves; public functional geometry H/S only, no evaluation; "
        "original P/S/no-pair/Gram controls.",
        "One public five-kappa nuclear virtual extension; original-P complement/gap/minimum/sign/Gram/no-pair "
        "checks; archive raw and canonical vectors, all old24 bitwise unchanged.",
        "Exactly3 public complete dense ci_solve calls (3 implicitH) and3 explicit public matrix archives "
        "for residual/nestedH controls. Save complete arrays before target/numeric checks; no gradients/optimizer.",
        "Same immutable original tracker for each arm; same equal0.2 physical targets/majority/ambiguity/cluster "
        "rules. Expanded local root may differ while original physical reference stays fixed.",
        "Report lower/upper absolute shifts and transition(deltaEupper-deltaElower), D/P degeneracy centroids/" 
        "splittings, four nominal gaps, nested-H/Ritz controls. Separate DC arms; no addition to static531 or003QED."]
    contract = dict(
        record_id="frozen-accepted160-native-rank-versus-balanced-virtual-three-point",
        created_before_physical_execution_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="prepared_only_pending_independent_root_execution_binding",
        hypothesis="Complete5p double-hole and a balanced native positive virtual layer give distinguishable "
        "same-point lower/upper, centroid and splitting responses; discriminate these axes before choosing "
        "the next correlated MCDHF space.", fixed_conditions=fixed, resources=resources, acceptance=acceptance,
        inputs_sha256=inputs, input_roles=binding["files"], procedure=procedure,
        audited_critical_source_sha256={"src/atomx/"+key: source["all_atomx_python_sha256"]["src/atomx/"+key]
                                       for key in critical},
        distinguishing_outcomes={
            "rank_response_larger": "Prioritize complete5p double-hole retention in subsequent native MCDHF; "
            "does not prove all-core rank dominates.",
            "virtual_response_larger": "Prioritize balanced radial extension for subsequent native MCDHF; "
            "one nuclear correlation layer does not prove radial convergence.",
            "both_comparable_or_opposing": "Need separately contracted combined arm to measure nonadditivity.",
            "both_small": "These finite axes do not explain current gap; next viable complete4f/mixed-core "
            "rank or boundary/operator discriminator needed; no physics-completeness inference."},
        decision_reporting="Measured per-line response signs, RMS, centroids/splittings and nominal gap "
        "reduction; no forced winner. Effects below the measured replay/nested numerical scale are unresolved.",
        stopping=["Any identity/resource/complete-block/numeric/signedtracking failure: archive and stop, "
                  "no newcalls/backend/pruning/threshold changes.",
                  "Stop after3points; new MCDHF/combinedspace/operators/boundary/grid require newcontract."],
        not_claimed=["Expanded MCDHF stationarity", "Missing-physics/convergence bound",
                     "Experimental/all-spectrum/globalground certificate", "Additivity onto static/003",
                     "Controlled/parallel/GPU speedup", "Nuclear boundary equivalent to screened/GRASP"])
    write("contract.json", contract)
    write("input_binding_proposal.json", dict(
        record_id="root-review-required-native-rank-versus-virtual", execution_authorized_by_root=False,
        input_status="frozen_verified_public531_checkpoint", pending_exact_root_execution_contract=True,
        preparation_contract_sha256=sha(HERE / "contract.json"), driver_sha256=None,
        expected_checkpoint_manifest_sha256=fixed["checkpoint"]["manifest_sha256"],
        expected_checkpoint_objective_sha256=fixed["checkpoint"]["objective_sha256"],
        expected_source_map_sha256=source["source_map_sha256"],
        tracker_inputs_attested_original_objective_by_root=False, scientific_inputs_sha256=inputs,
        review_required="Root writes a NEW exact reviewed binding. This proposal and oldsector approval "
        "never authorize new science."))
    print(json.dumps({"contract_sha256": sha(HERE / "contract.json"), "no_physical_import_or_evaluation": True}))


if __name__ == "__main__":
    main()
