"""Prepare SOURCE/JSON-only finite saved-matrix diagnostic; no NPZ access."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "execution_preparation002"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


if (HERE / "preparation_manifest.json").exists():
    raise FileExistsError("Frozen diagnostic preparation exists; preserve it")
old = read(PARENT / "contract.json")
failure = read(PARENT / "actual/failure.json")
roles = {
    key: old["input_roles"][key]
    for key in (
        "source_identity",
        "new_layout",
        "new_objective",
        "models",
        "canonical_basis",
        "reference_models",
        "tracking_binding",
        "baseline_owner",
        "baseline_basis",
        "baseline_state",
        "stageB_contract",
        "stageB_result",
        "stageB_receipt",
        "literal13_manifest",
        "native_result",
        "native_receipt",
        "prepared_models",
        "baseline_whole_CI",
        "tracking_ci",
        "geometry_workspace",
        "screened_geometry",
    )
}
for role, relative in {
    "parent_contract": "contract.json",
    "parent_driver": "run.py",
    "parent_manifest": "preparation_manifest.json",
    "parent_receipt": "root_execution.json",
    "parent_failure": "actual/failure.json",
    "parent_nested_H_control": "actual/nested_H_control.json",
    "parent_basis_maps": "actual/complete_basis_and_configuration_maps.json",
    "parent_nested_embedding": "actual/nested_embedding.json",
    "parent_baseline_point": "actual/baseline_conditional_point.json",
    "parent_baseline_tracking": "actual/baseline_tracking.json",
    "parent_signed_admission": "actual/full_signed_overlap_admission.json",
    "parent_representation_controls": "actual/source_representation_controls.json",
}.items():
    path = PARENT / relative
    roles[role] = dict(
        path=str(path.relative_to(ROOT)), sha256=sha(path), deferred_numerical=False
    )
row = next(
    row for row in failure["archives"] if row["path"] == "expanded_complete_H.npz"
)
roles["saved_expanded_H"] = dict(
    path=str((PARENT / "actual/expanded_complete_H.npz").relative_to(ROOT)),
    sha256=row["sha256"],
    deferred_numerical=True,
)
plans = {
    key: old["array_access"][key]
    for key in (
        "baseline_whole_CI",
        "tracking_ci",
        "geometry_workspace",
        "screened_geometry",
    )
}
plans["geometry_workspace"] = dict(
    plans["geometry_workspace"], basis_prefix=False, fields_max=3
)
plans["saved_expanded_H"] = dict(
    compressed_bytes_max=67108864,
    fields_max=4,
    fields={
        "H": dict(shape=[2574, 2574], dtype="<f8"),
        "keys": dict(shape=[31, 2], dtype="<i8"),
        "P": dict(shape=[31, 123], dtype="<f8"),
        "Q": dict(shape=[31, 123], dtype="<f8"),
    },
)
counts = read(ROOT / roles["prepared_models"]["path"])["counts"]
contract = dict(
    status="SOURCE_ONLY_saved_matrix_diagnostic_ready_for_root_review",
    execution_authorized=False,
    diagnostic_label="SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC",
    parent_directory=str(PARENT.relative_to(ROOT)),
    source_archive=old["source_archive"],
    source_map_sha256=old["source_map_sha256"],
    full_source_count=356,
    fixed=old["fixed"],
    input_roles=roles,
    array_access=plans,
    upfront_census=dict(
        NR_configs=[91, 100],
        parent_CSFs=[3789, 5741],
        selected_CSFs=[1595, 2574],
        radial_keys=31,
        electrons=69,
        radial_dimension=123,
        complete_blocks=counts["blocks"],
        original_tracker_reference_CSFs=531,
        new_complete_block_eigh=5,
        new_H=0,
        radial_key_evaluations=0,
        gradients=0,
        Dirac_spectra=0,
        optimizer=0,
    ),
    resources=dict(
        CPU=27,
        threads=1,
        AS_GiB=2.5,
        inner_seconds=180,
        outer_seconds=195,
        minimum_host_available_GiB=3,
        max_input_archive_loads=5,
        max_input_decoded_bytes=134217728,
        max_complete_block_eigh=5,
        max_tracker_signed_overlaps=2,
        max_tracker_resolve=1,
        max_tracking_SVD=5,
        max_public_projections=20,
        max_angular_null_space=1024,
        max_signed_overlap_pairs=1000000,
        expected_reference_self_pairs=1525,
        expected_reference_expanded_pairs=10132,
        exact_difference_entry_capacity=1595 * 1595,
        new_H=0,
        new_hS=0,
        radial_key_evaluations=0,
        gradients=0,
        Dirac_spectra=0,
        optimizer=0,
        timing_claim_valid=False,
        caps_are_stops_not_completion_guarantees=True,
    ),
    diagnostic_acceptance=dict(
        exact_input_shapes_types_finite=True,
        stored_full_H_exact_symmetric=True,
        full_same_radial_key_PQ_bytes=True,
        complete_signed_basis_maps_and_embedding=True,
        baseline_CI_residual_Ha=1e-8,
        baseline_CI_Gram=1e-10,
        snapshot_CI_residual_Ha=1e-8,
        snapshot_CI_Gram=1e-10,
        cross_symmetry_zero_Ha=1e-14,
        original_public_tracker_majority_strict_min=0.5,
        original_public_tracker_unique_boundary_strict_min=1e-10,
        original_public_tracker_cluster_gap_Ha=1e-10,
        equal_five_UID_weights=0.2,
        inherited_exact_nested_H_gate_remains_failed=True,
        no_native_model_or_source_qualification=True,
        no_stationarity_upgrade=True,
        nominal_experimental_improvement_required=False,
    ),
    prerequisite=dict(
        parent_closed_exit1=True,
        parent_failed_gate="expanded_exact_nested1595_H_bytes",
        parent_control_census=dict(PASS=157, FAIL=1),
        parent_new_CI_completed=0,
        parent_complete_CI_eigh=0,
        saved_native_H_sha256=row["sha256"],
        parent_max_nested_entry_difference_Ha=8.881784197001252e-16,
        stageB_ordinary_FD_and_original_stationarity_failures_preserved=True,
    ),
    finite_bound=dict(
        definition="Delta_ij = Fraction.from_float(saved_H[embed_i,embed_j]) - Fraction.from_float(baseline_H[i,j]); subtract exact represented binary64 operands, never a rounded subtraction.",
        row_sum="For all 1595 rows and every mathematically nonzero entry: r_i=sum_j abs(Delta_ij) exactly in Fraction. rho=max_i r_i; also each complete sector rho_s. Save all exact row ratios, affected rows, entry count/max and symmetry checks.",
        proof="Both matrices are finite real symmetric. For any eigenvector of Delta, choose its largest component: abs(lambda)<=max row absolute sum=rho, hence norm2(Delta)<=rho. Injecting this old-block Delta into the full2574 space preserves the nonzero spectrum and gives the same norm bound. Variational min-max bounds each ordered eigenvalue change by its sector rho_s.",
        comparison_operator="Only a conceptual finite K=H_saved-E Delta E^T for the theorem. Never construct, save, diagonalize, inject or use K; all actual old-to-new Q couplings and new Q-Q entries remain the native saved values.",
        excitation_bound="For fixed ordered upper/lower sector roots, absolute change due solely to nested arithmetic <=rho_upper+rho_lower. Centroid uses declared (2J+1) weights; splitting <=rho_a+rho_b. Bound is for exact finite matrix eigenvalues, excludes numerical eigensolver error, physical truncation error and physical-state re-identification under hypothetical K.",
        units="Hartree exact dyadic ratios; cm^-1 bound uses the exact rational of the represented native HARTREE_CM1 scalar, not a metrological constant interval.",
        display="Convert exact nonnegative Fraction to binary64 then nextafter(+infinity) when needed; preserve exact numerator/denominator and upward display hex. No heuristic norm or estimated roundoff bound.",
    ),
    eigensolve_semantics=dict(
        source="frozen f043 atomx.methods.ci complete dense branch",
        group_order="native _group_csfs_by_J_parity, ascending2J/even beforeodd",
        block_solver="scipy.linalg.eigh(block,subset_by_index=None,driver=evd), all roots",
        global_order="np.argsort(values,kind=stable); full2574 signed block mixing assembled exactly as native dense branch",
        snapshot_class="ordinary CIResult(E,C,provenance=SAVED_MATRIX_CONDITIONAL_DIAGNOSTIC,hamiltonian_snapshot=None); never fabricate a retained physical owner",
        blocked_public_ci_solve_calls=0,
    ),
    archive_order="Save every completed block E/C immediately and whole2574 E/C before tracker. Observe and archive complete public signed overlap before native tracker endpoint majority/uniqueness checks. Save successful tracker weights/diagnostics before custom full-snapshot residual/Gram/UID/energy endpoint gates. Unavoidable native tracker full-CI validation remains unchanged.",
    stop="First identity/resource/numerical/tracker diagnostic failure preserved; no H rerun, matrix correction, source adoption, original gate relaxation, automatic retry or physics-success inference.",
    preparation=dict(
        science_imports=0,
        numerical_archive_reads=0,
        numerical_archive_rehashes=0,
        numerical_ZIP_header_reads=0,
        numerical_calls=0,
    ),
)
write("contract.json", contract)
print(
    json.dumps(
        dict(
            status=contract["status"],
            saved_native_H_sha256=row["sha256"],
            new_H=0,
            eigh=5,
            NPZ_access=0,
        )
    )
)
