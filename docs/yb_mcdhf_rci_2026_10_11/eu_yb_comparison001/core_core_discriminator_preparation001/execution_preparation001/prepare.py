"""SOURCE/JSON-only preparation; never import science or access numerical archives."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/atomx')
HERE = Path(__file__).resolve().parent
TOP = ROOT / 'docs/yb_mcdhf_rci_2026_10_11'
STAGE = TOP / 'screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageB_execution_preparation001'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')

if (HERE / 'preparation_manifest.json').exists():
    raise FileExistsError('Frozen preparation exists; preserve it and use a successor')

old = read(STAGE / 'contract.json')
result = read(STAGE / 'actual/result.json')
proposal = read(HERE.parent / 'proposal.json')
roles = {key: old['input_roles'][key] for key in ('source_identity', 'new_layout', 'new_objective', 'models', 'canonical_basis', 'reference_models', 'tracking_binding', 'geometry_workspace', 'tracking_ci', 'screened_geometry', 'literal13_manifest', 'native_result', 'native_receipt')}
for role, path in {
    'stageB_contract': STAGE / 'contract.json',
    'stageB_result': STAGE / 'actual/result.json',
    'stageB_receipt': STAGE / 'root_execution.json',
    'baseline_owner': STAGE / 'actual/baseline_owner.json',
    'baseline_tracking': STAGE / 'actual/baseline_tracking.json',
    'baseline_basis': STAGE / 'actual/basis_identity.json',
    'baseline_state': STAGE / 'actual/state_identity.json',
    'core_core_proposal': HERE.parent / 'proposal.json',
    'resource_proposal': HERE.parent / 'resource_admission_proposal001.json',
    'historical_result': TOP / 'rank_only_discriminator/result.json',
}.items():
    roles[role] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), deferred_numerical=False)
archives = {Path(row['path']).name: row['sha256'] for row in result['archives']}
for role, name in (('baseline_point', 'baseline_point.npz'), ('baseline_whole_CI', 'baseline_whole_CI.npz'), ('baseline_geometry', 'immutable_original_geometry_metric.npz')):
    roles[role] = dict(path=str((STAGE / 'actual' / name).relative_to(ROOT)), sha256=archives[name], deferred_numerical=True)
# Copy bindings from completed JSON only. No NPZ read, stat or rehash occurs here.
models = read(ROOT / roles['models']['path'])
baseline = models['models']['complete5f']['configs']
new = baseline + proposal['candidate_NR_configs']
write('models.json', dict(selected_sectors=models['selected_sectors'], baseline_configs=baseline, expanded_configs=new, counts=proposal['counts'], expanded_definition='All 91 existing NR configurations plus all 9 declared 5p-double NR configurations. No new radial orbital, omitted signed CSF or amplitude cut.'))
roles['prepared_models'] = dict(path=str((HERE / 'models.json').relative_to(ROOT)), sha256=sha(HERE / 'models.json'), deferred_numerical=False)
def field(shape, dtype='<f8'):
    return dict(shape=shape, dtype=dtype)
array_access = {
    'geometry_workspace': dict(compressed_bytes_max=67108864, fields_max=64, basis_prefix=True, fields=dict(orbital_keys=field([24, 2], '<i8'), P=field([24, 123]), Q=field([24, 123]))),
    'tracking_ci': dict(compressed_bytes_max=8388608, fields_max=2, fields=dict(energies_Ha=field([531]), mixing=field([531, 531]))),
    'screened_geometry': dict(compressed_bytes_max=16777216, fields_max=7, fields={f's_{k}': field([246, 246]) for k in old['fixed']['kappas']}),
    'baseline_whole_CI': dict(compressed_bytes_max=67108864, fields_max=6, fields=dict(H=field([1595, 1595]), energies=field([1595]), mixing=field([1595, 1595]), keys=field([31, 2], '<i8'), P=field([31, 123]), Q=field([31, 123]))),
    'baseline_point': dict(compressed_bytes_max=8388608, fields_max=4, fields=dict(x=field([3689]), keys=field([31, 2], '<i8'), P=field([31, 123]), Q=field([31, 123]))),
    'baseline_geometry': dict(compressed_bytes_max=16777216, fields_max=9, fields=dict(x=field([3689]), inverse_diagonal=field([3689]), **{f'projector_{i}': field([246, 119]) for i in range(7)})),
}
uid = old['fixed']['original_UIDs']
historical = read(ROOT / roles['historical_result']['path'])
nominal = {row['label']: row['NIST_nominal_cm1'] for row in historical['stages'][0]['rows']}
contract = dict(
    status='SOURCE_ONLY_once_executor_ready_for_root_review', execution_authorized=False,
    hypothesis='Measure differential effects of the declared 5p-double correlation sector at the identical accepted13 31-orbital point, before increasing radial cutoffs.',
    source_archive=old['source_archive'], source_map_sha256=old['fixed']['source_map_sha256'], full_source_count=356,
    fixed=dict(Z=70, A=175, electrons=69, ns=123, radial_orbitals=31, coordinates=3689, nucleus='native sampled Fermi A175', operator='unshifted native Dirac-Coulomb', accepted_iteration=13, kappas=old['fixed']['kappas'], original_UIDs=uid, equal_weights=[.2]*5, metric_sha256=old['fixed']['new_metric_sha256'], geometry_sha256=old['fixed']['new_geometry_sha256'], objective_identity=old['fixed']['objective_identity'], baseline_original_stationarity_passed=False, baseline_fullsigned_metrics=result['baseline_fullsigned_metrics'], baseline=result['points']['baseline'], nominal_NIST_cm1=nominal, nominal_NIST_uncertainty_cm1=None),
    resources=dict(CPU=27, threads=1, AS_GiB=8, inner_seconds=900, outer_seconds=915, minimum_host_available_GiB=10, raw_determinants_per_block=2659, max_connected_pairs_total=29850774, max_connected_pairs_per_block=7070281, max_coefficient_table_nnz_per_block=20000000, max_coefficient_table_nnz_total=50000000, max_coefficient_tables=5, max_signed_overlap_pairs=1000000, max_signed_compatible_pairs_sum=66654, max_input_decoded_bytes=201326592, max_input_archive_loads=6, max_complete_H_builds=1, max_complete_CI_calls=1, max_complete_CI_eigh=5, max_native_hS_builds=7, max_hS_requests=100, max_angular_null_space_calls=1024, max_tracking_SVD_calls=10, max_public_overlap_calls=4, max_projection_calls=40, max_radial_key_evaluations=1000000, optimizer=0, gradients=0, Dirac_spectra=0, GPU=0, timing_claim_valid=False, table_caps_checked_after_return='AS and outer wall guard protect Python SET/list/CSR construction before a completed table can be counted.'),
    acceptance=dict(exact_nested_old_H_bytes=True, baseline_source_owner_basis_point_UID_identity=True, all_complete_CI_residual_Ha=1e-8, all_complete_CI_Gram=1e-10, matrix_hermiticity_Ha=1e-10, cross_symmetry_zero_Ha=1e-14, full_same_kappa_radial_Gram=1e-10, saved_P_metric_Gram=1e-10, original_P_leak_S_squared=1e-12, inherited_ordered_Ritz_increase_Ha=1e-8, original_public_tracker_principal_squared_strict_min=.5, original_public_tracker_score_boundary_strict_min=1e-10, original_public_tracker_cluster_gap_Ha=1e-10, nominal_improvement_required=False, stationarity_upgraded=False),
    prerequisite=dict(stageB_expected_status=result['status'], stageB_overall_passed=False, exactly_preserved_ordinary_FD_failures=['ordinary_reported_FD_h0','ordinary_reported_FD_h1'], all_other_declared_StageB_controls_passed=True, stageB_closed_exit_code=1, baseline_is_exact_accepted13=True),
    input_roles=roles, array_access=array_access,
    native_call_chain=['SOURCE/JSON full356 source and exact root seal; fresh exclusive actual directory; CPU27/thread1/8GiB/900s plus root915s hard foreground guardian.', 'Read declared complete baseline H/E/C and unchanged31 P/Q, original projectors/metric and immutable531 tracker inputs, with complete bounded field census.', 'Enumerate all 3789 baseline and 5741 expanded parent CSFs; retain every CSF of five complete blocks 1595->2574; save exact NR and signed-CSF/occupation/order maps.', 'Check original31 radial bytes, original P@x/S metric/owner/UID; baseline residual/Gram; unchanged sector union spinor ordering and common closed core. Complete public signed raw supports and overlap pair admission precede H.', 'Construct original531 native tracker once; resolve saved1595 baseline without H build or optimizer. Preserve exact StageB target energy, UIDs and weights.', 'One public native expanded full dense ci_solve(retain_hamiltonian=True,max_determinants=2659). Its unmodified physical builder observer archives H and tests byte-exact embedded1595 H before CI diagonalization. No new baseline H.', 'Archive complete2574 H/E/C, native owner, source/configuration mapping and counters before CI gates. Resolve same original531 tracker; archive signed samepoint baseline-five vs all2574 overlap without root selection or amplitude threshold.', 'Report absolute upper/lower shifts, four transitions, multiplet centroids/splittings and nominal NIST gaps/RMS as conditional fixed-orbital evidence only.'],
    representation_control_boundary='Same radial/order/closed-core source conditions are necessary, not a numerical proof. If the exact nested H byte gate fails, stop before expanded CI diagonalization; preserve complete expanded H and map. A separate source representation control is required, no tolerance relaxation or automatic retry.',
    stop='First identity/resource/CI/tracker failure ends this once-only run. Preserve completed arrays and failure. Native raw or CSR caps reject whole experiment, never remove configurations or amplitudes. No fallback/backend/parameter retry.',
    preparation=dict(science_imports=0, NPZ_reads=0, NPZ_rehashes=0, H_builds=0, CI_calls=0),
    outputs=['complete expanded H/E/C plus native real owner', 'all original baseline arrays remain bound at their closed StageB paths', 'all parent/selected signed CSF and configuration/occupation maps, exact nested embedding', 'complete signed five-state-to-all2574 overlap and immutable original tracker diagnostics', 'conditional upper/lower/four-transition/centroid/splitting/NIST nominal comparison', 'attempted/completed producers, work/cache/projection census, first-failure receipts'],
)
write('contract.json', contract)
print(json.dumps(dict(status=contract['status'], NR=[len(baseline), len(new)], CSFs=[1595,2574], science=0, NPZ_access=0, contract_sha256=sha(HERE/'contract.json'))))
