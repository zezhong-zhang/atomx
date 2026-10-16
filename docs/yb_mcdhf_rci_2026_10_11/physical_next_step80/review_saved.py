"""Metadata/source-only review. No array or scientific imports/reads."""
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / 'docs/yb_mcdhf_rci_2026_10_11'
OUT = Path(__file__).resolve().parent
SOURCE = BASE / 'rank_aware_cell_discriminator_preparation/candidate_source/atomx'
LABELS = ('6s_1/2', '5d_3/2', '5d_5/2', '6p_1/2', '6p_3/2')
UIDS = dict(zip(LABELS, ((1, 1, 0), (3, 1, 0), (5, 1, 0), (1, -1, 0), (3, -1, 0)), strict=True))
inputs = {}
checks = []


def read(path):
    content = path.read_bytes()
    inputs[str(path.relative_to(REPO))] = hashlib.sha256(content).hexdigest()
    return content.decode()


def load(relative):
    return json.loads(read(BASE / relative))


def check(name, condition):
    checks.append({'name': name, 'passed': bool(condition)})
    if not condition:
        raise ValueError(name)


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


read(Path(__file__).resolve())
contract = json.loads(read(OUT / 'contract.json'))
completion = load('correlated29_optimization_run004_preparation/completion.json')
point = load('correlated29_optimization_run004_preparation/actual/checkpoints/accepted_0064.json')
models = load('screened_core_virtual_discriminator002/models.json')
balanced = load('balanced_cv_replay/result.json')
rank = load('rank_only_discriminator/result.json')
virtual = load('virtual_only_discriminator/result.json')
screened = load('screened_core_virtual_discriminator002/result.json')
static = load('public531_endpoint_static_rci/result.json')
corrections = load('correction_budget/same_orbital001/result.json')
inner = load('rank_aware_cell_discriminator_preparation/actual/result.json')
sa = read(SOURCE / 'physics/csf/state_average.py')
tracking = read(SOURCE / 'physics/csf/tracking.py')
functional = read(SOURCE / 'methods/csf/variational.py')
optimizer = read(SOURCE / 'numerics/bspline/orbital_optimization.py')
checkpoint_consumer = read(BASE / 'correlated29_optimization_run005_preparation/run.py')
solver = read(SOURCE / 'methods/csf/solver.py')
check('closed_global80_metadata', completion['actual_foreground_exit_code'] == 0 and
      completion['total_global_accepted_steps'] == 80 and point['global_accepted_step'] == 80)
rows = {r['label']: r for r in point['targets']}
check('five_physical_UIDs_not_column_order', set(rows) == set(LABELS) and
      all(tuple(rows[label]['UID']) == UIDS[label] and
          rows[label]['tracking']['reference_levels'] == [list(UIDS[label])] and
          rows[label]['tracking']['per_level_weight'] == .2 for label in LABELS))
diag = completion['optimizer_final_diagnostics']
check('original_both_stationarity_gates_fail', not point['stationarity_passed'] and
      diag['orbital_residual'] > diag['orbital_tolerance'] == 1e-7 and
      diag['estimated_energy_error'] > diag['energy_tolerance'] == 1e-8)
check('private_f043_and_original_metric', point['source_map_sha256'] ==
      'f0435dd68e31a8c5cca5a4edd033c061f7bcff605dcb233c7afce190139496fe' and
      point['metric_sha256'] == 'a96f1aecd3e18316376121d8255c156a5f5b4c01a21fd0ba86ca91131ae66d73')
configs = models['models']['balanced_virtual']['configs']
core = dict(re.findall(r'(\d+[spdf])(\d+)', models['core_NR']))
core = {key: int(value) for key, value in core.items()}
holes = Counter()
particle_counts = Counter()
for row in configs:
    occ = {key: int(value) for key, value in re.findall(r'(\d+[spdf])(\d+)', row['config'])}
    missing = {key: value - occ.get(key, 0) for key, value in core.items() if value != occ.get(key, 0)}
    if missing:
        check_name = len(missing) == 1 and next(iter(missing.values())) == 1
        if not check_name:
            raise ValueError('declared family contains unexpected core-hole rank')
        holes[next(iter(missing))] += 1
    else:
        holes['none'] += 1
    particle_counts[sum(value for key, value in occ.items() if key not in core)] += 1
check('existing69_family_exactly_bare_and_single_core_holes', len(configs) == 69 and
      holes == Counter({'none': 6, '4f': 21, '5p': 21, '5s': 21}) and
      particle_counts == Counter({1: 6, 2: 63}))
check('public_state_average_and_tracking_capability',
      'class CIStateAverage' in sa and 'states:' in sa and 'statistical: bool = False' in sa and
      'class CIBasisTracking' in tracking and 'reference never' in tracking and
      'level_tracking.state_average != state_average' in functional)
parsed = ast.parse(optimizer)
opt = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'optimize_orbitals')
check('public_objective_change_explicit_and_secants_cleared',
      'allow_objective_change' in [a.arg for a in opt.args.args + opt.args.kwonlyargs] and
      'history = []' in optimizer and 'changed objective_identity requires allow_objective_change=True' in optimizer)
parsed_solver = ast.parse(solver)
perblock = next(n for n in ast.walk(parsed_solver) if isinstance(n, ast.FunctionDef) and n.name == '_solve_per_block_orbitals')
check('other_per_block_path_is_DF_not_this_variational_optimizer',
      any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'DFSolver' for n in ast.walk(perblock)) and
      not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'optimize_orbitals' for n in ast.walk(perblock)))

scale = []
for stage in balanced['stages']:
    level_rows = {r['label']: r for r in stage['tracked_levels']['DC']}
    scale.append({'name': stage['name'], 'origin': 'same native005 bare five-CSF stationary orbitals; nested fixed-orbital DC RCI',
                  'source_result': 'balanced_cv_replay/result.json', 'stage': stage['name'],
                  'lower_delta_Ha': level_rows['6s_1/2']['energy_delta_from_previous_Ha'],
                  'upper_delta_Ha': {label: level_rows[label]['energy_delta_from_previous_Ha'] for label in LABELS[1:]},
                  'line_delta_cm1': {label: level_rows[label]['transition_delta_from_previous_cm1'] for label in LABELS[1:]},
                  'multiplet_delta_cm1': {label: {'centroid': m['centroid_delta_from_previous_cm1'], 'splitting': m['splitting_delta_from_previous_cm1']} for label, m in stage['multiplets']['DC'].items()},
                  'additive_at_global80': False})
for name, data in [('5p_double_hole', rank), ('nuclear_auxiliary_virtuals', virtual), ('screened_closed68_auxiliary_virtuals', screened)]:
    stage = next(s for s in data['stages'] if 'rows' in s and s['name'] != 'baseline')
    level_rows = {r['label']: r for r in stage['rows']}
    check(name + '_physical_rows_complete', set(level_rows) == set(LABELS) and
          all((r['twoJ'], r['parity'], r['local_root']) == UIDS[label] for label, r in level_rows.items()))
    scale.append({'name': name, 'origin': 'same nonstationary full531 accepted160 old24 orbitals; fixed-orbital scalar DC discriminator',
                  'source_result': {'5p_double_hole': 'rank_only_discriminator/result.json', 'nuclear_auxiliary_virtuals': 'virtual_only_discriminator/result.json', 'screened_closed68_auxiliary_virtuals': 'screened_core_virtual_discriminator002/result.json'}[name],
                  'stage': stage['name'], 'lower_delta_Ha': level_rows['6s_1/2']['absolute_delta_lower_Ha'],
                  'upper_delta_Ha': {label: level_rows[label]['absolute_delta_upper_Ha'] for label in LABELS[1:]},
                  'line_delta_cm1': {label: level_rows[label]['transition_delta_upper_minus_lower_cm1'] for label in LABELS[1:]},
                  'multiplet_delta_cm1': {m['label']: {'centroid': m['delta_centroid_cm1'], 'splitting': m['delta_splitting_cm1']} for m in stage['multiplets']},
                  'additive_at_global80': False})
static_rows = {r['label']: r for r in static['static_rows']}
scale.append({'name': 'static_Breit', 'origin': 'same nonstationary full531 accepted160 point; full static Gaunt plus zero-frequency retardation RCI',
              'source_result': 'public531_endpoint_static_rci/result.json',
              'lower_delta_Ha': static_rows['6s_1/2']['same_lower_static_minus_DC_Ha'],
              'upper_delta_Ha': {label: static_rows[label]['absolute_upper_static_minus_DC_Ha'] for label in LABELS[1:]},
              'line_delta_cm1': {label: static_rows[label]['static_minus_DC_transition_cm1'] for label in LABELS[1:]},
              'multiplet_delta_cm1': {m['label']: {'centroid': m['static_minus_DC_centroid_cm1'], 'splitting': m['static_minus_DC_splitting_cm1']} for m in static['multiplets']}, 'additive_at_global80': False})
correction_rows = {r['label']: r for r in corrections['state_correction_ledger']}
for component in ('dynamic_minus_static_Breit', 'Uehling_point_source', 'WK_local_model', 'SE_indelicato_overlap_model', 'SE_shabaev_local_partial', 'NMS_Breit_point_Coulomb', 'SMS_Breit_point_Coulomb'):
    scale.append({'name': component, 'origin': corrections['input_run'] + '; five bare CSFs,24 radial spinors; orbital residual failed',
                  'source_result': 'correction_budget/same_orbital001/result.json',
                  'lower_delta_Ha': correction_rows['6s_1/2']['components'][component]['absolute_Ha'],
                  'upper_delta_Ha': {label: correction_rows[label]['components'][component]['absolute_Ha'] for label in LABELS[1:]},
                  'line_delta_cm1': {label: correction_rows[label]['components'][component]['relative_to_6s_cm1'] for label in LABELS[1:]},
                  'component_coverage': corrections['component_coverage'].get({'dynamic_minus_static_Breit': 'dynamic_Breit', 'Uehling_point_source': 'Uehling', 'WK_local_model': 'WK', 'SE_indelicato_overlap_model': 'SE_default', 'SE_shabaev_local_partial': 'SE_local_discriminator', 'NMS_Breit_point_Coulomb': 'NMS', 'SMS_Breit_point_Coulomb': 'SMS'}[component]),
                  'additive_at_global80': False, 'uncertainty_bound': False})
scale.append({'name': 'positive_inner_quadrature_order_fix', 'origin': 'same initial corrected29 reference; fixed signed CI expectations only, not new optimized roots',
              'source_result': 'rank_aware_cell_discriminator_preparation/actual/result.json',
              'line_delta_cm1': {r['label']: r['delta_transition_cm1'] for r in inner['targets'] if r['label'] != '6s_1/2'},
              'lower_delta_Ha': inner['targets'][0]['delta_expectation_Ha'], 'additive_at_global80': False,
              'uncertainty_bound': False})

nominal = {label: rows[label]['NIST_nominal_cm1'] for label in LABELS}
multiplets = []
for name, labels, weights in [('5d', ('5d_3/2', '5d_5/2'), (4, 6)), ('6p', ('6p_1/2', '6p_3/2'), (2, 4))]:
    m = next(m for m in point['multiplets'] if m['label'] == name)
    centroid = sum(w * nominal[label] for w, label in zip(weights, labels, strict=True)) / sum(weights)
    splitting = nominal[labels[1]] - nominal[labels[0]]
    multiplets.append(dict(m, NIST_nominal_centroid_cm1=centroid, NIST_nominal_splitting_cm1=splitting,
                          centroid_gap_cm1=m['centroid_cm1']-centroid, splitting_gap_cm1=m['splitting_cm1']-splitting))
result = {
    'scope': contract['scope'], 'scientific_calls': 0, 'array_reads': 0, 'new_physics_qualified': False,
    'current_point': {'global_accepted_step': 80, 'source_map_sha256': point['source_map_sha256'],
                      'metric_sha256': point['metric_sha256'], 'geometry_sha256': point['geometry_sha256'],
                      'objective_sha256': point['objective_sha256'], 'reported_EO_Ha': diag['energy_history'][-1],
                      'orbital_residual': diag['orbital_residual'], 'orbital_tolerance': diag['orbital_tolerance'],
                      'energy_indicator_Ha': diag['estimated_energy_error'], 'energy_tolerance_Ha': diag['energy_tolerance'],
                      'stationarity_passed': False, 'indicator_is_strict_error_bound': False,
                      'nominal_four_gap_RMS_cm1': point['nominal_four_gap_RMS_cm1'], 'targets': [rows[label] for label in LABELS],
                      'multiplets': multiplets,
                      'global16_to80': {'EO_change_Ha': completion['reported_EO_delta_Ha'], 'target_changes': completion['terminal_target_changes']}},
    'declared_space': {'electron_number': 69, 'radial_spinors': 29, 'coordinates': 3451,
                       'NR_configurations': len(configs), 'target_complete_CSFs': 1046,
                       'full_all_J_parent_CSFs': 2163, 'computed_other_J_sectors': False,
                       'other_J_omission_is_target_operator_coupling_error': False,
                       'core_hole_config_counts': dict(holes), 'external_particle_counts': dict(particle_counts),
                       'external_NR_labels': models['base_particle_shells'] + models['new_particle_shells'],
                       'two_core_holes_present': False, 'higher_hole_rank_present': False,
                       'deeper_core_holes_present': False, 'new_f_g_higher_l_virtual_layers_present': False},
    'samepoint_historical_scales': scale,
    'scales_are_additive_at80_or_error_bounds': False,
    'correction_coverage_unresolved': corrections['unsupported'],
    'native_interface_review': {
        'method_reference': 'GRASP rmcdhf EOL -> rci active-set correlation/operator workflow; native Atomx implementation, not a backend or numerical-equivalence claim',
        'shared_cross_parity_EOL_legal': True,
        'state_specific_or_subgroup_variational_interface_exists': True,
        'route': 'CIStateAverage selected frozen UID subset + CIBasisTracking from the original reference + CSFOrbitalFunctional + optimize_orbitals with original P/layout/inverse and explicit changed objective',
        'production_equal5_weights_change_authorized_by_this_review': False,
        'per_block_DF_path_is_current_full_RDM_MCDHF_alternative': False,
        'separate_orbital_transition_rates_qualified_here': False,
        'history_restoration_api_exists': False},
    'nonstationary_saved_only_next': {
        'current_manifest_has_per_kappa_or_orbital_gradient_decomposition': False,
        'frozen_checkpoint_array_fields': ['keys', 'radial_gradient', 'raw_coordinate_gradient', 'tangent_gradient', 'preconditioned_gradient', 'x', 'projector_i', 'inverse_blocks_i'],
        'source_only_array_field_confirmation': all(key in checkpoint_consumer for key in ('radial_gradient', 'raw_coordinate_gradient', 'tangent_gradient', 'preconditioned_gradient', 'projector_', 'inverse_blocks_')),
        'no_array_read_or_execution': True,
        'if005_nonstationary': 'Prospective saved-only per-kappa/column ||pg_j||^2 and signed g_j dot pg_j; core19/barevalence5/new5; keep Stiefel TMT-induced cross terms if decomposing independent drives. No numerical inference of core dominance from totalcoreenergy.',
        'new_physical_or_metric_bound': False},
    'next_discriminators_after_original_stationarity': [
        {'name': 'common_orbital_compromise', 'minimum': 'same1046/29/P/metric; first compare common five-state EOL against separately declared even6s+D and oddP EOL; singleton only if needed, no NIST-selected weights',
         'distinguishes': 'state-specific orbital relaxation versus a common-orbital compromise in the same finite Hamiltonian space',
         'report': 'each converged state energy; lower/upper difference; D/P centroid and splitting; UID/overlap and same original gates; no additive patch at80'},
        {'name': 'one_missing_correlation_class', 'minimum': 'same converged common point; one prospectively balanced nested core-double family at fixed existing pool, before another indiscriminate large virtual layer; resource preflight needed separately',
         'distinguishes': 'core-hole excitation rank versus radial layer; old5pdouble and virtual increments motivate a samepoint comparison, not a correction prediction',
         'report': 'all five complete symmetry blocks and unchanged UID; full signed CI; lower/upper shifts and centroid/splitting; expanded-space orbital nonstationarity remains explicit'},
        {'name': 'samepoint_RCI_operator_stack', 'minimum': 'same converged model and spinors; DC then full static Breit, then declared finite-frequency prescription and supported QED/recoil pieces individually',
         'distinguishes': 'operator correction from correlation/relaxation; actual manybody screened/nonlocal and finite-charge QED gaps stay unsupported',
         'report': 'absolute lower/upper shifts and transition difference; static is replaced by dynamic, not added twice; no cross-point splice or QED model spread treated as bound'}],
    'references': [
        {'title': 'GRASP Manual for Users, section3.6 and4.3–4.4', 'url': 'https://www.mdpi.com/2218-2004/11/4/68', 'use': 'Primary theory/workflow reference only: EOL may share orbitals across both parities; orbital-set and excitation-class convergence are distinct.'},
        {'title': 'An Introduction to Relativistic Theory as Implemented in GRASP, section8', 'url': 'https://doi.org/10.3390/atoms11010007', 'use': 'Primary reference only: separately optimized orbital sets require compatible nonorthogonal transition treatment; no foreign backend invocation.'},
        {'title': 'Itano2006 MCDHF/RCI Yb+ quadrupole moments and hyperfine constants, sectionIII.D', 'url': 'https://tf.nist.gov/general/pdf/2107.pdf', 'use': 'Primary Yb+ method reference:4f/5s/5p CV virtual layers then deeperCV/CC; not energy accuracy certification for Atomx.'},
        {'title': 'Porsev,Safronova,Kozlov2012 Correlation effects in Yb+', 'url': 'https://doi.org/10.1103/PhysRevA.86.022504', 'use': 'Primary correlation reference: some properties sensitive to single-particle/hole-two-particle mixing; not an external backend choice or Atomx error bound.'}],
    'inputs_sha256': inputs,
    'interpretation': 'Global80 is not a converged finite-model answer. Historical shifts show differential core correlation/relaxation can be hundreds of cm−1 while some operator corrections are smaller; this neither identifies the missing term nor excludes a larger unmeasured one. Increasing radial/CSF cutoffs cannot repair an absent Hamiltonian operator, and convergence within fixed restrictions does not imply physical completeness.'
}
save('result.json', result)
save('verification.json', {'all_passed': all(c['passed'] for c in checks), 'checks': len(checks), 'controls': checks,
                           'scope': 'stdlib saved JSON identity/source capability checks only; no independent array numerical replay or physics qualification',
                           'inputs_sha256': inputs, 'result_sha256': hashlib.sha256((OUT/'result.json').read_bytes()).hexdigest(),
                           'scientific_calls': 0, 'array_reads': 0})
print(json.dumps({'checks': len(checks), 'all_passed': True, 'scientific_calls': 0, 'array_reads': 0}))
