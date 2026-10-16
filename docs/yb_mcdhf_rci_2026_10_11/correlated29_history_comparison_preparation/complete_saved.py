"""Close two immutable matched history pilots with stdlib saved-ledger arithmetic.

No array numerical replay, physical import, producer, eigenproblem or optimizer.
Saved finite-H certificates are checked algebraically, not recomputed from H/C.
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def exact(row):
    return Fraction(int(row['numerator']), int(row['denominator']))


def close_branch(branch):
    HERE = ROOT / branch
    result = load(HERE / 'actual/result.json')
    receipt = load(HERE / 'root_execution.json')
    contract = load(HERE / 'contract.json')
    controls, inputs, records = [], {}, []

    def check(name, passed):
        controls.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise ValueError(name)

    def bind(path, expected=None):
        path = Path(path)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('Changed frozen input: ' + str(path))
        inputs[str(path.relative_to(REPO))] = actual
        return actual

    check('foreground_closed_explicit_status', receipt['status'] == 'foreground_closed' and type(receipt['exit_code']) is int)
    for name, expected in receipt['actual_files_sha256'].items():
        bind(HERE / name, expected)
    for name in ('root_execution.json', 'execution_contract.json', 'contract.json',
                 'preparation_verification.json', 'actual/objective_identity.json'):
        bind(HERE / name)
    check('receipt_driver_contract_execution_bound',
          result['driver_sha256'] == receipt['driver_sha256'] == sha(ROOT / 'run.py')
          and result['contract_sha256'] == receipt['contract_sha256'] == sha(HERE / 'contract.json')
          and result['execution_sha256'] == receipt['execution_contract_sha256'] == sha(HERE / 'execution_contract.json'))
    bind(ROOT / 'run.py', receipt['driver_sha256'])
    check('branch_and_search_capacity_matches_frozen_contract', result['comparison_branch'] == branch
          and result['search_memory_policy'] == contract['search_memory']
          and contract['search_memory']['history_size'] == int(branch.removeprefix('history')))
    accepted = result['trajectory']
    successful_return = result['status'] == 'bounded_public_optimizer_returned' and result['error'] is None
    check('actual_status_and_foreground_exit_consistent', (successful_return and receipt['exit_code'] == 0)
          or (not successful_return and receipt['exit_code'] != 0))
    check('actual_qualified_trajectory_contiguous_within_declared32steps',
          [t['iteration'] for t in accepted] == list(range(len(accepted))) and len(accepted) <= 33
          and all(t['global_accepted_step'] == 208 + t['iteration'] for t in accepted))
    failed_live_controls = [row for row in result['controls'] if row['passed'] is not True]
    # Bind failed/partial checkpoints too: never promote them into the qualified trajectory.
    forensic = []
    for path in sorted((HERE / 'actual').glob('checkpoints/accepted_*.json')):
        if path.name.endswith('_signed_density.json'):
            continue
        m = load(path)
        bind(path)
        bind(path.with_suffix('.npz'), m['array_sha256'])
        if m['checkpoint_qualified'] is not True:
            forensic.append({'path': str(path.relative_to(REPO)), 'checkpoint_qualified': False,
                             'iteration': m['iteration'], 'global_accepted_step': m['global_accepted_step'],
                             'array_sha256': m['array_sha256'], 'diagnostics': m['diagnostics'],
                             'exact_acceptance': m.get('exact_acceptance')})
    for path in sorted((HERE / 'actual').glob('accepted_trial_before_gradient_*.json')):
        m = load(path)
        bind(path.with_suffix('.npz'), m['array_sha256'])
    expected_uids = {'6s_1/2': [1, 1, 0], '5d_3/2': [3, 1, 0], '5d_5/2': [5, 1, 0],
                     '6p_1/2': [1, -1, 0], '6p_3/2': [3, -1, 0]}
    previous = None
    for t in result['trajectory']:
        path = REPO / t['manifest_path']
        bind(path, t['manifest_sha256'])
        m = load(path)
        bind(path.with_suffix('.npz'), m['array_sha256'])
        rows = {row['label']: row for row in m['targets']}
        if not (len(m['targets']) == 5 and set(rows) == set(expected_uids)
                and all(rows[k]['UID'] == uid and rows[k]['selected_level'][:2] == uid[:2]
                        for k, uid in expected_uids.items())):
            raise ValueError('Changed physical target UID')
        if not (m['checkpoint_qualified'] is True
                and m['source_map_sha256'] == result['source_map_sha256'] == contract['source']['candidate356_map_sha256']
                and m['metric_sha256'] == contract['metric']['residual_metric_sha256']
                and m['geometry_sha256'] == contract['metric']['geometry_sha256']
                and m['objective_sha256'] == t['diagnostics']['objective_identity']
                and m['diagnostics'] == t['diagnostics']
                and m['search_memory_policy'] == contract['search_memory']
                and m['diagnostics']['initial_history_size'] == 0
                and m['diagnostics']['resumed'] is True and m['diagnostics']['objective_changed'] is False
                and m['diagnostics']['orbital_tolerance'] == 1e-7
                and m['diagnostics']['energy_tolerance'] == 1e-8
                and m['experimental_certification'] is False
                and m['physical_completeness_certification'] is False):
            raise ValueError('Changed qualified point lineage')
        a = m['exact_acceptance']
        if previous is None:
            if a is not None:
                raise ValueError('Initial point has acceptance')
        else:
            ls = m['diagnostics']['accepted_line_search']
            e0, e1 = previous['diagnostics']['energy_ha'], m['diagnostics']['energy_ha']
            change, rhs = Fraction(ls['energy_change_ha']), Fraction(ls['armijo_rhs_ha'])
            exact_rhs = Fraction(1e-4) * Fraction(ls['step']) * Fraction(ls['directional_derivative'])
            margin = (abs(exact(previous['exact_point']['quotient_minus_actual_EO']))
                      + abs(exact(m['exact_point']['quotient_minus_actual_EO']))
                      + abs(Fraction(e1) - Fraction(e0) - change) + abs(exact_rhs - rhs))
            if not (a['qualified'] is True and a['actual_optimizer_comparison_passed'] is True
                    and exact(a['margin_exact']) == margin
                    and exact(a['guard_lhs']) == change + margin
                    and exact(a['recorded_rhs']) == rhs and change + margin <= rhs
                    and exact(a['stored_quotient_change']) ==
                        exact(m['exact_point']['weighted_quotient']) - exact(previous['exact_point']['weighted_quotient'])
                    and exact(a['stored_quotient_change']) < 0
                    and e1 < e0 and ls['parent_energy_ha'] == e0 and ls['trial_energy_ha'] == e1
                    and ls['stable_callback_used'] is False):
                raise ValueError('Saved exact acceptance ledger mismatch')
        rms = math.sqrt(math.fsum(row['nominal_gap_cm1']**2 for label, row in rows.items() if label != '6s_1/2') / 4)
        if rms != m['nominal_four_gap_RMS_cm1']:
            raise ValueError('Physical four-line RMS mismatch')
        expected_multiplets = []
        for label, low, high, weights in (
                ('5d', '5d_3/2', '5d_5/2', [4, 6]), ('6p', '6p_1/2', '6p_3/2', [2, 4])):
            low_excitation, high_excitation = rows[low]['excitation_cm1'], rows[high]['excitation_cm1']
            expected_multiplets.append({'label': label, 'degeneracy_weights': weights,
                                       'centroid_cm1': (weights[0] * low_excitation + weights[1] * high_excitation) / sum(weights),
                                       'splitting_cm1': high_excitation - low_excitation})
        if m['multiplets'] != expected_multiplets:
            raise ValueError('Matched physical centroid/splitting mismatch')
        if m['stationarity_passed'] != (m['diagnostics']['orbital_residual'] < 1e-7
                                       and m['diagnostics']['estimated_energy_error'] < 1e-8):
            raise ValueError('Checkpoint original stationarity gate mismatch')
        records.append({'state_sha256': m['state_sha256'], 'owner_input_sha256': m['owner_input_sha256'],
                        'owner_H_sha256': m['owner_H_sha256'], 'metric_sha256': m['metric_sha256'],
                        'geometry_sha256': m['geometry_sha256'], 'objective_sha256': m['objective_sha256'],
                        'manifest_sha256': sha(path), 'array_sha256': m['array_sha256'],
                        'exact_point': m['exact_point'], 'accepted_line_search': m['diagnostics']['accepted_line_search'],
                        'iteration': m['iteration'], 'global_accepted_step': m['global_accepted_step'],
                        'stationarity_passed': m['stationarity_passed'], 'reported_EO_Ha': m['diagnostics']['energy_ha'],
                        'orbital_residual': m['diagnostics']['orbital_residual'],
                        'energy_indicator_Ha': m['diagnostics']['estimated_energy_error'],
                        'nominal_four_gap_RMS_cm1': rms, 'targets': [{k: v for k, v in row.items() if k != 'tracking'}
                                                                  for row in m['targets']],
                        'multiplets': m['multiplets'], 'exact_acceptance_qualified': None if a is None else a['qualified']})
        previous = m
    check('all_actual_qualified_manifests_arrays_UID_metric_and_saved_exact_ledgers', True)
    if accepted:
        check('last_checkpoint_is_actual_terminal_not_best_gap', result['last_qualified_checkpoint'] == accepted[-1]['manifest_path'])
        last_gate = (previous['diagnostics']['orbital_residual'] < 1e-7
                     and previous['diagnostics']['estimated_energy_error'] < 1e-8)
        check('original_stationarity_boolean_matches_terminal_measurement',
              previous['stationarity_passed'] == last_gate and result['stationarity_passed'] == last_gate)
    else:
        check('no_checkpoint_means_no_stationarity_or_terminal_claim', result['last_qualified_checkpoint'] is None
              and result['stationarity_passed'] is False)
    resource_keys = {'H': 'max_H', 'CI': 'max_CI', 'gradient': 'max_full_gradient',
                     'optimizer': 'max_optimizer_calls', 'validation_Dirac': 'max_validation_Dirac',
                     'initial_hS': 'max_initial_hS', 'CI_internal_hS': 'max_CI_internal_hS',
                     'geometry_eigvalsh': 'max_geometry_eigvalsh', 'scipy_eigh': 'max_scipy_eigh',
                     'tracking_resolution': 'max_tracking_resolution', 'tracking_SVD': 'max_tracking_SVD',
                     'tracking_overlap': 'max_tracking_overlap', 'polar_retraction': 'max_polar_retraction',
                     'polar_numpy_eigh': 'max_polar_numpy_eigh',
                     'unfiltered_target_density': 'max_unfiltered_target_density_contractions'}
    check('actual_producer_counts_preserved_within_contract', all(
          0 <= result['completed_calls'][key] <= result['attempted_calls'][key] <= contract['resources'][limit]
          for key, limit in resource_keys.items()))
    check('actual_global_accepted_count_not_assumed32', result['initial_global_accepted_step'] == 208
          and result['total_accepted_steps'] == 208 + max(0, len(accepted) - 1))
    check('oldFAIL_ordinary_reported_and_no_callback_preserved', result['original_reported_derivative_failure_preserved'] is True
          and result['ordinary_reported_objective_changed'] is False and result['energy_callback_used'] is False)
    required_baseline = ['baseline_corrected_source_bitwise_' + key for key in ('H', 'energies', 'mixing', 'P', 'Q')]
    required_baseline += ['same_saved_x_bytes', 'same_residual_metric', 'same_physical_geometry',
                         'baseline_raw_coordinate_g_whole', 'baseline_radial_g_whole',
                         'baseline_original_pg_whole', 'baseline_corrected_owner_canonical',
                         'baseline_corrected_owner_input', 'baseline_corrected_owner_matrix']
    control_by_name = {row['name']: row for row in result['controls']}
    check('same_literal208_live_baseline_controls_preserved', all(
          name in control_by_name and control_by_name[name]['passed'] is True for name in required_baseline))
    check('returned_memory_capacity_and_terminal_not_NIST_selected',
          (not successful_return or result['optimizer_final_diagnostics']['history_size'] == contract['search_memory']['history_size'])
          and result['terminal_point_selected_without_gap_optimization'] is True)
    budget = result['exact_guard_budget']
    check('exact_consumer_resource_budget_and_point_counts',
          budget['points'] <= 33 and budget['quotients'] <= 165
          and budget['examined_pairs'] <= 4_000_000 and budget['nonzero_H_pairs'] <= budget['examined_pairs']
          and budget['points'] >= len(records) and budget['quotients'] == 5 * budget['points'])
    terminal_changes = []
    if records:
        first, last = records[0], records[-1]
        first_rows = {r['label']: r for r in first['targets']}
        terminal_changes = [{'label': row['label'], 'UID': row['UID'],
                             'delta_energy_Ha': row['energy_Ha'] - first_rows[row['label']]['energy_Ha'],
                             'transition_shift_cm1': row['excitation_cm1'] - first_rows[row['label']]['excitation_cm1'],
                             'terminal_excitation_cm1': row['excitation_cm1'], 'terminal_nominal_gap_cm1': row['nominal_gap_cm1']}
                            for row in last['targets']]
    check('zero_physical_imports', not any(n == 'numpy' or n.startswith('numpy.') or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    summary = {'all_completion_metadata_checks_passed': True, 'checks': len(controls), 'controls': controls,
               'inputs_sha256': inputs, 'completion_driver_sha256': sha(__file__),
               'branch': branch, 'search_memory_policy': result['search_memory_policy'],
               'actual_live_controls': len(result['controls']), 'failed_live_controls': failed_live_controls,
               'all_actual_live_controls_passed': not failed_live_controls,
               'actual_status': result['status'], 'actual_error': result['error'],
               'actual_foreground_exit_code': receipt['exit_code'], 'optimizer_final_diagnostics': result['optimizer_final_diagnostics'],
               'forensic_unqualified_candidates': forensic, 'actual_new_accepted_steps': max(0, len(accepted) - 1),
               'total_global_accepted_steps': result['total_accepted_steps'], 'live_control_scope': 'Sole public run; closure is metadata/saved-ledger arithmetic, not independent H/C/g replay',
               'scientific_calls_in_completion': 0, 'array_numerical_replays': 0,
               'trajectory': records, 'terminal_checkpoint': result['last_qualified_checkpoint'],
               'terminal_array_sha256': previous['array_sha256'] if previous else None,
               'terminal_manifest_sha256': sha(REPO / result['last_qualified_checkpoint']) if previous else None,
               'reported_EO_delta_Ha': last['reported_EO_Ha'] - first['reported_EO_Ha'] if records else None,
               'four_line_RMS_delta_cm1': last['nominal_four_gap_RMS_cm1'] - first['nominal_four_gap_RMS_cm1'] if records else None,
               'terminal_target_changes': terminal_changes, 'completed_calls': result['completed_calls'], 'attempted_calls': result['attempted_calls'],
               'exact_guard_budget': result['exact_guard_budget'],
               'all_qualified_finite_stored_H_acceptances_qualified': True, 'qualified_finite_acceptance_count': max(0, len(records) - 1),
               'stationarity_passed': result['stationarity_passed'], 'experimental_certification': False, 'physical_completeness_certification': False,
               'historical_derivative_failures_unchanged': True, 'optimizer_successor_authorized': False,
               'timing_scope': 'uncontrolled correctness measurement', 'actual_elapsed_uncontrolled_s': result['elapsed_uncontrolled_s'],
               'outer_elapsed_uncontrolled_s': receipt['elapsed_uncontrolled_seconds'], 'peak_RSS_MiB': result['peak_RSS_MiB']}
    (HERE / 'completion.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def physical_identity(record):
    """Saved numerical identity excluding branch policy, paths and file wrappers."""
    return {k: record[k] for k in ('state_sha256', 'owner_input_sha256', 'owner_H_sha256',
            'metric_sha256', 'geometry_sha256', 'objective_sha256', 'reported_EO_Ha',
            'orbital_residual', 'energy_indicator_Ha', 'exact_point')} | {
            'physical_targets': {row['label']: {k: row[k] for k in ('UID', 'selected_level', 'energy_Ha')}
                                 for row in record['targets']}}


def horizon_changes(baseline, point):
    old = {row['label']: row for row in baseline['targets']}
    return [{'label': row['label'], 'UID': row['UID'],
             'delta_energy_Ha': row['energy_Ha'] - old[row['label']]['energy_Ha'],
             'transition_shift_cm1': row['excitation_cm1'] - old[row['label']]['excitation_cm1'],
             'excitation_cm1': row['excitation_cm1'], 'nominal_gap_cm1': row['nominal_gap_cm1']}
            for row in point['targets']]


def work_summary(summary):
    lines = [p['accepted_line_search'] for p in summary['trajectory'][1:]]
    histogram = {}
    for line in lines:
        key = str(line['step'])
        histogram[key] = histogram.get(key, 0) + 1
    return {'completed_calls': summary['completed_calls'],
            'attempted_calls': summary['attempted_calls'],
            'accepted_steps': len(lines), 'accepted_step_histogram': histogram,
            'accepted_metadata_backtracks_sum': sum(row['backtrack'] for row in lines),
            'fallback_attempt_acceptances': sum(row['attempt'] == 1 for row in lines),
            'actual_postbaseline_CI_evaluations': summary['completed_calls']['CI'] - 1,
            'postbaseline_CI_above_accepted_count': summary['completed_calls']['CI'] - 1 - len(lines),
            'actual_retained_secant_pair_count': None,
            'retained_pair_count_scope': 'Public saved diagnostics do not export pair retention, transport drops or history resets; capacity is not a pair count.',
            'elapsed_uncontrolled_s': summary['actual_elapsed_uncontrolled_s'],
            'outer_elapsed_uncontrolled_s': summary['outer_elapsed_uncontrolled_s'],
            'wall_speedup_claim_valid': False}


def compare(left, right):
    a, b = left['trajectory'], right['trajectory']
    if not a or not b:
        raise ValueError('Cannot establish matched comparison without both qualified baseline points')
    if physical_identity(a[0]) != physical_identity(b[0]):
        raise ValueError('Saved same-literal208 baseline identity mismatch')
    horizon = min(len(a), len(b)) - 1
    rows, first_difference = [], None
    for i in range(horizon + 1):
        pa, pb = a[i], b[i]
        ia, ib = physical_identity(pa), physical_identity(pb)
        changed = [key for key in ia if ia[key] != ib[key]]
        if changed and first_difference is None:
            first_difference = {'accepted_index': i, 'global_accepted_step': 208 + i,
                                'changed_saved_identity_fields': changed,
                                'scope': 'Saved coordinate/state/H-owner/energy ledger identity; no array numerical replay.'}
        by_label_a = {r['label']: r for r in pa['targets']}
        spectra = [{'label': r['label'], 'UID': r['UID'],
                    'history40_minus_history10_energy_Ha': r['energy_Ha'] - by_label_a[r['label']]['energy_Ha'],
                    'history40_minus_history10_excitation_cm1': r['excitation_cm1'] - by_label_a[r['label']]['excitation_cm1']}
                   for r in pb['targets']]
        rows.append({'accepted_index': i, 'global_accepted_step': 208 + i,
                     'same_saved_numerical_identity': not changed,
                     'history10': {k: pa[k] for k in ('reported_EO_Ha', 'orbital_residual', 'energy_indicator_Ha',
                           'stationarity_passed', 'nominal_four_gap_RMS_cm1', 'multiplets')},
                     'history40': {k: pb[k] for k in ('reported_EO_Ha', 'orbital_residual', 'energy_indicator_Ha',
                           'stationarity_passed', 'nominal_four_gap_RMS_cm1', 'multiplets')},
                     'history40_minus_history10_EO_Ha': pb['reported_EO_Ha'] - pa['reported_EO_Ha'],
                     'matched_physical_target_differences': spectra})
    terminal_a, terminal_b = a[horizon], b[horizon]
    return {'same_literal208_baseline_qualified': True, 'common_accepted_horizon': horizon,
            'common_global_accepted_step': 208 + horizon,
            'comparison_selection': 'Largest common qualified accepted-index horizon; no NIST or energy minimum selection.',
            'first_observed_numerical_divergence': first_difference,
            'all_common_horizon_points': rows,
            'horizon_changes_from_common208': {'history10': horizon_changes(a[0], terminal_a),
                                             'history40': horizon_changes(b[0], terminal_b)},
            'producer_work': {'history10': work_summary(left), 'history40': work_summary(right)},
            'history_capacity_interpretation': {
                'configured_capacities': {'history10': 10, 'history40': 40},
                'both_entry_histories_empty': True, 'actual_retained_pairs_observed': False,
                'capacity40_fully_exercised': False,
                'first_possible_capacity_effect_direction': 12,
                'maximum_directions_after_first_possible_difference': max(0, horizon - 11),
                'scope': '32 updates cannot populate capacity40; actual positive-curvature acceptance/transport/reset counts are not exported. Divergence is evidence for these matched trajectories, not memory40 superiority.'},
            'single_winner_selected': False, 'NIST_used_to_select_memory': False,
            'controlled_timing_or_parallel_speedup_claim': False,
            'new_stationarity_or_experimental_certification': False,
            'historical_derivative_failures_unchanged': True,
            'scope': 'Two sole public ordinary-reported native runs, followed by stdlib byte hashes and saved Fraction/UID metadata arithmetic; no H/C/g independent replay.'}


def main():
    contract = load(ROOT / 'completion_contract.json')
    if sha(__file__) != contract['script_sha256']:
        raise ValueError('Changed prospective closure driver')
    for name, expected in contract['inputs_sha256'].items():
        if sha(REPO / name) != expected:
            raise ValueError('Changed closed input: ' + name)
    pair = load(ROOT / 'root_pair_execution.json')
    if pair['status'] != 'foreground_pair_closed' or pair['timing_claim_valid'] is not False:
        raise ValueError('Pair is not closed or timing scope changed')
    for name in ('history10', 'history40'):
        if pair['branch_receipts_sha256'][name] != sha(ROOT / name / 'root_execution.json'):
            raise ValueError('Pair receipt identity mismatch')
    left, right = close_branch('history10'), close_branch('history40')
    comparison = compare(left, right)
    summary = {'all_completion_metadata_checks_passed': True,
               'branch_metadata_checks': {s['branch']: s['checks'] for s in (left, right)},
               'branches': {s['branch']: {'completion_sha256': sha(ROOT / s['branch'] / 'completion.json'),
                           'actual_status': s['actual_status'], 'actual_foreground_exit_code': s['actual_foreground_exit_code'],
                           'all_live_controls_passed': s['all_actual_live_controls_passed'],
                           'failed_live_controls': s['failed_live_controls'],
                           'actual_qualified_points': len(s['trajectory']),
                           'actual_new_accepted_steps': s['actual_new_accepted_steps'],
                           'forensic_unqualified_candidates': s['forensic_unqualified_candidates'],
                           'stationarity_passed': s['stationarity_passed']} for s in (left, right)},
               'comparison': comparison, 'inputs_sha256': contract['inputs_sha256'],
               'completion_driver_sha256': sha(__file__), 'completion_contract_sha256': sha(ROOT / 'completion_contract.json'),
               'scientific_calls_in_completion': 0, 'array_numerical_replays': 0,
               'same_baseline_identity_check_passed': True, 'no_physical_imports': True,
               'automatic_successor_authorized': False}
    (ROOT / 'completion.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_completion_metadata_checks_passed': True,
                      'branch_metadata_checks': summary['branch_metadata_checks'],
                      'common_accepted_horizon': comparison['common_accepted_horizon'],
                      'first_observed_numerical_divergence': comparison['first_observed_numerical_divergence'],
                      'scientific_calls': 0, 'completion_sha256': sha(ROOT / 'completion.json')}))


if __name__ == '__main__':
    main()
