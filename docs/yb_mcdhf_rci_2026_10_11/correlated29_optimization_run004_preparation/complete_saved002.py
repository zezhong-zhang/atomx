"""Close immutable run004 records using stdlib hashes and saved-ledger arithmetic.

No array numerical replay, physical import, producer, eigenproblem or optimizer.
Saved finite-H certificates are checked algebraically, not recomputed from H/C.
"""
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


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


def main():
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
    for name in ('root_execution.json', 'execution_contract.json', 'run.py', 'contract.json',
                 'preparation_verification.json', 'actual/objective_identity.json'):
        bind(HERE / name)
    check('receipt_driver_contract_execution_bound',
          result['driver_sha256'] == receipt['driver_sha256'] == sha(HERE / 'run.py')
          and result['contract_sha256'] == receipt['contract_sha256'] == sha(HERE / 'contract.json')
          and result['execution_sha256'] == receipt['execution_contract_sha256'] == sha(HERE / 'execution_contract.json'))
    accepted = result['trajectory']
    successful_return = result['status'] == 'bounded_public_optimizer_returned' and result['error'] is None
    check('actual_status_and_foreground_exit_consistent', (successful_return and receipt['exit_code'] == 0)
          or (not successful_return and receipt['exit_code'] != 0))
    check('actual_qualified_trajectory_contiguous_within_declared64steps',
          [t['iteration'] for t in accepted] == list(range(len(accepted))) and len(accepted) <= 65
          and all(t['global_accepted_step'] == 16 + t['iteration'] for t in accepted))
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
                and m['diagnostics'] == t['diagnostics']):
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
        records.append({'iteration': m['iteration'], 'global_accepted_step': m['global_accepted_step'],
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
                     'optimizer': 'max_optimizer_calls', 'validation_Dirac': 'max_validation_Dirac'}
    check('actual_producer_counts_preserved_within_contract', all(
          0 <= result['completed_calls'][key] <= result['attempted_calls'][key] <= contract['resources'][limit]
          for key, limit in resource_keys.items()))
    check('actual_global_accepted_count_not_assumed64', result['initial_global_accepted_step'] == 16
          and result['total_accepted_steps'] == 16 + max(0, len(accepted) - 1))
    check('oldFAIL_ordinary_reported_and_no_callback_preserved', result['original_reported_derivative_failure_preserved'] is True
          and result['ordinary_reported_objective_changed'] is False and result['energy_callback_used'] is False)
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
               'terminal_target_changes': terminal_changes, 'completed_calls': result['completed_calls'],
               'exact_guard_budget': result['exact_guard_budget'],
               'all_qualified_finite_stored_H_acceptances_qualified': True, 'qualified_finite_acceptance_count': max(0, len(records) - 1),
               'stationarity_passed': result['stationarity_passed'], 'experimental_certification': False, 'physical_completeness_certification': False,
               'historical_derivative_failures_unchanged': True, 'optimizer_successor_authorized': False,
               'timing_scope': 'uncontrolled correctness measurement', 'actual_elapsed_uncontrolled_s': result['elapsed_uncontrolled_s'],
               'outer_elapsed_uncontrolled_s': receipt['elapsed_uncontrolled_seconds'], 'peak_RSS_MiB': result['peak_RSS_MiB']}
    (HERE / 'completion.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_completion_metadata_checks_passed': True, 'checks': len(controls), 'scientific_calls': 0,
                      'completion_sha256': sha(HERE / 'completion.json')}))


if __name__ == '__main__':
    main()
