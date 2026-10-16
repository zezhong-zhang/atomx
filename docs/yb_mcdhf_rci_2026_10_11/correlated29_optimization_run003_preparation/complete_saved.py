"""Close immutable run003 records using stdlib hashes and saved-ledger arithmetic.

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

    check('foreground_closed_exit0', receipt['status'] == 'foreground_closed' and receipt['exit_code'] == 0)
    for name, expected in receipt['actual_files_sha256'].items():
        bind(HERE / name, expected)
    for name in ('root_execution.json', 'execution_contract.json', 'run.py', 'contract.json',
                 'preparation_verification.json', 'actual/objective_identity.json'):
        bind(HERE / name)
    check('receipt_driver_contract_execution_bound',
          result['driver_sha256'] == receipt['driver_sha256'] == sha(HERE / 'run.py')
          and result['contract_sha256'] == receipt['contract_sha256'] == sha(HERE / 'contract.json')
          and result['execution_sha256'] == receipt['execution_contract_sha256'] == sha(HERE / 'execution_contract.json'))
    check('successful_bounded_return', result['status'] == 'bounded_public_optimizer_returned' and result['error'] is None)
    check('all1175_actual_controls_passed', len(result['controls']) == 1175 and all(c['passed'] for c in result['controls']))
    check('complete0_to16_trajectory', [t['iteration'] for t in result['trajectory']] == list(range(17)))
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
                and all(rows[k]['UID'] == uid for k, uid in expected_uids.items())):
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
        records.append({'iteration': m['iteration'], 'reported_EO_Ha': m['diagnostics']['energy_ha'],
                        'orbital_residual': m['diagnostics']['orbital_residual'],
                        'energy_indicator_Ha': m['diagnostics']['estimated_energy_error'],
                        'nominal_four_gap_RMS_cm1': rms, 'targets': [{k: v for k, v in row.items() if k != 'tracking'}
                                                                  for row in m['targets']],
                        'multiplets': m['multiplets'], 'exact_acceptance_qualified': None if a is None else a['qualified']})
        previous = m
    check('all17_manifests_arrays_bound_UID_metric_geometry_and16_exact_ledgers', True)
    check('terminal16_not_best_gap_selected', result['last_qualified_checkpoint'] == result['trajectory'][-1]['manifest_path'])
    diag = result['optimizer_final_diagnostics']
    check('original_stationarity_failure_not_upgraded', diag['converged'] is False
          and result['stationarity_passed'] is False and diag['termination_reason'] == 'maximum_iterations'
          and diag['orbital_residual'] >= 1e-7 and diag['estimated_energy_error'] >= 1e-8)
    check('exactscope27H27CI17g_1optimizer', all(result['completed_calls'][k] == v for k, v in
          {'H': 27, 'CI': 27, 'gradient': 17, 'optimizer': 1, 'validation_Dirac': 7}.items())
          and result['attempted_calls'] == result['completed_calls'])
    check('oldFAIL_ordinary_reported_and_no_callback_preserved', result['original_reported_derivative_failure_preserved'] is True
          and result['ordinary_reported_objective_changed'] is False and result['energy_callback_used'] is False)
    first, last = records[0], records[-1]
    first_rows = {r['label']: r for r in first['targets']}
    terminal_changes = [{ 'label': row['label'], 'UID': row['UID'],
                        'delta_energy_Ha': row['energy_Ha'] - first_rows[row['label']]['energy_Ha'],
                        'transition_shift_cm1': row['excitation_cm1'] - first_rows[row['label']]['excitation_cm1'],
                        'terminal_excitation_cm1': row['excitation_cm1'], 'terminal_nominal_gap_cm1': row['nominal_gap_cm1']}
                       for row in last['targets']]
    check('zero_physical_imports', not any(n == 'numpy' or n.startswith('numpy.') or n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    summary = {'all_completion_metadata_checks_passed': True, 'checks': len(controls), 'controls': controls,
               'inputs_sha256': inputs, 'completion_driver_sha256': sha(__file__),
               'actual_live_controls': 1175, 'live_control_scope': 'Sole public run; closure is metadata/saved-ledger arithmetic, not independent H/C/g replay',
               'scientific_calls_in_completion': 0, 'array_numerical_replays': 0,
               'trajectory': records, 'terminal_checkpoint': result['last_qualified_checkpoint'],
               'terminal_array_sha256': previous['array_sha256'], 'terminal_manifest_sha256': sha(REPO / result['last_qualified_checkpoint']),
               'reported_EO_delta_Ha': last['reported_EO_Ha'] - first['reported_EO_Ha'],
               'four_line_RMS_delta_cm1': last['nominal_four_gap_RMS_cm1'] - first['nominal_four_gap_RMS_cm1'],
               'terminal_target_changes': terminal_changes, 'completed_calls': result['completed_calls'],
               'exact_guard_budget': result['exact_guard_budget'], 'all16_finite_stored_H_acceptances_qualified': True,
               'stationarity_passed': False, 'experimental_certification': False, 'physical_completeness_certification': False,
               'historical_derivative_failures_unchanged': True, 'optimizer_successor_authorized': False,
               'timing_scope': 'uncontrolled correctness measurement', 'actual_elapsed_uncontrolled_s': result['elapsed_uncontrolled_s'],
               'outer_elapsed_uncontrolled_s': receipt['elapsed_uncontrolled_seconds'], 'peak_RSS_MiB': result['peak_RSS_MiB']}
    (HERE / 'completion.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_completion_metadata_checks_passed': True, 'checks': len(controls), 'scientific_calls': 0,
                      'completion_sha256': sha(HERE / 'completion.json')}))


if __name__ == '__main__':
    main()
