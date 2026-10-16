"""Constructed JSON/file provenance controls; no Atomx import or arrays loaded."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import run as driver

HERE = Path(__file__).resolve().parent


def main():
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    with tempfile.TemporaryDirectory(prefix='atomx-start160-guard-') as directory:
        root = Path(directory)
        previous_root = driver.ROOT
        driver.ROOT = root

        def save(name, value):
            (root / name).write_text(json.dumps(value, indent=2) + '\n')

        payload = {'synthetic_full_signed_objective': ['A', 'B'], 'weight': .2}
        objective = {'new': payload, 'new_sha256': driver.digest(payload)}
        parent = {'inputs_sha256': {'synthetic/radial': 'bound-fixture'}}
        save('parent.json', parent)
        save('source.json', {'synthetic_source': 'bound-fixture'})
        save('execution.json', {'synthetic_authorization': True})
        save('objective.json', objective)
        (root / 'arrays.npz').write_bytes(b'constructed-byte-fixture-never-loaded-as-array')
        manifest = {'iteration': 160, 'accepted_complete_point': True,
            'array_sha256': driver.sha(root / 'arrays.npz'), 'objective_sha256': driver.digest(payload),
            'source_identity_sha256': driver.sha(root / 'source.json'),
            'preparation_contract_sha256': driver.sha(root / 'parent.json'),
            'execution_contract_sha256': driver.sha(root / 'execution.json'),
            'frozen_scientific_inputs_sha256': parent['inputs_sha256'],
            'original_metric_sha256': 'original-metric-fixture', 'geometry_sha256': 'geometry-fixture',
            'diagnostics': {'objective_identity': driver.digest(payload)}}
        save('point.json', manifest)
        proof = {'all_identity_replay_checks_passed': True, 'checks_count': 2,
            'checks': [{'passed': True}, {'passed': True}],
            'new_source_identity_sha256': driver.sha(root / 'source.json'),
            'preparation_contract_sha256': driver.sha(root / 'parent.json'),
            'execution_contract_sha256': driver.sha(root / 'execution.json'),
            'objective_identity_file_sha256': driver.sha(root / 'objective.json'),
            'accepted_checkpoint_sha256': driver.sha(root / 'point.json'),
            'accepted_arrays_sha256': driver.sha(root / 'arrays.npz'),
            'independently_rebuilt_objective_sha256': driver.digest(payload),
            'independently_rebuilt_objective_payload': payload,
            'actual_original_metric_orbital_residual': 4.7e-5,
            'actual_original_metric_energy_indicator_Ha': 6.9e-9,
            'original_stationarity_gates_passed': False}
        save('proof.json', proof)
        contract = {'start_checkpoint': {'manifest': 'point.json', 'arrays': 'arrays.npz',
            'parent_contract': 'parent.json', 'parent_source_identity': 'source.json',
            'parent_execution_contract': 'execution.json', 'parent_objective_identity': 'objective.json',
            'independent_verification': 'proof.json', 'reference_iteration': 160,
            'same_objective_sha256': driver.digest(payload), 'starting_residual': 4.7e-5,
            'starting_energy_indicator_Ha': 6.9e-9, 'starting_original_stationarity_passed': False},
            'fixed_conditions': {'original_projector_and_preconditioner_metric_sha256': 'original-metric-fixture',
                                 'same_geometry_sha256': 'geometry-fixture'}}

        def passes(value=contract):
            try:
                driver.start_checkpoint_status(value)
            except (ValueError, KeyError, TypeError):
                return False
            return True

        check('complete_constructed_nonstationary_start_is_permitted', passes())
        check('start_status_retains_failed_stationarity',
              driver.start_checkpoint_status(contract)['original_stationarity_passed'] is False)
        for field in manifest:
            changed = copy.deepcopy(manifest)
            changed[field] = False if field == 'accepted_complete_point' else 'changed'
            save('point.json', changed)
            # Bind the altered JSON bytes so this tests its content as well as SHA.
            rebound_proof = copy.deepcopy(proof)
            rebound_proof['accepted_checkpoint_sha256'] = driver.sha(root / 'point.json')
            save('proof.json', rebound_proof)
            check('manifest_content_tampering_rejected:' + field, not passes())
            save('point.json', manifest)
            save('proof.json', proof)
        changed = copy.deepcopy(manifest)
        changed['accepted_complete_point'] = 1
        save('point.json', changed)
        check('truthy_nonboolean_complete_point_is_rejected', not passes())
        save('point.json', manifest)
        for field in proof:
            changed = copy.deepcopy(proof)
            changed[field] = False if field == 'all_identity_replay_checks_passed' else 'changed'
            save('proof.json', changed)
            check('independent_proof_tampering_rejected:' + field, not passes())
            save('proof.json', proof)
        changed = copy.deepcopy(proof)
        changed['checks'][0]['passed'] = False
        save('proof.json', changed)
        check('failed_independent_identity_check_rejected', not passes())
        save('proof.json', proof)
        for field in ('reference_iteration', 'same_objective_sha256', 'starting_residual',
                      'starting_energy_indicator_Ha', 'starting_original_stationarity_passed'):
            changed = copy.deepcopy(contract)
            changed['start_checkpoint'][field] = 'changed'
            check('declared_start_point_tampering_rejected:' + field, not passes(changed))
        for field in contract['fixed_conditions']:
            changed = copy.deepcopy(contract)
            changed['fixed_conditions'][field] = 'changed'
            check('original_geometry_metric_tampering_rejected:' + field, not passes(changed))
        for name in ('arrays.npz', 'source.json', 'parent.json', 'execution.json', 'objective.json'):
            original = (root / name).read_bytes()
            (root / name).write_bytes(original + b' ')
            check('actual_file_byte_change_rejected:' + name, not passes())
            (root / name).write_bytes(original)
        driver.ROOT = previous_root

    groups = [{'reference_levels': [(1, 1, 0)], 'selected_levels': [(1, 1, 0)],
               'per_level_weight': .2, 'principal_overlap_squared': [.9]},
              {'reference_levels': [(3, -1, 0)], 'selected_levels': [(3, -1, 0)],
               'per_level_weight': .2, 'principal_overlap_squared': [.8]}]
    canonical = driver.canonical_physical_target_groups(groups)
    check('runtime_tuple_and_JSON_list_targets_are_identical',
          canonical == driver.canonical_physical_target_groups(json.loads(json.dumps(groups))))
    check('target_group_order_is_not_physical_identity',
          canonical == driver.canonical_physical_target_groups(groups[::-1]))
    mixed = json.loads(json.dumps(groups))
    mixed[0]['selected_levels'][0] = tuple(mixed[0]['selected_levels'][0])
    check('mixed_runtime_JSON_target_containers_are_identical',
          canonical == driver.canonical_physical_target_groups(mixed))
    changed = copy.deepcopy(groups)
    changed[0]['principal_overlap_squared'] = [.6]
    check('measured_overlap_diagnostics_are_not_target_ID',
          canonical == driver.canonical_physical_target_groups(changed))
    for field in ('reference_levels', 'selected_levels'):
        for position, value in enumerate((5, -1, 2)):
            changed = copy.deepcopy(groups)
            changed[0][field] = [list(changed[0][field][0])]
            changed[0][field][0][position] = value
            check('actual_target_UID_change_detected:' + field + ':' + str(position),
                  canonical != driver.canonical_physical_target_groups(changed))
    changed = copy.deepcopy(groups)
    changed[0]['per_level_weight'] = .3
    check('actual_target_weight_change_detected',
          canonical != driver.canonical_physical_target_groups(changed))
    for label, changed in (
            ('duplicate_group', groups + [groups[0]]),
            ('rank_change', [{**groups[0], 'selected_levels': [(1, 1, 0), (1, 1, 1)]}]),
            ('missing_UID', [{**groups[0], 'selected_levels': []}]),
            ('boolean_UID', [{**groups[0], 'selected_levels': [(True, 1, 0)]}]),
            ('noninteger_UID', [{**groups[0], 'selected_levels': [(1., 1, 0)]}]),
            ('negative_root', [{**groups[0], 'selected_levels': [(1, 1, -1)]}]),
            ('invalid_parity', [{**groups[0], 'selected_levels': [(1, 0, 0)]}]),
            ('nan_weight', [{**groups[0], 'per_level_weight': float('nan')}]),
            ('boolean_weight', [{**groups[0], 'per_level_weight': True}]),
            ('zero_weight', [{**groups[0], 'per_level_weight': 0.}]),
            ('empty_groups', [])):
        try:
            driver.canonical_physical_target_groups(changed)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check('invalid_complete_target_metadata_rejected:' + label, rejected)
    span = [{'reference_levels': [(1, 1, 0), (1, 1, 1)],
             'selected_levels': [(1, 1, 2), (1, 1, 3)], 'per_level_weight': .2}]
    reordered = [{**span[0], 'reference_levels': span[0]['reference_levels'][::-1],
                  'selected_levels': span[0]['selected_levels'][::-1]}]
    check('complete_target_span_order_independent_without_column_pairing',
          driver.canonical_physical_target_groups(span) == driver.canonical_physical_target_groups(reordered))

    output = {'all_passed': all(c['passed'] for c in checks), 'checks_count': len(checks), 'checks': checks,
              'driver_sha256': driver.sha(HERE / 'run.py'), 'contract_sha256': driver.sha(HERE / 'contract.json'),
              'script_sha256': driver.sha(Path(__file__)), 'Atomx_imports': 0, 'numerical_arrays_loaded': 0,
              'physical_H_builds': 0, 'CI_solves': 0, 'gradients': 0, 'optimizer_calls': 0,
              'scope': 'Temporary synthetic JSON/file content and provenance only; actual runtime point not evaluated.'}
    (HERE / 'starting_checkpoint_guard_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'all_passed': output['all_passed'], 'checks_count': len(checks)}))
    if not output['all_passed']:
        raise RuntimeError('Preserved failed synthetic starting-point guard')


if __name__ == '__main__':
    main()
