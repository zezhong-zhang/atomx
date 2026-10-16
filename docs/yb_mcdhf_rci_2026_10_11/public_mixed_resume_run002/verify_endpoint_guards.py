"""Bounded synthetic checkpoint guards; no Atomx physics import or eigensolve."""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from verify_endpoint import (
    canonical_sha,
    checkpoint_bindings,
    rebuilt_objective_payload,
    saved_eigenpair_checks,
)

HERE = Path(__file__).resolve().parent


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np

    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    with tempfile.TemporaryDirectory(prefix='atomx-checkpoint-guard-') as directory:
        paths = [Path(directory) / name for name in ('source.json', 'preparation.json', 'execution.json')]
        for index, path in enumerate(paths):
            path.write_text(json.dumps({'synthetic_fixture_only': True, 'index': index}) + '\n')
        inputs = {'frozen/reference.npz': 'reference-fixture-sha'}
        expected = dict(source_sha256=file_sha(paths[0]), contract_sha256=file_sha(paths[1]),
                        execution_sha256=file_sha(paths[2]), inputs=inputs)
        checkpoint = {'accepted_complete_point': True, 'source_identity_sha256': expected['source_sha256'],
                      'preparation_contract_sha256': expected['contract_sha256'],
                      'execution_contract_sha256': expected['execution_sha256'],
                      'frozen_scientific_inputs_sha256': inputs.copy()}
        check('synthetic_actual_file_bindings_accepted', all(checkpoint_bindings(checkpoint, **expected).values()))
        for field in checkpoint:
            tampered = copy.deepcopy(checkpoint)
            tampered[field] = False if field == 'accepted_complete_point' else 'different'
            check('tampered_manifest_rejected:' + field, not all(checkpoint_bindings(tampered, **expected).values()))
        tampered = copy.deepcopy(checkpoint)
        tampered['accepted_complete_point'] = 1
        check('truthy_nonboolean_accepted_flag_rejected', not all(checkpoint_bindings(tampered, **expected).values()))
        tampered = copy.deepcopy(checkpoint)
        del tampered['frozen_scientific_inputs_sha256']['frozen/reference.npz']
        check('missing_frozen_input_binding_rejected', not all(checkpoint_bindings(tampered, **expected).values()))
        for key, path in zip(('source_sha256', 'contract_sha256', 'execution_sha256'), paths, strict=True):
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            changed = dict(expected, **{key: file_sha(path)})
            check('actual_immutable_file_change_rejected:' + key,
                  not all(checkpoint_bindings(checkpoint, **changed).values()))
            path.write_bytes(original)

    records = [{'index': 0, 'coupling': [[1, 1, 0]], 'label': 'signed-A'},
               {'index': 1, 'coupling': [[1, 1, 1]], 'label': 'signed-B'}]
    targets = [{'label': '6s_1/2', 'twoJ': 1, 'parity': 1, 'models': {'DC': {'local_root': 0}}}]
    objective = rebuilt_objective_payload(records, targets, radial_sha256='radial', source_map_sha256='source',
        reference_ci_sha256='CI', reference_binding_sha256='binding')
    check('independent_objective_explicit_physics', objective['Z'] == 70 and objective['electrons'] == 69
          and objective['operator'] == 'full declared no-pair Dirac-Coulomb'
          and objective['nucleus'] == 'native005 Fermi A175'
          and objective['target_identity'][0] == {'label': '6s_1/2', 'twoJ': 1, 'parity': 1, 'root': 0, 'weight': .2})
    expected_digest = hashlib.sha256(json.dumps(objective, separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    check('canonical_serialization_independently_checked', canonical_sha(objective) == expected_digest)
    check('canonical_mapping_order_invariant', canonical_sha(dict(reversed(list(objective.items())))) == expected_digest)
    for field in ('CSFs', 'target_identity', 'Z', 'electrons', 'nucleus', 'operator', 'radial_archive_sha256',
                  'source_map_sha256', 'physical_reference_CI_sha256', 'physical_reference_binding_sha256'):
        changed = copy.deepcopy(objective)
        changed[field] = 'changed'
        check('objective_identity_sensitive:' + field, canonical_sha(changed) != expected_digest)
    changed = copy.deepcopy(objective)
    changed['CSFs'].reverse()
    check('signed_CSF_order_cannot_be_permuted', canonical_sha(changed) != expected_digest)
    changed = copy.deepcopy(objective)
    changed['CSFs'][0]['coupling'][0][2] += 1
    check('signed_CSF_coupling_phase_cannot_change', canonical_sha(changed) != expected_digest)
    for field in ('label', 'twoJ', 'parity', 'root', 'weight'):
        changed = copy.deepcopy(objective)
        changed['target_identity'][0][field] = 'changed'
        check('physical_target_identity_sensitive:' + field, canonical_sha(changed) != expected_digest)

    gates = json.loads((HERE / 'contract.json').read_text())['acceptance']
    energies = np.array([-3., -1., -1., .5])
    archive = {'matrix_Ha': np.diag(energies), 'energies_Ha': energies.copy(), 'mixing': np.eye(4)}

    def passes(value):
        return all(item['passed'] for item in saved_eigenpair_checks(value, gates, 4))

    check('complete_constructed_exact_eigenpairs_pass', passes(archive))
    rotated = copy.deepcopy(archive)
    angle = .731
    rotated['mixing'][1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    check('degenerate_subspace_rotation_passes_without_column_equality', passes(rotated))
    signed = copy.deepcopy(rotated)
    signed['mixing'][:, ::2] *= -1
    check('signed_column_phase_choice_passes', passes(signed))
    permuted = copy.deepcopy(rotated)
    order = [3, 2, 0, 1]
    permuted['energies_Ha'] = permuted['energies_Ha'][order]
    permuted['mixing'] = permuted['mixing'][:, order]
    check('consistent_full_eigenpair_order_passes', passes(permuted))
    changed = copy.deepcopy(archive)
    changed['mixing'][1:3, 1:3] = [[1., 1.], [0., 0.]]
    found = {item['name']: item for item in saved_eigenpair_checks(changed, gates, 4)}
    check('degenerate_duplicate_vectors_residual_alone_can_pass', found['saved_full_eigenpair_residual_Ha']['passed'])
    check('degenerate_duplicate_vectors_rejected_by_Gram', not found['saved_full_signed_CI_Gram']['passed'])
    changed = copy.deepcopy(archive)
    changed['mixing'][[0, 3], :] = changed['mixing'][[3, 0], :]
    check('orthogonal_wrong_nondegenerate_eigenvectors_rejected', not passes(changed))
    changed = copy.deepcopy(archive)
    changed['energies_Ha'][0] += 1e-5
    check('wrong_saved_energy_rejected', not passes(changed))
    changed = copy.deepcopy(archive)
    changed['matrix_Ha'][0, 1] = 1e-4
    check('nonhermitian_saved_H_rejected', not passes(changed))
    for name in archive:
        changed = copy.deepcopy(archive)
        changed[name] = changed[name][:-1]
        check('incomplete_saved_array_rejected:' + name, not passes(changed))
        changed = copy.deepcopy(archive)
        changed[name].flat[0] = np.nan
        check('nonfinite_saved_array_rejected:' + name, not passes(changed))
        changed = copy.deepcopy(archive)
        changed[name] = changed[name].astype(np.float32)
        check('changed_saved_array_precision_rejected:' + name, not passes(changed))
    result = {'all_passed': all(item['passed'] for item in checks), 'checks_count': len(checks), 'checks': checks,
              'verifier_sha256': file_sha(HERE / 'verify_endpoint.py'),
              'guard_runner_sha256': file_sha(Path(__file__)), 'contract_sha256': file_sha(HERE / 'contract.json'),
              'scope': 'Constructed four-dimensional linear-algebra arrays and temporary-file provenance only.',
              'Atomx_imports': 0, 'physical_H_builds': 0, 'CI_solves': 0, 'full_gradients': 0, 'optimizer_calls': 0,
              'actual_endpoint_verified': False, 'runtime_source_sealed': False}
    (HERE / 'endpoint_guard_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks),
                      'failed': [item for item in checks if not item['passed']]}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
