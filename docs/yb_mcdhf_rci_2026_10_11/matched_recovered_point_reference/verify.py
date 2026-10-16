"""Read-only identity/algebra verification; no physical operator evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
BASE = ROOT / 'docs/yb_mcdhf_rci_2026_10_11'
contract = json.loads((OWN / 'contract.json').read_text())
result = json.loads((OWN / 'result.json').read_text())
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name, ok, value=None):
    checks.append({'name': name, 'passed': bool(ok), 'observed': value})
    assert ok, (name, value)


original = json.loads((BASE / 'native_run/nuclear_eol005_projected_increment/source_identity.json').read_text())
frozen = Path(original['import_root'])
for relative, expected in contract['inputs_sha256'].items():
    check('input:' + relative, sha(ROOT / relative) == expected)
for relative, expected in original['all_atomx_python_sha256'].items():
    check('frozen_old_source:' + relative, sha(frozen / Path(relative).relative_to('src')) == expected)
check('old_source_epoch', result['original005_source_map_sha256'] == contract['old_source_map_sha256'])
for module, record in result['loaded_atomx_modules'].items():
    path = Path(record['file'])
    relative = 'src/' + str(path.relative_to(frozen))
    check('loaded_only_frozen_old:' + module, sha(path) == record['sha256'] == original['all_atomx_python_sha256'][relative])
check('contract_bound', result['contract_sha256'] == sha(OWN / 'contract.json'))
check('producer_bound', result['driver_sha256'] == sha(OWN / 'run.py'))
check('exactly_two_gradients', result['full_bare_gradient_calls'] == 2)
check('no_optimizer', result['optimizer_calls'] == 0)
check('CPU1_enforced', len(result['enforced_affinity']) == 1)
check('memory_cap_enforced', result['enforced_address_space_GiB'] == 3.95)
check('completed_inside_hard60s', result['seconds_uncontrolled'] < 60, result['seconds_uncontrolled'])
check('observed_RSS_inside_plan', result['observed_peak_RSS_MiB'] < 1.4 * 1024, result['observed_peak_RSS_MiB'])
rows = json.loads((BASE / 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json').read_text())
metric = np.load(BASE / 'native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz')
archived = np.load(BASE / 'native_run/nuclear_eol005_projected_increment/stationarity_replay002/recovered_coordinates.npz')
workspace = np.load(BASE / 'native_run/nuclear_eol005_projected_increment/orbitals_workspace.npz')
all_point_g = {}
for label, filename in [('physical', 'physical_point_arrays.npz'), ('actual', 'arrays.npz')]:
    data = np.load(OWN / filename)
    saved = result['physical_original005' if label == 'physical' else 'actual_x_unpack_originalpoint']
    check(label + ':NPZ_identity', sha(OWN / filename) == saved['arrays_sha256'])
    for name in data.files:
        check(label + ':finite:' + name, np.isfinite(data[name]).all())
    check(label + ':archived_x_bitwise', np.array_equal(data['x'], archived['x']))
    check(label + ':x_raw_bytes_identity', hashlib.sha256(data['x'].tobytes()).hexdigest() == result['x_raw_bytes_sha256'])
    check(label + ':complete_radial_keys', data['orbital_keys'].shape == (24, 2)
          and np.array_equal(data['orbital_keys'], workspace['orbital_keys']))
    check(label + ':equal_full_five_weights', np.array_equal(data['state_weights'], np.full(5, .2)))
    check(label + ':CI_energy_objective', abs(float(data['state_weights'] @ data['ci_energies_Ha'])
                                             - float(data['EOL_energy_Ha'])) < 1e-9)
    check(label + ':full_CI_orthonormal', np.linalg.norm(data['ci_mixing'].T @ data['ci_mixing'] - np.eye(5)) < 1e-10)
    digest = hashlib.sha256()
    g = np.zeros_like(data['x'])
    inverse = np.zeros_like(g)
    for i, row in enumerate(rows):
        keys, shape = [tuple(k) for k in row['keys']], tuple(row['shape'])
        where = slice(*row['slice'])
        p, s = data[f'projector_{i}'], data[f'metric_{i}']
        check(label + f':projector_{i}_bitwise', np.array_equal(p, metric[f'projector_{i}']))
        check(label + f':inverse_blocks_{i}_bitwise', np.array_equal(data[f'inverse_blocks_{i}'], metric[f'inverse_blocks_{i}']))
        digest.update(repr((keys, shape)).encode())
        digest.update(np.ascontiguousarray(p).tobytes())
        y = data['x'][where].reshape(shape)
        columns = [list(map(tuple, data['orbital_keys'])).index(k) for k in keys]
        c = np.column_stack([np.r_[data['P'][col], data['Q'][col]] for col in columns])
        if label == 'actual':
            check(f'actual:direct_Px_group_{i}_bitwise', np.array_equal(c, p @ y))
        else:
            original_c = np.column_stack([np.r_[workspace['P'][col], workspace['Q'][col]] for col in columns])
            check(f'physical:original_wave_group_{i}_bitwise', np.array_equal(c, original_c))
        check(label + f':physical_Gram_{i}', np.linalg.norm(c.T @ s @ c - np.eye(shape[1]), ord=np.inf) < 1e-10)
        derivatives = data['radial_gradient'][columns].T
        projected = p.T @ derivatives
        g[where] = (projected - y @ ((y.T @ projected + projected.T @ y) / 2)).ravel()
    digest.update(data['inverse_diagonal'].tobytes())
    for i, _ in enumerate(rows):
        for block in data[f'inverse_blocks_{i}']:
            digest.update(np.ascontiguousarray(block).tobytes())
    check(label + ':legacy_complete_metric_identity', digest.hexdigest() == result['original_metric_sha256'])
    for i, row in enumerate(rows):
        where, shape = slice(*row['slice']), tuple(row['shape'])
        y, derivatives = data['x'][where].reshape(shape), g[where].reshape(shape)
        # Assemble the full block diagonal inverse action independently.
        blocks = data[f'inverse_blocks_{i}']
        big = np.zeros((len(blocks) * shape[0],) * 2)
        for col, block in enumerate(blocks):
            slot = slice(col * shape[0], (col + 1) * shape[0])
            big[slot, slot] = block
        action = (big @ derivatives.T.ravel()).reshape(shape[1], shape[0]).T
        inverse[where] = (action - y @ ((y.T @ action + action.T @ y) / 2)).ravel()
    g_error = float(np.max(abs(g - data['g'])))
    inverse_error = float(np.max(abs(inverse - data['preconditioned_gradient'])))
    check(label + ':saved_full_radial_to_coordinate_g', g_error < 1e-9, g_error)
    check(label + ':independent_block_diagonal_pg', inverse_error < 1e-10, inverse_error)
    check(label + ':original_stationarity', np.linalg.norm(inverse) < 1e-7 and .5 * abs(g @ inverse) < 1e-8)
    check(label + ':residual_scalar', abs(np.linalg.norm(inverse) - saved['original_metric_orbital_residual']) < 1e-10)
    all_point_g[label] = data['g']
check('physical_coordinate_g_matches_archive', np.max(abs(all_point_g['physical'] - archived['g'])) < 1e-9)
check('actual_coordinate_g_intentionally_differs_old_physical_point', np.max(abs(all_point_g['actual'] - archived['g'])) > 1e-9)
check('point_difference_explanation_supported', result['different_radial_point_explanation_supported'])
for name, value in result['comparisons'].items():
    if name.startswith('old_source_'):
        check('stopped_new_scalar:' + name, value == 0., value)
check('cross_source_elementwise_not_falsely_claimed', result['comparisons']['complete_old_current_elementwise_g_comparison_obtained'] is False)
output = {'all_passed': True, 'checks_count': len(checks), 'checks': checks,
          'contract_sha256': sha(OWN / 'contract.json'), 'result_sha256': sha(OWN / 'result.json'),
          'arrays_sha256': sha(OWN / 'arrays.npz'), 'physical_arrays_sha256': sha(OWN / 'physical_point_arrays.npz'),
          'verifier_sha256': sha(Path(__file__)), 'verification_scope': 'separate saved-array identity and algebra replay',
          'new_full_gradient_evaluations': 0, 'new_CI_solves': 0, 'optimizer_calls': 0,
          'strict_g_gate': 1e-9, 'strict_pg_gate': 1e-10,
          'source_regression_elementwise_gate_remains_for_future_consumer': True,
          'expanded_model_or_experimental_certification': False}
(OWN / 'verification.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
print(json.dumps({'all_passed': True, 'checks': len(checks), 'verification_sha256': sha(OWN / 'verification.json')}))
