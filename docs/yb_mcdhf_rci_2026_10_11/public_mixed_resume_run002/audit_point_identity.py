"""Read-only source and archived coordinate reconstruction; no physics solve."""

import ast
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = ROOT / 'docs/yb_mcdhf_rci_2026_10_11'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np

    native = PREFIX / 'native_run/nuclear_eol005_projected_increment'
    old_replay = native / 'stationarity_replay002'
    source_path = old_replay / 'diagnostic_source.py'
    source = source_path.read_text()
    tree = ast.parse(source)
    evaluations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == 'evaluate']
    saved_arrays = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'savez_compressed']
    workspace = np.load(native / 'orbitals_workspace.npz')
    recovered = np.load(old_replay / 'recovered_coordinates.npz')
    original_metric = PREFIX / 'native_run/nuclear_eol004_stable_delta_trace'
    metric = np.load(original_metric / 'failed_coordinates.npz')
    layout = json.loads((original_metric / 'failed_coordinate_layout.json').read_text())
    original_waves = {tuple(k): np.concatenate((p, q)) for k, p, q in
                      zip(workspace['orbital_keys'], workspace['P'], workspace['Q'], strict=True)}
    maximum = 0.
    digest = hashlib.sha256()
    for index, group in enumerate(layout):
        keys, shape = [tuple(key) for key in group['keys']], tuple(group['shape'])
        projector = metric[f'projector_{index}']
        direct = projector @ recovered['x'][slice(*group['slice'])].reshape(shape)
        original = np.column_stack([original_waves[key] for key in keys])
        maximum = max(maximum, float(np.max(abs(direct - original))))
        digest.update(repr((keys, shape)).encode())
        digest.update(np.ascontiguousarray(projector).tobytes())
    digest.update(metric['inverse_diagonal'].tobytes())
    for index in range(len(layout)):
        for block in metric[f'inverse_blocks_{index}']:
            digest.update(np.ascontiguousarray(block).tobytes())
    predecessor = PREFIX / 'public_mixed_resume_run'
    controls = json.loads((predecessor / 'new_source_baseline_controls.json').read_text())
    checks = [
        {'name': 'old_gradient_evaluated_on_workspace_orbitals',
         'passed': len(evaluations) == 1 and ast.unparse(evaluations[0].args[0]) == 'orbitals'
         and 'zip(workspace["orbital_keys"], workspace["P"], workspace["Q"], strict=True)' in source},
        {'name': 'old_gradient_evaluation_precedes_coordinate_recovery',
         'passed': source.index('functional.evaluate(orbitals)') < source.index('p.astype(np.longdouble).T')},
        {'name': 'old_saved_g_is_projected_existing_physical_wave_gradient',
         'passed': 'derivative = p.T @ np.column_stack([gradient[key] for key in keys])' in source
         and len(saved_arrays) == 1 and {'x', 'g', 'preconditioned_gradient'}
         <= {kw.arg for kw in saved_arrays[0].keywords}},
        {'name': 'direct_unpacked_x_and_original_physical_waves_differ', 'passed': 0 < maximum < 1e-12},
        {'name': 'independent_reconstruction_matches_first_launch_difference',
         'passed': maximum == controls['original_x_minus_saved_physical_wave_max']},
        {'name': 'original_metric_byte_identity_unchanged',
         'passed': digest.hexdigest() == controls['metric_sha256']},
        {'name': 'failed_original_different_point_assertion_preserved',
         'passed': controls['bare_recovered_gradient_replay_max'] > 1e-9
         and (predecessor / 'execution_failure_pre_solve.json').exists()},
    ]
    result = {'all_passed': all(item['passed'] for item in checks), 'checks_count': len(checks), 'checks': checks,
              'source_read_sha256': sha(source_path), 'source_evaluate_line': evaluations[0].lineno,
              'original005_workspace_sha256': sha(native / 'orbitals_workspace.npz'),
              'recovered_coordinate_archive_sha256': sha(old_replay / 'recovered_coordinates.npz'),
              'original_metric_sha256': digest.hexdigest(),
              'direct_P_times_x_minus_original_physical_wave_max': maximum,
              'predecessor_failed_gradient_difference': controls['bare_recovered_gradient_replay_max'],
              'predecessor_failed_gradient_limit': 1e-9,
              'predecessor_baseline_controls_sha256': sha(predecessor / 'new_source_baseline_controls.json'),
              'predecessor_failure_sha256': sha(predecessor / 'execution_failure_pre_solve.json'),
              'conclusion': 'The old saved g/pg uses original physical-wave gradient projected at recovered y; it is not a gradient evaluation on direct P*x. A same-point old-source reference is required for a source-equality gate.',
              'not_proven': 'No spectral-stiffness explanation or absence of a source derivative defect is certified before the independently frozen matched reference.',
              'physics_H_builds': 0, 'CI_solves': 0, 'full_gradients': 0, 'optimizer_calls': 0,
              'script_sha256': sha(Path(__file__))}
    (HERE / 'point_identity_source_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks), 'maximum_wave_difference': maximum}))
    if not result['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
