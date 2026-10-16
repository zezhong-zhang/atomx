"""Once-only saved-coordinate algebra; no Atomx, Hamiltonian or spectral calls."""
import argparse
import datetime
import hashlib
import json
import os
import resource
import signal
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
UIDS = {'6s_1/2': (1, 1, 0), '5d_3/2': (3, 1, 0), '5d_5/2': (5, 1, 0),
        '6p_1/2': (1, -1, 0), '6p_3/2': (3, -1, 0)}
BARE_VALENCE = {(6, -1), (5, -3), (5, 2), (6, -2), (6, 1)}
NEW_VIRTUAL = {(7, -1), (6, -3), (6, 2), (7, -2), (7, 1)}
GROUPS = ('core19', 'bare_valence5', 'new_virtual5')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class AuditStop(ValueError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--execution-contract', required=True)
    args = parser.parse_args()
    contract_path, execution_path = Path(args.contract).resolve(), Path(args.execution_contract).resolve()
    contract, execution = load_json(contract_path), load_json(execution_path)
    if contract['execution_ready'] is not True or contract['terminal_point'] is None:
        raise AuditStop('Terminal closure/input hashes pending; no NumPy import or array read permitted')
    if (execution['authorized_once'] is not True or execution['contract_sha256'] != sha(contract_path)
            or execution['driver_sha256'] != sha(__file__)
            or execution['resources'] != contract['resources']):
        raise AuditStop('Exact once-only root seal does not match driver/contract/resources')
    roles = contract['inputs']
    paths = {}
    for role, item in roles.items():
        path = REPO / item['path']
        if item['sha256'] is None or sha(path) != item['sha256']:
            raise AuditStop(f'Input identity mismatch: {role}')
        paths[role] = path
    receipt, producer = load_json(paths['run005_closed_receipt']), load_json(paths['run005_closed_result'])
    if receipt['status'] != 'foreground_closed' or receipt['exit_code'] != contract['terminal_producer_exit_code']:
        raise AuditStop('run005 must be explicitly closed with the prospectively bound actual exit code')
    last = producer['trajectory'][-1]
    if (last['manifest_path'] != roles['terminal_manifest']['path'] or
            last['manifest_sha256'] != roles['terminal_manifest']['sha256'] or
            last['array_sha256'] != roles['terminal_arrays']['sha256'] or
            last['global_accepted_step'] != contract['terminal_point']['global_accepted_step'] or
            producer['last_qualified_checkpoint'] != last['manifest_path']):
        raise AuditStop('Terminal must be producer-declared last qualified point, never selected by gap')
    layout_rows = load_json(paths['layout'])
    manifest80, manifest_last = load_json(paths['global80_manifest']), load_json(paths['terminal_manifest'])
    for name, role, manifest in [('global80', 'global80_arrays', manifest80), ('terminal', 'terminal_arrays', manifest_last)]:
        if (manifest['checkpoint_qualified'] is not True or manifest['source_map_sha256'] != contract['identity']['source_map_sha256'] or
                manifest['metric_sha256'] != contract['identity']['metric_sha256'] or
                manifest['geometry_sha256'] != contract['identity']['geometry_sha256'] or
                manifest['objective_sha256'] != contract['identity']['objective_sha256'] or
                manifest['array_sha256'] != roles[role]['sha256']):
            raise AuditStop(f'Saved point/model/source/metric mismatch: {name}')
        rows = {row['label']: row for row in manifest['targets']}
        if (len(manifest['targets']) != 5 or set(rows) != set(UIDS) or any(tuple(rows[key]['UID']) != uid or
                rows[key]['tracking']['reference_levels'] != [list(uid)] or
                rows[key]['tracking']['per_level_weight'] != .2 for key, uid in UIDS.items())):
            raise AuditStop(f'Saved physical target identity mismatch: {name}')
        if (manifest['diagnostics']['orbital_tolerance'] != 1e-7 or
                manifest['diagnostics']['energy_tolerance'] != 1e-8):
            raise AuditStop('Original stationarity thresholds changed')
    if manifest80['global_accepted_step'] != 80:
        raise AuditStop('First point is not actual global80')

    resources = contract['resources']
    available = sorted(os.sched_getaffinity(0))
    if resources['CPU'] not in available or any(os.environ.get(v) != '1' for v in THREAD_VARS):
        raise AuditStop('CPU or BLAS/thread resource admission mismatch')
    affinity_before = available
    os.sched_setaffinity(0, {resources['CPU']})
    memory_bytes = resources['address_space_bytes']
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(AuditStop('Offline inner wall cap')))
    signal.alarm(resources['inner_wall_seconds'])
    output = HERE / 'actual'
    output.mkdir(exist_ok=False)
    state = {'status': 'running', 'controls': [], 'points': [], 'scientific_calls': 0,
             'array_reads_attempted': 0, 'array_reads_completed': 0,
             'no_eigensolver_or_Atomx_import': True,
             'driver_sha256': sha(__file__), 'contract_sha256': sha(contract_path),
             'execution_sha256': sha(execution_path), 'inputs_sha256': roles,
             'environment': {'CPU_affinity_before': affinity_before, 'CPU_affinity_actual': sorted(os.sched_getaffinity(0)),
                             'threads': {v: os.environ[v] for v in THREAD_VARS},
                             'address_space_limit_bytes': memory_bytes, 'inner_wall_seconds': resources['inner_wall_seconds'],
                             'load_average': os.getloadavg(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()},
             'scope': 'Saved coordinate algebra only; no physical error bound or producer/stationarity certification upgrade'}

    def flush(phase):
        state['phase'] = phase
        write(output / 'progress.json', state)

    def check(name, value, tolerance=0.):
        passed = value <= tolerance
        state['controls'].append({'name': name, 'value': float(value), 'tolerance': tolerance, 'passed': bool(passed)})
        if not passed:
            flush(name + '_failed')
            raise AuditStop(name)

    try:
        flush('admitted_before_numpy_or_arrays')
        import numpy as np

        def read_arrays(role):
            state['array_reads_attempted'] += 1
            flush(role + '_before_read')
            # Lazy member loads intentionally exclude saved H/energies/mixing.
            required = ('keys', 'x', 'P', 'Q', 'radial_gradient', 'raw_coordinate_gradient',
                        'tangent_gradient', 'preconditioned_gradient', 'inverse_diagonal')
            with np.load(paths[role], allow_pickle=False) as saved:
                values = {key: saved[key].copy() for key in required}
                values.update({f'projector_{i}': saved[f'projector_{i}'].copy() for i in range(7)})
                values.update({f'inverse_blocks_{i}': saved[f'inverse_blocks_{i}'].copy() for i in range(7)})
            state['array_reads_completed'] += 1
            flush(role + '_after_read')
            return values

        reference = None
        for point_name, role, manifest in [('global80', 'global80_arrays', manifest80),
                                            ('run005_declared_last', 'terminal_arrays', manifest_last)]:
            arrays = read_arrays(role)
            keys = tuple(tuple(map(int, key)) for key in arrays['keys'])
            check(point_name + '_29_unique_ordered_keys', not (len(keys) == len(set(keys)) == 29 and keys == tuple(sorted(keys))))
            check(point_name + '_group_membership', not (set(keys) >= BARE_VALENCE and set(keys) >= NEW_VIRTUAL
                  and len(set(keys) - BARE_VALENCE - NEW_VIRTUAL) == 19))
            for key in ('x', 'raw_coordinate_gradient', 'tangent_gradient', 'preconditioned_gradient', 'inverse_diagonal'):
                check(point_name + '_shape_dtype_' + key, not (arrays[key].shape == (3451,) and arrays[key].dtype == np.float64))
            for key in ('P', 'Q'):
                check(point_name + '_shape_dtype_' + key, not (arrays[key].shape == (29, 123) and arrays[key].dtype == np.float64))
            check(point_name + '_radial_gradient_shape', not (arrays['radial_gradient'].shape == (29, 246)
                  and arrays['radial_gradient'].dtype == np.float64))
            check(point_name + '_finite_arrays', not all(np.all(np.isfinite(a)) for a in arrays.values()))
            layout, covered, stop = [], [], 0
            fingerprint = hashlib.sha256()
            for i, row in enumerate(layout_rows):
                group_keys, shape = tuple(map(tuple, row['keys'])), tuple(row['shape'])
                where = slice(*row['slice'])
                projector, blocks = arrays[f'projector_{i}'], arrays[f'inverse_blocks_{i}']
                check(point_name + f'_layout_complete_{i}', not (where.start == stop and where.stop - where.start == shape[0] * shape[1]
                      and shape == (119, len(group_keys)) and len({key[1] for key in group_keys}) == 1
                      and projector.shape == (246, 119) and projector.dtype == np.float64
                      and blocks.shape == (len(group_keys), 119, 119) and blocks.dtype == np.float64))
                stop = where.stop
                covered.extend(group_keys)
                layout.append((group_keys, projector, shape, where, blocks))
                fingerprint.update(repr((list(group_keys), shape)).encode())
                fingerprint.update(np.ascontiguousarray(projector).tobytes())
            # Layout groups are ordered by kappa, while saved radial rows are
            # ordered by (n,kappa). Preserve both recorded orders exactly.
            check(point_name + '_layout_complete_keys_and_end', not (len(covered) == len(set(covered)) == 29
                  and set(covered) == set(keys) and stop == 3451 and len(layout) == 7))
            fingerprint.update(arrays['inverse_diagonal'].tobytes())
            for *_, blocks in layout:
                for block in blocks:
                    fingerprint.update(np.ascontiguousarray(block).tobytes())
            check(point_name + '_original_metric_fingerprint', fingerprint.hexdigest() != contract['identity']['metric_sha256'])
            if reference is None:
                reference = arrays
            else:
                for key in ('keys', 'inverse_diagonal') + tuple(f'projector_{i}' for i in range(7)) + tuple(f'inverse_blocks_{i}' for i in range(7)):
                    check(point_name + '_original_immutable_geometry_' + key, not np.array_equal(arrays[key], reference[key]))

            def tangent(vector, point_layout=layout, point_arrays=arrays):
                result = vector.copy()
                for _, _, shape, where, _ in point_layout:
                    y, v = point_arrays['x'][where].reshape(shape), vector[where].reshape(shape)
                    a = y.T @ v
                    result[where] = (v - .5 * y @ (a + a.T)).ravel()
                return result

            def inverse(vector, point_layout=layout):
                result = np.zeros_like(vector)
                for _, _, shape, where, blocks in point_layout:
                    v = vector[where].reshape(shape)
                    result[where] = np.column_stack([block @ v[:, j] for j, block in enumerate(blocks)]).ravel()
                return result

            raw = np.zeros(3451)
            key_to_row = {key: j for j, key in enumerate(keys)}
            indices = {name: [] for name in GROUPS}
            orbital_rows, kappa_rows = [], []
            for group_keys, projector, shape, where, _ in layout:
                y = arrays['x'][where].reshape(shape)
                check(point_name + f'_coordinate_Gram_{group_keys[0][1]}', np.max(abs(y.T @ y - np.eye(shape[1]))), 1e-10)
                c = projector @ y
                check(point_name + f'_direct_P_times_x_{group_keys[0][1]}', not np.array_equal(c, np.column_stack([
                    np.concatenate((arrays['P'][key_to_row[k]], arrays['Q'][key_to_row[k]])) for k in group_keys])))
                raw[where] = (projector.T @ np.column_stack([arrays['radial_gradient'][key_to_row[k]] for k in group_keys])).ravel()
                for j, key in enumerate(group_keys):
                    name = 'bare_valence5' if key in BARE_VALENCE else 'new_virtual5' if key in NEW_VIRTUAL else 'core19'
                    column_indices = where.start + np.arange(shape[0]) * shape[1] + j
                    indices[name].extend(column_indices.tolist())
                    g_j, pg_j = arrays['tangent_gradient'][column_indices], arrays['preconditioned_gradient'][column_indices]
                    l = -key[1]-1 if key[1] < 0 else key[1]
                    label = f'{key[0]}{"spdfgh"[l]}_{2*abs(key[1])-1}/2'
                    orbital_rows.append({'key': key, 'label': label, 'group': name, 'pg_norm_squared': float(pg_j @ pg_j),
                                         'signed_g_dot_pg': float(g_j @ pg_j), 'g_norm': float(np.linalg.norm(g_j))})
                g_k, pg_k = arrays['tangent_gradient'][where], arrays['preconditioned_gradient'][where]
                kappa_rows.append({'kappa': group_keys[0][1], 'pg_norm_squared': float(pg_k @ pg_k),
                                   'signed_g_dot_pg': float(g_k @ pg_k), 'g_norm': float(np.linalg.norm(g_k))})
            g, pg = tangent(raw), tangent(inverse(tangent(raw)))
            check(point_name + '_saved_raw_g_whole', np.max(abs(raw - arrays['raw_coordinate_gradient'])), 1e-9)
            check(point_name + '_saved_tangent_g_whole', np.max(abs(g - arrays['tangent_gradient'])), 1e-9)
            check(point_name + '_saved_pg_original_metric_whole', np.max(abs(pg - arrays['preconditioned_gradient'])), 1e-10)
            saved_g, saved_pg = arrays['tangent_gradient'], arrays['preconditioned_gradient']
            residual_squared, signed_work = float(saved_pg @ saved_pg), float(saved_g @ saved_pg)
            check(point_name + '_saved_residual', abs(float(np.linalg.norm(saved_pg)) - manifest['diagnostics']['orbital_residual']), 1e-10)
            check(point_name + '_saved_energy_indicator', abs(.5 * abs(signed_work) - manifest['diagnostics']['estimated_energy_error']), 1e-10)
            check(point_name + '_29_column_residual_squared_sum', abs(sum(r['pg_norm_squared'] for r in orbital_rows) - residual_squared), 1e-10)
            check(point_name + '_29_column_signed_work_sum', abs(sum(r['signed_g_dot_pg'] for r in orbital_rows) - signed_work), 1e-10)
            # Groups mask only raw coordinate input; both tangent projections remain full same-kappa maps.
            group_g, group_pg = [], []
            for name in GROUPS:
                masked = np.zeros_like(raw)
                masked[indices[name]] = raw[indices[name]]
                tangent_component = tangent(masked)
                drive = tangent(inverse(tangent_component))
                group_g.append(tangent_component)
                group_pg.append(drive)
                for group_keys, _, shape, where, _ in layout:
                    y, d = arrays['x'][where].reshape(shape), drive[where].reshape(shape)
                    check(point_name + f'_group_tangent_{name}_{group_keys[0][1]}', np.max(abs(y.T @ d + d.T @ y)), 1e-10)
            group_g, group_pg = np.array(group_g), np.array(group_pg)
            check(point_name + '_group_g_sum_whole', np.max(abs(group_g.sum(axis=0) - g)), 1e-9)
            check(point_name + '_group_pg_sum_whole', np.max(abs(group_pg.sum(axis=0) - pg)), 1e-10)
            cross_work, cross_residual = group_g @ group_pg.T, group_pg @ group_pg.T
            check(point_name + '_full_signed_cross_work_sum', abs(float(cross_work.sum()) - float(g @ pg)), 1e-10)
            check(point_name + '_full_cross_residual_sum', abs(float(cross_residual.sum()) - float(pg @ pg)), 1e-10)
            norm_shares = [{**row, 'residual_squared_coordinate_share': row['pg_norm_squared'] / residual_squared if residual_squared else None}
                           for row in orbital_rows]
            result = {'point': point_name, 'global_accepted_step': manifest['global_accepted_step'],
                      'array_sha256': roles[role]['sha256'], 'metric_sha256': fingerprint.hexdigest(),
                      'geometry_sha256': manifest['geometry_sha256'], 'source_map_sha256': manifest['source_map_sha256'],
                      'objective_sha256': manifest['objective_sha256'], 'reported_EO_Ha': manifest['diagnostics']['energy_ha'],
                      'orbital_residual': float(np.linalg.norm(saved_pg)), 'orbital_tolerance': 1e-7,
                      'signed_g_dot_pg': signed_work, 'energy_indicator_Ha': .5 * abs(signed_work), 'energy_tolerance_Ha': 1e-8,
                      'stationarity_passed': manifest['stationarity_passed'],
                      'per_orbital_columns': norm_shares, 'per_kappa': kappa_rows, 'group_order': GROUPS,
                      'TMT_group_signed_work_matrix': cross_work.tolist(),
                      'TMT_group_residual_cross_matrix': cross_residual.tolist(),
                      'TMT_group_drive_norms': [float(np.linalg.norm(v)) for v in group_pg],
                      'coordinate_masked_pg_norms': {name: float(np.linalg.norm(saved_pg[ids])) for name, ids in indices.items()},
                      'no_pair_validation_scope': 'Saved original nuclear positive-P/source and physical geometry lineage; exact two-point P equality and direct unpack, no new spectral or physical-S validation',
                      'not_a_state_energy_error_bound': True, 'core_dominance_inferred_without_measurement': False}
            numeric_path = output / (point_name + '_algebra.npz')
            np.savez_compressed(numeric_path, reconstructed_raw=raw, reconstructed_g=g, reconstructed_pg=pg,
                                group_g=group_g, group_pg=group_pg, cross_work=cross_work, cross_residual=cross_residual)
            result['saved_algebra_sha256'] = sha(numeric_path)
            state['points'].append(result)
            flush(point_name + '_completed')
        check('exactly_two_declared_archive_reads', state['array_reads_completed'] != 2)
        state.update(status='completed', all_declared_controls_passed=True,
                     changed_producer_stationarity_or_physics_flags=False)
    except Exception as error:
        state.update(status='failed', error={'type': type(error).__name__, 'message': str(error)}, all_declared_controls_passed=False)
        raise
    finally:
        signal.alarm(0)
        state.update(elapsed_uncontrolled_s=time.monotonic()-started,
                     peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                     finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(output / 'result.json', state)


if __name__ == '__main__':
    main()
