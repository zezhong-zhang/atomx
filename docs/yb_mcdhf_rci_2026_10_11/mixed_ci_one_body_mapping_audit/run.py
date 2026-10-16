"""Root-sealed saved-matrix representation audit; zero Atomx/builder calls."""
import hashlib
import json
import os
import resource
import signal
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
PRIOR = BASE.parent / 'mixed_ci_finite_increment_implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    contract = json.loads((BASE / 'contract.json').read_text())
    execution = json.loads((BASE / 'execution_contract.json').read_text())
    if (execution.get('saved_mapping_audit_execution_authorized') is not True
            or execution.get('contract_sha256') != sha(BASE / 'contract.json')
            or execution.get('runner_sha256') != sha(Path(__file__))):
        raise ValueError('explicit root saved-array execution seal required')
    if len(os.sched_getaffinity(0)) != 1 or any(os.environ.get(k) != '1'
            for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('CPU/thread admission failed')
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    def stop(signum, frame):
        raise TimeoutError('saved-matrix audit30s wall cap exhausted')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(30)
    for name, expected in contract['inputs_sha256'].items():
        if sha(REPO / name) != expected:
            raise ValueError('bound input changed: ' + name)
    import numpy as np
    geometry = np.load(PRIOR / 'actual/geometry_inputs.npz', allow_pickle=False)
    keys = tuple(tuple(int(v) for v in key) for key in geometry['orbital_keys'])
    checks, rows = [], []
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
    def pairs(left, right, defect, metric, left_names, right_names, *, normalize_ket):
        values = left.T @ defect @ right
        scales = np.linalg.norm(left, axis=0)[:, None] * np.linalg.norm(right, axis=0)[None, :]
        relative = np.zeros_like(values)
        np.divide(np.abs(values), scales * np.max(np.abs(defect)),
                  out=relative, where=scales * np.max(np.abs(defect)) != 0)
        norms = np.diag(right.T @ metric @ right)
        normalized = values / norms[None, :] if normalize_ket else values
        return dict(signed_pairs=[dict(left=a, right=b, signed_Ha=float(values[i, j]),
            absolute_Ha=float(abs(values[i, j])), ket_normalized_Ha=float(normalized[i, j]),
            relative_to_coefficient_norms_times_max_defect_entry=float(relative[i, j]),
            left_norm=float(np.linalg.norm(left[:, i])), right_norm=float(np.linalg.norm(right[:, j])),
            ket_metric_norm=float(norms[j]))
            for i, a in enumerate(left_names) for j, b in enumerate(right_names)],
            matrix=values.tolist(), maximum_absolute_Ha=float(np.max(np.abs(values))),
            maximum_ket_normalized_Ha=float(np.max(np.abs(normalized))))
    for k in (-2, -1, 1):
        raw, metric, P = geometry[f'h_{k}'], geometry[f'S_{k}'], geometry[f'P_{k}']
        mapped = np.tril(raw) + np.tril(raw, -1).T
        defect = raw - mapped
        check(f'k{k}_lower_triangle_unchanged', np.array_equal(np.tril(mapped), np.tril(raw)))
        check(f'k{k}_mapped_exactly_symmetric', np.array_equal(mapped, mapped.T))
        check(f'k{k}_diagonal_unchanged', np.array_equal(np.diag(mapped), np.diag(raw)))
        check(f'k{k}_raw_float64_finite', raw.dtype == np.float64 and np.isfinite(raw).all())
        active = [key for key in keys if key[1] == k]
        projector = pairs(P, P, defect, metric, list(range(P.shape[1])), list(range(P.shape[1])),
                          normalize_ket=False)
        check(f'k{k}_projector_matrix_representation_gate',
              projector['maximum_absolute_Ha'] <= contract['gates']['P_matrix_Ha'])
        row = dict(kappa=k, raw_upper_lower_disagreement_Ha=float(np.max(np.abs(defect))),
                   raw_minus_mapped_nonzero=int(np.count_nonzero(defect)),
                   lower_mapping='tril(raw)+tril(raw,-1).T; no average/energy shift',
                   original_P=projector, points=[], directions=[])
        for index in range(13):
            C = np.column_stack([np.concatenate(geometry[f'waves_{index}'][keys.index(key)])
                                 for key in active])
            point = pairs(C, C, defect, metric, active, active, normalize_ket=True)
            # N=2 row-sum bound: no cancellation of signed orbital pairs.
            matrix = np.asarray(point['matrix'])
            bound = 2 * float(np.sqrt(np.max(np.sum(np.abs(matrix), axis=0))
                                    * np.max(np.sum(np.abs(matrix), axis=1))))
            check(f'k{k}_point{index}_every_signed_pair_finite_gate',
                  point['maximum_ket_normalized_Ha'] <= contract['gates']['actual_wave_pair_Ha'])
            check(f'k{k}_point{index}_two_electron_noncancelling_bound',
                  bound <= contract['gates']['actual_wave_pair_Ha'])
            row['points'].append(dict(index=index, pairs=point, two_electron_bound_Ha=bound))
        C = np.column_stack([np.concatenate(geometry['waves_0'][keys.index(key)]) for key in active])
        for direction in (0, 1):
            T = P @ geometry[f'eta{direction}_{k}']
            left = pairs(T, C, defect, metric, active, active, normalize_ket=True)
            right = pairs(C, T, defect, metric, active, active, normalize_ket=False)
            a, b = np.asarray(left['matrix']), np.asarray(right['matrix'])
            absolute_sum = np.abs(a) + np.abs(b)
            bound = 2 * float(np.sqrt(np.max(np.sum(absolute_sum, axis=0))
                                    * np.max(np.sum(absolute_sum, axis=1))))
            check(f'k{k}_direction{direction}_every_signed_pair_derivative_gate',
                  max(left['maximum_absolute_Ha'], right['maximum_absolute_Ha'])
                  <= contract['gates']['tangent_pair_Ha'])
            check(f'k{k}_direction{direction}_two_electron_noncancelling_derivative_bound',
                  bound <= contract['gates']['tangent_pair_Ha'])
            row['directions'].append(dict(direction=direction, tangent_bra=left,
                tangent_ket=right, two_electron_noncancelling_bound_Ha=bound))
        rows.append(row)
    for index in range(13):
        bound = sum(row['points'][index]['two_electron_bound_Ha'] for row in rows)
        check(f'point{index}_all_kappa_noncancelling_bound',
              bound <= contract['gates']['actual_wave_pair_Ha'])
    for direction in (0, 1):
        bound = sum(row['directions'][direction]['two_electron_noncancelling_bound_Ha'] for row in rows)
        check(f'direction{direction}_all_kappa_noncancelling_bound',
              bound <= contract['gates']['tangent_pair_Ha'])
    for name, expected in contract['inputs_sha256'].items():
        check('exit_' + name, sha(REPO / name) == expected)
    passed = all(c['passed'] for c in checks)
    result = dict(all_passed=passed, checks_count=len(checks), checks=checks, kappa_rows=rows,
        contract_sha256=sha(BASE / 'contract.json'), runner_sha256=sha(Path(__file__)),
        execution_sha256=sha(BASE / 'execution_contract.json'), Atomx_imports=0,
        H_CI_gradient_radial_builders=0, eigensolves=0, SVD=0,
        original_guard002_status='failed before H; immutable',
        scope='saved raw/L representation defects in original P and actual waves/tangents',
        decision='admit preparation of strict-mapping003' if passed else
                 'material defect/identity gate failed; stop before003 physical work')
    output = BASE / 'actual'
    output.mkdir(exist_ok=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    signal.alarm(0)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checks', 'kappa_rows')}))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
