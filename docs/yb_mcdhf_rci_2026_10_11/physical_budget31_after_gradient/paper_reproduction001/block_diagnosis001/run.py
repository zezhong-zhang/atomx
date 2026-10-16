"""One declared saved-matrix diagnosis; no atomx imports or operator builds."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
INPUTS = [HERE.parent / name / "actual/failed_arrays.npz"
          for name in ("yb_high_l002", "yb_high_l003")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    contract = json.loads((HERE / "contract.json").read_text())
    caps = contract["resource_caps"]
    assert os.getloadavg()[0] <= caps["maximum_load1"]
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable'] >= caps['minimum_MemAvailable_bytes']
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(key) == '1'
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))
    import numpy as np
    import scipy
    import scipy.linalg as la

    expected_hashes = [
        '8e32133bd4c917d51f12b9f286d53acc1cbcdb30afbd046ec43fb268c0adb37e',
        '76148f5821e0fa2cf1ab9fbada2ff898aca95382f19382ee38d05def858275c9',
    ]
    assert [sha(p) for p in INPUTS] == expected_hashes
    with np.load(INPUTS[0], allow_pickle=False) as a, np.load(INPUTS[1], allow_pickle=False) as b:
        hfull = a['Fock_raw_4']
        sfull = a['S_4']
        e0 = a['full_electron_E_4']
        vfull = a['full_electron_V_4']
        for left, right in [('Fock_raw_4', 'Fock_raw_4'), ('S_4', 'S_4'),
                            ('full_electron_E_4', 'original_full_electron_E_4'),
                            ('full_electron_V_4', 'original_full_electron_V_4')]:
            assert np.array_equal(a[left], b[right])
        e1 = b['full_electron_E_4']
        v1full = b['full_electron_V_4']
        selected = b['fixed_band_indices_4']
    keep = np.r_[np.arange(1, 120), np.arange(124, 244)]
    assert len(keep) == 239 and hfull.shape == sfull.shape == (246, 246)
    assert e0.shape == e1.shape == (119,) and vfull.shape == v1full.shape == (246, 119)
    assert np.array_equal(selected, np.flatnonzero(e0 < 50.0)) and len(selected) == 67
    other = np.setdiff1d(np.arange(246), keep)
    assert np.all(vfull[other] == 0) and np.all(v1full[other] == 0)
    raw = hfull[np.ix_(keep, keep)]
    h = np.tril(raw) + np.tril(raw, -1).T
    s = sfull[np.ix_(keep, keep)]
    v0, v1 = vfull[keep], v1full[keep]
    assert np.array_equal(s, s.T)
    assert all(np.all(np.isfinite(x)) for x in (h, s, e0, e1, v0, v1))
    hd, sd, vd = (x.astype(np.longdouble) for x in (h, s, v0))
    chol = la.cholesky(s, lower=True)
    q, _ = la.qr(chol.T @ v0, mode='economic')
    gaps = np.min(np.where(np.eye(119, dtype=bool), np.inf, abs(e0[:, None]-e0)), axis=1)
    arrays = {'original_band_indices': selected, 'original_E': e0, 'refined_E': e1}
    out = {'status': 'completed_saved_matrix_diagnosis', 'contract_sha256': sha(HERE/'contract.json'),
           'driver_sha256': sha(Path(__file__)), 'inputs_sha256': expected_hashes,
           'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
           'longdouble_mantissa_bits': int(np.finfo(np.longdouble).nmant),
           'active_dimension': 239, 'original_electron_span_dimension': 119, 'complete_band_roots': 67,
           'atomic_operator_calls': 0, 'eigh_calls': 0, 'routes': {},
           'metric_condition_2': float(np.linalg.cond(s)),
           'original_full_energy_range_Ha': [float(e0.min()), float(e0.max())]}

    def measure(name, e, v):
        assert e.shape == (119,) and v.shape == (239, 119)
        assert np.array_equal(e < 50, e0 < 50) and np.all(np.diff(e) > 0)
        g = v.T @ s @ v
        off = g - np.diag(np.diag(g))
        frame = chol.T @ v
        projector = frame @ frame.T
        overlap = abs(np.sum(v0 * (s @ v), axis=0))
        shift_gap = abs(e-e0)/gaps
        ordinary = h @ v - (s @ v)*e
        wide = np.asarray(hd @ v.astype(np.longdouble)
                          - (sd @ v.astype(np.longdouble))*e.astype(np.longdouble), dtype=float)
        item = {'full_Gram_error': float(np.max(abs(g-np.eye(119)))),
                'diagonal_Gram_error': float(np.max(abs(np.diag(g)-1))),
                'offdiagonal_Gram_error': float(np.max(abs(off))),
                'projector_idempotence': float(np.max(abs(projector@projector-projector))),
                'minimum_original_column_overlap': float(overlap.min()),
                'maximum_energy_shift_vs_gap': float(shift_gap.max())}
        for kind, residual in [('ordinary', ordinary), ('wide_contracted', wide)]:
            dual = la.solve(s, residual, assume_a='pos', check_finite=True)
            norms = np.sqrt(np.maximum(0, np.sum(residual*dual, axis=0)))
            white = la.solve_triangular(chol, residual, lower=True)
            inside = q @ (q.T @ white)
            outside = white-inside
            innorm, outnorm = np.linalg.norm(inside, axis=0), np.linalg.norm(outside, axis=0)
            arrays[name+'_'+kind+'_full_Sdual'] = norms
            arrays[name+'_'+kind+'_in_span_Sdual'] = innorm
            arrays[name+'_'+kind+'_out_span_Sdual'] = outnorm
            worst = int(selected[np.argmax(norms[selected])])
            item[kind] = {'complete_band_max_Sdual_Ha': float(norms[selected].max()),
                          'full_branch_max_Sdual_Ha': float(norms.max()),
                          'worst_original_column': worst,
                          'worst_original_energy_Ha': float(e0[worst]),
                          'worst_in_span_Sdual_Ha': float(innorm[worst]),
                          'worst_out_span_Sdual_Ha': float(outnorm[worst]),
                          'complete_band_max_in_span_Sdual_Ha': float(innorm[selected].max()),
                          'complete_band_max_out_span_Sdual_Ha': float(outnorm[selected].max()),
                          'positive_solve_vs_Cholesky_norm_max_difference': float(np.max(abs(norms-np.linalg.norm(white, axis=0)))),
                          'original_complete_band_gate_PASS': bool(norms[selected].max() <= 1e-8)}
        item['all_original_gates_PASS'] = bool(item['full_Gram_error'] <= 1e-10
               and item['projector_idempotence'] <= 1e-10
               and overlap.min() >= .99 and shift_gap.max() <= .25
               and item['ordinary']['original_complete_band_gate_PASS'])
        out['routes'][name] = item

    measure('original', e0, v0)
    measure('independent_refinement', e1, v1)
    for route in ['ordinary_projection', 'wide_projection']:
        if route == 'ordinary_projection':
            g = v0.T @ (s @ v0)
            k = v0.T @ (h @ v0)
        else:
            g = np.asarray(vd.T @ (sd @ vd), dtype=float)
            k = np.asarray(vd.T @ (hd @ vd), dtype=float)
        out[route+'_projected_H_asymmetry'] = float(np.max(abs(k-k.T)))
        kg = np.tril(k)+np.tril(k, -1).T
        gg = np.tril(g)+np.tril(g, -1).T
        lower = la.cholesky(gg, lower=True)
        invl = la.solve_triangular(lower, np.eye(119), lower=True)
        whitened = invl @ kg @ invl.T
        values, u = la.eigh(whitened, lower=True, check_finite=True)
        out['eigh_calls'] += 1
        coordinate = la.solve_triangular(lower.T, u, lower=False)
        lifted = v0 @ coordinate
        arrays[route+'_coordinate'] = coordinate
        arrays[route+'_Ritz_E'] = values
        measure(route+'_Ritz_E', values, lifted)
        lifted_wide = lifted.astype(np.longdouble)
        rayleigh = np.asarray(np.sum(lifted_wide*(hd@lifted_wide), axis=0)
                              / np.sum(lifted_wide*(sd@lifted_wide), axis=0), dtype=float)
        arrays[route+'_Rayleigh_E'] = rayleigh
        measure(route+'_Rayleigh_E', rayleigh, lifted)
    assert out['eigh_calls'] == 2
    assert [sha(p) for p in INPUTS] == expected_hashes
    np.savez_compressed(HERE/'diagnostic_arrays.npz', **arrays)
    out['arrays_sha256'] = sha(HERE/'diagnostic_arrays.npz')
    out['elapsed_seconds_uncontrolled'] = time.monotonic()-start
    out['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (HERE/'result.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({key: out[key] for key in ('status', 'eigh_calls', 'elapsed_seconds_uncontrolled', 'peak_RSS_MiB')}))
    for name, item in out['routes'].items():
        print(name, json.dumps(item))


if __name__ == '__main__':
    main()
