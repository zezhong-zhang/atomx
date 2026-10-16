"""Read-only verification of once-only outputs; never rebuilds H or solves CI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/data/code/atomx')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'actual'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    c, r = read(HERE / 'contract.json'), read(OUT / 'result.json')
    seal, manifest = read(HERE / 'root_execution_seal.json'), read(HERE / 'current_source_identity.json')
    checks, values, bound = [], {}, {}

    def check(name, value, limit=None):
        passed = bool(value) if limit is None else bool(value <= limit)
        checks.append(dict(name=name, observed=value, limit=limit, PASS=passed))
        if not passed:
            raise RuntimeError('independent result gate failed: ' + name)

    def bind(path):
        bound[str(path.relative_to(ROOT))] = sha(path)

    try:
        check('finished_PASS', r['status'] == 'PASS_current_cached_streamed_full_basis_Davidson_fixed_R0')
        check('exact_driver_seal', seal['driver_sha256'] == sha(HERE / 'run.py'))
        check('exact_contract_seal', seal['contract_sha256'] == sha(HERE / 'contract.json'))
        check('same_current_source_manifest', r['source_map_sha256'] == c['source_map_sha256'] == manifest['source_map_sha256'])
        check('same_resource_seal', seal['resources'] == c['resources'])
        check('all_reported_gates', all(row['PASS'] and
              (bool(row['observed']) if row['limit'] is None else row['observed'] <= row['limit'])
              for row in r['checks']))
        check('four_modes', set(r['points']) == {'A_cached', 'A_streamed', 'B_cached', 'B_streamed'})
        check('no_accuracy_or_tracker_claim', not any(r[k] for k in (
            'experimental_accuracy_qualified', 'larger_4f12_calculated',
            'public_iterative_owner_or_tracker_qualified', 'controlled_timing')))
        for label, exact in dict(H=4, Davidson=20, H_projection=20, angular_table=10,
                                 streamed_numeric=10, pair_kernel=20, angular_nullspace=43,
                                 NPZ_load=5, NPZ_save=4).items():
            check('exact_calls/' + label, r['counts'][label] == exact)
        for label, maximum in c['resources']['calls'].items():
            check('entry_cap/' + label, r['attempted'].get(label, 0), maximum)
        for name, source in r['actual_imported_sources'].items():
            check('import_hash/' + name,
                  source['sha256'] == manifest['all_atomx_python_sha256'][source['path']])
        for path, expected in manifest['all_atomx_python_sha256'].items():
            if sha(ROOT / path) != expected:
                raise RuntimeError('current source changed after execution: ' + path)
        check('unchanged_current_source_files', True)
        for name in ('run.py', 'contract.json', 'root_execution_seal.json', 'current_source_identity.json'):
            bind(HERE / name)
        bind(OUT / 'result.json')
        for role, spec in c['inputs'].items():
            path = ROOT / spec['path']
            check('unchanged_input/' + role, sha(path) == spec['sha256'])
        if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
            raise RuntimeError('output-array verification requires single-thread controls')
        import numpy as np

        maps = read(ROOT / c['inputs']['basis_maps']['path'])
        numeric = {}
        for arm, model, n in (('A', 'baseline', 1595), ('B', 'expanded', 2574)):
            oldpath = ROOT / c['inputs'][arm + '_R0']['path']
            with np.load(oldpath, allow_pickle=False) as archive:
                old = {k: archive[k] for k in ('energies', 'mixing', 'H', 'keys', 'P', 'Q')}
            signatures = [json.loads(v) for v in maps[model]['selected_signatures']]
            sectors = {tuple(v[-2:]) for v in signatures}
            result_modes = {}
            for mode in ('cached', 'streamed'):
                tag = arm + '_' + mode
                path = OUT / (tag + '_selected.npz')
                bind(path)
                bind(OUT / (tag + '.json'))
                with np.load(path, allow_pickle=False) as archive:
                    new = {key: archive[key] for key in archive.files}
                check(tag + '/shapes', new['mixing'].shape == (n, 10)
                      and new['energies'].shape == (10,) and new['levels'].shape == (10, 3))
                check(tag + '/finite', all(np.isfinite(v).all() for v in new.values()))
                check(tag + '/all_sectors_roots', set(map(tuple, new['levels'])) ==
                      {(*sector, root) for sector in sectors for root in (0, 1)})
                check(tag + '/unaltered_radials', all(np.array_equal(new[k], old[k]) for k in ('keys', 'P', 'Q')))
                gram = float(np.max(abs(new['mixing'].T @ new['mixing'] - np.eye(10))))
                check(tag + '/full_saved_Gram', gram, c['gates']['selected_Gram'])
                residual = float(np.max(np.linalg.norm(
                    old['H'] @ new['mixing'] - new['mixing'] * new['energies'], axis=0)))
                check(tag + '/saved_selected_dense_oracle_L2', residual, 1e-8)
                delta, wave, projector, leakage = 0., 0., 0., 0.
                for index, (j2, parity, root) in enumerate(new['levels']):
                    rows = np.array([i for i, sig in enumerate(signatures) if tuple(sig[-2:]) == (j2, parity)])
                    probability = np.sum(old['mixing'][rows] ** 2, axis=0)
                    cols = np.flatnonzero(probability > .5)
                    check(tag + '/complete_old_sector/' + str((int(j2), int(parity), int(root))), len(cols) == len(rows))
                    cols = cols[np.argsort(old['energies'][cols])]
                    target = old['mixing'][:, cols[root]]
                    actual = new['mixing'][:, index]
                    delta = max(delta, float(abs(new['energies'][index] - old['energies'][cols[root]])))
                    wave = max(wave, float(np.max(abs(actual - target))))
                    projector = max(projector, float(np.max(abs(
                        np.outer(actual[rows], actual[rows]) - np.outer(target[rows], target[rows])))))
                    outside = np.ones(n, dtype=bool)
                    outside[rows] = False
                    leakage = max(leakage, float(np.max(abs(actual[outside]))))
                check(tag + '/saved_energy', delta, c['gates']['selected_energy_max_abs_Ha'])
                check(tag + '/saved_signed_wave', wave, c['gates']['selected_signed_wave_max_abs'])
                check(tag + '/saved_projector', projector, c['gates']['selected_projector_max_abs'])
                check(tag + '/saved_cross_sector_exact_zero', leakage, 0.)
                numeric[tag] = dict(Gram=gram, dense_oracle_residual_L2_Ha=residual,
                                    energy_difference_Ha=delta, signed_wave_difference=wave,
                                    projector_difference=projector)
                result_modes[mode] = new
            check(arm + '/cached_streamed_same_levels', np.array_equal(result_modes['cached']['levels'], result_modes['streamed']['levels']))
            check(arm + '/cached_streamed_saved_signed_waves', float(np.max(abs(result_modes['cached']['mixing'] - result_modes['streamed']['mixing']))), 1e-8)
            check(arm + '/cached_streamed_saved_energies', float(np.max(abs(result_modes['cached']['energies'] - result_modes['streamed']['energies']))), 1e-8)
        values.update(saved_outputs=numeric, reported_checks=len(r['checks']),
                      report_elapsed_seconds=r['elapsed_seconds'], report_peak_RSS_MiB=r['peak_RSS_MiB'])
        result = dict(status='PASS_independent_output_review', checks=checks, values=values,
                      bound_files=bound, no_H_rebuild=True, no_CI_solve=True, no_numerical_retry=True)
    except Exception as error:
        result = dict(status='FAILED_preserved_independent_output_review', error=repr(error),
                      checks=checks, values=values, bound_files=bound, no_numerical_retry=True)
    (HERE / 'result_review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], checks=len(checks), passed=sum(v['PASS'] for v in checks))))
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
