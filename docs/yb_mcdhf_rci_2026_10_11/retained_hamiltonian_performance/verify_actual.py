"""Independent saved-array and timing-metadata audit; no Atomx/science calls."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def targets(groups):
    return sorted((tuple(tuple(level) for level in group['reference_levels']),
                   tuple(tuple(level) for level in group['selected_levels']), float(group['per_level_weight']))
                  for group in groups)


def main():
    started = time.monotonic()
    contract = load(HERE / 'verification_contract.json')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3 or any(contract['resources'][key] != 0 for key in
                              ('H_builds', 'CI_solves', 'gradients', 'optimizer_calls', 'new_performance_measurements')):
        raise ValueError('Saved-array audit contract changed')
    if contract['verifier_sha256'] != sha(__file__):
        raise ValueError('Prospective saved-array verifier identity changed')
    os.sched_setaffinity(0, {affinity[-2]})
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    cap = int(contract['resources']['address_space_cap_GiB'] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Independent saved60s cap')))
    signal.alarm(contract['resources']['wall_cap_s'])
    import numpy as np

    checks = []

    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
        if not passed:
            raise ValueError('Saved-array audit failed: ' + name)

    def close(name, value, limit=0.):
        value = float(value)
        check(name, math.isfinite(value) and value <= limit, error=value, limit=limit)

    for name, expected in contract['inputs_sha256'].items():
        check('immutable:' + name, sha(ROOT / name) == expected)
    prospective = load(HERE / 'contract.json')
    approval = load(HERE / 'execution_contract.json')
    result = load(HERE / 'actual/result.json')
    qualified_path = ROOT / prospective['prerequisites']['qualification_result']['path']
    qualification = load(qualified_path)
    qualified_rdm = load(qualified_path.parent / 'signed_public_RDM.json')
    qualified = np.load(qualified_path.parent / 'arrays.npz', allow_pickle=False)
    point_path = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/checkpoints/accepted_0160.npz'
    point = np.load(point_path, allow_pickle=False)
    layout = load(ROOT / 'docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json')
    basis = load(ROOT / 'docs/yb_mcdhf_rci_2026_10_11/balanced_cv_replay/basis_identity.json')['CSFs']
    groups = {}
    for row in basis:
        groups.setdefault((row['twoJ'], row['parity']), []).append(row['index'])
    check('complete signed531/16-sector identity', len(basis) == 531 and len(groups) == 16)
    source = load(ROOT / prospective['candidate_source_identity_path'])
    source_dir = ROOT / prospective['candidate_frozen_source_path']
    actual_source = {'src/' + str(p.relative_to(source_dir)): sha(p)
                     for p in sorted((source_dir / 'atomx').rglob('*.py'))}
    check('actual frozen354 source map', actual_source == source['all_atomx_python_sha256']
          and len(actual_source) == 354 and canonical(actual_source) == contract['source_map_sha256'])
    check('actual result direct execution/source/qualification chain', result['contract_sha256'] == sha(HERE / 'contract.json')
          and result['execution_contract_sha256'] == sha(HERE / 'execution_contract.json')
          and approval['driver_sha256'] == sha(HERE / 'run.py')
          and result['candidate_source_map_sha256'] == contract['source_map_sha256']
          and result['prerequisite_qualification_sha256'] == sha(qualified_path))
    expected_root_flags = ('execution_authorized_by_root', 'paired_chain_performance_authorized',
        'complete531_qualification_required', 'exclusive_resource_window_authorized', 'root_physics_diagnostic_completed')
    check('exact actual root flags', all(approval[key] is True for key in expected_root_flags))
    symmetry = [(row['twoJ'], row['parity']) for row in basis]
    cross = np.array([[a != b for b in symmetry] for a in symmetry])
    gates = contract['gates']
    payloads, records, hosts, counts, clusters_checked = {}, [], [], dict(H=0, CI=0, gradient=0), 0
    paths_by_kind = {}
    qualification_columns = {}
    reference_mix = qualified['ordinary_mixing']
    for key, row_indices in groups.items():
        probabilities = np.sum(reference_mix[row_indices]**2, axis=0)
        cols = np.flatnonzero(probabilities > .5)
        cols = cols[np.argsort(qualified['ordinary_energies_Ha'][cols], kind='stable')]
        check('reference sector root dimension:' + repr(key), len(cols) == len(row_indices))
        qualification_columns[key] = cols

    for child in result['children']:
        kind = child['kind']
        worker = load(HERE / 'actual' / kind / 'result.json')
        paths_by_kind[kind] = worker
        check(kind + '/exit0/caps/complete/science', child['exit_code'] == 0 and worker['actual_runtime_caps_passed']
              and worker['physics_passed'] and worker['complete_declared_cases'] and worker['controlled'])
        expected_count = prospective['resources']['worker_caps'][kind]
        check(kind + '/full started/completed counters', worker['counters'] == {
            'started': expected_count, 'completed': expected_count})
        for name in counts:
            counts[name] += worker['counters']['completed'][name]
        check(kind + '/four actual helper constructors', worker['setup_functional_construction']['calls'] ==
              worker['setup_functional_construction']['completed'] == 4)
        check(kind + '/finite setup/RSS', 0 < worker['setup_seconds'] < worker['worker_seconds']
              and worker['peak_RSS_MiB'] < prospective['resources']['address_space_cap_GiB'] * 1024)
        hosts.extend((child['before_host'], child['after_host']))
        for item in worker['records']:
            records.append(item)
            hosts.extend((item['before_host'], item['after_host']))
            path = HERE / item['path']
            header = load(path.with_suffix('.json'))
            prefix = item['path']
            check(prefix + '/own array+manifest SHA', sha(path) == item['arrays_sha256'] == header['arrays_sha256']
                  and sha(path.with_suffix('.json')) == item['manifest_sha256'])
            check(prefix + '/manifest gates retained', all(row['passed'] is True for row in header['checks'])
                  and header['source_map_sha256'] == contract['source_map_sha256']
                  and header['original_metric_sha256'] == prospective['fixed_conditions']['original_metric_sha256']
                  and header['qualification_result_sha256'] == sha(qualified_path))
            check(prefix + '/same source-only objective evolution', header['epoch_identity']['physical_payload_except_source_unchanged'] is True
                  and header['epoch_identity']['original_actual_objective_sha256'] == prospective['fixed_conditions']['parent_actual_objective_sha256']
                  and header['epoch_identity']['candidate_actual_objective_sha256'] == canonical(
                      header['epoch_identity']['candidate_actual_objective_payload']))
            archive = np.load(path, allow_pickle=False)
            a = {key: archive[key] for key in archive.files}
            archive.close()
            payloads[item['path']] = a
            check(prefix + '/complete finite archive/schema', set(a) == set(header['complete_array_keys'])
                  and all(np.all(np.isfinite(v)) for v in a.values()))
            check(prefix + '/full dimensions', a['matrix_Ha'].shape == a['mixing'].shape == (531, 531)
                  and a['energies_Ha'].shape == a['state_weights'].shape == (531,)
                  and a['radial_gradient'].shape == (24, 246) and a['x'].shape == (2856,)
                  and a['original_projected_gradient'].shape == a['original_preconditioned_gradient'].shape == (2856,))
            check(prefix + '/actual complete float64 arrays', all(a[name].dtype == np.dtype('float64') for name in
                ('matrix_Ha', 'mixing', 'energies_Ha', 'radial_gradient', 'original_projected_gradient',
                 'original_preconditioned_gradient', 'state_weights', 'signed_RDM_values')))
            for name in a:
                if name in point.files and name not in ('matrix_Ha', 'energies_Ha', 'mixing',
                                                       'original_projected_gradient', 'original_preconditioned_gradient'):
                    check(prefix + '/same original geometry:' + name, np.array_equal(a[name], point[name]))
            h, v, energy = a['matrix_Ha'], a['mixing'], a['energies_Ha']
            check(prefix + '/actual H bitwise full qualified and oldpoint', np.array_equal(h, point['matrix_Ha'])
                  and np.array_equal(h, qualified['ordinary_Ha']))
            close(prefix + '/H hermiticity', np.max(abs(h - h.T)), gates['matrix_hermiticity_Ha'])
            close(prefix + '/complete cross-sector H zero', np.max(abs(h[cross])), gates['cross_symmetry_zero_Ha'])
            hv = h @ v
            rayleigh = np.sum(v * hv, axis=0) / np.sum(v**2, axis=0)
            close(prefix + '/independent rootwise Rayleigh', np.max(abs(rayleigh - energy)), gates['CI_residual_Ha'])
            close(prefix + '/rootwise centered eigenpair', np.max(np.linalg.norm(hv - v * rayleigh, axis=0)), gates['CI_residual_Ha'])
            close(prefix + '/CI Gram', np.max(abs(v.T @ v - np.eye(531))), gates['CI_orthogonality'])
            close(prefix + '/full531 saved energies replay', np.max(abs(energy - point['energies_Ha'])),
                  gates['all_roots_vs_frozen320_max_Ha'])
            columns = {}
            for key, row_indices in groups.items():
                probability = np.sum(v[row_indices]**2, axis=0)
                cols = np.flatnonzero(probability > .5)
                cols = cols[np.argsort(energy[cols], kind='stable')]
                columns[key] = cols
                check(prefix + '/complete sector dimension:' + repr(key), len(cols) == len(row_indices))
                close(prefix + '/complete sector support:' + repr(key), np.max(abs(probability[cols] - 1)), gates['CI_orthogonality'])
                anchor = qualification_columns[key]
                lo = 0
                while lo < len(cols):
                    hi = lo + 1
                    while hi < len(cols) and abs(energy[cols[hi]] - energy[cols[hi-1]]) <= gates['cluster_gap_Ha']:
                        hi += 1
                    left, right = v[:, cols[lo:hi]], reference_mix[:, anchor[lo:hi]]
                    overlap = right.T @ left
                    # Signed overlap and two residual span actions, without
                    # fixed-column phases or constructing N^2 projector arrays.
                    close(prefix + '/signed root span:' + repr((key, lo, hi)), max(np.linalg.norm(
                        left - right @ overlap), np.linalg.norm(right - left @ overlap.T)), gates['signed_cluster_span_max'])
                    clusters_checked += 1
                    lo = hi
            check(prefix + '/original targets metadata', targets(header['physical_tracking']) == targets(qualification['physical_tracking'])
                  and len(header['physical_tracking']) == 5
                  and all(min(group['principal_overlap_squared']) > .5 for group in header['physical_tracking']))
            weights = np.zeros(531)
            for group in header['physical_tracking']:
                for twoj, parity, root in group['selected_levels']:
                    weights[columns[(twoj, parity)][root]] = group['per_level_weight']
            check(prefix + '/independent tracked five weights', np.array_equal(weights, a['state_weights'])
                  and np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
            rdm_keys = [ast.literal_eval(text) for text in header['signed_RDM_keys']]
            check(prefix + '/complete signed density keys values order', json.loads(json.dumps(rdm_keys)) == [r['key'] for r in qualified_rdm['terms']]
                  and np.array_equal(a['signed_RDM_values'], np.array([r['value'] for r in qualified_rdm['terms']])))
            check(prefix + '/signed RDM negative coverage', len(rdm_keys) == qualified_rdm['count'] == 2173
                  and int(np.sum(a['signed_RDM_values'] < 0)) == qualified_rdm['negative_terms'] == 1219)
            positions = {tuple(map(int, key)): index for index, key in enumerate(a['orbital_keys'])}
            gradient_positions = {tuple(map(int, key)): index for index, key in enumerate(a['radial_gradient_keys'])}
            check(prefix + '/complete all24 gradient keys', set(gradient_positions) == set(positions) and len(positions) == 24)
            native_radial = qualified['ordinary_radial_gradient']
            native_positions = {tuple(map(int, key)): index for index, key in enumerate(qualified['radial_gradient_keys'])}
            check(prefix + '/all24 radial gradient exact qualification', all(np.array_equal(
                a['radial_gradient'][gradient_positions[key]], native_radial[native_positions[key]]) for key in positions))
            independent_g, independent_pg, metric_digest = np.zeros(2856), np.zeros(2856), hashlib.sha256()
            for i, row in enumerate(layout):
                keys, shape, bounds = [tuple(k) for k in row['keys']], tuple(row['shape']), slice(*row['slice'])
                p, s = a[f'projector_{i}'], qualified[f'actual_S_{i}']
                y = a['x'][bounds].reshape(shape)
                physical = np.column_stack([np.r_[a['P'][positions[k]], a['Q'][positions[k]]] for k in keys])
                metric_digest.update(repr((keys, shape)).encode())
                metric_digest.update(p.tobytes())
                check(prefix + f'/original projector x waves:{i}', np.array_equal(p @ y, physical))
                close(prefix + f'/radial Gram:{i}', np.max(abs(physical.T @ s @ physical - np.eye(shape[1]))), gates['radial_Gram_error'])
                outside = physical - p @ (p.T @ s @ physical)
                close(prefix + f'/no-pair actual original span:{i}', np.sqrt(max(0., float(np.max(np.sum(
                    outside * (s @ outside), axis=0))))), gates['projector_leakage_norm'])
                # Independent per-column radial projection and tangent action.
                projected = np.column_stack([p.T @ a['radial_gradient'][gradient_positions[k]] for k in keys])
                block = projected - y @ (.5 * (y.T @ projected + projected.T @ y))
                preconditioned = np.column_stack([a[f'inverse_blocks_{i}'][j] @ block[:, j] for j in range(shape[1])])
                preconditioned -= y @ (.5 * (y.T @ preconditioned + preconditioned.T @ y))
                independent_g[bounds], independent_pg[bounds] = block.ravel(), preconditioned.ravel()
            metric_digest.update(a['inverse_diagonal'].tobytes())
            for i in range(len(layout)):
                for inverse in a[f'inverse_blocks_{i}']:
                    metric_digest.update(inverse.tobytes())
            check(prefix + '/original metric bytes exact', metric_digest.hexdigest() == header['original_metric_sha256'])
            close(prefix + '/independent radial_to2856g', np.max(abs(independent_g - a['original_projected_gradient'])),
                  gates['radial_to_projected_g_archive_max'])
            close(prefix + '/independent original2856pg', np.max(abs(independent_pg - a['original_preconditioned_gradient'])),
                  gates['projected_to_original_pg_archive_max'])
            close(prefix + '/complete frozen320g replay', np.max(abs(a['original_projected_gradient'] -
                  point['original_projected_gradient'])), 1e-9)
            close(prefix + '/complete frozen320pg replay', np.max(abs(a['original_preconditioned_gradient'] -
                  point['original_preconditioned_gradient'])), 1e-10)
            residual, indicator = float(np.linalg.norm(independent_pg)), .5 * abs(float(independent_g @ independent_pg))
            close(prefix + '/saved residual agrees independent metric', abs(residual - header['original_residual']), 1e-10)
            close(prefix + '/saved indicator agrees independent metric', abs(indicator - header['original_indicator_Ha']), 1e-12)
            check(prefix + '/original stationarity FAIL preserved', residual > 1e-7 and indicator < 1e-8
                  and header['original_stationarity_passed'] is False)
            owned_digest = hashlib.sha256(repr((h.shape, h.dtype.str)).encode() + h.tobytes(order='C')).hexdigest()
            check(prefix + '/saved physical H ownership digest', owned_digest == header['owner_matrix_sha256'])
            check(prefix + '/owner input binding appropriate', header['owner_input_sha256'] == qualification['owner_input_sha256']
                  if item['mode'] == 'retained' else header['owner_input_sha256'] is None)
            check(prefix + '/exact public calls', item['actual_calls'] == dict(H=1 if item['mode'] == 'retained' else 2, CI=1, gradient=1))
            stage_keys = ('identity_seconds', 'energy_trial_seconds', 'before_gradient_full_eigenpair_validation_seconds',
                'accepted_gradient_seconds', 'callback_tracking_and_H_seconds', 'validation_seconds', 'full_archive_write_fsync_readback_seconds')
            check(prefix + '/timing scope includes all stages', all(math.isfinite(item[k]) and item[k] >= 0 for k in stage_keys)
                  and sum(item[k] for k in stage_keys) <= item['whole_chain_seconds'])
            check(prefix + '/binding subscope not double added', item['binding_subscope_overlaps_stages'] is True
                  and item['binding_seconds_subscope'] <= item['whole_chain_seconds'])
    check('all fourteen complete chains with exact calls', len(records) == 14 and counts == result['actual_calls'] == dict(H=21, CI=14, gradient=14))
    check('cold modes and explicit warmups exactly declared', [(r['category'], r['mode']) for r in records[:4]] ==
          [('cold', 'ordinary'), ('cold', 'retained'), ('warmup', 'ordinary'), ('warmup', 'retained')])
    check('five alternating order pairs retained', [(r['pair'], r['mode']) for r in records[4:]] ==
          [(i, mode) for i in range(1, 6) for mode in (('ordinary', 'retained') if i % 2 else ('retained', 'ordinary'))])
    check('all34 host endpoints independent replay', len(hosts) == 34)
    for i, host in enumerate(hosts):
        denominator = len(host['available_cpus'])
        close(f'host:{i}/original CPU load ratio', abs(host['load1_per_available_cpu'] - host['load_average'][0] / denominator))
        check(f'host:{i}/actual admission arithmetic', host['admitted'] is True and denominator >= 3
              and host['available_cpus'] == approval['available_cpus'] and host['chosen_cpu'] == approval['chosen_cpu']
              and host['chosen_cpu'] in host['available_cpus']
              and host['host_available_GiB'] >= prospective['resources']['minimum_host_available_GiB']
              and host['load1_per_available_cpu'] <= prospective['resources']['max_load1_per_available_CPU'])
    pairs = []
    for i in range(1, 6):
        rows = {r['mode']: r for r in records if r['category'] == 'measured' and r['pair'] == i}
        left, right = payloads[rows['ordinary']['path']], payloads[rows['retained']['path']]
        for name in left:
            if name != 'mixing':
                check(f'pair{i}/complete same physics array:' + name, np.array_equal(left[name], right[name]))
        ratio = rows['ordinary']['whole_chain_seconds'] / rows['retained']['whole_chain_seconds']
        close(f'pair{i}/saved statistics ratio exact', abs(ratio - result['warm_pairs'][i-1]['ordinary_over_retained']))
        pairs.append(dict(pair=i, ordinary_seconds=rows['ordinary']['whole_chain_seconds'],
                          retained_seconds=rows['retained']['whole_chain_seconds'], ordinary_over_retained=ratio))
    ratios = [r['ordinary_over_retained'] for r in pairs]
    median = statistics.median(ratios)
    close('saved median exact', abs(median - result['median_warm_ratio']))
    check('saved ratio range exact', [min(ratios), max(ratios)] == result['warm_ratio_range'])
    decision = all(value > 1. for value in ratios) and median >= prospective['performance_decision']['minimum_median_ratio']
    check('prospective controlled chain performance gate exact', decision is True and result['controlled_warm_chain_speedup_supported'] is True
          and result['controlled_admission_passed'] is True and result['scientific_parity_passed'] is True and result['stopping_condition'] is None)
    check('actual overall elapsed/caps and root exit0', result['observed_panel_seconds'] < prospective['resources']['whole_panel_wall_cap_s']
          and load(HERE / 'root_execution.json')['exit_code'] == 0)
    check('no optimizer GPU parallel stationarity extrapolation', result['optimizer_calls'] == 0
          and result['GPU_speedup_claim'] is False and result['whole_optimizer_speedup_claim'] is False
          and result['original_stationarity_passed'] is False)
    for name, expected in contract['inputs_sha256'].items():
        check('unchanged after saved replay:' + name, sha(ROOT / name) == expected)
    check('no Atomx imports / new physical calls', not any(n == 'atomx' or n.startswith('atomx.') for n in sys.modules))
    check('saved audit resource cap', time.monotonic() - started < contract['resources']['wall_cap_s']
          and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1024 * 1024 and len(os.sched_getaffinity(0)) == 1)
    summary = dict(all_passed=True, check_count=len(checks), checks=checks, inputs_sha256=contract['inputs_sha256'],
        verifier_sha256=sha(__file__), verification_contract_sha256=sha(HERE / 'verification_contract.json'),
        actual_result_sha256=sha(HERE / 'actual/result.json'), actual_driver_sha256=sha(HERE / 'run.py'),
        prospective_contract_sha256=sha(HERE / 'contract.json'), execution_contract_sha256=sha(HERE / 'execution_contract.json'),
        source_map_sha256=contract['source_map_sha256'], full_archives_verified=14, signed_clusters_verified=clusters_checked,
        warm_pairs=pairs, median_warm_ratio=median, range_warm_ratios=[min(ratios), max(ratios)],
        median_ratio_implied_elapsed_reduction=1 - 1 / median,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, seconds=time.monotonic() - started,
        H_builds=0, CI_solves=0, gradients=0, optimizer_calls=0, new_performance_measurements=0,
        scope='Independent saved-array/statistics verification; existing controlled warm checked-chain claim only',
        whole_optimizer_speedup_claim=False, GPU_parallel_claim=False, original_stationarity_passed=False)
    (HERE / 'verification.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('checks', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
