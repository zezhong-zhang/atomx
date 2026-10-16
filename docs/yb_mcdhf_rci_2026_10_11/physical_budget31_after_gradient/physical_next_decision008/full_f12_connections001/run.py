"""One complete B+4f12 structural graph; prepare is integer/stdlib only.

Execution requires an exact root seal. No angular coefficients, CSF amplitudes,
projection, radial arrays, Hamiltonian, CI, eigensolver or NPZ are evaluated.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import resource
import signal
import struct
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
BASE = HERE.parent.parent
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
INPUTS = {
    'prior_actual': BASE / 'physical_next_decision007/full_f12_connections001/actual/result.json',
    'prior_contract': BASE / 'physical_next_decision007/full_f12_connections001/contract.json',
    'missing': BASE / 'physical_next_decision005/missing_4f12_budget.json',
    'angular_budget': BASE / 'physical_next_decision006/angular_budget.json',
    'pattern_helper': BASE / 'physical_next_decision006/angular_budget_count.py',
    'basis_maps': ROOT / 'docs/yb_mcdhf_rci_2026_10_11/eu_yb_comparison001/'
                  'core_core_discriminator_preparation001/execution_preparation002/'
                  'actual/complete_basis_and_configuration_maps.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


@lru_cache(None)
def shell_m(kappa, q):
    counts = {(0, 0): 1}
    for m in range(-(2*abs(kappa)-1), 2*abs(kappa), 2):
        updated = dict(counts)
        for (n, total), count in counts.items():
            if n < q:
                key = n+1, total+m
                updated[key] = updated.get(key, 0) + count
        counts = updated
    return tuple((m, number) for (n, m), number in counts.items() if n == q)


def pattern_m(pattern):
    counts = {0: 1}
    for _n, kappa, q in pattern:
        updated = Counter()
        for m, number in counts.items():
            for local, count in shell_m(kappa, q):
                updated[m+local] += number*count
        counts = updated
    return counts


def model_patterns():
    """Use the exact approved 82 NR rows and existing original B occupations."""
    old = read(INPUTS['missing'])
    saved = read(INPUTS['basis_maps'])
    rows = old['complete_missing_family']['complete_NR_occupancy_rows']
    if len(rows) != 82 or len(saved['expanded']['configurations']) != 100:
        raise RuntimeError('complete NR class or B model changed')
    node = next(n for n in ast.parse(INPUTS['pattern_helper'].read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'missing_patterns')
    namespace = {'itertools': itertools}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(INPUTS['pattern_helper']),
                 'exec'), namespace)
    missing = namespace['missing_patterns'](rows)
    if len(missing) != 1080:
        raise RuntimeError('complete missing jj occupation count changed')
    existing = {tuple(map(tuple, p)) for p in saved['expanded']['selected_occupations']}
    output, raw, csfs = [], 0, 0
    for pattern in sorted(existing | missing):
        if sum(q for _n, _k, q in pattern) != 69:
            raise RuntimeError('electron count changed')
        parity = (-1)**sum((k if k > 0 else -k-1)*q for _n, k, q in pattern)
        if parity != 1:
            continue
        counts = pattern_m(pattern)
        multiplicity = counts.get(1, 0)-counts.get(3, 0)
        if multiplicity <= 0:
            continue
        output.append(pattern)
        raw += counts.get(1, 0)
        csfs += multiplicity
    if (len(output), raw, csfs) != (738, 92715, 5717):
        raise RuntimeError('integer full model census mismatch')
    return tuple(output), raw, csfs


def source_check(c):
    if sha(HERE / 'run.py') != c['driver_sha256']:
        raise RuntimeError('driver changed')
    for relative, expected in c['sources'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('source changed: '+relative)
    for item in c['inputs'].values():
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('input changed: '+item['path'])


def prepare():
    start = time.perf_counter()
    patterns, raw, csfs = model_patterns()
    budget = read(INPUTS['angular_budget'])['models']['expanded_plus_complete_4f12'][
        '2J=1 parity=1']
    if (budget['raw_M_determinants'], budget['CSFs'], budget['connected_ordered_pairs_exact']) != (
            raw, csfs, 43720175):
        raise RuntimeError('approved sparse census mismatch')
    prior = read(INPUTS['prior_actual'])
    if not prior['success'] or prior['pairs'] != 43720175 or prior['ordered_pair_sha256'] != (
            '9a49475f11b27a0889a1d187c9a5a18a445db0e92510688362f0b131d4a290e4'):
        raise RuntimeError('closed prior graph control mismatch')
    if (prior['patterns_sha256'], prior['raw_determinants'], prior['CSFs_integer_native']) != (
            canonical_hash(patterns), raw, csfs):
        raise RuntimeError('prior graph physical pattern identity differs')
    k = budget['spectator_bucket_memberships_exact']
    v = budget['spectator_bucket_ordered_pair_visits_before_dedup_exact']
    lower_u = (k*k+v-1)//v
    bucket_floor = lower_u*(sys.getsizeof(1)+sys.getsizeof([])+24)+8*k
    c = {
        'status': 'SOURCE_ONLY_INTEGER_PREPARATION_NO_NATIVE_ENUMERATION',
        'hypothesis': 'Singleton-compressed spectator storage preserves every ordered '
                      'connection and exact prior007 pair digest for the same complete B+4f12 '
                      'model, under the unchanged4GiB cap and without angular coefficients.',
        'fixed': {'arm': 'B_plus_complete_4f12', 'electrons': 69, 'NR_configurations': 182,
                  'sector': [1, 1], 'jj_patterns': 738, 'raw_determinants': raw,
                  'CSFs_integer_only': csfs, 'frozen_spinors': 46,
                  'active_electrons': 23, 'triangular': False,
                  'patterns_sha256': canonical_hash(patterns),
                  'ordered_pairs': 43720175, 'ordered_pairs_by_replacements':
                  {'0': 92715, '1': 1206876, '2': 42420584},
                  'spectator_memberships': k, 'bucket_ordered_pair_visits': v,
                  'original_unique_buckets': prior['spectator_index']['unique_buckets'],
                  'prior_ordered_pair_sha256': prior['ordered_pair_sha256'],
                  'prior_determinant_sha256': prior['determinant_sha256']},
        'resources': {'CPU': 28, 'threads': 1, 'AS_bytes': 4*1024**3,
                      'alarm_seconds': 300, 'outer_seconds': 310,
                      'host_load1_max': 16, 'host_MemAvailable_GiB_min': 10},
        'memory_before_admission': {
            'prior007_peak_RSS_KiB': prior['RSS_highwater_KiB'],
            'prior007_owned_bucket_bytes': prior['spectator_index']['map_key_list_owned_bytes'],
            'new_construction_scope': 'First occurrences share existing row integer objects; '
                                      'promote only repeated keys to lists. Filter singleton '
                                      'keys before streaming; dictionary rebuild old+new overlap '
                                      'must remain in whole-process peak RSS. Prior bucket floor '
                                      'below describes old fullmap, not new retained-map floor.',
            'bucket_count_integer_lower_bound': lower_u,
            'bucket_count_integer_upper_bound': k,
            'membership_reference_payload_bytes': 8*k,
            'conditional_min_CPython_bucket_floor_bytes': bucket_floor,
            'floor_scope': 'Cauchy U>=ceil(K²/V); minimum positive integer and empty list '
                           'headers plus24B/hash-entry and8B/reference. Excludes key bitwidth, '
                           'hash capacity/indices, list overallocation, shared row ints, '
                           'native import and determinant/cached layout storage.',
            'peak_estimate': 'Unknown: actual unique buckets and Python allocator capacity '
                             'can exceed this lower bound.4GiB is an explicit bounded trial '
                             'cap, not a proven fit or a peak prediction.2GiB not assumed.',
            'output_memory': 'Native integer-first map during construction, promoted-only '
                             'map after pruning, one row set/sorted list, <=64KiB hash '
                             'bytearray and aggregate counters. No emitted-pair list.',
        },
        'gates': {'all_census_values_and_patterns_hash_exact': True,
                  'rows_cols_strict_lexicographic_order': True,
                  'every_row_has_diagonal': True, 'all_distances_0_2_4': True,
                  'pair_count_and_replacement_counts_exact': True,
                  'whole_ordered_pair_sha256_equals_prior007': True,
                  'retained_buckets_minimum_length_2': True,
                  'retained_plus_inferred_singletons_reconstruct_original_U_K_V': True},
        'timing_scope': 'One cold native graph enumeration with scalar distance checks '
                        'and SHA256 streaming.007 and008 are separate single-run graph '
                        'diagnostics; no controlled fullgraph timing or speedup claim.',
        'forbidden': ['angular coefficient table', 'CSF vectors', 'projection', 'H', 'CI',
                      'eigensolver', 'radial solve', 'NPZ', 'old global connection graph'],
        'stop': 'First source,input,census,order,resource or timeout failure stops. '
                'Keep partial report; no retry, changed cap/tolerance/model or shortened graph.',
        'inputs': {name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
                   for name, path in INPUTS.items()},
        'sources': {str(path.relative_to(ROOT)): sha(path)
                    for path in sorted((ROOT / 'src/atomx').rglob('*.py'))},
        'driver_sha256': sha(HERE / 'run.py'),
    }
    write(HERE / 'contract.json', c)
    source_check(c)
    print(json.dumps({'status': c['status'], 'driver_sha256': c['driver_sha256'],
                      'contract_sha256': sha(HERE / 'contract.json'),
                      'seconds': time.perf_counter()-start}), flush=True)


def host():
    available = next(int(line.split()[1])*1024 for line in
                     Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return {'load1': os.getloadavg()[0], 'MemAvailable_bytes': available}


def execute(seal_path):
    c = read(HERE / 'contract.json')
    seal = read(seal_path)
    if seal != {'execution_authorized_by_root': True, 'once_only': True,
                'driver_sha256': sha(HERE / 'run.py'),
                'contract_sha256': sha(HERE / 'contract.json'), 'resources': c['resources']}:
        raise RuntimeError('exact root seal mismatch')
    source_check(c)
    cap = c['resources']
    if set(os.sched_getaffinity(0)) != {cap['CPU']} or any(
            os.environ.get(name) != '1' for name in THREADS):
        raise RuntimeError('CPU/thread environment mismatch')
    actual = HERE / 'actual'
    actual.mkdir()
    report = {'success': False, 'stage': 'admission', 'host_before': host(), 'pairs': 0}
    write(actual / 'result.json', report)
    resource.setrlimit(resource.RLIMIT_AS, (cap['AS_bytes'], cap['AS_bytes']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError('full graph300s hard stop')))
    signal.alarm(cap['alarm_seconds'])
    start = time.perf_counter()
    iterator = None
    try:
        admission = report['host_before']
        if admission['load1'] > cap['host_load1_max'] or admission[
                'MemAvailable_bytes'] < cap['host_MemAvailable_GiB_min']*1024**3:
            raise RuntimeError('host load/memory admission guard failed')
        sys.path.insert(0, str(ROOT / 'src'))
        from atomx.angular import csf_hamiltonian as angular

        def forbidden(*_args, **_kwargs):
            raise RuntimeError('forbidden amplitude/table/H path entered')

        for name in ('_coefficient_table', '_block_structure', 'block_terms',
                     '_block_csf_sparse_matrix', '_block_csf_matrix', 'csf_matrix',
                     'csf_block_matrix', 'highest_weight_states', 'exact_block_matrix'):
            setattr(angular, name, forbidden)
        patterns, raw, csfs = model_patterns()
        if canonical_hash(patterns) != c['fixed']['patterns_sha256']:
            raise RuntimeError('pattern identity changed')
        offsets = dict(angular._union_offsets(patterns))
        dets, raw_native, csfs_native = [], 0, 0
        for pattern in patterns:
            counts = dict(angular.pattern_m_counts(pattern))
            raw_native += counts.get(1, 0)
            csfs_native += counts.get(1, 0)-counts.get(3, 0)
            lay = angular.pattern_layout(pattern)
            dets.extend(angular._remap(mask, lay, offsets) for mask in lay.by_m2[1])
        if (len(dets), raw_native, csfs_native) != (raw, raw, csfs):
            raise RuntimeError('native magnetic determinant/CSF census mismatch')
        if len(set(dets)) != raw:
            raise RuntimeError('duplicate determinant across complete patterns')
        frozen = dets[0]
        for mask in dets:
            frozen &= mask
        if frozen.bit_count() != 46 or any(mask.bit_count() != 69 for mask in dets):
            raise RuntimeError('electron/common-core identity changed')
        if canonical_hash(dets) != c['fixed']['prior_determinant_sha256']:
            raise RuntimeError('native full determinant order differs from prior007')
        report.update(stage='native_graph', raw_determinants=raw,
                      CSFs_integer_native=csfs_native, patterns_sha256=canonical_hash(patterns),
                      determinant_sha256=canonical_hash(dets),
                      enumeration_seconds=time.perf_counter()-start,
                      RSS_before_graph_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(actual / 'result.json', report)
        graph_start = time.perf_counter()
        iterator = angular._connected_pairs(dets, triangular=False)
        first = next(iterator)
        if first != (0, 0):
            raise RuntimeError('first graph coordinate differs')
        # Read only scalar ownership/census metadata of the live native map.
        buckets = iterator.gi_frame.f_locals['spectators']
        memberships = visits = owned_bytes = largest = 0
        smallest = raw+1
        for key, indices in buckets.items():
            if not isinstance(indices, list) or len(indices) < 2:
                raise RuntimeError('singleton/unpromoted bucket retained')
            number = len(indices)
            smallest = min(smallest, number)
            memberships += number
            visits += number*number
            largest = max(largest, number)
            owned_bytes += sys.getsizeof(key)+sys.getsizeof(indices)
        owned_bytes += sys.getsizeof(buckets)
        inferred_singletons = c['fixed']['original_unique_buckets']-len(buckets)
        if inferred_singletons < 0:
            raise RuntimeError('retained map exceeds original unique-bucket control')
        report['spectator_index'] = {
            'retained_unique_buckets': len(buckets), 'retained_memberships': memberships,
            'retained_ordered_visits': visits, 'smallest_bucket': smallest,
            'largest_bucket': largest,
            'implicit_singletons_inferred_from_prior_U': inferred_singletons,
            'inference_scope': 'Prior007 originalU minus actual retainedU; not a new '
                               'production counter or separately materialized singleton set.',
            'reconstructed_original_U': len(buckets)+inferred_singletons,
            'reconstructed_original_K': memberships+inferred_singletons,
            'reconstructed_original_V': visits+inferred_singletons,
            'retained_map_key_list_owned_bytes': owned_bytes,
            'owned_bytes_scope': 'After pruning sys.getsizeof(map)+retained keys+lists; '
                                 'construction fullmap and rebuild transient are absent from '
                                 'this payload but included in whole-run RSS highwater. Shared '
                                 'row integer objects, masks, caches and arenas excluded.',
            'RSS_after_first_pair_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'first_pair_and_index_census_seconds': time.perf_counter()-graph_start,
        }
        del buckets
        if (memberships+inferred_singletons, visits+inferred_singletons) != (
                c['fixed']['spectator_memberships'], c['fixed']['bucket_ordered_pair_visits']):
            raise RuntimeError('live spectator index census differs')
        write(actual / 'result.json', report)
        h, buffer = hashlib.sha256(), bytearray()
        previous = (-1, -1)
        by_replacement = Counter()
        rows_seen = diagonal = count = 0
        for row, col in itertools.chain((first,), iterator):
            if not (0 <= row < raw and 0 <= col < raw) or (row, col) <= previous:
                raise RuntimeError('duplicate/non-row-major/out-of-range coordinate')
            if row != previous[0]:
                rows_seen += 1
            if row == col:
                diagonal += 1
            difference = (dets[row] ^ dets[col]).bit_count()
            if difference not in (0, 2, 4):
                raise RuntimeError('non Slater-Condon connection')
            by_replacement[str(difference//2)] += 1
            buffer.extend(struct.pack('<QQ', row, col))
            if len(buffer) >= 65536:
                h.update(buffer)
                buffer.clear()
            previous = row, col
            count += 1
            if count % 1_000_000 == 0:
                report.update(pairs=count, elapsed_seconds=time.perf_counter()-start)
                write(actual / 'result.json', report)
        h.update(buffer)
        if (count, rows_seen, diagonal) != (43720175, raw, raw) or dict(
                by_replacement) != c['fixed']['ordered_pairs_by_replacements']:
            raise RuntimeError('full ordered graph count differs from integer census')
        if h.hexdigest() != c['fixed']['prior_ordered_pair_sha256']:
            raise RuntimeError('whole ordered graph hash differs from closed007 control')
        source_check(c)
        imported = {}
        for name, module in tuple(sys.modules.items()):
            if name != 'atomx' and not name.startswith('atomx.'):
                continue
            path = getattr(module, '__file__', None)
            if path is None:
                continue
            path = Path(path).resolve()
            relative = str(path.relative_to(ROOT))
            if not path.is_relative_to(ROOT / 'src/atomx') or sha(path) != c['sources'].get(relative):
                raise RuntimeError('imported source identity mismatch')
            imported[relative] = sha(path)
        report.update(success=True, stage='complete', pairs=count, rows_seen=rows_seen,
                      diagonals=diagonal, ordered_by_replacements=dict(by_replacement),
                      ordered_pair_sha256=h.hexdigest(), last_pair=previous,
                      prior007_ordered_pair_sha256_match=True,
                      graph_seconds=time.perf_counter()-graph_start,
                      imported_sources=imported,
                      claim_scope='Complete structural graph only; no angular coefficients, '
                                  'full H/CI admission, energy or NIST accuracy result.')
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        if iterator is not None:
            iterator.close()
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.perf_counter()-start,
                      RSS_highwater_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      host_after=host())
        write(actual / 'result.json', report)
    print(json.dumps({k: report[k] for k in
                      ('success', 'pairs', 'elapsed_seconds', 'RSS_highwater_KiB')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.execute)
    else:
        print(json.dumps({'status': 'SOURCE_ONLY_NOT_EXECUTED',
                          'driver_sha256': sha(HERE / 'run.py')}))


if __name__ == '__main__':
    main()
