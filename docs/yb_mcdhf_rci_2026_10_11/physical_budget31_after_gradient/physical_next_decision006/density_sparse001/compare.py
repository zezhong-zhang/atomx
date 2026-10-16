"""Once-only angular density storage qualification; no H/CI/radial solve."""

from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/data/code/atomx')
HERE = Path(__file__).resolve().parent
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def main():
    contract = read(HERE / 'contract.json')
    cap, fixed = contract['resources'], contract['fixed']
    for relative, expected in contract['source_map'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('current source changed before launch: ' + relative)
    for spec in contract['inputs'].values():
        if sha(ROOT / spec['path']) != spec['sha256']:
            raise ValueError('whole input changed: ' + spec['path'])
    if sha(HERE / 'source_before.py') != contract['before_sha256']:
        raise ValueError('before source changed')
    if os.sched_getaffinity(0) != {cap['CPU']} or any(
            os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('CPU/thread admission failed')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if os.getloadavg()[0] > cap['host_load1_max'] or available < cap['host_MemAvailable_GiB_min'] * 1024**3:
        raise ValueError('host admission failed')
    actual = HERE / 'actual'
    actual.mkdir()  # Deliberate once-only marker; never reuse a failed run.
    resource.setrlimit(resource.RLIMIT_AS, (cap['AS_bytes'], cap['AS_bytes']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('300 s cap')))
    signal.alarm(cap['seconds'])
    started = time.monotonic()
    counters, checks = collections.Counter(), []
    result = dict(status='RUNNING', contract_sha256=sha(HERE / 'contract.json'),
                  driver_sha256=sha(__file__), checks=checks, models={}, counters=counters,
                  forbidden_science_calls=0, controlled_timing=False,
                  scope=contract['scope'], input_bindings=contract['inputs'],
                  host_load=os.getloadavg(), available_bytes=available)

    def check(name, observed, limit=None):
        passed = bool(observed) if limit is None else bool(observed <= limit)
        checks.append(dict(name=name, observed=plain(observed), limit=limit, PASS=passed))
        if not passed:
            raise RuntimeError('gate failed: ' + name)

    def counted(module, name, label, limit):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            counters[label] += 1
            if counters[label] > limit:
                raise RuntimeError('entry cap: ' + label)
            return original(*args, **kwargs)

        setattr(module, name, wrapped)

    def forbidden(*args, **kwargs):
        result['forbidden_science_calls'] += 1
        raise RuntimeError('H/CI/gradient/radial solve is forbidden')

    try:
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np
        from scipy.sparse import issparse

        import atomx.angular.mscheme as mscheme

        counted(mscheme, 'null_space', 'angular_nullspace', cap['angular_nullspace'])
        highest = mscheme.highest_weight_states
        check('cold_highest_weight_cache', highest.cache_info().currsize == 0)
        allowed = {(r['twoj'], r['q']) for r in contract['angular_census']}
        queries = set()

        def guarded_highest(tj, q):
            if (tj, q) not in allowed:
                raise ValueError('unbudgeted highest-weight query')
            queries.add((tj, q))
            return highest(tj, q)

        mscheme.highest_weight_states = guarded_highest
        import atomx.angular.csf_hamiltonian as angular
        import atomx.physics.csf.matrix as matrix
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.methods import ci as ci_module
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.radial_density import CIRadialDensity

        counted(angular, 'block_terms', 'block_terms', cap['block_terms'])
        counted(angular, '_coefficient_table', 'coefficient_tables', cap['coefficient_tables'])
        counted(angular, 'csf_block_matrix', 'projection', cap['projection'])
        for module, names in ((matrix, ('build_ci_matrix', 'build_dirac_matrices', '_radial_evaluator')),
                              (ci_module, ('ci_solve', 'davidson_eigh'))):
            for name in names:
                setattr(module, name, forbidden)
        spec = importlib.util.spec_from_file_location('density_source_before', HERE / 'source_before.py')
        before_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before_module)
        check('before_after_same_angular_module', before_module.exact_ci is angular)
        check('before_after_same_matrix_module', before_module.matrix is matrix)
        frozen = ROOT / 'docs/yb_mcdhf_rci_2026_10_11/rank_aware_cell_discriminator_preparation/candidate_source'
        check('angular_exact_f043', sha(angular.__file__) == sha(frozen / 'atomx/angular/csf_hamiltonian.py'))
        model_specs = read(ROOT / contract['inputs']['models']['path'])
        maps = read(ROOT / contract['inputs']['basis_maps']['path'])
        sectors = set(map(tuple, model_specs['selected_sectors']))
        decoded = 0
        rng = np.random.default_rng(fixed['random_seed'])

        def compare(label, old, new):
            union = sorted(set(old) | set(new), key=repr)
            errors = [abs(old.get(k, 0.) - new.get(k, 0.)) for k in union]
            maximum = max(errors, default=0.)
            at = None if not union else union[int(np.argmax(errors))]
            record = dict(label=label, before_keys=len(old), after_keys=len(new),
                          union_keys=len(union), maximum_abs_difference=maximum,
                          maximum_key=plain(at), before_only=[plain(k) for k in sorted(set(old)-set(new), key=repr)],
                          after_only=[plain(k) for k in sorted(set(new)-set(old), key=repr)],
                          negative_before=sum(v < 0 for v in old.values()),
                          negative_after=sum(v < 0 for v in new.values()),
                          complete_values=[[plain(k), old.get(k, 0.), new.get(k, 0.)] for k in union])
            check(label + '/complete_union_radial_keys', maximum, fixed['radial_keys_tolerance'])
            return record

        for arm, model in (('A', 'baseline'), ('B', 'expanded')):
            declared = fixed['models'][arm]
            configs = model_specs[model + '_configs']
            check(arm + '/NR_configs', len(configs) == declared['nr'])
            full = [z for row in configs for z in enumerate_jj_csfs(row['config'])]
            csfs = [z for z in full if (z.total_J2, z.parity) in sectors]
            signatures = [json.dumps(plain((s.subshells, s.coupling, z.total_J2, z.parity)), separators=(',', ':'))
                          for z in csfs for s in (angular.csf_signature(z),)]
            check(arm + '/parent_CSFs', len(full) == declared['full'])
            check(arm + '/all_signed_CSFs', signatures == maps[model]['selected_signatures'])
            check(arm + '/selected_CSFs', len(csfs) == declared['selected'])
            check(arm + '/electrons', all(z.n_electrons == 69 for z in full))
            point = read(ROOT / contract['inputs'][arm + '_json']['path'])
            check(arm + '/same31keys', list(map(list, sorted(get_shared_orbital_basis(csfs)))) == point['owner']['orbital_keys'])
            path = ROOT / contract['inputs'][arm + '_npz']['path']
            with zipfile.ZipFile(path) as z:
                check(arm + '/NPZ_members', set(z.namelist()) == {'H.npy', 'energies.npy', 'mixing.npy', 'keys.npy', 'P.npy', 'Q.npy'})
                check(arm + '/decoded_C_preallocation', z.getinfo('mixing.npy').file_size, cap['decoded_bytes'])
            with np.load(path, allow_pickle=False) as z:
                mixing, energies = z['mixing'].copy(), z['energies'].copy()
            decoded += mixing.nbytes + energies.nbytes
            check(arm + '/decoded_input_total', decoded, cap['decoded_bytes'])
            check(arm + '/full_C_E_shape', mixing.shape == (len(csfs), len(csfs)) and energies.shape == (len(csfs),))
            check(arm + '/real_finite', mixing.dtype == np.dtype('float64') and energies.dtype == np.dtype('float64')
                  and np.isfinite(mixing).all() and np.isfinite(energies).all())
            check(arm + '/full_C_Gram', float(np.max(abs(mixing.T @ mixing - np.eye(len(csfs))))), fixed['full_C_Gram_tolerance'])
            ci = SimpleNamespace(state_energies=energies, mixing_coefficients=mixing, ground_energy=float(energies[0]))
            _, blocks = ci_level_blocks(csfs, ci, require_full=True)
            target_cols, uid_records = [], []
            for row in point['rows']:
                uid = tuple(row['UID'])
                selected = tuple(row['selected_level'])
                check(arm + '/UID/' + str(uid), uid == selected and selected[2] == 0 and row['weight'] == .2)
                col = int(blocks[selected[:2]][selected[2]])
                check(arm + '/column/' + str(uid), col == row['column'] and energies[col] == row['energy_Ha'])
                target_cols.append(col)
                uid_records.append(dict(UID=uid, column=col, energy_Ha=float(energies[col])))
            check(arm + '/five_distinct_UIDs', len(target_cols) == len(set(target_cols)) == 5)
            weights = np.zeros(len(csfs))
            weights[target_cols] = fixed['five_equal_weights']
            before = before_module.CIRadialDensity(csfs, max_determinants=declared['detcap'])
            after = CIRadialDensity(csfs, max_determinants=declared['detcap'])
            counters['density_constructors'] += 2
            check('density_constructor_cap', counters['density_constructors'], cap['density_constructors'])
            model_result = dict(UIDs=uid_records, blocks=[], contractions=[])
            result['models'][arm] = model_result
            for index, ((s0, c0, rows0), (s1, c1, rows1)) in enumerate(zip(before.blocks, after.blocks, strict=True)):
                check(arm + '/shared_structure/' + str(index), s0 is s1 and rows0 == rows1)
                check(arm + '/projection_storage/' + str(index), isinstance(c0, np.ndarray) and issparse(c1))
                coo = c1.tocoo()
                difference = c0.copy()
                difference[coo.row, coo.col] -= coo.data
                check(arm + '/complete_signed_projection/' + str(index), float(np.max(abs(difference))), 0.)
                check(arm + '/projection_nonzero_count/' + str(index), np.count_nonzero(c0) == c1.nnz)
                model_result['blocks'].append(dict(J2=s0.j2, shape=c0.shape, nnz=int(c1.nnz),
                    before_dense_bytes=c0.nbytes, after_CSR_bytes=c1.data.nbytes+c1.indices.nbytes+c1.indptr.nbytes,
                    angular_table_nnz=s0.table.nnz, raw_determinants=len(s0.dets)))
            pairs = []
            for row, col in zip(point['rows'], target_cols, strict=True):
                uid = tuple(row['UID'])
                pairs.append(('state/' + str(uid), mixing[:, col], mixing[:, col]))
                same = blocks[uid[:2]]
                if len(same) > 1:
                    pairs.append(('transition_to_second/' + str(uid), mixing[:, col], mixing[:, same[1]]))
            random = rng.standard_normal((len(csfs), 2))
            random /= np.linalg.norm(random, axis=0)
            pairs.append(('random_signed_pair', random[:, 0], random[:, 1]))
            model_result['contractions'].append(compare(arm + '/five_equal_weight_contract', before.contract(ci, weights), after.contract(ci, weights)))
            for label, left, right in pairs:
                model_result['contractions'].append(compare(arm + '/unfiltered_bilinear/' + label,
                    before.contract_bilinear(left, right), after.contract_bilinear(left, right)))
            del before, after, ci, mixing, energies, pairs, random, difference
            write(actual / 'partial.json', result)
        reference_configs = read(ROOT / contract['inputs']['reference_models']['path'])['models']['baseline']['configs']
        reference = [z for row in reference_configs for z in enumerate_jj_csfs(row['config'])]
        check('original531_reference', len(reference) == fixed['reference_csfs'])
        check('all_allowed_highest_weight_queries', queries == allowed)
        check('whole43_nullspace_census', counters['angular_nullspace'] == cap['angular_nullspace'])
        imported = {}
        for name, module in list(sys.modules.items()):
            if name == 'atomx' or name.startswith('atomx.'):
                path = Path(module.__file__).resolve() if getattr(module, '__file__', None) else None
                if path is not None:
                    relative = str(path.relative_to(ROOT))
                    check('imported_source/' + name, relative in contract['source_map'] and sha(path) == contract['source_map'][relative])
                    imported[name] = dict(path=relative, sha256=sha(path))
        for path in ('src/atomx/physics/csf/radial_density.py', 'src/atomx/angular/csf_hamiltonian.py'):
            check('unchanged_runtime_source/' + path, sha(ROOT / path) == contract['source_map'][path])
        check('zero_forbidden_science', result['forbidden_science_calls'] == 0)
        result.update(status='PASS_actual_Yb_angular_density_storage_qualification', imported_source_bindings=imported,
                      before_sha256=contract['before_sha256'], after_sha256=contract['after_sha256'],
                      decoded_bytes=decoded, no_H_CI_gradient_radial_solve=True)
    except BaseException as error:
        result.update(status='FAIL_preserved', exception=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        result.update(elapsed_seconds_uncontrolled=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      checks_passed=sum(x['PASS'] for x in checks), checks_total=len(checks))
        write(actual / 'result.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'checks_passed', 'checks_total', 'elapsed_seconds_uncontrolled', 'peak_RSS_MiB')}, indent=2))
    return 0 if result['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
