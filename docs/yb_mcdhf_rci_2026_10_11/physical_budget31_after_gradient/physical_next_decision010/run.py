"""Source-bound signed density/gradient qualification; no optimizer or CI solve.

--prepare only seals metadata. --case needs that immutable prepared manifest;
each case writes to a new directory once. Selected vectors have explicitly
checked signed basis/radial/source ancestry; no CIResult is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
BASE = HERE.parent
PREV = BASE / 'physical_next_decision009'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def read(path):
    return json.loads(Path(path).read_text())


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return plain(x.tolist()) if hasattr(x, 'tolist') else x


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def source_map():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}


def prepare():
    bc = read(BASE / 'physical_next_decision006/execution001/contract.json')
    mapping = source_map()
    before = read(HERE / 'source_before.json')['all_atomx_python_sha256']
    changed = sorted(p for p in before.keys() | mapping.keys() if before.get(p) != mapping.get(p))
    if changed != ['src/atomx/methods/csf/variational.py', 'src/atomx/physics/csf/radial_density.py']:
        raise ValueError('undeclared source change')
    inputs = {v['path']: v['sha256'] for v in bc['inputs'].values()}
    for p in (HERE / 'contract.json', HERE / 'test_receipt.json', HERE / 'run.py',
              PREV / 'full_f12_preparation001/complete_models.json',
              PREV / 'full_f12_preparation001/plan.json',
              PREV / 'full_f12_preparation001/aggregate001/result.json'):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    for case in ('J1_even', 'J3_even', 'J5_even', 'J1_odd', 'J3_odd'):
        for name in ('result.json', 'basis.json', 'A_selected.npz', 'B_selected.npz'):
            p = PREV / ('full_f12_preparation001/actual_' + case) / name
            inputs[str(p.relative_to(ROOT))] = sha(p)
    if any(sha(ROOT / p) != expected for p, expected in inputs.items()):
        raise ValueError('frozen input mismatch')
    write(HERE / 'current_source_identity.json', {'all_atomx_python_sha256': mapping,
                                                'source_map_sha256': digest(mapping)})
    write(HERE / 'execution_contract.json', dict(
        source_map_sha256=digest(mapping), inputs=inputs, base_contract=bc,
        cases=['matched', 'J1_even', 'J3_even', 'J5_even', 'J1_odd', 'J3_odd'],
        derivative_action_abs_Ha=1e-9, all_key_abs=1e-12, full_gradient_abs=1e-9,
        energy_reconstruction_abs_Ha=1e-7,
        limits=dict(AS_bytes=8*1024**3, seconds=1200, decoded_bytes=256*1024**2,
                    radial_keys=384706, raw_determinants=92715,
                    load1_max=32, available_GiB_min=32),
        interpretation='Fixed R0 real DC bilinears and analytic radial gradients only; failed ordinary FD005 remains failed. No full-spectrum owner, eigensolve, optimizer or experimental admission.'))
    print(json.dumps(dict(status='SOURCE_PREPARED', science_calls=0,
                          source_map_sha256=digest(mapping), bound_inputs=len(inputs))))


def execute(case):
    c = read(HERE / 'execution_contract.json')
    m = read(HERE / 'current_source_identity.json')
    if case not in c['cases'] or source_map() != m['all_atomx_python_sha256']:
        raise ValueError('case or current source mismatch')
    if digest(m['all_atomx_python_sha256']) != c['source_map_sha256']:
        raise ValueError('source manifest mismatch')
    for path, expected in c['inputs'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('bound input changed: ' + path)
    seal = read(HERE / ('seal_' + case + '.json'))
    if seal != dict(case=case, CPU=seal['CPU'], once_only=True,
                    source_map_sha256=c['source_map_sha256'],
                    execution_contract_sha256=sha(HERE / 'execution_contract.json')):
        raise ValueError('root execution seal mismatch')
    if (os.sched_getaffinity(0) != {seal['CPU']} or
            any(os.environ.get(k) != '1' for k in THREADS) or
            os.path.abspath(sys.executable) != '/data/code/atomx/.venv/bin/python'):
        raise ValueError('interpreter/CPU/thread guard')
    available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith('MemAvailable:'))
    cap = c['limits']
    if os.getloadavg()[0] > cap['load1_max'] or available < cap['available_GiB_min']*1024**3:
        raise ValueError('host load/memory guard')
    if (HERE / 'STOP').exists():
        raise ValueError('a prior case failed')
    if case != 'matched' and not read(HERE / 'actual_matched/result.json')['status'].startswith('PASS'):
        raise ValueError('complete matched qualification required')
    out = HERE / ('actual_' + case)
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (cap['AS_bytes'], cap['AS_bytes']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1200s stop')))
    signal.alarm(cap['seconds'])
    started = time.monotonic()
    report = dict(status='RUNNING', case=case, source_map_sha256=c['source_map_sha256'],
                  checks=[], counts={}, states={}, host_load=os.getloadavg(), CPU=seal['CPU'],
                  ordinary_FD005_superseded=False, stationarity_qualified=False,
                  experimental_accuracy_qualified=False, public_full_spectrum_owner=False,
                  controlled_timing=False)

    def checkpoint():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write(out / 'partial.json', report)

    def check(name, observed, limit=None):
        passed = bool(observed) if limit is None else bool(observed <= limit)
        report['checks'].append(dict(name=name, observed=observed, limit=limit, PASS=passed))
        if not passed:
            raise ValueError('scientific gate: ' + name)

    def count(label, limit=None):
        counts = report['counts']
        counts[label] = counts.get(label, 0) + 1
        if limit is not None and counts[label] > limit:
            raise RuntimeError('work cap: ' + label)

    try:
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np

        from atomx.angular import csf_hamiltonian as exact
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline import orbital_optimization as opt
        from atomx.numerics.bspline.convol import compute_slater_rk_cell
        from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
        from atomx.physics.csf import matrix
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.potentials.nuclear import make_nuclear_potential
        bc = c['base_contract']
        decoded = 0

        def archive(role, fields):
            nonlocal decoded
            spec = bc['arrays'][role]
            path = ROOT / bc['inputs'][role]['path']
            with zipfile.ZipFile(path) as z:
                if sum(x.file_size for x in z.infolist()) > spec['uncompressed_cap']:
                    raise ValueError('archive header capacity')
            result = {}
            with np.load(path, allow_pickle=False) as z:
                for field in fields:
                    value, expected = z[field], spec['fields'][field]
                    shapes = [expected['shape']] if 'shape' in expected else expected['shape_choices']
                    if (value.dtype.str != expected['dtype'] or list(value.shape) not in shapes
                            or not np.isfinite(value).all()):
                        raise ValueError('array schema: ' + role + '/' + field)
                    decoded += value.nbytes
                    if decoded > cap['decoded_bytes']:
                        raise ValueError('decoded bytes cap')
                    result[field] = value
            return result

        wavebank = archive('B_R0', ('keys', 'P', 'Q'))
        geometry = archive('workspace', bc['arrays']['workspace']['fields'])
        cp = archive('checkpoint_arrays', bc['arrays']['checkpoint_arrays']['fields'])
        metric = archive('saved_S', bc['arrays']['saved_S']['fields'])
        data = {k[6:]: v.item() if v.ndim == 0 else v for k, v in geometry.items()
                if k.startswith('basis_')}
        keys = tuple(map(tuple, wavebank['keys'].tolist()))
        waves = {k: (p, q) for k, p, q in zip(keys, wavebank['P'], wavebank['Q'], strict=True)}
        nucleus = make_nuclear_potential(70, 'fermi', A=175)
        binding = asdict(CIReferenceBinding.capture(
            [(*waves[k], k[1]) for k in sorted(keys)], {}, data, 70, 123, nucleus))
        checkpoint_meta = read(ROOT / bc['inputs']['checkpoint']['path'])
        check('owned original radial/nuclear binding', plain(binding) == checkpoint_meta['owner']['radial_binding'])
        layout = tuple((tuple(map(tuple, row['keys'])), cp['projector_' + str(i)],
                        tuple(row['shape']), slice(*row['slice'])) for i, row in enumerate(
                            read(ROOT / bc['inputs']['layout']['path'])))
        inverse = tuple(tuple(cp['inverse_blocks_' + str(i)]) for i in range(7))
        check('original metric identity', opt._residual_metric_sha256(
            layout, cp['inverse_diagonal'], inverse) == bc['fixed']['metric_sha256'])
        radial_direction = np.zeros((31, 246))
        for group, projector, shape, sl in layout:
            v = np.column_stack([np.concatenate(waves[k]) for k in group])
            s = metric['s_' + str(group[0][1])]
            check('radial Gram ' + str(group[0][1]), float(abs(v.T@s@v-np.eye(len(group))).max()), 1e-10)
            check('coordinate wave ' + str(group[0][1]), float(abs(v-projector@cp['x'][sl].reshape(shape)).max()), 1e-12)
            d = projector @ (-cp['full_normalized_pg'][sl].reshape(shape))
            check('tangent direction ' + str(group[0][1]), float(abs(v.T@s@d+d.T@s@v).max()), 1e-10)
            for j, key in enumerate(group):
                radial_direction[keys.index(key)] = d[:, j]
        directions = {k: (d[:123], d[123:]) for k, d in zip(keys, radial_direction, strict=True)}
        report['radial_binding'] = binding
        report['direction_sha256'] = hashlib.sha256(radial_direction.tobytes()).hexdigest()
        if case == 'matched':
            models = read(ROOT / bc['inputs']['models']['path'])
            configs = [row['config'] for row in models['expanded_configs']]
            sector = None
        else:
            j, parity = case.split('_')
            sector = (int(j[1:]), 1 if parity == 'even' else -1)
            models = read(PREV / 'full_f12_preparation001/complete_models.json')['models']
            configs = [row['configuration'] for row in models['B']['NR_configurations']]
        uids = {tuple(uid[:2]) for _, uid in bc['UIDs']}
        csfs, nr = [], []
        for i, config in enumerate(configs):
            selected = [csf for csf in enumerate_jj_csfs(config)
                        if ((csf.total_J2, csf.parity) in uids if sector is None
                            else (csf.total_J2, csf.parity) == sector)]
            csfs.extend(selected)
            nr.extend([i]*len(selected))
        sigs = [json.dumps(plain((s.subshells, s.coupling, csf.total_J2, csf.parity)),
                           separators=(',', ':')) for csf in csfs for s in (exact.csf_signature(csf),)]
        check('unique full signed CSF order', len(set(sigs)) == len(sigs))
        states, vectors, references = [], [], []
        if case == 'matched':
            maps = read(ROOT / bc['inputs']['basis_maps']['path'])
            check('complete original B signed basis', sigs == maps['expanded']['selected_signatures'])
            a_indices = np.array([i for i, n in enumerate(nr) if n < 91])
            check('complete original CSF counts', len(csfs) == 2574 and len(a_indices) == 1595)
            for arm, tag in (('A', 'baseline'), ('B', 'expanded')):
                indices = a_indices if arm == 'A' else np.arange(len(csfs))
                check(arm + ' original ordered basis', [sigs[i] for i in indices] == maps[tag]['selected_signatures'])
                bank = archive(arm + '_R0', ('mixing',))
                rows = read(ROOT / bc['inputs'][arm + '_R0_record']['path'])['rows']
                own = CSFOrbitalFunctional([csfs[int(i)] for i in indices], data, 70,
                                          nuclear_potential=nucleus, max_determinants=92715)
                for row in rows:
                    v = np.zeros(len(csfs))
                    v[indices] = bank['mixing'][:, row['column']]
                    vectors.append(v)
                    states.append(dict(arm=arm, UID=row['UID'], energy_Ha=row['energy_Ha']))
                    references.append((own, own.radial_density.contract_bilinear(v[indices], v[indices])))
        else:
            prior = PREV / ('full_f12_preparation001/actual_' + case)
            record, basis = read(prior / 'result.json'), read(prior / 'basis.json')
            check('prior source-bound sector PASS', record['status'].startswith('PASS')
                  and all(row['PASS'] for row in record['checks']))
            check('complete 4f12 signed basis', sigs == basis['B_signatures'])
            for arm in ('A', 'B'):
                selected = record['selected'][arm]
                with np.load(prior / (arm + '_selected.npz'), allow_pickle=False) as z:
                    indices, mixing = z['B_nested_indices'], z['mixing']
                    check(arm + ' exact saved nesting', np.array_equal(indices, basis['A_nested_indices']
                                                                     if arm == 'A' else np.arange(len(csfs))))
                    column = selected['selected_local_root']
                    check(arm + ' original residual acceptance', float(z['residual_L2_Ha'].max()), 1e-8)
                    check(arm + ' original norm', float(abs(mixing.T@mixing-np.eye(2)).max()), 1e-10)
                    check(arm + ' original signed overlap', z['signed_overlap'][column]**2 > .5)
                    check(arm + ' original energy binding', float(z['energies'][column]) == selected['energy_Ha'])
                    v = np.zeros(len(csfs))
                    v[indices] = mixing[:, column]
                vectors.append(v)
                states.append(dict(arm=arm, UID=selected['UID'], energy_Ha=selected['energy_Ha']))
        def forbidden(*_args, **_kw):
            count('forbidden_angular_table')
            raise RuntimeError('retained angular coefficient table forbidden')
        exact.block_terms = forbidden
        exact._coefficient_table = forbidden
        connected = exact._connected_pairs
        plan = read(PREV / 'full_f12_preparation001/plan.json')
        pair_cap = (5_000_000 if sector is None else plan['memory_per_B_sector'][
            f'2J={sector[0]} parity={sector[1]}']['ordered_structural_pairs'])
        def guarded_pairs(dets, **kw):
            count('connection_graphs', 5 if sector is None else 1)
            if len(dets) > cap['raw_determinants']:
                raise RuntimeError('raw determinant cap')
            for i, j in connected(dets, **kw):
                count('ordered_pairs', pair_cap)
                if report['counts']['ordered_pairs'] % 1_000_000 == 0:
                    if (HERE / 'STOP').exists():
                        raise RuntimeError('another worker failed')
                    checkpoint()
                yield i, j
        exact._connected_pairs = guarded_pairs
        functional = CSFOrbitalFunctional(csfs, data, 70, nuclear_potential=nucleus,
                                          max_determinants=92715, density_storage='streamed')
        check('all 31 radial coordinates', set(functional.keys) == set(keys))
        report['stage'] = 'streamed signed density'
        checkpoint()
        raw_weights = functional.radial_density.contract_bilinears(np.column_stack(vectors), np.column_stack(vectors))
        check('one graph traversal per block', report['counts']['connection_graphs'] == (5 if sector is None else 1))
        if sector is not None:
            check('complete ordered pair census', report['counts']['ordered_pairs'] == pair_cap)
        raw_blocks = {kappa: build_dirac_matrices(123, kappa, 70, data, nucleus)
                      for kappa in {k for _, k in keys}}
        radial_evaluate = matrix._radial_evaluator(waves, data, 70, 123, nuclear_potential=nucleus)
        integral_values, derivative_values = {}, {}

        def forward(term):
            if term not in derivative_values:
                if term[0] == 'h':
                    _, na, kappa, nb = term
                    a, b = (na, kappa), (nb, kappa)
                    va, vb = np.concatenate(waves[a]), np.concatenate(waves[b])
                    da, db = np.concatenate(directions[a]), np.concatenate(directions[b])
                    h, s = raw_blocks[kappa]
                    norm, numerator = vb@s@vb, va@h@vb
                    value = (da@h@vb+va@h@db)/norm - numerator*(db@s@vb+vb@s@db)/norm**2
                else:
                    _, rank, *orbs = term
                    pieces = []
                    for slot in range(4):
                        args = [directions[k] if i == slot else waves[k] for i, k in enumerate(orbs)]
                        pieces.append(float(compute_slater_rk_cell(
                            *args[0], *args[1], *args[2], *args[3], data, rank, cache=functional.cache)))
                    value = math.fsum(pieces)
                derivative_values[term] = float(value)
                if len(derivative_values) > cap['radial_keys']:
                    raise RuntimeError('forward radial key capacity')
            return derivative_values[term]

        gradients = []
        for index, (state, vector, table) in enumerate(zip(states, vectors, raw_weights, strict=True)):
            report['stage'] = 'gradient ' + str(index)
            checkpoint()
            norm = sum((Fraction.from_float(float(v))**2 for v in vector), Fraction())
            check('positive actual CI norm ' + str(index), norm > 0)
            weights = {key: float(Fraction.from_float(float(v))/norm) for key, v in table.items()}
            check('complete finite key capacity ' + str(index), bool(weights) and len(weights) <= cap['radial_keys']
                  and all(np.isfinite(v) and v != 0 for v in weights.values()))
            g = functional.gradient_from_weights(waves, weights, record_diagnostics=False)
            array = np.array([g[key] for key in keys])
            check('complete finite gradient ' + str(index), array.shape == (31, 246) and np.isfinite(array).all())
            if references:
                own, ref = references[index]
                error = max(abs(table.get(key, 0.)-ref.get(key, 0.)) for key in table.keys() | ref.keys())
                check('complete cached radial key difference ' + str(index), float(error), c['all_key_abs'])
                ref = {k: float(Fraction.from_float(float(v))/norm) for k, v in ref.items()}
                gg = own.gradient_from_weights(waves, ref, record_diagnostics=False)
                check('complete cached gradient difference ' + str(index),
                      float(max(abs(g[k]-gg[k]).max() for k in keys)), c['full_gradient_abs'])
            for key in weights:
                if key not in integral_values:
                    integral_values[key] = float(radial_evaluate(key))
            energy = math.fsum(value*integral_values[key] for key, value in weights.items())
            check('owned energy reconstruction ' + str(index), abs(energy-state['energy_Ha']), c['energy_reconstruction_abs_Ha'])
            action = float(np.sum(array.astype(np.longdouble)*radial_direction.astype(np.longdouble)))
            expected = math.fsum(value*forward(key) for key, value in weights.items())
            check('independent four-slot derivative ' + str(index), abs(action-expected), c['derivative_action_abs_Ha'])
            components = {kind: math.fsum(value*forward(key) for key, value in weights.items() if key[0] == kind)
                          for kind in ('h', 'r')}
            gradients.append(array)
            report['states'][str(index)] = dict(**state, actual_CI_norm=float(norm), radial_keys=len(weights),
                reconstruction_energy_Ha=energy, directional_action_Ha=action,
                independent_forward_Ha=expected, forward_components_Ha=components,
                gradient_norm=float(np.linalg.norm(array)), weights_sha256=digest(list(weights.items())))
            write(out / ('weights_' + str(index) + '.json'), list(weights.items()))
        np.savez_compressed(out / 'gradients.npz', keys=np.asarray(keys),
                            gradients=np.array(gradients), radial_direction=radial_direction,
                            signed_vectors=np.column_stack(vectors))
        check('unchanged current source after science', source_map() == m['all_atomx_python_sha256'])
        check('no retained angular table calls', report['counts'].get('forbidden_angular_table', 0) == 0)
        report.update(status='PASS_complete_signed_density_and_analytic_radial_gradient', stage='complete',
                      distinct_radial_values=len(integral_values), distinct_forward_derivatives=len(derivative_values),
                      gradient_arrays_sha256=sha(out / 'gradients.npz'))
        checkpoint()
        write(out / 'result.json', report)
        print(json.dumps(dict(status=report['status'], case=case, checks=len(report['checks']),
                              seconds=report['elapsed_seconds'], peak_RSS_MiB=report['peak_RSS_MiB'])))
    except BaseException:
        report.update(status='FAIL', traceback=traceback.format_exc())
        checkpoint()
        write(out / 'result.json', report)
        (HERE / 'STOP').write_text(case + '\n')
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.prepare and args.case is None:
        prepare()
    elif args.case and not args.prepare:
        execute(args.case)
    else:
        parser.error('choose --prepare or a sealed --case')
