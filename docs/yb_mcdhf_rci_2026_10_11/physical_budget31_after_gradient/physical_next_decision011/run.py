"""Saved-H exact-dyadic FD decomposition; no Atomx import or new H/CI."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import resource
import signal
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
OLD = HERE.parent / 'physical_next_decision005/execution001'
LATEST = HERE.parent / 'physical_next_decision010'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def parts(value):
    n, d = float(value).as_integer_ratio()
    return n, -(d.bit_length() - 1)


class Dyadic:
    """Accumulate signed integer times powers of two, with no rounded products."""

    def __init__(self):
        self.n, self.e = 0, 0

    def add(self, n, e):
        if not n:
            return
        if not self.n:
            self.n, self.e = n, e
        elif e < self.e:
            self.n = (self.n << (self.e - e)) + n
            self.e = e
        else:
            self.n += n << (e - self.e)

    def fraction(self):
        return (Fraction(self.n << self.e) if self.e >= 0 else
                Fraction(self.n, 1 << -self.e))


def exact_quadratic(matrix, vector, reference):
    """Exact normalized c.T (H-reference*I)c for literal binary64 arrays.

    The caller verifies exact real symmetry first. Every upper-triangle entry
    is visited; only exact-zero products are omitted. Centering is an arithmetic
    identity, not a changed physical Hamiltonian or another eigensolve.
    """
    c = [parts(v) for v in vector]
    norm, total = Dyadic(), Dyadic()
    for i, (ni, ei) in enumerate(c):
        norm.add(ni * ni, 2 * ei)
        diagonal = Fraction.from_float(float(matrix[i, i])) - reference
        total.add(ni * ni * diagonal.numerator,
                  2 * ei - (diagonal.denominator.bit_length() - 1))
        if not ni:
            continue
        for j in range(i + 1, len(c)):
            nj, ej = c[j]
            h = float(matrix[i, j])
            if nj and h:
                nh, eh = parts(h)
                total.add(2 * ni * nj * nh, ei + ej + eh)
    n = norm.fraction()
    if n <= 0:
        raise ValueError('zero saved vector norm')
    return total.fraction() / n, n


def self_tests(np):
    rng = np.random.default_rng(1101)
    checks = 0
    for size in (1, 2, 4, 6):
        for _ in range(8):
            h = rng.normal(size=(size, size))
            h = h + h.T
            v = rng.normal(size=size)
            if size > 1:
                v[-1] = -1e-100
            ref = Fraction.from_float(-14000.125)
            vv = [Fraction.from_float(float(x)) for x in v]
            norm = sum((x*x for x in vv), Fraction())
            oracle = sum((vv[i] * vv[j] * Fraction.from_float(float(h[i, j]))
                          for i in range(size) for j in range(size)), Fraction())
            got, got_norm = exact_quadratic(h, v, ref)
            assert got == oracle / norm - ref and got_norm == norm
            assert exact_quadratic(h, -v, ref)[0] == got
            assert exact_quadratic(h, 2*v, ref)[0] == got
            checks += 3
    h = np.array([[2., -0.75], [-0.75, 3.]])
    v = np.array([0.6, -0.8])
    assert exact_quadratic(h, v, Fraction())[0] != exact_quadratic(
        np.diag(np.diag(h)), v, Fraction())[0]
    return checks + 1


def prepare():
    old = read(OLD / 'contract.json')
    files = [OLD / name for name in ('contract.json', 'run.py',
                                     'root_execution_seal.json', 'actual/result.json')]
    for arm in ('A', 'B'):
        for tag in ('R0', 'plus_h', 'minus_h', 'plus_half', 'minus_half'):
            files.extend(OLD / 'actual' / (arm+'_'+tag+suffix)
                         for suffix in ('.json', '.npz'))
        files.append(OLD / 'actual' / (arm+'_R0_state_gradients.json'))
    files.extend([LATEST / 'current_source_identity.json',
                  ROOT / old['inputs']['source_identity']['path']])
    dims = {}
    for arm in ('A', 'B'):
        meta = read(OLD / 'actual' / (arm+'_R0.json'))
        counts = collections.Counter((x[2][1], x[1]) for x in meta['owner']['CSFs'])
        dims[arm] = [[list(key), value] for key, value in sorted(counts.items())]
    visits = 8 * sum(n*(n+1)//2 for rows in dims.values() for _, n in rows)
    contract = dict(
        hypothesis='Reported roots, finite CI-vector response and fixed-vector operator FD make separately measurable contributions to the three original005 failures.',
        fixed='Literal005 full A1595/B2574 H, signed C, basis/owner, all five UIDs, original31-wave two-step polar endpoints and analytic actions; no fresh radial/H/CI/optimizer.',
        steps=[0.002, 0.001], dimensions=dims,
        arithmetic='Exact binary64 integer-ratio products/sums and Fraction actual-norm division; no magnitude cutoff or rounded matrix centering.',
        decomposition='reported-analytic = (reported-exact_Rayleigh) + (exact_Rayleigh-fixed_C0) + (fixed_C0-analytic), separately per state and upper-minus-lower line.',
        distinguish='If reported-exact_Rayleigh accounts for the failed line, target root/reporting arithmetic. Otherwise retain the remaining finite-response/operator terms; two steps alone do not uniquely separate operator curvature, endpoint reconstruction and H assembly.',
        acceptance={'dyadic_oracle':'exact equality, signed/off-diagonal/tiny/scaling controls',
                    'saved_owner':'literal whole-file hashes plus exact matrix/identity frame hashes',
                    'selected_residual_abs_Ha':1e-8, 'norm_abs':1e-10,
                    'outside_sector':0, 'decomposition':'exact Fraction equality',
                    'ordinary_replay_abs_Ha':1e-15,
                    'old_scientific_gate':'unchanged max(1e-8,1e-5*abs(analytic)); classifications are diagnostics, do not supersede005'},
        stopping='First identity/resource/oracle/closure failure stops; preserve evidence, no automatic retries or new steps. Scientific decomposition outcomes all retained.',
        limits={'CPU':27, 'threads':1, 'AS_bytes':4*1024**3, 'seconds':300,
                'load_max':24, 'available_min_bytes':8*1024**3,
                'NPZ_loads':10, 'archive_decoded_bytes_max':120*1024**2,
                'quadratic_calls':80, 'triangle_entries_upper':visits},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in files},
        driver_sha256=sha(__file__), original005_failure_superseded=False)
    save(HERE / 'contract.json', contract)
    print(json.dumps({'status':'PREPARED_METADATA_ONLY', 'inputs':len(files),
                      'triangle_entries_upper':visits, 'dimensions':dims}))


def execute():
    contract = read(HERE / 'contract.json')
    if sha(__file__) != contract['driver_sha256']:
        raise ValueError('changed driver')
    for path, digest in contract['inputs'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('changed input: ' + path)
    current = read(LATEST / 'current_source_identity.json')
    old = read(OLD / 'contract.json')
    frozen = read(ROOT / old['inputs']['source_identity']['path'])
    for name, digest in current['all_atomx_python_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('changed current source: ' + name)
    for name, digest in frozen['all_atomx_python_sha256'].items():
        if sha(ROOT / old['source_archive'] / name.removeprefix('src/')) != digest:
            raise ValueError('changed historical source: ' + name)
    cap = contract['limits']
    if (str(Path(sys.executable).absolute()) != str(ROOT / '.venv/bin/python')
            or os.sched_getaffinity(0) != {cap['CPU']}
            or any(os.environ.get(k) != '1' for k in THREADS)):
        raise ValueError('interpreter/CPU/thread admission')
    available = next(int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()
                     if r.startswith('MemAvailable:'))
    if available < cap['available_min_bytes'] or os.getloadavg()[0] > cap['load_max']:
        raise ValueError('host admission')
    out = HERE / 'actual001'
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS, (cap['AS_bytes'], cap['AS_bytes']))
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(TimeoutError('300s stop')))
    signal.alarm(cap['seconds'])
    started = time.monotonic()
    checks, points, comparisons = [], {}, []
    report = dict(status='RUNNING', checks=checks, points=points, comparisons=comparisons,
                  original005_failure_superseded=False, optimizer_admitted=False,
                  atomx_imported=False, new_H_CI_radial_calls=0,
                  source_map_sha256=current['source_map_sha256'],
                  original_source_map_sha256=old['source_map_sha256'],
                  host_load=os.getloadavg(), available_bytes=available)
    calls = collections.Counter()

    def check(name, value):
        checks.append(dict(name=name, PASS=bool(value)))
        if not value:
            raise ValueError('gate: ' + name)

    def archive(name):
        calls['NPZ_loads'] += 1
        check('NPZ work '+name, calls['NPZ_loads'] <= cap['NPZ_loads'])
        path = OLD / 'actual' / (name+'.npz')
        with zipfile.ZipFile(path) as z:
            check('archive size '+name, sum(v.file_size for v in z.infolist()) <= cap['archive_decoded_bytes_max'])
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k] for k in ('H', 'mixing', 'energies', 'keys', 'P', 'Q')}
        n = 1595 if name[0] == 'A' else 2574
        shapes = dict(H=(n,n), mixing=(n,n), energies=(n,), keys=(31,2), P=(31,123), Q=(31,123))
        for key, value in arrays.items():
            check('shape/dtype/finite '+name+'/'+key,
                  value.shape == shapes[key] and value.dtype.str == ('<i8' if key == 'keys' else '<f8')
                  and np.isfinite(value).all())
        check('exact symmetry '+name, np.array_equal(arrays['H'], arrays['H'].T))
        meta = read(OLD / 'actual' / (name+'.json'))
        frame = repr((arrays['H'].shape, arrays['H'].dtype.str)).encode() + arrays['H'].tobytes(order='C')
        check('matrix owner '+name, hashlib.sha256(frame).hexdigest() == meta['owner_matrix_sha256'])
        payload = json.dumps(meta['owner'],sort_keys=True,separators=(',',':')).encode()
        check('input owner '+name, hashlib.sha256(payload).hexdigest() == meta['owner_input_sha256'])
        check('complete five UID '+name, [[r['label'],r['UID']] for r in meta['rows']] == old['UIDs'])
        check('radial key identity '+name, arrays['keys'].tolist() == meta['owner']['orbital_keys'])
        return arrays, meta

    try:
        import numpy as np

        report['independent_fraction_oracle_checks'] = self_tests(np)
        old_result = read(OLD / 'actual/result.json')
        check('old005 remains failed', old_result['ordinary_FD_passed'] is False)
        for arm in ('A', 'B'):
            baseline, base_meta = archive(arm+'_R0')
            owner = base_meta['owner']
            sector_labels = [(s[2][1],s[1]) for s in owner['CSFs']]
            base_vectors, sectors = {}, {}
            reference = Fraction.from_float(base_meta['rows'][0]['energy_Ha'])
            for row in base_meta['rows']:
                label = row['label']
                ix = np.array([i for i,s in enumerate(sector_labels) if s == tuple(row['UID'][:2])])
                check('complete base sector '+arm+label, len(ix) == dict((tuple(s),n) for s,n in contract['dimensions'][arm])[tuple(row['UID'][:2])])
                vector = baseline['mixing'][:,row['column']]
                other = np.ones(len(vector),bool)
                other[ix] = False
                check('base exact support '+arm+label, np.count_nonzero(vector[other]) == 0)
                check('base residual '+arm+label, float(np.max(abs(baseline['H'] @ vector - row['energy_Ha']*vector))) <= 1e-8)
                base_vectors[label], sectors[label] = vector[ix].copy(), ix
            del baseline
            gradients = read(OLD / 'actual' / (arm+'_R0_state_gradients.json'))
            values = {}
            for tag in ('plus_h', 'minus_h', 'plus_half', 'minus_half'):
                name = arm+'_'+tag
                arrays, meta = archive(name)
                check('same complete basis '+name, meta['owner']['CSFs'] == owner['CSFs'])
                check('same operator '+name, meta['owner']['hamiltonian_terms'] == ['dirac-coulomb'])
                values[tag], points[name] = {}, {}
                for row in meta['rows']:
                    label = row['label']
                    ix = sectors[label]
                    vector = arrays['mixing'][:,row['column']]
                    other = np.ones(len(vector),bool)
                    other[ix] = False
                    check('exact support '+name+label, np.count_nonzero(vector[other]) == 0)
                    check('selected energy bytes '+name+label, arrays['energies'][row['column']] == row['energy_Ha'])
                    residual = float(np.max(abs(arrays['H'] @ vector - row['energy_Ha']*vector)))
                    check('selected residual '+name+label, residual <= 1e-8)
                    block, v = arrays['H'][np.ix_(ix,ix)], vector[ix]
                    trial, norm = exact_quadratic(block, v, reference)
                    fixed, _ = exact_quadratic(block, base_vectors[label], reference)
                    calls['quadratic_calls'] += 2
                    calls['triangle_entries_upper'] += len(ix)*(len(ix)+1)
                    check('exact norm '+name+label, abs(float(norm)-1) <= 1e-10)
                    signed_overlap = float(v @ base_vectors[label])
                    check('same-basis overlap '+name+label, abs(signed_overlap) > 0.9)
                    reported = Fraction.from_float(row['energy_Ha']) - reference
                    values[tag][label] = dict(reported=reported, rayleigh=trial, fixed=fixed)
                    points[name][label] = dict(centered_Rayleigh_Ha=float(trial),
                        centered_fixed_C0_Ha=float(fixed), reported_minus_Rayleigh_Ha=float(reported-trial),
                        exact_norm=[str(norm.numerator),str(norm.denominator)], residual_max_Ha=residual,
                        same_basis_signed_overlap=signed_overlap, block_dimension=len(ix),
                        centered_Rayleigh_fraction=[str(trial.numerator),str(trial.denominator)],
                        centered_fixed_C0_fraction=[str(fixed.numerator),str(fixed.denominator)])
                del arrays
                save(out/(name+'.json'),points[name])
            for plus, minus, step in (('plus_h','minus_h',0.002), ('plus_half','minus_half',0.001)):
                state_slopes = {}
                for label, _uid in old['UIDs']:
                    slopes = {k:(values[plus][label][k]-values[minus][label][k])/(2*Fraction.from_float(step))
                              for k in ('reported','rayleigh','fixed')}
                    state_slopes[label] = slopes
                ground = old['UIDs'][0][0]
                for kind in ('state','line'):
                    for label, uid in old['UIDs']:
                        if kind == 'line' and label == ground:
                            continue
                        slopes = state_slopes[label].copy()
                        analytic_float = gradients['states'][label]['action_Ha']
                        if kind == 'line':
                            slopes = {k:v-state_slopes[ground][k] for k,v in slopes.items()}
                            analytic_float -= gradients['states'][ground]['action_Ha']
                        analytic = Fraction.from_float(analytic_float)
                        root = slopes['reported']-slopes['rayleigh']
                        response = slopes['rayleigh']-slopes['fixed']
                        operator = slopes['fixed']-analytic
                        check('exact decomposition '+arm+str(step)+kind+label,
                              root+response+operator == slopes['reported']-analytic)
                        saved = next(r for r in old_result['finite_directional_response'] if
                                     r['arm']==arm and r['step']==step and r['kind']==kind and r['label']==label)
                        check('original ordinary replay '+arm+str(step)+kind+label,
                              abs(float(slopes['reported'])-saved['central_Ha']) <= 1e-15)
                        tolerance = max(1e-8,1e-5*abs(analytic_float))
                        check('original classification '+arm+str(step)+kind+label,
                              (abs(float(slopes['reported']-analytic)) <= tolerance) == saved['PASS'])
                        comparisons.append(dict(arm=arm,step=step,kind=kind,label=label,UID=uid,
                            analytic_Ha=analytic_float,reported_FD_Ha=float(slopes['reported']),
                            exact_Rayleigh_FD_Ha=float(slopes['rayleigh']),fixed_C0_FD_Ha=float(slopes['fixed']),
                            reported_minus_analytic_Ha=float(slopes['reported']-analytic),
                            reported_minus_Rayleigh_Ha=float(root), finite_CI_response_Ha=float(response),
                            fixed_C0_minus_analytic_Ha=float(operator),
                            Rayleigh_minus_analytic_Ha=float(slopes['rayleigh']-analytic),
                            original_tolerance_Ha=tolerance,original_PASS=saved['PASS'],
                            exact_Rayleigh_comparison_PASS=abs(float(slopes['rayleigh']-analytic)) <= tolerance))
        check('exact budget', all(calls[k] == cap[k] for k in ('NPZ_loads','quadratic_calls','triangle_entries_upper')))
        check('no Atomx imports', not any(m=='atomx' or m.startswith('atomx.') for m in sys.modules))
        for path,digest in contract['inputs'].items():
            check('unchanged input '+path, sha(ROOT/path)==digest)
        for name,digest in current['all_atomx_python_sha256'].items():
            if sha(ROOT/name) != digest:
                raise ValueError('current source changed after readback')
        for name,digest in frozen['all_atomx_python_sha256'].items():
            if sha(ROOT/old['source_archive']/name.removeprefix('src/')) != digest:
                raise ValueError('historical source changed after readback')
        report['status'] = 'PASS_exact_saved_H_decomposition_original_failures_preserved'
    except BaseException as exc:
        report['status'] = 'FAILED_DIAGNOSTIC_PRESERVED'
        report['error'] = repr(exc)
        raise
    finally:
        report.update(calls=dict(calls),elapsed_seconds=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save(out/'result.json',report)
        print(json.dumps({k:report[k] for k in ('status','calls','elapsed_seconds','peak_RSS_MiB')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    if args.prepare == args.execute:
        raise SystemExit('choose exactly one --prepare or --execute')
    prepare() if args.prepare else execute()
