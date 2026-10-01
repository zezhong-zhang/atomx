"""Live RADIAL DCOULF check of continuum normalization and Coulomb references.

The probe reads signed potential strength, E, kappa, r and writes those inputs,
FU, FL, GU, GL, error_F, error_G, delta. Build it against the supplied RADIAL
source; --reference-c records the constant in that build. The integrated-wave
comparison requires the same speed of light as Atomx.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from atomx.constants import C
from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_shooting
from atomx.numerics.shooting.dirac_coulomb import dirac_coulomb_outgoing


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(probe, cases, out, name, *, allow_reference_failures=False):
    inputs = ''.join(f"{-v['z']:.17g} {v['e']:.17g} {v['k']} {v['r']:.17g}\n" for v in cases)
    (out/f'{name}.in').write_text(inputs)
    start = time.perf_counter()
    process = subprocess.run([str(probe.resolve())], input=inputs, text=True,
                             capture_output=True, check=True, timeout=120)
    seconds = time.perf_counter()-start
    (out/f'{name}.out').write_text(process.stdout)
    (out/f'{name}.stderr').write_text(process.stderr)
    data = np.array([[float(x) for x in line.split()]
                     for line in process.stdout.splitlines()])
    if data.shape != (len(cases), 11):
        raise RuntimeError('RADIAL returned incomplete output')
    expected = [[-v['z'], v['e'], v['k'], v['r']] for v in cases]
    np.testing.assert_allclose(data[:, :4], expected, rtol=2e-15, atol=0)
    if not allow_reference_failures and (not np.all(np.isfinite(data))
                                         or np.max(data[:, 8:10]) > 1e-8):
        raise RuntimeError('RADIAL reference error exceeds comparison budget')
    return data, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--radial-source', type=Path, required=True)
    parser.add_argument('--reference-c', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=False, parents=True)
    report = dict(reference_c=args.reference_c, atomx_c=C,
                  probe_sha256=sha(args.probe), radial_source_sha256=sha(args.radial_source),
                  runner_sha256=sha(__file__), platform=platform.platform(),
                  threads={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                  source_sha256={str(p): sha(p) for p in Path('src/atomx/numerics/shooting').glob('*.py')})
    cases = [dict(z=z, e=e, k=k, r=r)
             for z in [0., 1., 17., 73.] for e in [.001, .1, 10., 150., 7500.]
             for k in [-1, 1, -3, 5] for r in [5., 75.]]
    ref, seconds = reference(args.probe, cases, out, 'reference')
    start = time.perf_counter()
    for case, row in zip(cases, ref, strict=True):
        actual = dirac_coulomb_outgoing(case['e'], case['k'], case['z'], case['r'], c=args.reference_c)
        expected = row[6:8]+1j*row[4:6]
        case['reference'] = row[4:8].tolist()
        case['reference_error'] = row[8:10].tolist()
        case['relative_error'] = float(np.max(abs(actual-expected))/np.max(abs(expected)))
        determinant = actual[0].imag*actual[1].real-actual[1].imag*actual[0].real
        target = -np.sqrt(case['e']/(case['e']+2*args.reference_c**2))
        case['determinant_relative_error'] = float(abs(determinant/target-1))
    report['reference'] = dict(cases=cases, atomx_seconds=time.perf_counter()-start,
                               radial_process_seconds=seconds)
    if args.reference_c == C:
        rows, requests = [], []
        # Warm the JIT independently of the measured solves.
        solve_dirac_continuum_shooting(1., -1, lambda r: -1/r, n_grid=4000)
        for z in [0., 1., 17., 73.]:
            for e in [.001, .1, 10., 7500.]:
                for k in [-1, 1, -3]:
                    for box in [5., 75.]:
                        start = time.perf_counter()
                        r, p, q, phase = solve_dirac_continuum_shooting(
                            e, k, lambda r, *, z=z: -z/r, r_max=box, n_grid=4000, Z_residual=z)
                        seconds = time.perf_counter()-start
                        ix = [int(np.argmin(abs(r-x))) for x in [.01, .1, 1., .75*box, box]]
                        rows.append(dict(z=z, e=e, k=k, box=box, phase=phase, seconds=seconds,
                                         r=r[ix].tolist(), p=p[ix].tolist(), q=q[ix].tolist()))
                        requests.extend(dict(z=z, e=e, k=k, r=float(r[i])) for i in ix)
        ref, seconds = reference(args.probe, requests, out, 'integrated',
                                 allow_reference_failures=True)
        valid = np.all(np.isfinite(ref), axis=1) & (np.max(ref[:, 8:10], axis=1) <= 1e-8)
        rejected = [dict(**request, reason='DCOULF nonfinite value or reported error > 1e-8')
                    for request, ok in zip(requests, valid, strict=True) if not ok]
        for index, row in enumerate(rows):
            target = ref[index*5:index*5+5, 4:6].T
            accepted = valid[index*5:index*5+5]
            if not np.any(accepted):
                raise RuntimeError('No valid RADIAL reference points for continuum case')
            e = row['e']
            norm = np.sqrt((2+e/C**2)/(np.pi*np.sqrt(e*(2+e/C**2))))
            actual = np.array([row['p'], row['q']])/norm
            sign = 1 if np.vdot(actual[:, accepted], target[:, accepted]) >= 0 else -1
            row['reference_regular'] = [[float(v) if ok else None for v, ok in zip(component, accepted, strict=False)]
                                        for component in target]
            row['reference_valid'] = accepted.tolist()
            row['error_over_asymptotic_amplitude'] = float(np.max(abs(actual[:, accepted]-sign*target[:, accepted])))
            row['phase_error_mod_pi'] = float(abs(np.arctan(np.tan(row['phase']))))
        report['integrated'] = dict(cases=rows, radial_process_seconds=seconds, n_grid=4000,
                                    reference_points_accepted=int(valid.sum()),
                                    reference_points_rejected=rejected)
    else:
        report['integrated'] = dict(skipped='Matching constants required for integrated-wave check')
    (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Reference max relative error:', max(v['relative_error'] for v in cases))
    if 'cases' in report['integrated']:
        print('Integrated max absolute error / asymptotic amplitude:',
              max(v['error_over_asymptotic_amplitude'] for v in rows))
        print('Integrated max phase error modulo pi:', max(v['phase_error_mod_pi'] for v in rows))
        print('RADIAL points accepted/rejected:', int(valid.sum()), len(rejected))


if __name__ == '__main__':
    main()
