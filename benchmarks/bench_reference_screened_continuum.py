"""Live screened-field RADIAL DFREE comparison, including dense wave sampling.

The reference probe reads NV, NOUTPUT, E, kappa, tolerance, followed by NV
(r,rV) pairs and NOUTPUT requested radii. It prints a header with IER, ILAST,
phase, CPU seconds, then (r,P,Q) at each requested radius. Its only RADIAL
source change must be the speed-of-light constant matched to Atomx.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from atomx.constants import C
from atomx.numerics.shooting.dirac_continuum import (
    _prepare_potential, solve_dirac_continuum_shooting)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--radial-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--r-max', type=float, default=50.)
    parser.add_argument('--n-grid', type=int, default=1000)
    parser.add_argument('--n-output', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or args.n_output < 2:
        parser.error('repeat >= 1 and n-output >= 2 required')
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    report = dict(c=C, r_max=args.r_max, n_grid=args.n_grid, n_output=args.n_output,
        probe_sha256=sha(args.probe), radial_source_sha256=sha(args.radial_source),
        runner_sha256=sha(__file__), source_sha256={str(p):sha(p) for p in
        Path('src/atomx/numerics/shooting').glob('*.py')}, cases=[])
    target = np.r_[0., np.geomspace(1e-7, args.r_max, args.n_output-1)]
    for z, tail, scale in [(10., 0., 1.), (26., 17., .3), (82., 73., .1)]:
        def potential(r):
            return -(tail+(z-tail)*np.exp(-r/scale))/r
        prep = _prepare_potential(potential, args.r_max, args.n_grid, tail)
        source_r, source_rv = prep[0], prep[-1]
        for energy in [.01, 10., 150.]:
            for kappa in [-1, 1, -3, 5]:
                folder = out/f'Z{int(z):03d}-E{energy}-k{kappa}'
                folder.mkdir()
                durations = []
                for _ in range(args.repeat+1):
                    start = time.perf_counter()
                    r, p, q, phase = solve_dirac_continuum_shooting(
                        energy, kappa, potential, r_max=args.r_max, n_grid=args.n_grid,
                        Z_residual=tail, interp_grid=target)
                    durations.append(time.perf_counter()-start)
                inputs = f'{len(source_r)} {len(target)} {energy} {kappa} 1e-10\n'
                inputs += ''.join(f'{a:.17g} {b:.17g}\n' for a,b in zip(source_r, source_rv))
                inputs += ''.join(f'{x:.17g}\n' for x in target)
                (folder/'input.txt').write_text(inputs)
                start = time.perf_counter()
                process = subprocess.run([str(args.probe.resolve())], input=inputs,
                                         capture_output=True, text=True, timeout=120)
                process_seconds = time.perf_counter()-start
                (folder/'stdout.txt').write_text(process.stdout)
                (folder/'stderr.txt').write_text(process.stderr)
                process.check_returncode()
                header = next(line for line in process.stdout.splitlines() if line.startswith('#')).split()
                if int(header[1]) != 0:
                    raise RuntimeError('RADIAL DFREE reported nonzero IER')
                ref = np.array([[float(x) for x in line.split()] for line in process.stdout.splitlines()
                                if not line.startswith('#')])
                if ref.shape != (len(target), 3) or not np.all(np.isfinite(ref)):
                    raise RuntimeError('incomplete or nonfinite DFREE reference')
                np.testing.assert_allclose(ref[:,0], target, rtol=2e-15, atol=0)
                amplitude = np.sqrt((2+energy/C**2)/(np.pi*np.sqrt(energy*(2+energy/C**2))))
                actual, expected = np.array([p,q])/amplitude, ref[:,1:].T
                sign = 1 if np.vdot(actual, expected) >= 0 else -1
                row = dict(z=z, tail=tail, scale=scale, energy=energy, kappa=kappa,
                    first_native_seconds=durations[0], repeated_native_seconds=durations[1:],
                    median_native_seconds=float(np.median(durations[1:])),
                    radial_cpu_seconds=float(header[-1]), radial_process_seconds=process_seconds,
                    amplitude_error=float(np.max(abs(actual-sign*expected))),
                    phase_error=float(abs(np.arctan(np.tan(phase-float(header[-2]))))),
                    native_phase=phase, reference_phase=float(header[-2]),
                    raw_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
                report['cases'].append(row)
                (out/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Cases:', len(report['cases']))
    print('Max wave error / asymptotic amplitude:', max(r['amplitude_error'] for r in report['cases']))
    print('Max phase error modulo pi:', max(r['phase_error'] for r in report['cases']))
    print('Native median seconds:', float(np.median([r['median_native_seconds'] for r in report['cases']])))


if __name__ == '__main__':
    main()
