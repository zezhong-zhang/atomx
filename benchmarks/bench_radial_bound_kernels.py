"""Time matched frozen-field DBOUND solves; compare native and Salvat RADIAL.

Both codes receive the same potential samples, output grid, initial energy,
series tolerance and speed of light. Potential preparation and process/JIT
startup are excluded from kernel timings. This is not a nonlocal HF benchmark.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import atomx
from atomx.constants import C
from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import vint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--radial-source', type=Path)
    parser.add_argument('--repeat', type=int, default=50)
    args = parser.parse_args()
    if args.repeat < 1 or bool(args.probe) != bool(args.radial_source):
        parser.error('repeat must be positive; supply both probe and radial-source')
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(atomx.__file__).resolve().parent
    report = dict(c=C, repetitions=args.repeat, tolerance=1e-11,
        source_sha256={str(p.relative_to(source)):sha(p)
                       for p in (source/'numerics/shooting').glob('*.py')},
        runner_sha256=sha(__file__),
        reference_sha256=({str(p):sha(p) for p in (args.probe, args.radial_source)}
                          if args.probe else {}), cases=[])
    r, _, count = sgrid(75., 1e-7, .5, 1000)
    states = [(1,-1),(2,-1),(2,1),(2,-2),(3,-1),(3,1),(3,-2),(3,2),(3,-3)]
    for z, tail, scale in [(2.,2.,1.),(26.,17.,.3),(98.,89.,.1)]:
        rv = -(tail+(z-tail)*np.exp(-r/scale))
        prep = vint(r, rv, count)
        for n, kappa in states:
            guess = -z*z/(2*n*n)
            arguments = (guess, 1e-11, n, kappa, *prep, r, count)
            identity = (z, n, kappa)
            def solve(arguments=arguments, identity=identity):
                result = dbound(*arguments)
                if result[1] != 0:
                    raise RuntimeError(f'native DBOUND failed: state={identity}, ier={result[1]}')
                return result
            first = solve()  # Exclude compilation and first use.
            times = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(args.repeat):
                    actual = solve()
                times.append((time.perf_counter()-start)/args.repeat)
            np.testing.assert_array_equal(first[2], actual[2])
            np.testing.assert_array_equal(first[3], actual[3])
            row = dict(z=z, tail=tail, scale=scale, n=n, kappa=kappa,
                energy_Ha=float(actual[0]), native_times_s=times,
                median_native_s=float(np.median(times)),
                wave_sha256=hashlib.sha256(np.array(actual[2:4]).tobytes()).hexdigest())
            if args.probe:
                folder = args.out/f'Z{int(z)}-n{n}-k{kappa}'
                folder.mkdir()
                inputs = f'{count} {count} {guess} {n} {kappa} 1e-11 {args.repeat}\n'
                inputs += ''.join(f'{a:.17g} {b:.17g}\n' for a,b in zip(r,rv, strict=True))
                inputs += ''.join(f'{x:.17g}\n' for x in r)
                (folder/'input.txt').write_text(inputs)
                proc = subprocess.run([str(args.probe.resolve())], input=inputs,
                    capture_output=True, text=True, timeout=120)
                (folder/'stdout.txt').write_text(proc.stdout)
                (folder/'stderr.txt').write_text(proc.stderr)
                proc.check_returncode()
                header = next(line for line in proc.stdout.splitlines() if line.startswith('#')).split()
                if int(header[1]):
                    raise RuntimeError('RADIAL DBOUND reported a failed solve')
                reference = np.array([[float(x) for x in line.split()]
                    for line in proc.stdout.splitlines() if not line.startswith('#')])
                np.testing.assert_allclose(reference[:,0], r, rtol=2e-15, atol=0)
                expected = reference[:,1:].T
                waves = np.array(actual[2:4])
                sign = 1 if np.vdot(waves, expected) >= 0 else -1
                row.update(reference_energy_Ha=float(header[-2]),
                    energy_difference_Ha=float(actual[0]-float(header[-2])),
                    relative_wave_error=float(np.max(abs(waves-sign*expected))/np.max(abs(expected))),
                    radial_cpu_s=float(header[-1]),
                    raw_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
            report['cases'].append(row)
            (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Cases:',len(report['cases']), flush=True)
    print('Median kernel seconds:',float(np.median([c['median_native_s'] for c in report['cases']])), flush=True)


if __name__ == '__main__':
    main()
