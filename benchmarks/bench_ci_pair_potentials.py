"""Time native SLAG6 CI radial integrals with and without pair-potential reuse.

The scalar reference uses the same production SLAG6 kernels and source/target
order. Timings include every requested radial integral once per fresh build;
no evaluator-result cache is reused between repetitions. Common FAC spinors
cover the independent CI operator panel. This is a kernel benchmark, not SCF.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from atomx.numerics.shooting.coulomb import _slater_integral_slag6
from atomx.physics.csf.matrix import _radial_evaluator
from benchmarks.bench_reference_ci_operator import common_grid, freeze
from benchmarks.bench_reference_structure import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--points', type=int, default=8000)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root/'tests/data'
    reference = json.loads((data/'fac_ci_operator.json').read_text())
    report = dict(scope=__doc__, points=args.points, repeats=args.repeats, cases=[],
                  source_sha256={str(p.relative_to(root)): sha256(p) for p in (
                      Path(__file__), root/'src/atomx/physics/csf/matrix.py',
                      root/'src/atomx/numerics/shooting/coulomb.py')})
    with np.load(data/'fac_ci_spinors.npz') as arrays:
        for case in reference['cases']:
            sources = {(o['n'], o['kappa']): arrays[o['array']] for o in case['orbitals']}
            waves, ws = common_grid(sources, case['z'], args.points, alpha=case['fac_alpha'])
            keys = [freeze(row['key']) for row in case['radial'] if row['key'][0] == 'r']
            def scalar():
                values = []
                for _, rank, a, c, b, d in keys:
                    pa, qa = waves[a]; pc, qc = waves[c]
                    pb, qb = waves[b]; pd, qd = waves[d]
                    values.append(_slater_integral_slag6(pa*pc+qa*qc, pb*pd+qb*qd,
                                                         rank, ws.r, ws.difr, ws.n))
                return np.array(values)
            def reused():
                evaluate = _radial_evaluator(waves, ws, case['z'], ws.n)
                return np.array([evaluate(key) for key in keys])
            timings = {name: [] for name in ('scalar', 'reused')}
            for repeat in range(args.repeats+1):
                for name, method in [('scalar', scalar), ('reused', reused)][::1 if repeat % 2 else -1]:
                    start = time.perf_counter()
                    value = method()
                    elapsed = time.perf_counter()-start
                    if repeat:
                        timings[name].append(elapsed)
            error = float(np.max(np.abs(scalar()-reused())))
            if error > 1e-10:
                raise AssertionError(f'pair-potential reuse changed quadrature: {error}')
            medians = {name: statistics.median(values) for name, values in timings.items()}
            row = dict(z=case['z'], sequence=case['sequence'], radial_integrals=len(keys),
                unique_source_potentials=len({(key[1], tuple(sorted(key[2:4]))) for key in keys}),
                max_difference_Ha=error, timing_samples_s=timings, medians_s=medians,
                speedup=medians['scalar']/medians['reused'])
            report['cases'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'timing_samples_s'}), flush=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
