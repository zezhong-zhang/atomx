"""Warm radial operator latency with source fingerprints and saved outputs.

Select the implementation through PYTHONPATH; this driver does not modify it.
Run before/after in separate processes with one BLAS/OMP thread. Preparation
and compilation are excluded from the batched timings. End-to-end performance
must be measured separately with bench_df_solver.py.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np

import atomx
from atomx.numerics.shooting.dirac_green import scaled_green_solve
from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.operators import _ds_derivative, one_body_element


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--points', type=int, nargs='+', default=[1000, 4000, 12000])
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--batch', type=int, default=100)
    args = parser.parse_args()
    if min(args.points) < 9 or min(args.samples, args.batch) < 1:
        parser.error('points must be >= 9 and samples/batch positive')
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(atomx.__file__).resolve().parent
    report = dict(python=platform.python_version(), source=str(source),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(source.rglob('*.py'))},
        threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        timing_scope='warm operator action; preparation and compilation excluded',
        batch=args.batch, samples=args.samples, measurements={})
    outputs = {}

    def measure(name, operation):
        outputs[name] = np.asarray(operation())
        seconds = []
        for _ in range(args.samples):
            start = time.perf_counter()
            for _ in range(args.batch):
                operation()
            seconds.append((time.perf_counter()-start)/args.batch)
        report['measurements'][name] = dict(seconds=seconds)

    for points in args.points:
        r, jacobian, n = sgrid(30., 1e-7, .1, points)
        p, q = r*np.exp(-r), .01*r*r*np.exp(-r)
        pb, qb = r*(1-r)*np.exp(-r/2), -.02*r*np.exp(-r/2)
        rv = np.full(n, -2.)
        response = DiracResponse(-5., -1, r, jacobian, rv, boundary='box')
        source_term = np.array([p, q])
        measure(f'derivative_{n}', partial(_ds_derivative, p))
        measure(f'diagonal_one_body_{n}',
                partial(one_body_element, p, q, p, q, -1, r, jacobian, rv))
        measure(f'offdiagonal_one_body_{n}',
                partial(one_body_element, p, q, pb, qb, -1, r, jacobian, rv))
        measure(f'green_action_{n}', partial(scaled_green_solve,
            *source_term, response.scaled_pair, jacobian, response.scaled_quadrature))
    np.savez(args.out/'outputs.npz', **outputs)
    report['outputs_sha256'] = hashlib.sha256((args.out/'outputs.npz').read_bytes()).hexdigest()
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
