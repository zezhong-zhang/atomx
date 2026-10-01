"""Measure current DF solve latency, energies and convergence in one process.

Run with PYTHONPATH=src:. and set OPENBLAS_NUM_THREADS=1 / OMP_NUM_THREADS=1
before starting Python. First solves may load/compile JIT kernels; repetitions
reuse process caches. Constructor and import times are excluded. --profile
adds instrumentation overhead and is for hotspot analysis, not latency claims.
"""
import argparse
import cProfile
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import atomx
from atomx import Atom, Basis, Correlation, RadialRefinement, Sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--z', nargs='+', type=int, default=[10, 6, 98])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--refine', action='store_true',
                        help='check radial grid and automatic box convergence')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(atomx.__file__).resolve().parent
    report = dict(python=platform.python_version(), source=str(source),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(source.rglob('*.py'))},
        threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        settings=dict(engine='shooting', exchange='hf', averaging='hamiltonian',
                      corrections='default Breit + QED', tol=1e-5, max_iter=40),
        timing_scope='Atom.solve only; import and constructor excluded',
        profiled=args.profile, radial_refinement=args.refine, cases=[])
    for z in args.z:
        for repeat in range(args.repeats):
            atom = Atom(z, engine='shooting', exchange='hf',
                        correlation=Correlation(averaging='hamiltonian'),
                        basis=Basis(sampling=Sampling(radial_refinement=RadialRefinement()))
                              if args.refine else None)
            profile = cProfile.Profile() if args.profile else None
            start = time.perf_counter()
            if profile:
                profile.enable()
            atom.solve()
            if profile:
                profile.disable()
            seconds = time.perf_counter()-start
            result = atom.result
            row = dict(z=z, repeat=repeat, seconds=seconds,
                energy_Ha=atom.total_energy, converged=bool(atom.converged),
                iterations=result.iterations, result=type(result).__name__,
                orbital_energies_Ha=atom.orbital_energies,
                radial_convergence=result.radial_convergence)
            report['cases'].append(row)
            (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            if profile:
                profile.dump_stats(str(args.out/f'Z{z}-{repeat}.prof'))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
