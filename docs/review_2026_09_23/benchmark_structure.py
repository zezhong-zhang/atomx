"""Measure initializer and correlated optimization costs in a fresh process.

Run with BLAS/OpenMP threads fixed to one for comparable local timings.
This is a same-machine diagnostic, not a cross-hardware performance claim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from atomx import Atom, Basis  # noqa: E402
from atomx.methods.mcdhf import variational  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Z', type=int, default=82)
    parser.add_argument('--ns', type=int)
    args = parser.parse_args()
    timing = {'Z': args.Z, 'requested_ns': args.ns}
    original = variational.optimize_orbitals
    started = time.perf_counter()

    def measured(functional, orbitals, **kwargs):
        timing['initializer_seconds'] = time.perf_counter() - started
        timing['ns'] = functional.ns
        optimized = time.perf_counter()
        result = original(functional, orbitals, **kwargs)
        timing['optimizer_seconds'] = time.perf_counter() - optimized
        timing['stationarity'] = {k: v for k, v in result[2].items()
                                   if k != 'energy_history'}
        return result

    variational.optimize_orbitals = measured
    try:
        kwargs = {} if args.ns is None else {'basis': Basis(ns=args.ns)}
        atom = Atom(args.Z, engine='bspline', corrections='none', **kwargs).solve()
    finally:
        variational.optimize_orbitals = original
    timing.update(total_seconds=time.perf_counter() - started,
                  energy=float(atom.total_energy), converged=bool(atom.converged))
    print(json.dumps(timing, indent=2))


if __name__ == '__main__':
    main()
