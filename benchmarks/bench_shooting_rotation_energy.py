"""Measure fixed-state rotation bookkeeping separately from complete DF solves.

Use PYTHONPATH to select baseline/candidate source in fresh processes. The
baseline interface builds a context per pair; the candidate builds immutable
metadata once per sweep. Both keep radial scratch private to each pair. The
compatibility dispatch here exists only to replay the archived baseline.
Preparation is measured separately; JIT warmup and Atom.solve are excluded.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np

import atomx
from atomx import Atom
from atomx.config.jj_coupling import generate_jj_average
from atomx.physics.csf.matrix import GridCIWorkspace
from atomx.physics.csf.radial_fock import CSFRadialFock
from atomx.physics.hartree_fock import RadialHartreeFock
from atomx.potentials.radial_nucleus import screened_nuclear_rv


def rotation_factory(field, p, q, r, jacobian, rv):
    if hasattr(field, 'rotation_energy_factory'):
        return field.rotation_energy_factory(p, q, r, jacobian, rv)
    return lambda a, b: field.rotation_energy(p, q, r, jacobian, rv, a, b)


def prepare_sweep(make_factory, pairs):
    make_pair = make_factory()
    for a, b, _ in pairs:
        make_pair(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--z', type=int, nargs='+', default=[11, 98])
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--batch', type=int, default=50)
    args = parser.parse_args()
    if min(args.batch, args.samples) < 1:
        parser.error('batch and samples must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(atomx.__file__).resolve().parent
    report = dict(python=platform.python_version(), source=str(source),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(source.rglob('*.py'))},
        threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        batch=args.batch, samples=args.samples,
        settings=dict(engine='shooting', averaging='hamiltonian', corrections='none'),
        timing_scope='warm fixed-state metadata, pair preparation and energy; solve/JIT excluded',
        measurements={}, cases=[])
    outputs = {}

    def measure(name, operation):
        operation()
        seconds = []
        for _ in range(args.samples):
            start = time.perf_counter()
            for _ in range(args.batch):
                operation()
            seconds.append((time.perf_counter()-start)/args.batch)
        report['measurements'][name] = dict(seconds=seconds)

    for z in args.z:
        atom = Atom(z, engine='shooting', corrections='none').solve()
        if not atom.converged:
            raise RuntimeError(f'nonconverged Z={z}')
        result = atom.result
        shells, ensemble = generate_jj_average(atom._config_strings[0])
        r, jacobian = result._grid, result._difr
        rv = screened_nuclear_rv(z, r, jacobian, len(r), nuclear='fermi')
        reference = RadialHartreeFock(deepcopy(shells), ensemble)
        field = reference
        if hasattr(result, 'csfs'):
            field = CSFRadialFock(deepcopy(shells), result.csfs,
                                 GridCIWorkspace(r, jacobian, rv), reference)
            field.set_state(result)
        p, q = np.array([[atom.orbitals[shell.label].P, atom.orbitals[shell.label].Q]
                         for shell in field.subshells]).transpose(1, 0, 2)
        groups = {}
        for a, shell in enumerate(field.subshells):
            groups.setdefault(shell.kappa, []).append(a)
        pairs = tuple(field.rotation_pairs(groups))
        if not pairs:
            raise RuntimeError(f'no occupied rotation pairs for Z={z}')

        make_factory = partial(rotation_factory, field, p, q, r, jacobian, rv)

        # Warm every context before latency measurements. Save the full
        # nine-angle coherent energy probes, including signs/odd harmonics.
        factory = make_factory()
        for pair_index, (a, b, _) in enumerate(pairs):
            energy = factory(a, b)
            ap, aq, bp, bq = (v.copy() for v in (p[a], q[a], p[b], q[b]))
            values = []
            for angle in np.arange(9)*2*np.pi/9:
                c, s = np.cos(angle), np.sin(angle)
                p[a], q[a] = c*ap+s*bp, c*aq+s*bq
                p[b], q[b] = c*bp-s*ap, c*bq-s*aq
                values.append(energy())
            outputs[f'Z{z}_pair{pair_index}'] = np.asarray(values)
            p[a], q[a], p[b], q[b] = ap, aq, bp, bq

        a, b, _ = pairs[0]
        measure(f'Z{z}_sweep_metadata', make_factory)
        measure(f'Z{z}_pair_prepare', partial(factory, a, b))
        energy = factory(a, b)
        measure(f'Z{z}_energy', energy)

        measure(f'Z{z}_sweep_prepare', partial(prepare_sweep, make_factory, pairs))
        report['cases'].append(dict(z=z, points=len(r), orbitals=len(p),
            functional=type(field).__name__, rotation_pairs=len(pairs),
            ordered_coulomb_terms=len(tuple(field.coulomb_energy_terms()))))
    np.savez(args.out/'outputs.npz', **outputs)
    report['outputs_sha256'] = hashlib.sha256((args.out/'outputs.npz').read_bytes()).hexdigest()
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
