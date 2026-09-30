"""Replay shooting energies, response, photon and collision observables.

Run in fresh processes with PYTHONPATH selecting the before/after source tree.
The small CE/EII partial-wave spaces are regression controls, not converged
cross-section predictions. The collision/PI/RR APIs currently require
coefficient-based bound states, so these explicitly use B-spline bound states
and shooting continua. Native shooting bound states are tested separately.
No external-reference accuracy is inferred here.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import atomx
from atomx import Atom, Correlation
from atomx.constants import HARTREE_EV
from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_shooting
from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.exponential_quadrature import exponential_prefix_weights
from atomx.numerics.shooting.grid import sgrid
from atomx.scattering import ContinuumModel
from atomx.scattering.distorted_wave.solver import dw_excitation_cross_section
from atomx.scattering.electron_ionization import eii_cross_section
from atomx.scattering.radiative_recombination import (
    _shooting_photoionization_cross_section,
    rr_cross_section,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(atomx.__file__).resolve().parent
    report = dict(python=platform.python_version(), source=str(source),
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(source.rglob('*.py'))},
        threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        settings=dict(engine='shooting', exchange='hf', averaging='hamiltonian',
                      corrections='none', scf_tol=1e-9,
                      scattering_bound_engine='bspline',
                      native_bound_scattering='not supported by these coefficient APIs',
                      collision_kappa_max=3, sharing_nodes=4, radial_refinement=2,
                      ce_topup=False, continuum_radius=75, continuum_points=1200),
        measurements={})
    arrays = {}

    def measure(name, operation, repeats=1):
        times = []
        result = None
        for _ in range(repeats):
            start = time.perf_counter()
            result = operation()
            times.append(time.perf_counter()-start)
        report['measurements'][name] = dict(seconds=times)
        print(name, times, flush=True)
        return result

    # Exclude JIT warmup explicitly from the repeated kernel measurements.
    for slope in (0., .1, 2., 8., 20., 100.):
        logs = slope*np.arange(2000.)
        exponential_prefix_weights(logs)
        measure(f'prefix_slope_{slope}',
                lambda logs=logs: exponential_prefix_weights(logs), repeats=30)

    radius, jacobian, n = sgrid(30., 1e-7, .1, 2000)
    potential = np.full(n, -2.)
    source_term = np.array([radius*np.exp(-radius), .01*radius**2*np.exp(-radius)])
    def response():
        return DiracResponse(-5., -1, radius, jacobian, potential,
                             boundary='box').apply(source_term)
    response()
    arrays['response'] = measure('response', response, repeats=20)

    def solve(z, config, engine='shooting'):
        atom = Atom(z, config=config, engine=engine, exchange='hf',
                    corrections='none', correlation=Correlation(averaging='hamiltonian'))
        atom.solve(tol=1e-9, max_iter=150)
        if not atom.converged:
            raise RuntimeError(f'nonconverged {z=} {config=}')
        report['measurements'][f'bound_{z}_{config}'] = dict(
            energy_Ha=atom.total_energy, orbital_energies_Ha=atom.orbital_energies,
            iterations=atom.result.iterations, result=type(atom.result).__name__)
        for label, orbital in atom.orbitals.items():
            key = f'bound_{z}_{config}_{label}'
            arrays[key] = np.array([orbital.P, orbital.Q])
        return atom

    neon = measure('neon_solve', lambda: solve(10, '1s2 2s2 2p6'))
    lines = measure('neon_transitions', lambda: neon.transitions(
        multipoles=['E1', 'M1', 'E2'], gauge='both'))
    report['transitions'] = [asdict(line) for line in lines]
    helium = measure('helium_solve', lambda: solve(2, '1s2', 'bspline'))
    excited = measure('helium_excited_solve', lambda: solve(2, '1s1 2p1', 'bspline'))
    energies = np.array([100., 200.])
    arrays['ce_sigma_cm2'], arrays['ce_omega'] = measure('ce', lambda:
        dw_excitation_cross_section(helium, excited, energies, L_max=2,
            kappa_max=3, shooting_n_grid=1200, radial_refinement=2, topup=False))
    model = ContinuumModel.shooting(n_grid=1200, r_max=75.)
    eii = measure('eii', lambda: eii_cross_section(helium.result, energies,
        method='DW', kappa_max=3, n_outgoing_energies=4,
        continuum_model=model, radial_refinement=2))
    arrays['eii_sigma_cm2'] = eii.sigma_total_cm2
    rr_energies = np.array([1., 10., 100.])
    rr = measure('rr', lambda: rr_cross_section(helium.result, rr_energies,
        capture_orbitals=['1s1/2'], continuum_model=model, multipoles=['E1']))
    arrays['rr_sigma_a0_sq'] = rr.sigma_total_a0_sq
    sh = helium.result.subshells[0]
    p, q, _ = helium.result.orbitals_raw[sh.kappa][0]
    arrays['pi_sigma_a0_sq'] = measure('pi', lambda:
        _shooting_photoionization_cross_section(helium.result, sh, p, q,
            rr_energies/HARTREE_EV-helium.orbital_energies[sh.label], model))

    def field(r):
        return -(2.+8.*np.exp(-r))/r
    for energy in (.01, 10., 150.):
        for kappa in (-1, 1, -3, 5):
            name = f'continuum_{energy}_{kappa}'
            r, p, q, phase = measure(name, lambda energy=energy, kappa=kappa:
                solve_dirac_continuum_shooting(energy, kappa, field,
                    Z_residual=2., n_grid=3000, r_max=200.))
            arrays[name] = np.array([r, p, q])
            report['measurements'][name]['phase_rad'] = phase
    np.savez(args.out/'arrays.npz', **arrays)
    report['arrays_sha256'] = hashlib.sha256((args.out/'arrays.npz').read_bytes()).hexdigest()
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
