"""Diagnose full-interval EII quadrature independently of its production rule."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from atomx import Atom
from atomx.constants import BOHR_RADIUS_M, C_SQ, HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization import _eii_dw_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--z', type=int, nargs='+', default=[2, 82])
    parser.add_argument('--nodes', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--mappings', nargs='+', choices=['linear', 'square', 'sinh'],
                        default=['linear', 'square'])
    parser.add_argument('--factors', type=float, nargs='+', default=[2.])
    parser.add_argument('--kappa-max', type=int, default=10)
    args = parser.parse_args()
    report = dict(scope='Gauss-Legendre full half-sharing interval; fixed partial waves',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path('src/atomx').rglob('*.py')}, cases=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for z in args.z:
        atom = Atom(z, config='1s1', engine='bspline', nuclear='point',
                    corrections='none').solve(tol=1e-10)
        if not atom.converged:
            raise RuntimeError('bound state not converged')
        threshold = -atom.orbital_energies['1s1/2']
        for mapping in args.mappings:
            for n in args.nodes:
                x, w = np.polynomial.legendre.leggauss(n)
                t, w = (x+1)/2, w/2
                for factor in args.factors:
                    if mapping == 'sinh':
                        extent = np.arcsinh(np.sqrt((factor-1)/2))
                        fractions = np.sinh(extent*t)**2/(factor-1)
                        weights = extent*np.sinh(2*extent*t)*w/(factor-1)
                    else:
                        power = 1 if mapping == 'linear' else 2
                        fractions = .5*t**power
                        weights = .5*power*t**(power-1)*w
                    capture = {}
                    start = time.perf_counter()
                    result = _eii_dw_total(atom.result,
                        [factor*threshold*HARTREE_EV],
                        kappa_max=args.kappa_max, n_outgoing_energies=n,
                        partial_wave_tail='none',
                        continuum_model=ContinuumModel.shooting(n_grid=3000),
                        e1_grid_frac=fractions, _capture=capture)
                    seconds = time.perf_counter()-start
                    incident = float(result.energies_eV[0])
                    grid, integrand, available = capture['1s1/2', incident]
                    energy = incident/HARTREE_EV
                    p2 = 2*energy+energy**2/C_SQ
                    # One electron in j=1/2; half the sharing domain counts
                    # each antisymmetrized outgoing pair once.
                    prefactor = np.pi/p2*.5*BOHR_RADIUS_M**2*1e4
                    sigma = prefactor*available*np.dot(weights, integrand)
                    # The production result still integrates the custom nodes
                    # by trapezoids. Keep both rules to expose that difference.
                    case = dict(z=z, factor=factor, nodes=n, mapping=mapping, tail='none',
                        kappa_max=args.kappa_max, seconds=seconds,
                        threshold_Ha=threshold, sigma_cm2=float(sigma),
                        trapezoid_sigma_cm2=float(result.sigma_total_cm2[0]),
                        energy_Ha=grid.tolist(), integrand=integrand.tolist(),
                        fractional_weights=weights.tolist())
                    report['cases'].append(case)
                    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                    print({k:v for k,v in case.items() if k not in
                           ('energy_Ha','integrand','fractional_weights')}, flush=True)


if __name__ == '__main__':
    main()
