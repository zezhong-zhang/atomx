"""Separate source-grid and energy-sharing errors in the live Coulomb panel."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization.dw import dw_eii_cross_section


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--z', type=int, nargs='+', default=[2, 82])
    parser.add_argument('--source-points', type=int, nargs='+', default=[3000, 6000])
    parser.add_argument('--sharing-points', type=int, nargs='+', default=[25, 49, 97])
    parser.add_argument('--kappa-max', type=int, nargs='+', default=[10])
    parser.add_argument('--factors', type=float, nargs='+')
    parser.add_argument('--tail', choices=['none', 'geometric'], default='none')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    report = dict(reference=str(args.reference),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path('src/atomx').rglob('*.py')}, cases=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for z in args.z:
        atom = Atom(z, config='1s1', engine='bspline', nuclear='point',
                    corrections='none').solve(tol=1e-10)
        if not atom.converged:
            raise RuntimeError(f'Z={z} bound state not converged')
        ref = next(c for c in reference['cases'] if c['z'] == z)
        threshold = -atom.orbital_energies['1s1/2'] * HARTREE_EV
        np.testing.assert_allclose(threshold, ref['threshold_eV'], rtol=1e-12)
        # Vary one control at a time around the first requested setting.
        baseline = (args.source_points[0], args.sharing_points[0], args.kappa_max[0])
        settings = dict.fromkeys(
            [(g, baseline[1], baseline[2]) for g in args.source_points] +
            [(baseline[0], n, baseline[2]) for n in args.sharing_points] +
            [(baseline[0], baseline[1], k) for k in args.kappa_max])
        for grid, sharing, kappa in settings:
            start = time.perf_counter()
            factors = args.factors or ref['factors']
            result = dw_eii_cross_section(atom.result, np.array(factors) * threshold,
                kappa_max=kappa, n_outgoing_energies=sharing,
                partial_wave_tail=args.tail,
                continuum_model=ContinuumModel.shooting(n_grid=grid))
            sigma = result.sigma_total_cm2
            case = dict(z=z, source_points=grid, sharing_points=sharing, kappa_max=kappa,
                        factors=factors, tail=args.tail, seconds=time.perf_counter()-start,
                        sigma_cm2=sigma.tolist(),
                        partial_waves=result.provenance['per_orbital']['1s1/2']['partial_wave_diagnostics'])
            if factors == ref['factors']:
                case['ratios_fac'] = (sigma/ref['fac_sigma_cm2']).tolist()
                case['ratios_baseline'] = (sigma/ref['atomx_sigma_cm2']).tolist()
            report['cases'].append(case)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(case, flush=True)


if __name__ == '__main__':
    main()
