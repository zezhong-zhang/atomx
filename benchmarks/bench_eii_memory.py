"""Measure EII peak memory and wall time on identical collision settings."""
import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numba

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization import _eii_dw_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--implementation', type=Path)
    parser.add_argument('--cutoff', type=int, default=24)
    parser.add_argument('--sharing', type=int, default=16)
    parser.add_argument('--radius', type=float, default=200.)
    args = parser.parse_args()
    implementation = _eii_dw_total
    if args.implementation:
        spec = importlib.util.spec_from_file_location('eii_before_streaming', args.implementation)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        implementation = module._eii_dw_total
    numba.set_num_threads(4)
    atom = Atom(2, config='1s1', engine='bspline', nuclear='point',
                corrections='none').solve(tol=1e-10)
    if not atom.converged:
        raise RuntimeError('bound state not converged')
    threshold = -atom.orbital_energies['1s1/2']*HARTREE_EV
    start = time.perf_counter()
    result = implementation(atom.result, [5*threshold], kappa_max=args.cutoff,
        n_outgoing_energies=args.sharing, radial_refinement=4,
        continuum_model=ContinuumModel.shooting(n_grid=3000, r_max=args.radius))
    report = dict(cutoff=args.cutoff, sharing=args.sharing, radius=args.radius,
        seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sigma_cm2=result.sigma_total_cm2.tolist(),
        implementation=str(args.implementation) if args.implementation else 'current',
        implementation_sha256=(hashlib.sha256(args.implementation.read_bytes()).hexdigest()
                               if args.implementation else None),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path('src/atomx').rglob('*.py')})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='source_sha256'}, flush=True)


if __name__ == '__main__':
    main()
