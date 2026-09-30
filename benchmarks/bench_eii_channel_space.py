"""He+ EII in the finite angular space of the explicit FAC comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numba
from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization import _eii_dw_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=200.)
    parser.add_argument('--refinement', type=int, default=4)
    parser.add_argument('--sharing', type=int, default=32)
    args = parser.parse_args()
    numba.set_num_threads(4)
    atom = Atom(2, config='1s1', nuclear='point', engine='bspline',
                corrections='none').solve(tol=1e-10)
    if not atom.converged:
        raise RuntimeError('bound state not converged')
    threshold = -atom.orbital_energies['1s1/2']*HARTREE_EV
    # FAC: incident l<=36, slow l<=8, transfer rank<=9. Its fast j loop
    # permits l through 36+9+1=46. For a 1s target rank 9 includes the
    # entire recoupled slow-electron space. Keep both antisymmetric terms.
    limits = (36, 8, 46)
    start = time.perf_counter()
    result = _eii_dw_total(atom.result, [5*threshold], kappa_max=max(limits),
        _channel_limits=limits, n_outgoing_energies=args.sharing,
        radial_refinement=args.refinement,
        continuum_model=ContinuumModel.shooting(n_grid=3000, r_max=args.radius))
    report = dict(threshold_eV=threshold, incident_factor=5., channel_l_limits=limits,
        sharing=args.sharing, refinement=args.refinement, radius=args.radius,
        seconds=time.perf_counter()-start, sigma_cm2=result.sigma_total_cm2.tolist(),
        per_orbital=result.provenance['per_orbital'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path('src/atomx').rglob('*.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['source_sha256','per_orbital']}, flush=True)


if __name__ == '__main__':
    main()
