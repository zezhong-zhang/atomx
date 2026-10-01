"""Separate Slater integration-mesh convergence from continuum propagation."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numba
import numpy as np

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization.dw import dw_eii_cross_section


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--z", type=int, nargs="+", default=[2, 82])
    p.add_argument("--refinement", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--cutoff", type=int, default=8)
    p.add_argument("--sharing", type=int, default=8)
    p.add_argument("--factors", type=float, nargs="+", default=[2.0, 5.0])
    p.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Continuum and collision radius; bound orbitals stay fixed",
    )
    p.add_argument("--shooting-points", type=int, default=3000)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    numba.set_num_threads(4)
    report = dict(
        cases=[],
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path("src/atomx").rglob("*.py")
        },
    )
    for z in args.z:
        atom = Atom(
            z, config="1s1", engine="bspline", nuclear="point", corrections="none"
        ).solve(tol=1e-10)
        if not atom.converged:
            raise RuntimeError(f"Z={z} bound state not converged")
        threshold = -atom.orbital_energies["1s1/2"] * HARTREE_EV
        for refinement in args.refinement:
            start = time.perf_counter()
            result = dw_eii_cross_section(
                atom.result,
                np.array(args.factors) * threshold,
                kappa_max=args.cutoff,
                n_outgoing_energies=args.sharing,
                radial_refinement=refinement,
                partial_wave_tail="none",
                continuum_model=ContinuumModel.shooting(
                    n_grid=args.shooting_points, r_max=args.r_max
                ),
            )
            case = dict(
                z=z,
                refinement=refinement,
                threshold_eV=threshold,
                factors=args.factors,
                cutoff=args.cutoff,
                sharing=args.sharing,
                r_max=args.r_max,
                shooting_points=args.shooting_points,
                seconds=time.perf_counter() - start,
                sigma=result.sigma_total_cm2.tolist(),
                quadrature=result.provenance["per_orbital"]["1s1/2"],
            )
            report["cases"].append(case)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print({k: v for k, v in case.items() if k != "quadrature"}, flush=True)


if __name__ == "__main__":
    main()
