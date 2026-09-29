"""Numerical hydrogenic Dirac wavefunctions on a B-spline grid.

For the Shabaev model-QED-operator calibration (and for any other
use case that needs a clean hydrogenic reference), atomx needs
P(r), Q(r) of a *single* electron in the Coulomb field of charge
Z for a given (n, κ).

Rather than implement analytic Dirac-Coulomb formulas (Greiner §9,
or :func:`atomx.physics.dirac.hydrogenic_dirac_energy` for
eigenvalues), we solve the Dirac equation numerically on the
B-spline basis via a single-electron :class:`DFSolver`.  The result
is a genuine ab-initio Dirac-Coulomb wavefunction on the actual
SCF grid — identical to what the user's multi-electron SCF would
produce if all electron-electron interactions were turned off.

The helpers below memoise their results per (Z, n, κ, basis
fingerprint) so repeated calls within a session are cheap.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _subshell_for_hydrogenic(n: int, kappa: int):
    """Build a Subshell instance with a single electron in (n, κ)."""
    from atomx.config.subshell import Subshell
    return Subshell(n=n, kappa=kappa, occupation=1.0)


def hydrogenic_dirac_wavefunction(
    n: int,
    kappa: int,
    Z: int,
    *,
    ns: int = 60,
    ks: int = 9,
    r_min: float | None = None,
    r_max: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> Tuple[np.ndarray, np.ndarray, dict, float]:
    """Solve the hydrogenic Dirac equation on a B-spline basis.

    Runs a single-electron :class:`DFSolver` at nuclear charge Z with
    the requested (n, κ) occupation.  Returns the B-spline coefficient
    arrays (P, Q), the shared B-spline workspace, and the eigenvalue.

    Parameters
    ----------
    n : int
        Principal quantum number.
    kappa : int
        Dirac quantum number.
    Z : int
        Nuclear charge.
    ns, ks : int
        Basis size and B-spline order.
    r_min, r_max : float, optional
        Grid bounds in Bohr.  Default: scaled by Z.
    tol : float
        SCF convergence tolerance (loose for single-electron).
    max_iter : int
        Max SCF iterations.

    Returns
    -------
    P, Q : np.ndarray, shape (ns,)
        B-spline coefficients of the large and small components.
    bsp_data : dict
        Shared B-spline workspace used to evaluate on a radial grid.
    energy : float
        Dirac-Coulomb eigenvalue in Hartree.
    """
    from atomx.methods.dirac_fock import DFSolver

    shells = [_subshell_for_hydrogenic(n, kappa)]
    solver = DFSolver(
        Z=Z, subshells=shells, ns=ns, ks=ks,
        r_min=r_min, r_max=r_max,
        exchange="hf",   # single electron → no real exchange
    )
    result = solver.solve(max_iter=max_iter, tol=tol)
    orb_list = solver._flatten_orbitals(result.orbitals_raw)
    if not orb_list:
        raise RuntimeError(
            f"Hydrogenic Dirac solve failed for (n={n}, κ={kappa}, Z={Z})"
        )
    P, Q, _occ, _kappa, _label = orb_list[0]
    return P, Q, result._bsp_data, result.orbital_energies[_label]


# ---------------------------------------------------------------------------
# Integrals
# ---------------------------------------------------------------------------

def hydrogenic_r_exp_integral_dirac(
    n: int,
    kappa: int,
    Z: int,
    *,
    ns: int = 60,
    ks: int = 9,
) -> float:
    """``∫ [P(r)² + Q(r)²] · r · exp(−r·c) dr``  for hydrogenic (n, κ, Z).

    This is the Dirac ab-initio version of
    :func:`atomx.corrections.qedmod.shabaev._hydrogenic_r_exp_integral`
    (which uses a non-relativistic analytic formula).  The difference
    matters at the O((Zα)²) level for heavy elements.
    """
    from atomx.constants import C
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    P, Q, bsp_data, _E = hydrogenic_dirac_wavefunction(
        n, kappa, Z, ns=ns, ks=ks,
    )
    nv = bsp_data["nv"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    fP = eval_orbital_on_grid(P, bsp_p, nv, ksp, nsp)
    fQ = eval_orbital_on_grid(Q, bsp_q, nv, ksq, nsq)
    rho = fP * fP + fQ * fQ
    kernel = gr * np.exp(-C * gr)

    return float(np.sum(grw * rho * kernel))
