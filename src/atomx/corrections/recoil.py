"""One-body nuclear recoil radial integrals in the Breit approximation.

The four radial terms follow GRASP RIS4 ``rinti_nms.f90`` and
Gaidamauskas et al., J. Phys. B 44, 175003 (2011), arXiv:1106.1988.
They include p²/2 and its relativistic correction, to first order in 1/M.
This point-Coulomb recoil operator omits the two-body specific mass shift,
finite-nuclear-size recoil, and radiative recoil. Using finite-nucleus
orbitals does not supply those omitted recoil operators.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import C
from atomx.corrections.workspace import eval_orbital_pair, is_grid_workspace


def hydrogenic_mass_remainder(Z: int, n: int, mass: float) -> float:
    """Exact nonrelativistic point-Coulomb recoil beyond first order, Ha.

    E_NR(M)=E_inf*M/(M+1). Removing E_inf and its first-order recoil
    -E_inf/M leaves E_inf/[M(M+1)]. This supplies neither relativistic
    higher-order recoil nor finite-size/radiative recoil. It applies only
    to a single electron in a spectroscopically identified n shell.
    """
    import math

    if not isinstance(Z, (int, np.integer)) or Z < 1:
        raise ValueError('Z must be a positive integer')
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('nuclear mass must be positive and finite in electron masses')
    return -0.5*(Z/n)**2/mass/(mass+1)


def _radial_values(P, Q, workspace):
    p, q = eval_orbital_pair(P, Q, workspace)
    if is_grid_workspace(workspace):
        from scipy.interpolate import CubicSpline
        r = workspace["gr"]
        dp = CubicSpline(r, p)(r, 1)
        dq = CubicSpline(r, q)(r, 1)
    else:
        from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
        nv, ns, ks = (workspace[k] for k in ("nv", "ns", "ks"))
        dp = eval_orbital_on_grid(
            P, workspace["bsp_p_d"], nv, workspace.get("ksp", ks),
            workspace.get("nsp", ns))
        dq = eval_orbital_on_grid(
            Q, workspace["bsp_q_d"], nv, workspace.get("ksq", ks),
            workspace.get("nsq", ns))
    return p, q, dp, dq


def recoil_radial_matrix(orbitals, workspace, Z, *, relativistic=True):
    """Return mass-independent K_ab; the one-body shift is Tr(ρ K)/M.

    ``orbitals`` is an ordered sequence of ``(P, Q, kappa)`` tuples.
    Its radial normalization is the solver's normalization; no per-orbital
    renormalization is performed, preserving bilinearity under rotations.
    Both off-diagonal radial integrals and the small component are retained.
    ``relativistic=False`` returns p²/2 alone on the same Dirac spinors.
    """
    r, w = workspace["gr"], workspace["grw"]
    values = [_radial_values(P, Q, workspace) for P, Q, _ in orbitals]
    matrix = np.zeros((len(orbitals), len(orbitals)))
    for a, (_, _, kappa) in enumerate(orbitals):
        l = kappa if kappa > 0 else -kappa - 1
        lt = 2 * abs(kappa) - 1 - l
        pa, qa, dpa, dqa = values[a]
        for b in range(a, len(orbitals)):
            if orbitals[b][2] != kappa:
                continue
            pb, qb, dpb, dqb = values[b]
            integrand = (dpa * dpb + dqa * dqb
                         + (l * (l + 1) * pa * pb
                            + lt * (lt + 1) * qa * qb) / r**2)
            if relativistic:
                integrand = (integrand - 2 * Z / C * (qa * dpb + qb * dpa) / r
                             - Z / C * (kappa - 1) * (qa * pb + pa * qb) / r**2)
            matrix[a, b] = matrix[b, a] = 0.5 * np.sum(w * integrand)
    return matrix


def recoil_provenance(mass):
    return {
        "operator": "one-body Breit recoil (GRASP RIS4 RINTI_NMS)",
        "nuclear_mass_me": float(mass),
        "order": "first order in 1/M",
        "reference": "https://arxiv.org/abs/1106.1988",
        "omitted": ["two-body specific mass shift", "finite-size recoil",
                    "radiative recoil", "higher orders in 1/M"],
    }


def recoil_state_parameters(csfs, mixing, orbitals_dict, workspace, Z):
    """First-order K expectation of every CI state, including coherence."""
    from atomx.methods.mcdhf.one_body import scalar_one_body_expectations
    keys = tuple(sorted(orbitals_dict))
    radial = recoil_radial_matrix(
        [(*orbitals_dict[key], key[1]) for key in keys], workspace, Z)
    return scalar_one_body_expectations(csfs, mixing, keys, radial)
