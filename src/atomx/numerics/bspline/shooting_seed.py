"""Galerkin projection of shooting orbitals onto a Dirac B-spline basis.

This is an initial guess only. The final nonlocal DHF equations and energy
are solved by the B-spline solver on its own radial grid.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import cho_factor, cho_solve

from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
from atomx.potentials.nuclear import make_nuclear_potential


def project_shooting_orbitals(result, bsp_data, z: int,
                              nuclear: str = "fermi"):
    """Return ``DFSolver.seed_orbitals`` from a converged shooting result.

    Each spinor component is the weighted least-squares (Galerkin)
    projection onto its own kinetically balanced B-spline basis:
    ``S c = <B|psi_shooting>``. The overlap metric is computed on the
    *target* grid and factored once for all occupied orbitals. PCHIP samples
    the shooting functions at the target quadrature nodes without inventing
    extra radial nodes. A shooting grid that does not cover the target
    quadrature region is rejected.

    Returns
    -------
    seed : dict[int, list[tuple[ndarray, ndarray, float]]]
        Grouped in the shooting result's subshell order within each kappa.
    max_relative_projection_error : float
        Largest weighted spinor L2 error divided by its target L2 norm.
    """
    if not result.converged:
        raise ValueError("shooting seed must be converged")
    radial = np.asarray(result._grid, dtype=float)
    gr = np.asarray(bsp_data["gr"], dtype=float)
    if (radial.ndim != 1 or len(radial) < 2
            or not np.all(np.diff(radial) > 0)
            or gr.min() < radial[0] or gr.max() > radial[-1]):
        raise ValueError("shooting grid does not cover B-spline quadrature")

    ns = bsp_data["ns"]
    nsp = bsp_data.get("nsp", ns)
    nsq = bsp_data.get("nsq", ns)
    _, overlap = build_dirac_matrices(
        ns, -1, z, bsp_data, make_nuclear_potential(z, nuclear))
    metrics = (overlap[:nsp, :nsp],
               overlap[ns:ns + nsq, ns:ns + nsq])
    factors = [cho_factor(metric, check_finite=False) for metric in metrics]
    indices = np.arange(bsp_data["nv"])
    weights = bsp_data["grw"]

    def project(values, component):
        n_component = nsp if component == "p" else nsq
        splines = bsp_data["bsp_p" if component == "p" else "bsp_q"]
        sampled = PchipInterpolator(radial, values, extrapolate=False)(gr)
        if not np.all(np.isfinite(sampled)):
            raise ValueError("non-finite shooting orbital on target grid")
        rhs = np.zeros(n_component)
        for local in range(splines.shape[2]):
            global_index = indices + local
            valid = global_index < n_component
            moments = np.sum(
                weights * sampled * splines[:, :, local], axis=1)
            np.add.at(rhs, global_index[valid], moments[valid])
        coefficients = cho_solve(
            factors[0 if component == "p" else 1], rhs,
            check_finite=False)
        fitted = eval_orbital_on_grid(
            coefficients, splines, bsp_data["nv"], splines.shape[2],
            n_component)
        squared_error = np.sum(weights * (fitted - sampled) ** 2)
        squared_norm = np.sum(weights * sampled ** 2)
        padded = np.pad(coefficients, (0, ns - n_component))
        return padded, squared_error, squared_norm

    seed = {}
    worst = 0.0
    for shell, p, q in zip(result.subshells, result._pa, result._qa,
                           strict=True):
        large, p_error, p_norm = project(p, "p")
        small, q_error, q_norm = project(q, "q")
        norm = p_norm + q_norm
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError(f"zero or non-finite shooting orbital {shell.label}")
        relative_error = np.sqrt((p_error + q_error) / norm)
        worst = max(worst, float(relative_error))
        seed.setdefault(shell.kappa, []).append(
            (large, small, shell.occupation))
    return seed, worst
