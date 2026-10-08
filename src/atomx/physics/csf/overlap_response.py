"""Complete real CI eigensubspace response of physical radial overlaps.

The reference is fixed. Only selected-to-external energy denominators enter;
internal degeneracies and internal eigenvector gauge never require division.
This is a local response of a declared subspace, not a target selector.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from atomx.angular.csf_hamiltonian import csf_signature
from atomx.physics.csf.levels import CILevel, ci_level_blocks
from atomx.physics.csf.overlap import (
    ci_subspace_overlap_gradient,
    ci_subspace_overlaps,
)
from atomx.physics.csf.radial_density import CIRadialDensity


@dataclass(frozen=True)
class CIPrincipalOverlapResponse:
    """Derivative of trace(W O O.T), with the real symmetric W held fixed.

    Add basis_gradient to the radial gradient of hamiltonian_weights in the
    same orbital snapshot. For a simple minimum principal value W is its
    normalized eigenprojector. At a repeated minimum use the whole derivative
    matrix in that eigenspace; an arbitrary single eigenvector is insufficient.
    """

    overlaps: np.ndarray
    principal_matrix: np.ndarray
    basis_gradient: object
    hamiltonian_weights: object
    nearest_external_gap: float | None


def principal_overlap_response(
    bra_csfs, bra_ci, bra_orbitals, bra_levels,
    ket_csfs, ket_ci, ket_orbitals, ket_levels, metric, principal_weights,
    *, radial_density, max_determinant_pairs=1_000_000,
):
    """Local real complete-subspace derivative in a shared radial geometry.

    The selected levels must lie in one J/parity block and include every
    exact-degenerate partner. The supplied radial_density must use the exact
    ket CSF order. No basis or eigenvector response contribution is omitted.
    """
    bra_levels, ket_levels = tuple(bra_levels), tuple(ket_levels)
    if not bra_levels or not ket_levels:
        raise ValueError('nonempty response subspaces required')
    if any(not isinstance(x, CILevel) for x in bra_levels + ket_levels):
        raise TypeError('response subspaces require CILevel selectors')
    symmetries = {(x.J2, x.parity) for x in bra_levels + ket_levels}
    if len(symmetries) != 1:
        raise ValueError('one common J/parity response block required')
    if (not isinstance(radial_density, CIRadialDensity)
            or radial_density.signatures != tuple(csf_signature(c) for c in ket_csfs)):
        raise ValueError('response radial density requires the exact ket CSF basis')
    w = np.asarray(principal_weights)
    rank = len(bra_levels)
    if (w.shape != (rank, rank) or np.iscomplexobj(w)
            or not np.isfinite(w).all() or not np.array_equal(w, w.T)):
        raise ValueError('principal weights require a finite real symmetric matrix')
    energies, blocks = ci_level_blocks(ket_csfs, ket_ci, require_full=True)
    symmetry = next(iter(symmetries))
    columns = blocks[symmetry]
    roots = np.array([s.root for s in ket_levels], dtype=int)
    if np.any(roots >= len(columns)):
        raise ValueError('response subspace outside the CI spectrum')
    candidates = tuple(CILevel(*symmetry, i) for i in range(len(columns)))
    full_overlap = ci_subspace_overlaps(
        bra_csfs, bra_ci, bra_orbitals, bra_levels,
        ket_csfs, ket_ci, ket_orbitals, candidates, metric,
        max_determinant_pairs=max_determinant_pairs,
    )
    cotangent = 2 * w @ full_overlap[:, roots]
    overlap, basis = ci_subspace_overlap_gradient(
        bra_csfs, bra_ci, bra_orbitals, bra_levels,
        ket_csfs, ket_ci, ket_orbitals, ket_levels, metric, cotangent,
        max_determinant_pairs=max_determinant_pairs,
    )
    # The overlap kernel validates selected cluster closure and duplicates.
    # Summed internal rotations cancel for the symmetric principal functional.
    external = np.setdiff1d(np.arange(len(columns)), roots)
    weights = {}
    nearest = None
    if len(external):
        gaps = energies[columns[roots], None] - energies[columns[external]][None, :]
        nearest = float(np.min(abs(gaps)))
        if nearest < 1e-10:
            raise ValueError('response must include the complete degenerate cluster')
        for i, root in enumerate(roots):
            factors = (cotangent[:, i] @ full_overlap[:, external]) / gaps[i]
            left = ket_ci.mixing_coefficients[:, columns[external]] @ factors
            right = ket_ci.mixing_coefficients[:, columns[root]]
            terms = radial_density.contract_bilinear(left, right)
            for key, value in terms.items():
                weights[key] = weights.get(key, 0.) + value
    principal = overlap @ overlap.T
    for array in (overlap, principal, *basis.values()):
        array.flags.writeable = False
    return CIPrincipalOverlapResponse(
        overlap, principal, MappingProxyType(basis), MappingProxyType(weights), nearest,
    )
