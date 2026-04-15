"""Configuration-interaction diagonalisation on frozen orbitals.

This is the *CI half* of an MCDHF calculation — build the CI matrix
for a list of configuration-state functions (CSFs) that share a
common orbital basis, then diagonalise to obtain state energies and
mixing coefficients.  Call it on converged DHF orbitals to get a
pure-CI result (no orbital re-optimisation); MCDHF's outer loop
composes this with the Guerra average-level orbital-opt iteration.

Decoupling the CI step from orbital optimisation makes two things
possible:

1. **Pure CI on converged DHF orbitals** — much cheaper than a full
   MCDHF, useful for trial CSF spaces and for benchmarking the
   orbital-optimisation gain.
2. **Perturbative correlation methods** (MBPT, CC) built on the same
   infrastructure, calling :func:`ci_solve` with a truncated CSF
   list corresponding to their amplitude sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy import linalg


@dataclass
class CIResult:
    """Outcome of a CI diagonalisation.

    Attributes
    ----------
    state_energies
        Eigenvalues of the CI matrix, sorted in ascending order.
    mixing_coefficients
        CSF mixing coefficients, shape ``(n_csf, n_states)``;
        ``mixing[:, k]`` are the coefficients of the k-th state.
    ground_energy
        Shortcut for ``state_energies[0]``.
    provenance
        Free-form notes on the basis, CSF count, and whether Breit/QED
        were folded into the CI matrix.
    """

    state_energies: np.ndarray
    mixing_coefficients: np.ndarray
    ground_energy: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state_energies.size:
            self.ground_energy = float(self.state_energies[0])


def ci_solve(
    csfs,
    orbitals_dict: Dict[tuple, tuple],
    orbital_basis,
    bsp_data: Dict[str, Any],
    Z: int,
    ns: int,
) -> CIResult:
    """Build the CI matrix and diagonalise it.

    Parameters
    ----------
    csfs
        List of :class:`~atomx.methods.mcdhf.config.CSF` objects.
    orbitals_dict
        ``{(n, kappa): (P_coeffs, Q_coeffs)}`` — converged radial
        orbitals on the shared B-spline basis.
    orbital_basis
        Shared orbital basis descriptor (produced by
        :func:`~atomx.methods.mcdhf.config.get_shared_orbital_basis`).
    bsp_data
        B-spline workspace (grid, weights, basis).
    Z
        Nuclear charge.
    ns
        B-spline dimension.

    Returns
    -------
    :class:`CIResult`
    """
    from atomx.methods.mcdhf.matrix import build_ci_matrix

    H_ci = build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns)
    state_energies, mixing = linalg.eigh(H_ci)
    return CIResult(
        state_energies=np.asarray(state_energies),
        mixing_coefficients=np.asarray(mixing),
        provenance={"n_csf": len(csfs), "n_orbitals": len(orbitals_dict)},
    )


__all__ = ["CIResult", "ci_solve"]
