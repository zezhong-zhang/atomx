"""Public single-channel R-matrix driver.

``compute_single_channel`` wires the internal-region B-spline eigensolve
to the surface-matching layer and packages the result in
:class:`RMatrixResult` for downstream consumers.

Phase 1 of HYP-172: a self-contained Wigner R-matrix on a local radial
potential, valid for arbitrary short-range interactions and the
centrifugal barrier. The multi-channel close-coupling driver
(``compute(...)``) remains the scaffold ``NotImplementedError`` until
Phase 2 lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.scattering.r_matrix.internal_region import assemble_internal_region
from atomx.scattering.r_matrix.surface import (
    partial_wave_cross_section_au,
    phase_shift_from_R,
    r_matrix_from_eigenpairs,
)

# Bohr radius squared expressed in cm² — converts σ in a₀² to cm².
_A0_CM2 = (BOHR_RADIUS_M * 1.0e2) ** 2


@dataclass
class RMatrixResult:
    """Output of an R-matrix calculation.

    Phase 1 (single-channel) populates the scalar phase-shift and cross-
    section arrays. ``cross_sections_Mbarn`` and ``provenance`` are kept
    for forward-compatibility with the multi-channel driver and existing
    scaffold consumers.
    """

    a_au: float = 0.0
    energies_eV: np.ndarray = field(default_factory=lambda: np.empty(0))
    phase_shift_rad: np.ndarray = field(default_factory=lambda: np.empty(0))
    cross_section_cm2: np.ndarray = field(default_factory=lambda: np.empty(0))
    eigenvalues_E_n_au: np.ndarray = field(default_factory=lambda: np.empty(0))
    l: int = 0
    n_basis: int = 0
    cross_sections_Mbarn: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute_single_channel(
    V_func: Callable[[np.ndarray], np.ndarray],
    energies_eV: np.ndarray,
    a_au: float = 20.0,
    n_basis: int = 80,
    l: int = 0,
    k_spline_order: int = 6,
    n_gauss: int = 10,
) -> RMatrixResult:
    """Single-channel Wigner R-matrix solve for a local radial potential.

    Parameters
    ----------
    V_func :
        Callable ``V(r)`` returning the potential in Hartree at radii
        ``r`` (Bohr). Must be vectorised over ``r``.
    energies_eV :
        Projectile energies in eV (one open channel — interpreted as
        the kinetic energy of the scattered partial wave).
    a_au :
        R-matrix boundary radius in Bohr.
    n_basis :
        Number of clamped B-splines on ``[0, a]``.
    l :
        Partial-wave angular momentum (``ℓ ≥ 0``).
    k_spline_order :
        Polynomial order of the splines (must be ≥ 3 for a smooth
        kinetic-energy assembly).
    n_gauss :
        Gauss-Legendre quadrature points per knot cell.

    Returns
    -------
    RMatrixResult
        With ``phase_shift_rad`` wrapped to ``(−π/2, π/2]`` and
        ``cross_section_cm2 = (4π/k²)(2ℓ+1) sin²(δ_ℓ)`` converted to
        cm² via the CODATA Bohr radius.

    Notes
    -----
    Validity is verified analytically:

    * ``V ≡ 0`` reproduces the free-particle phase shift ``δ_ℓ = 0`` to
      within the basis-set residual.
    * The unitarity bound ``σ_ℓ ≤ 4π(2ℓ+1)/k²`` is enforced by
      construction (``sin²(δ_ℓ) ≤ 1``).
    """
    if a_au <= 0:
        raise ValueError(f"a_au must be positive, got {a_au}")
    if n_basis < k_spline_order:
        raise ValueError(
            f"n_basis ({n_basis}) must be >= spline order ({k_spline_order})"
        )
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")

    energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    if np.any(energies_eV <= 0):
        raise ValueError("energies_eV must be strictly positive (open channel)")

    energies_au = energies_eV / HARTREE_EV

    interior = assemble_internal_region(
        V_func=V_func,
        a_au=a_au,
        n_basis=n_basis,
        l=l,
        k_spline_order=k_spline_order,
        n_gauss=n_gauss,
    )

    R_E = r_matrix_from_eigenpairs(
        eigenvalues_au=interior.eigenvalues_au,
        surface_amplitudes=interior.surface_amplitudes,
        a_au=a_au,
        energies_au=energies_au,
    )

    delta, k = phase_shift_from_R(R_E, energies_au, a_au, l)
    sigma_a0 = partial_wave_cross_section_au(delta, k, l)
    sigma_cm2 = sigma_a0 * _A0_CM2

    return RMatrixResult(
        a_au=a_au,
        energies_eV=energies_eV,
        phase_shift_rad=delta,
        cross_section_cm2=sigma_cm2,
        eigenvalues_E_n_au=interior.eigenvalues_au,
        l=l,
        n_basis=n_basis,
        provenance={
            "module": "atomx.scattering.r_matrix.solver",
            "method": "single_channel_wigner_R_matrix",
            "bloch_constant_b": 0.0,
            "k_spline_order": k_spline_order,
            "n_gauss_per_cell": n_gauss,
        },
    )
