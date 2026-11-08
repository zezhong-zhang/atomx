"""ABACUS crystal-field bridge: DFT d-block projection -> atomx impurity.

Pipeline (the "ABACUS CF embedding" the user asked for):

    ABACUS LCAO DFT(+U) on the material
      -> d-block on-site Hamiltonian (real-harmonic LCAO basis)
      -> transform to complex spherical |m = -2..2>
      -> ImpuritySpec(cf_matrix=...) — Slater + SOC + the DFT crystal field
      -> Lehmann solve: spectrum WITH the many-body (multiplet) effects

Basis bookkeeping (the known trap, from the E1.5 kappa lesson): ABACUS LCAO
d orbitals are ordered (per its documentation and generated orbital files)

    index 0..4 = dz2, dxz, dyz, dx2-y2, dxy      (real cubic harmonics)

The spherical transform here is the SAME one validated in the independent
RIXS chain (cubic-harmonic rows: dz2, dxz, dyz, dx2y2, dxy).
"""
from __future__ import annotations

import math

import numpy as np

# ABACUS LCAO d-orbital order (real cubic harmonics)
ABACUS_D_ORDER = ("dz2", "dxz", "dyz", "dx2y2", "dxy")


def cubic_to_spherical_5() -> np.ndarray:
    """T: rows = cubic harmonics (ABACUS order), cols = |m=-2..2> spherical."""
    T = np.zeros((5, 5), dtype=complex)
    s2 = 1 / math.sqrt(2)
    T[0, 2] = 1.0                                    # dz2 = Y20
    T[1, 1], T[1, 3] = -s2, s2                       # dxz
    T[2, 1], T[2, 3] = 1j * s2, 1j * s2              # dyz
    T[3, 0], T[3, 4] = s2, s2                        # dx2-y2
    T[4, 0], T[4, 4] = 1j * s2, -1j * s2             # dxy
    return T


def abacus_dblock_to_spherical(h_dblock: np.ndarray) -> np.ndarray:
    """Transform a 5x5 d-block Hamiltonian from the ABACUS LCAO (real
    cubic) basis to the complex spherical basis used by the impurity."""
    H = np.asarray(h_dblock, dtype=complex)
    if H.shape != (5, 5):
        raise ValueError("d-block must be 5x5")
    T = cubic_to_spherical_5()
    # h_sph[m, m'] = sum_{a,b} T*[a, m] h[a, b] T[b, m']
    return T.conj().T @ H @ T


def impurity_from_abacus_dblock(
    h_dblock: np.ndarray,
    *,
    slater_eV: dict,
    zeta_d_eV: float,
    n_electrons: int = 8,
    temperature_K: float = 300.0,
) -> dict:
    """Build an ImpuritySpec with the ABACUS-derived crystal field and solve.

    Returns the atomic-limit solution dict (same as solve_atomic_limit)
    plus 'cf_matrix_spherical' for provenance.
    """
    from atomx.embedding.impurity import ImpuritySpec, solve_atomic_limit

    cf5 = abacus_dblock_to_spherical(h_dblock)
    spec = ImpuritySpec(
        n_electrons=n_electrons,
        slater_eV=slater_eV,
        zeta_d_eV=zeta_d_eV,
        cf_matrix=cf5,
        temperature_K=temperature_K,
    )
    out = solve_atomic_limit(spec)
    out["cf_matrix_spherical"] = cf5
    out["cf_input_provenance"] = {
        "model": "abacus:LCAO-d-block",
        "basis_in": "real cubic (ABACUS order: dz2 dxz dyz dx2y2 dxy)",
        "basis_out": "complex spherical |m=-2..2>",
        "note": "DFT crystal field embedded into the multiplet impurity",
    }
    return out
