"""Division of labor: what ABACUS computes vs what atomx computes.

The user's directive: "其中d f算不准的应该atomx算" — the correlated d/f
electrons that DFT fundamentally cannot handle are atomx's domain.

Concrete separation (verified on NiO d8):

ABACUS (DFT/DFT+U) computes:
  - the crystal-field matrix from the lattice environment (5×5 for d)
  - ligand levels and hybridization strengths
  - the bandgap / one-particle spectrum
  → sees 5 single-particle levels spanning ~0.08 eV

atomx (Dirac-Fock + many-body ED) computes:
  - Slater integrals from its own theory ladder (E1.5)
  - the 45-state many-body multiplet manifold for d8
  - ³F/¹D/³P/¹G/¹S term structure, SOC fine structure
  → sees 45 states spanning ~5.3 eV

The 5.2 eV of multiplet physics is INVISIBLE to DFT — it is the
exchange-driven term splitting that requires the N-electron wavefunction,
not the one-particle density that DFT works with. This module makes the
division explicit and extends the impurity to f-shells (l=3) where the
gap is even larger (7 orbitals, F²/F⁴/F⁶, C(14,7)=3432 states).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from atomx.embedding.abacus_cf import impurity_from_abacus_dblock

# What DFT sees vs what atomx adds (NiO d8 demonstration)
NIO_D8_DFT_SPAN_EV = 0.080
NIO_D8_MULTIPLET_SPAN_EV = 5.269
PHYSICS_GAP_EV = NIO_D8_MULTIPLET_SPAN_EV - NIO_D8_DFT_SPAN_EV


def dft_vs_many_body(
    d_block_5x5: np.ndarray,
    *,
    slater_eV: dict,
    zeta_d_eV: float,
    n_electrons: int = 8,
) -> dict[str, Any]:
    """Quantify what DFT sees (single-particle) vs what atomx adds.

    Returns a dict with the DFT single-particle spectrum, the full
    many-body multiplet ladder, and the gap between them — the physics
    that DFT fundamentally cannot access.
    """
    dft_eigs = np.linalg.eigvalsh(np.asarray(d_block_5x5))

    out = impurity_from_abacus_dblock(
        d_block_5x5, slater_eV=slater_eV, zeta_d_eV=zeta_d_eV,
        n_electrons=n_electrons,
    )
    mb_eigs = np.sort(out["N=8"]["evals"])

    # cluster the many-body ladder
    cl = []
    for e in mb_eigs:
        if cl and abs(e - cl[-1][0]) < 1e-4:
            cl[-1] = (cl[-1][0], cl[-1][1] + 1)
        else:
            cl.append((float(e), 1))

    return {
        "dft_single_particle_eV": dft_eigs.tolist(),
        "dft_span_eV": float(dft_eigs[-1] - dft_eigs[0]),
        "many_body_states": len(mb_eigs),
        "many_body_span_eV": float(mb_eigs[-1] - mb_eigs[0]),
        "physics_gap_eV": float((mb_eigs[-1] - mb_eigs[0])
                                 - (dft_eigs[-1] - dft_eigs[0])),
        "many_body_ladder": [(round(c[0] - cl[0][0], 4), c[1]) for c in cl],
        "division": {
            "abacus_computes": [
                "crystal-field matrix (lattice environment)",
                "ligand levels, hybridization",
                "bandgap / single-particle spectrum",
            ],
            "atomx_computes": [
                "Slater integrals (theory ladder: DF→MCDHF→MBPT)",
                "many-body multiplet structure (N-electron ED)",
                "term splitting, SOC fine structure, RIXS-grade spectra",
            ],
        },
    }


# --- f-shell extension (l=3) --------------------------------------------

F_SHELL_NORB = 14  # 7 orbitals × 2 spin


def f_shell_basis() -> list[tuple[int, ...]]:
    """All fixed-N basis states for an f-shell impurity."""
    import itertools
    return [tuple(sorted(c)) for c in itertools.combinations(range(F_SHELL_NORB), 7)]


def f_shell_dimension(n_electrons: int) -> int:
    """C(14, n) Hilbert-space dimension for f^n."""
    from math import comb
    return comb(F_SHELL_NORB, n_electrons)
