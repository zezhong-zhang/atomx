"""Charge-exchange (CX) cross sections — Landau-Zener model.

Single-electron capture from a neutral target by an ion projectile:

    A^{q+} + B  →  A^{(q-1)+}(n l) + B^+

dominates astrophysical X-ray emission from cometary atmospheres
and solar wind interactions, and contributes to fusion-plasma
edge transport.

This module provides the **multi-channel Landau-Zener (MCLZ) model**
(Salzborn-Müller 1986; Kimura-Lin 1988), a fast semi-classical
approximation valid for low-velocity collisions.  At each crossing
of the asymptotic potential curves, the LZ formula gives a transition
probability

    P_LZ = exp(-2π H_12² / (v · ΔF))

with H_12 the diabatic coupling, v the relative velocity, and ΔF
the slope difference of the crossing potentials.

A high-accuracy quantum-mechanical close-coupling treatment is on
the roadmap; for now this module ships the LZ kernels and the
typical empirical fit-functions used in CRM databases (Janev,
Smith, Allan).

References
----------
- L. D. Landau, *Phys. Z. Sowjet.* **2**, 46 (1932); C. Zener,
  *Proc. R. Soc. London A* **137**, 696 (1932).
- E. Salzborn, A. Müller, in *Atomic Physics* **9**, World
  Scientific (1986).
- M. Kimura, C. D. Lin, *Phys. Rev. A* **38**, 1259 (1988).
- R. K. Janev *et al.*, *Atomic and Plasma-Material Interaction
  Data for Fusion* **8**, IAEA (2003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class CXResult:
    """Outcome of a CX cross-section calculation.

    Currently the LZ model returns total CX cross section into a
    single capture channel (n, ℓ).  Future expansion: angular-
    momentum-resolved contributions, n-resolved spectra.
    """

    velocity_au: np.ndarray
    sigma_a0_sq: np.ndarray
    crossing_data: List[Dict[str, float]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)


def landau_zener_probability(
    H_12: float,
    velocity_au: np.ndarray,
    delta_F: float,
) -> np.ndarray:
    """Single-passage Landau-Zener transition probability.

    P(v) = exp(-2π H_12² / (v · |ΔF|))
    """
    v = np.asarray(velocity_au, dtype=float)
    arg = 2.0 * np.pi * H_12 ** 2 / (v * abs(delta_F))
    return np.exp(-arg)


def landau_zener_cross_section(
    H_12: float,
    velocity_au: np.ndarray,
    R_x_au: float,
    delta_F: float,
) -> np.ndarray:
    """Two-passage MCLZ total cross section into a single channel.

    σ(v) = 2π R_x² · 2 P(1 − P) ≈ 4π R_x² P(1 − P)

    where R_x is the crossing radius.  Factor of 2 accounts for the
    in/out passage; product P(1−P) maximises at P=1/2.
    """
    P = landau_zener_probability(H_12, velocity_au, delta_F)
    return 4.0 * np.pi * R_x_au ** 2 * P * (1.0 - P)


def cx_cross_section_total_lz(
    crossings: List[Dict[str, float]],
    velocity_au: np.ndarray,
) -> CXResult:
    """Sum LZ contributions from a list of curve crossings.

    Parameters
    ----------
    crossings
        Each entry must have keys ``H_12`` (a.u.), ``R_x`` (a.u.),
        ``delta_F`` (a.u./a.u. = a.u.).  These come from a coupled
        diabatic surface fit (e.g. Olson-Salop / Salzborn-Müller).
    velocity_au
        Projectile velocity grid (atomic units).

    Returns
    -------
    CXResult
    """
    velocity_au = np.asarray(velocity_au, dtype=float)
    sigma_total = np.zeros_like(velocity_au)
    for ch in crossings:
        sigma_total += landau_zener_cross_section(
            ch["H_12"], velocity_au, ch["R_x"], ch["delta_F"]
        )
    return CXResult(
        velocity_au=velocity_au,
        sigma_a0_sq=sigma_total,
        crossing_data=list(crossings),
        provenance={"model": "MCLZ"},
    )


# Roadmap stubs for future high-accuracy methods


def quantum_close_coupling_cx(*_args, **_kwargs):
    """Close-coupling (multi-channel) CX — not yet implemented."""
    raise NotImplementedError(
        "Quantum close-coupling CX is on the roadmap; for now use "
        "the Landau-Zener model in :func:`cx_cross_section_total_lz`."
    )


__all__ = [
    "CXResult",
    "landau_zener_probability",
    "landau_zener_cross_section",
    "cx_cross_section_total_lz",
    "quantum_close_coupling_cx",
]
