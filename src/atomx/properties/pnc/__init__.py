"""Parity non-conservation (PNC) amplitudes.

Matrix element of the weak-interaction Hamiltonian

    H_W = (G_F / (2√2)) · Q_W · γ_5 · ρ_nuc(r)

between opposite-parity atomic states.  The dominant PNC observable
in heavy alkalis (Cs, Yb, Fr) is the spin-independent ``E_PNC`` E1
amplitude

    E_PNC = Σ_n ⟨f|D|n p⟩⟨n p|H_W|i⟩/(E_i-E_np)
          + Σ_n ⟨f|H_W|n p⟩⟨n p|D|i⟩/(E_f-E_np)

via opposite-parity admixtures with intermediate p states.  That
amplitude dictates the parity-violating optical-rotation signal.

This module implements the **single-particle weak Hamiltonian**
expectation value via the radial integral

    ⟨P_a | H_W | P_b⟩  ∝  ⟨P_a | ρ_nuc(r) · (g_a · g_b ± f_a · f_b) | ⟩

with appropriate sign for s_{1/2} ↔ p_{1/2} parity-flipping pairs.
The result is dressed by core/correlation enhancements through
RPA + MBPT (planned), without which PNC results have ~30 % error.

References
----------
- M. A. Bouchiat, C. C. Bouchiat, *J. Physique* **35**, 899 (1974).
- V. A. Dzuba, V. V. Flambaum, M. G. Kozlov, *Phys. Rev. A* **66**,
  012512 (2002) — all-order PNC amplitudes for Cs.
- J. S. M. Ginges, V. V. Flambaum, *Phys. Rep.* **397**, 63 (2004).

This module provides:

- :func:`weak_hamiltonian_radial_me` — single-particle ⟨a|H_W|b⟩.
- :func:`pnc_amplitude_dirac_only` — reserved for the full sum-over-
  states PNC E1 amplitude; currently raises ``NotImplementedError``.
- :func:`weak_charge_qw` — Q_W from Z, N (default Standard Model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from atomx.numerics.bspline.evaluate import eval_P, eval_Q


G_F_AU = 2.222516e-14  # Fermi constant in atomic units
# Q_W(p) ≈ 0.0713; Q_W(n) ≈ -0.9890 (Standard Model)
Q_W_PROTON = 0.0713
Q_W_NEUTRON = -0.9890


@dataclass
class PNCResult:
    """PNC amplitude result for a transition n₁s ↔ n₂s."""

    E_pnc_au: float = 0.0
    E_pnc_iea: float = 0.0   # in 10⁻¹¹ i e a₀ Q_W convenience unit
    Q_W: float = 0.0
    label_initial: str = ""
    label_final: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)


def weak_charge_qw(Z: int, N: int) -> float:
    """Standard-model nuclear weak charge Q_W = Z·Q_W(p) + N·Q_W(n).

    Includes 1-loop radiative correction (Marciano-Sirlin 1990)
    via the empirical reduction:

        Q_W^{SM} ≈ Z (1 − 4 sin²θ_W) − N

    with the effective sin²θ_W = 0.2381.  Default uses the proton/
    neutron values above.
    """
    return float(Z * Q_W_PROTON + N * Q_W_NEUTRON)


def weak_hamiltonian_radial_me(
    P_a: np.ndarray,
    Q_a: np.ndarray,
    P_b: np.ndarray,
    Q_b: np.ndarray,
    bsp_data: dict,
    *,
    rho_nuc_func,
    Q_W: float,
) -> float:
    """Radial matrix element of the weak Hamiltonian between two
    Dirac orbitals.

    H_W = (G_F / (2√2)) · Q_W · γ_5 · ρ_nuc(r)

    For s_{1/2} ↔ p_{1/2} parity-flipping transitions (the only
    non-zero case for the spin-independent dominant term):

        ⟨a|H_W|b⟩ = (G_F / (2√2)) · Q_W ·
                    ∫₀^∞ ρ_nuc(r) · [P_a Q_b − Q_a P_b] dr

    with γ_5 in the Dirac representation acting as a parity-flip
    between large and small components.

    Parameters
    ----------
    rho_nuc_func
        Callable rho_nuc(r) returning the normalised nuclear charge
        density at radius r (atomic units), normalised to ∫ρ dr = 1.
        Use the Fermi or uniform models from
        :mod:`atomx.potentials.nuclear`.
    Q_W
        Nuclear weak charge.

    Returns
    -------
    float
        Matrix element in atomic units.
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    Pa = eval_P(P_a, bsp_data)
    Qa = eval_Q(Q_a, bsp_data)
    Pb = eval_P(P_b, bsp_data)
    Qb = eval_Q(Q_b, bsp_data)
    rho_grid = np.vectorize(rho_nuc_func)(gr.ravel()).reshape(gr.shape)
    integrand = rho_grid * (Pa * Qb - Qa * Pb)
    return (G_F_AU / (2.0 * np.sqrt(2.0))) * Q_W * float(np.sum(grw * integrand))


def pnc_amplitude_dirac_only(
    scf_result,
    initial_label: str,
    final_label: str,
    *,
    rho_nuc_func=None,
    N: Optional[int] = None,
) -> PNCResult:
    """Dirac-only PNC E1 amplitude.

    A physical same-parity ``ns -> n's`` PNC amplitude is not the
    direct weak matrix element between the two states. It is a sum
    over opposite-parity intermediate states containing both E1 and
    weak-Hamiltonian matrix elements. That machinery is not yet
    implemented, so this public entry point intentionally fails rather
    than returning a misleading number.

    Parameters
    ----------
    scf_result
        Converged SCFResult.
    initial_label, final_label
        Subshell labels (e.g. ``"6s1/2"``, ``"7s1/2"``).
    rho_nuc_func
        Nuclear charge density.  Default: point Coulomb δ(r) limit
        — the integrand collapses to a contact term at r=0.
    N
        Neutron number (for default Standard-Model Q_W).  Required
        if Q_W not supplied.

    Returns
    -------
    PNCResult
    """
    raise NotImplementedError(
        "PNC E1 amplitudes require a sum over opposite-parity "
        "intermediate states with E1 and weak matrix elements. Use "
        "weak_hamiltonian_radial_me() for the low-level single-particle "
        "weak matrix element."
    )


__all__ = [
    "PNCResult",
    "compute",
    "pnc_amplitude_dirac_only",
    "weak_hamiltonian_radial_me",
    "weak_charge_qw",
    "G_F_AU",
    "Q_W_PROTON",
    "Q_W_NEUTRON",
]


def compute(*_args, **_kwargs) -> PNCResult:
    """Backwards-compat placeholder for the full PNC amplitude."""
    raise NotImplementedError(
        "pnc compute() is not implemented; use weak_hamiltonian_radial_me() "
        "for low-level weak matrix elements."
    )
