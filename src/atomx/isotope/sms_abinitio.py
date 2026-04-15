"""Ab-initio Specific Mass Shift (SMS) for closed-shell Dirac–Fock atoms.

Derivation (first principles)
-----------------------------
The non-relativistic two-body mass-shift operator is

    H_SMS  =  −(1/M) · Σ_{i<j} p_i · p_j                              (1)

On a Slater determinant |Φ⟩ built from occupied spin-orbitals, Wick's
theorem gives

    ⟨Φ| Σ_{i<j} p_i·p_j |Φ⟩
        =  ½ · Σ_{a,b occ, a≠b} [ ⟨ab|p_1·p_2|ab⟩ − ⟨ab|p_1·p_2|ba⟩ ]  (2)

For **closed-shell** atoms the direct term vanishes because each
filled subshell is spherically symmetric so the summed single-
particle expectation ⟨p⟩ cancels over orientations.  The exchange
piece reduces to a sum over subshell pairs coupled by the rank-1
vector operator ∇:

    K_SMS  =  ⟨Φ|Σ_{i<j} p_i·p_j|Φ⟩
           =  − Σ_{a<b}  q_a q_b  | ⟨a || ∇ || b⟩ |²                  (3)

with selection rule ``l_a + l_b + 1`` even (≡ |Δl| = 1), using the
reduced matrix element in jj-coupling::

    ⟨a || ∇ || b⟩  =  ⟨κ_a || C^1 || κ_b⟩  ·  R_p(a, b)                (4)

    ⟨κ_a || C^1 || κ_b⟩
        =  (−1)^{j_a+½} · √((2 j_a+1)(2 j_b+1))
           · (j_a  1  j_b ; −½  0  +½)                               (5)

(Wigner-3j; reduces to the standard LS-coupling angular factor in
the non-relativistic limit).

The radial momentum integral R_p(a, b) uses the velocity-gauge E1
matrix element already implemented in :mod:`atomx.transitions.multipole`:

    R_p(a, b)  =  ⟨ψ_a | ∇_r | ψ_b⟩_radial                           (6)

This is identical (up to an i/ω factor) to the E1 velocity-gauge
radial matrix element; we simply reuse that machinery.

Result
------
E_SMS  =  K_SMS / M                                                  (7)

with ``M`` the nuclear mass in electron-mass units.  The sign of eq.
(3) matches the usual convention in which K_SMS > 0 for ground
states of light atoms (gives a blue shift for the lighter isotope).

Scope
-----
* Non-relativistic closed-shell implementation.  The jj-coupling
  angular reduction (eq. 5) is exact; the radial integral uses the
  atomx Dirac-Fock wavefunction in the velocity gauge.
* Valid for any atom whose configuration is fully-occupied subshells
  (no open shells).  Open-shell atoms will need a term for the
  partially-filled subshell plus a trivial "open shell" scalar.
* Relativistic corrections beyond the velocity-gauge E1 form (e.g.
  Breit, higher-order recoil) are NOT included; for heavy atoms
  those ~1 % effects matter for precision work and are flagged as
  future work.

References
----------
P. Indelicato, M. Gu, P. Beiersdorfer, *Atomic Data and Nuclear
    Data Tables* **99**, 294 (2013) — FAC SMS formulation.
M. G. Kozlov, *Nucl. Phys. A* **684**, 358 (2001) — jj-coupled SMS
    reduction for many-body methods.
J. C. Berengut, V. V. Flambaum, M. G. Kozlov, *Phys. Rev. A* **73**,
    012504 (2006) — benchmarks for Fe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from atomx.angular.wigner import wigner_3j
from atomx.transitions.multipole import radial_eL_velocity


@dataclass
class SMSResult:
    """Ab-initio SMS calculation result.

    Attributes
    ----------
    K_SMS_au : float
        Specific-mass-shift coefficient in atomic units (Hartree · m_e).
        Definition:  E_SMS = K_SMS / M  with M in electron masses.
    pair_contributions : dict
        ``{(label_a, label_b): K_SMS^{(a,b)}_au}`` — per-pair breakdown,
        useful for spotting dominant contributions.
    n_pairs : int
        Number of dipole-allowed (l_a, l_b) subshell pairs summed.
    method : str
        Human-readable tag of the computational method used.
    """

    K_SMS_au: float
    pair_contributions: Dict[tuple, float] = field(default_factory=dict)
    n_pairs: int = 0
    method: str = "ab-initio velocity-gauge closed-shell"

    def for_mass(self, mass_me: float) -> float:
        """Evaluate E_SMS (Hartree) at the given nuclear mass."""
        if mass_me <= 0:
            raise ValueError("mass_me must be positive")
        return self.K_SMS_au / mass_me


# ---------------------------------------------------------------------------
# Selection rules + angular factor
# ---------------------------------------------------------------------------

def _l_from_kappa(kappa: int) -> int:
    return kappa if kappa > 0 else -kappa - 1


def _j_from_kappa(kappa: int) -> float:
    return abs(kappa) - 0.5


def _allowed_sms_pair(kappa_a: int, kappa_b: int) -> bool:
    """Dipole selection rule |Δl|=1 for ⟨a||∇||b⟩."""
    l_a = _l_from_kappa(kappa_a)
    l_b = _l_from_kappa(kappa_b)
    if abs(l_a - l_b) != 1:
        return False
    # Triangle inequality on j (j_a − j_b ≤ 1 ≤ j_a + j_b)
    j_a = _j_from_kappa(kappa_a)
    j_b = _j_from_kappa(kappa_b)
    if abs(j_a - j_b) > 1.0 + 1e-9 or (j_a + j_b) < 1.0 - 1e-9:
        return False
    return True


def _reduced_C1(kappa_a: int, kappa_b: int) -> float:
    """Reduced matrix element ``⟨κ_a || C^1 || κ_b⟩`` (eq. 5).

    Zero when the parity selection rule is violated.
    """
    if not _allowed_sms_pair(kappa_a, kappa_b):
        return 0.0
    j_a = _j_from_kappa(kappa_a)
    j_b = _j_from_kappa(kappa_b)
    # Wigner-3j with twice-j convention (atomx.angular.wigner).
    tj_3j = wigner_3j(int(2 * j_a), 2, int(2 * j_b), -1, 0, 1)
    sign = (-1.0) ** (j_a + 0.5)
    return sign * np.sqrt((2 * j_a + 1) * (2 * j_b + 1)) * tj_3j


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def compute_sms_abinitio(
    orbitals,
    bsp_data,
) -> SMSResult:
    """Ab-initio SMS coefficient K_SMS from a closed-shell DHF calculation.

    Parameters
    ----------
    orbitals : list of ``(P, Q, occ, kappa, label)`` tuples
        As produced by :meth:`DFSolver._flatten_orbitals`.
    bsp_data : dict
        B-spline workspace.

    Returns
    -------
    :class:`SMSResult`

    Notes
    -----
    Implements eq. (3)–(6).  For each unordered pair (a, b) of
    occupied subshells with the dipole-allowed selection rule, the
    contribution to K_SMS is

        ΔK = − q_a q_b · |⟨κ_a||C^1||κ_b⟩|² · R_p(a,b)²

    and the total is the sum over all such pairs.
    """
    K_SMS = 0.0
    per_pair: Dict[tuple, float] = {}
    n_pairs = 0

    for i, (P_a, Q_a, occ_a, kappa_a, label_a) in enumerate(orbitals):
        for j, (P_b, Q_b, occ_b, kappa_b, label_b) in enumerate(orbitals):
            if i >= j:
                continue  # unordered pair a < b only

            if not _allowed_sms_pair(kappa_a, kappa_b):
                continue

            C_red = _reduced_C1(kappa_a, kappa_b)
            if abs(C_red) < 1e-14:
                continue

            # Radial momentum integral R_p(a,b) = ⟨a|∇_r|b⟩ (velocity gauge)
            R_p = radial_eL_velocity(
                P_b, Q_b, P_a, Q_a, bsp_data,
                L=1, kappa_i=kappa_b, kappa_f=kappa_a,
            )

            contribution = -occ_a * occ_b * (C_red ** 2) * (R_p ** 2)
            K_SMS += contribution
            per_pair[(label_a, label_b)] = contribution
            n_pairs += 1

    return SMSResult(
        K_SMS_au=K_SMS,
        pair_contributions=per_pair,
        n_pairs=n_pairs,
    )
