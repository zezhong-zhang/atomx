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
matrix element already implemented in :mod:`atomx.properties.transitions.multipole`:

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

from atomx.angular.coefficients import _l_from_kappa, reduced_C
from atomx.properties.transitions.multipole import radial_eL


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
    method: str = "mean-field DHF closed-shell (R_p = ω·⟨r⟩)"

    def for_mass(self, mass_me: float) -> float:
        """Evaluate E_SMS (Hartree) at the given nuclear mass."""
        if mass_me <= 0:
            raise ValueError("mass_me must be positive")
        return self.K_SMS_au / mass_me


# ---------------------------------------------------------------------------
# Selection rules + angular factor
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def compute_sms_abinitio(
    orbitals,
    bsp_data,
    orbital_energies,
) -> SMSResult:
    """Mean-field ab-initio SMS coefficient K_SMS from a closed-shell DHF calc.

    Parameters
    ----------
    orbitals : list of ``(P, Q, occ, kappa, label)`` tuples
        As produced by :meth:`DFSolver._flatten_orbitals`.
    bsp_data : dict
        B-spline workspace.
    orbital_energies : dict[str, float]
        ``{label: ε}`` orbital eigenvalues (Hartree), as in
        ``SCFResult.orbital_energies``.  Needed for the ``ω_ab`` factor of
        the momentum matrix element.

    Returns
    -------
    :class:`SMSResult`

    Notes
    -----
    Implements eq. (3)–(6).  For each unordered pair (a, b) of
    occupied subshells with the dipole-allowed selection rule, the
    contribution to K_SMS is

        ΔK = − q_a q_b · |⟨κ_a||C^1||κ_b⟩|² · R_p(a,b)²

    **Radial operator.**  The SMS building block is the momentum matrix
    element ``R_p = ⟨a|∇_r|b⟩``.  Using the exact operator identity
    ``∇ = -[H, r]`` (with r = r·C^1 a rank-1 tensor), this equals

        R_p(a, b) = ω_ab · ⟨a‖r‖b⟩_radial = ω_ab · radial_eL(a, b, L=1)

    with ``ω_ab = |ε_a − ε_b|`` and the *same* ``⟨κ_a‖C^1‖κ_b⟩`` angular
    factor already applied in eq. (5).  This form is **antihermitian** by
    construction (⟨a|r|b⟩ symmetric × ω_ab antisymmetric), as a momentum
    matrix element must be, and reduces to the standard velocity/length
    gauge relation.  (History: earlier code called ``radial_eL_velocity``
    *without* ``omega``, hitting a fallback branch that returned the
    O(α)-suppressed photon element — wrong by ~4 orders of magnitude on Ne.
    See ``docs/CRITICAL_REVIEW_2026-07-09.md`` §A2.)

    .. note::
       **Mean-field scope.**  This is the closed-shell Dirac–Fock (single
       determinant) SMS.  The literature repeatedly notes SMS is *very*
       sensitive to electron correlation (contributing already in second
       order); the correlated K_SMS therefore differs from this mean-field
       value and requires MBPT/CI on top.  Two further mean-field-level
       approximations: (i) ``∇ = -[H,r]`` is exact only for a *local*
       potential — HF's nonlocal exchange adds a ``⟨[V_x, r]⟩`` term
       (~%-level); (ii) non-relativistic reduction of the two-body recoil.
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

            # ⟨κ_a‖C^1‖κ_b⟩ (eq. 5) — canonical Grant-convention implementation.
            C_red = reduced_C(kappa_a, kappa_b, 1)
            if abs(C_red) < 1e-14:
                continue

            # Momentum matrix element R_p(a,b) = ⟨a|∇_r|b⟩ = ω_ab·⟨a|r|b⟩
            # via ∇ = -[H,r]; antihermitian, same C^1 angular factor.
            omega = abs(orbital_energies[label_a] - orbital_energies[label_b])
            R_r = radial_eL(P_a, Q_a, P_b, Q_b, bsp_data, L=1)
            R_p = omega * R_r

            contribution = -occ_a * occ_b * (C_red ** 2) * (R_p ** 2)
            K_SMS += contribution
            per_pair[(label_a, label_b)] = contribution
            n_pairs += 1

    return SMSResult(
        K_SMS_au=K_SMS,
        pair_contributions=per_pair,
        n_pairs=n_pairs,
    )
