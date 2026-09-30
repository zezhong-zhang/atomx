"""Closed-shell specific mass shift from direct radial momentum integrals.

The leading recoil Hamiltonian is +sum_{i<j} p_i.p_j/M. Its direct term
vanishes in closed parity eigenstates. Exchange gives negative K_SMS for
an independent closed-shell determinant. The radial Vinti integral contains
large- and small-component derivatives; it is not inferred from photon
matrix elements or orbital energy differences. In particular [H_D,r]
gives c*alpha, not canonical momentum, and nonlocal HF exchange contributes
to the commutator as well.

Supplying Z includes the point-Coulomb Breit SMS2/SMS3 terms, using the
same GRASP RIS4 conventions as the correction stack. Correlation, finite-size
recoil, radiative recoil and higher orders in 1/M are not supplied here.
For coupled/open-shell states use the CSF recoil matrix, or explicitly use
the configuration-average recoil API for an independently averaged ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from atomx.angular.coefficients import reduced_C
from atomx.corrections.specific_recoil import configuration_average_recoil_contributions


@dataclass
class SMSResult:
    """Mass-independent two-electron recoil coefficient, E_SMS = K_SMS/M."""
    K_SMS_au: float
    pair_contributions: dict[tuple, float] = field(default_factory=dict)
    n_pairs: int = 0
    method: str = "closed-shell direct Vinti momentum integrals"

    def for_mass(self, mass_me: float) -> float:
        """Energy correction in Hartree for nuclear mass in electron masses."""
        if not math.isfinite(mass_me) or mass_me <= 0:
            raise ValueError("mass_me must be positive and finite")
        return self.K_SMS_au / mass_me


def _allowed_sms_pair(kappa_a: int, kappa_b: int) -> bool:
    """Rank-one, odd-parity momentum selection rule."""
    return reduced_C(kappa_a, kappa_b, 1) != 0.


def compute_sms_abinitio(orbitals, workspace, *, Z=None) -> SMSResult:
    """Closed-shell recoil coefficient and its subshell-pair decomposition.

    Parameters
    ----------
    orbitals : sequence of (P, Q, occupation, kappa, label)
        Each occupied subshell must be full; zero-occupation virtuals are
        ignored. Supply B-spline coefficients or tabulated grid spinors.
    workspace : dict
        The corresponding radial workspace, including native quadrature.
    Z : int, optional
        Nuclear charge for point-Coulomb Breit recoil. Omitting Z explicitly
        selects the leading p_i.p_j operator evaluated on these spinors.

    No orbital energies are needed. The radial and angular implementation
    is shared with the two-electron recoil correction, including the
    magnetic-substate occupation normalization.
    """
    occupied = [o for o in orbitals if o[2] != 0.]
    for _, _, occupation, kappa, _ in occupied:
        if not math.isfinite(occupation) or abs(occupation-2*abs(kappa)) > 1e-12:
            raise ValueError('closed-shell SMS needs fully occupied subshells; '
                             'use CSF or configuration-average recoil for open shells')
    pairs = configuration_average_recoil_contributions(occupied, workspace, Z=Z)
    labeled = {(occupied[a][4], occupied[b][4]): value for (a, b), value in pairs.items()}
    return SMSResult(K_SMS_au=float(sum(pairs.values())),
                     pair_contributions=labeled, n_pairs=len(pairs),
                     method=('closed-shell point-Coulomb Breit recoil (VINTI + SMS2/SMS3)'
                             if Z is not None else 'closed-shell leading recoil (VINTI)'))
