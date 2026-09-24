"""Alignment and electric-dipole emission for axially symmetric populations.

Cross sections at |M| include both signs. P=(I_parallel-I_perpendicular)/I
at 90 degrees to the symmetry axis. Higher multipoles need higher-rank
statistical tensors and are not inferred from alignment alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from atomx.angular.wigner import wigner_3j, wigner_6j


def _twice_j(value: float) -> int:
    return int(round(2.0 * value))


def _integer_phase(exponent: float) -> float:
    n = int(round(exponent))
    return -1.0 if n % 2 else 1.0


def steffen_alder_F2(J_u: float, J_l: float, L: int = 1) -> float:
    """Steffen-Alder ``F_2`` coefficient for a radiative multipole line.

    The coefficient converts upper-level alignment into emitted-line angular
    anisotropy/polarisation.  This implementation uses the standard 6j
    angular factor and the Steffen-Alder normalization for rank 2.  It returns
    zero for forbidden angular-momentum triangles and for upper levels that
    cannot support rank-2 alignment.
    """
    if L < 1:
        raise ValueError("Radiative multipole order L must be >= 1")
    if J_u < 1.0:
        return 0.0
    if abs(J_u - J_l) > L or J_u + J_l < L:
        return 0.0

    Ju2 = _twice_j(J_u)
    Jl2 = _twice_j(J_l)
    L2 = 2 * int(L)
    six = wigner_6j(Ju2, Ju2, 4, L2, L2, Jl2)
    if abs(six) < 1e-14:
        return 0.0
    norm = np.sqrt(3.0 * J_u * (2.0 * J_u + 1.0) / (J_u + 1.0))
    phase = _integer_phase(J_u + J_l + L)
    return float(phase * norm * six)


def _m_values(J):
    if not np.isfinite(J) or J < 0 or abs(2*J-round(2*J)) > 1e-10:
        raise ValueError("J must be a nonnegative integer or half integer")
    return np.arange(int(2*J) % 2 / 2, J + 0.1, 1.0)


def _alignment_prefactor(J):
    return np.sqrt(5.0 / (3*J*(J+1)*(2*J-1)*(2*J+3)))


def linear_polarisation(alignment: float, J: float, J_lower: float = 0.0,
                        *, L: int = 1) -> float:
    """E1 polarization at 90 degrees from upper-level alignment.

    Supply the lower J explicitly except for decays to J_lower=0.
    The response is derived from dipole 3j probabilities; the alignment
    normalization is exactly that of alignment_from_msublevel_cross_sections.
    """
    M = _m_values(J)
    _m_values(J_lower)
    if L != 1:
        raise NotImplementedError("Higher multipoles require higher-rank tensors")
    if abs(J-J_lower) > 1 or J+J_lower < 1 or (2*J-2*J_lower) % 2:
        raise ValueError("Forbidden electric-dipole angular momentum triangle")
    if not np.isfinite(alignment):
        raise ValueError("alignment must be finite")
    if J < 1:
        if abs(alignment) > 1e-12:
            raise ValueError("J < 1 cannot support rank-two alignment")
        return 0.0
    pref = _alignment_prefactor(J)
    moments = 3*M*M-J*(J+1)
    avg = alignment / pref
    if avg < moments.min()-1e-10 or avg > moments.max()+1e-10:
        raise ValueError("alignment is incompatible with nonnegative populations")
    # q=0 dipole probability at stretched M=J. For E1 it is affine in M².
    w_stretched = (2*J+1)*wigner_3j(
        int(2*J_lower), 2, int(2*J), -int(2*J), 0, int(2*J))**2
    slope = (w_stretched-1/3) / (J*(2*J-1))
    w0 = 1/3 + slope*avg
    return float((3*w0-1)/(1+w0))


def alignment_from_msublevel_cross_sections(sigma_M: Sequence[float], J: float) -> float:
    """Rank-two alignment from combined ±M cross sections in ascending |M|.

    A = sqrt(5/[3J(J+1)(2J-1)(2J+3)]) <3M²-J(J+1)>.
    For integer J an isotropic population is proportional to [1,2,...,2].
    """
    M = _m_values(J)
    sigma = np.asarray(sigma_M, dtype=float)
    if sigma.shape != M.shape:
        raise ValueError(f"sigma_M length must match the {len(M)} |M| values for J={J}")
    if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
        raise ValueError("cross sections must be finite and nonnegative")
    total = float(sigma.sum())
    if J < 1 or total == 0:
        return 0.0
    return float(_alignment_prefactor(J) * np.dot(3*M*M-J*(J+1), sigma)/total)


def polarised_emission_intensity(polarisation: float, observation_angle_rad: float) -> tuple[float, float]:
    """E1 analyzer intensities for an axially symmetric ensemble.

    P is (I_parallel-I_perpendicular)/(I_parallel+I_perpendicular) at
    90 degrees. Intensities are normalized to unit total at that angle.
    On the axis the two analyzers agree; a pure axial dipole is dark there.
    """
    if not np.isfinite(polarisation) or abs(polarisation) > 1:
        raise ValueError("polarisation must lie in [-1,1]")
    if not np.isfinite(observation_angle_rad):
        raise ValueError("observation angle must be finite")
    transverse = (1-polarisation)/2
    parallel = transverse*np.cos(observation_angle_rad)**2 + (1+polarisation)/2*np.sin(observation_angle_rad)**2
    return float(parallel), float(transverse)


__all__ = ["steffen_alder_F2", "linear_polarisation",
           "alignment_from_msublevel_cross_sections", "polarised_emission_intensity"]
