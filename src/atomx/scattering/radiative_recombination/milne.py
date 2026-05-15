"""Milne detailed-balance σ_RR for occupied subshells.

This module re-exposes the σ_RR helper that builds the Milne σ_RR
from the photoionisation σ_PI of every occupied subshell of a solved
SCF reference.  The numerical kernel lives in the package's
``__init__.py`` (function ``rr_cross_section``); this thin module
gives users a clear, named entry point under the canonical
``milne`` label.

Public entry point: :func:`rr_milne_cross_section`.

Reference
---------
* E. A. Milne, *Phys. Rev.* **24**, 397 (1924) — original detailed-
  balance theorem.
* M. F. Gu, *Astrophys. J.* **589**, 1085 (2003) — FAC RR algorithm.
"""

from __future__ import annotations

from typing import List, Optional


def rr_milne_cross_section(
    scf_result,
    energies_eV,
    *,
    capture_orbitals: Optional[List[str]] = None,
    g_initial: Optional[int] = None,
    g_final: Optional[int] = None,
):
    """Total radiative-recombination cross section via Milne detailed balance.

    Builds σ_RR(E_e) from the photoionisation σ_PI of every occupied
    subshell.  Wrapper around the package-level ``rr_cross_section``
    so callers can do

        from atomx.scattering.radiative_recombination.milne import rr_milne_cross_section

    Parameters
    ----------
    scf_result
        Solved SCFResult / MCDHFResult satisfying
        :class:`atomx.scattering.protocols.AtomicResultLike`.
    energies_eV : array_like
        Free-electron kinetic energies, eV.
    capture_orbitals : list[str], optional
        Whitelist of subshell labels (default: every occupied shell).
    g_initial : int, optional
        Statistical weight 2J_ion+1 of the recombining ion (default 1).
    g_final : int, optional
        Statistical weight g_n of the recombined bound state.

    Returns
    -------
    RRResult
    """
    from atomx.scattering.radiative_recombination import rr_cross_section
    return rr_cross_section(
        scf_result, energies_eV,
        capture_orbitals=capture_orbitals,
        g_initial=g_initial,
        g_final=g_final,
    )


__all__ = ["rr_milne_cross_section"]
