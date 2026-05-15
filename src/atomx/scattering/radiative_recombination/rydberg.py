"""Full-Rydberg-series radiative recombination cross section.

The standard :func:`rr_cross_section` only sums Milne contributions
from the subshells that are *already occupied* in the SCF reference.
For neutral or near-neutral targets that's the right answer, but
for highly-charged ions the dominant RR rate comes from capture into
the *unoccupied Rydberg series* — empty (n', l') states with
n' > n_max(occupied).

This module re-exposes the Rydberg-summing variant
``rr_cross_section_rydberg`` defined in the package's ``__init__.py``
under a clear, named entry point.

Reference
---------
* M. F. Gu, *Astrophys. J.* **589**, 1085 (2003) — FAC's Rydberg-
  series RR algorithm.
"""

from __future__ import annotations


def rr_rydberg_cross_section(
    scf_result,
    energies_eV,
    *,
    n_max_per_kappa: int = 8,
    kappa_max: int = 4,
    g_initial: int = 1,
):
    """RR cross section including capture into the full Rydberg series.

    For each |κ| ≤ kappa_max, builds the V^{N-1} bound-state basis
    and sums Milne-inverted σ_PI over every bound (and hence every
    Rydberg) state down to the lowest n_max_per_kappa per κ.  This
    captures the Σ_n n^{-3} hydrogenic baseline that subshell-only
    σ_RR misses.

    Parameters
    ----------
    scf_result
        Solved SCF reference for the recombined ion.
    energies_eV : array_like
        Free-electron kinetic energies in eV.
    n_max_per_kappa : int
        Cap on the number of bound Rydberg states per κ included
        (default 8 — enough to cover the Σ n^{-3} tail).
    kappa_max : int
        Largest |κ| in the partial-wave sum (default 4).
    g_initial : int
        Statistical weight of the recombining ion (default 1 for
        closed-shell core).

    Returns
    -------
    RRResult
    """
    from atomx.scattering.radiative_recombination import (
        rr_cross_section_rydberg,
    )
    return rr_cross_section_rydberg(
        scf_result, energies_eV,
        n_max_per_kappa=n_max_per_kappa,
        kappa_max=kappa_max,
        g_initial=g_initial,
    )


__all__ = ["rr_rydberg_cross_section"]
