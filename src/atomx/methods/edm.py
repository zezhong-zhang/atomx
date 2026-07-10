"""High-level electron-EDM (eEDM) enhancement factor API.

This module is the user-facing entry point for T,P-violating
electron-EDM enhancement-factor calculations.  It adapts the
low-level Sternheimer sum-over-states machinery in
:mod:`atomx.properties.edm.sternheimer` to the :class:`~atomx.Atom`
high-level API: pass either a solved ``Atom`` or a bare
``SCFResult`` and the enhancement factor is returned as a
:class:`~atomx.properties.edm.SternheimerEDMResult`.

The physics — Sternheimer SOS, V^{N-1} virtual basis, Sandars
scaling — lives in :mod:`atomx.properties.edm`; see there for
formulae and references (Sandars 1965; Dzuba & Flambaum 2009).

Example
-------
    >>> from atomx import Atom
    >>> from atomx.methods.edm import compute_edm_enhancement
    >>> tl = Atom(Z=81, config="...")
    >>> tl.solve()
    >>> R = compute_edm_enhancement(tl, "6p1/2")
    >>> R.R_e   # enhancement factor
"""

from __future__ import annotations

from typing import Optional

from atomx.properties.edm import (
    SternheimerEDMResult,
    sternheimer_edm_enhancement,
)

__all__ = ["compute_edm_enhancement", "SternheimerEDMResult"]


def compute_edm_enhancement(
    atom_or_result,
    target_subshell: Optional[str] = None,
    *,
    E_dm: Optional[float] = None,
    kappa_max: int = 4,
    n_virt_max: Optional[int] = None,
    e_virt_max: Optional[float] = None,
    potential_mode: str = "VN-1",
    Z_eff: Optional[float] = None,
) -> SternheimerEDMResult:
    """Electron-EDM enhancement factor via Sternheimer sum-over-states.

    Computes the T,P-violating enhancement factor

        R_e = (2/3) Σ_a Σ_r ω_{ra} · |⟨a‖r·C¹‖r⟩|² · I_d(κ_a, κ_r) / ω_{ra}

    for the specified valence subshell(s) on a converged SCF reference.

    Parameters
    ----------
    atom_or_result
        Either a solved :class:`~atomx.Atom` (``.solve()`` is invoked
        automatically if it has not been solved yet) or a converged
        ``SCFResult``.
    target_subshell
        Subshell label (e.g. ``"6p1/2"``) for which to compute ``R_e``.
        If ``None``, the routine sums over all open / partially-filled
        subshells.
    E_dm
        Electron EDM value (e·cm).  If given, the routine also
        returns the atomic EDM ``d_atom = R_e · E_dm``.
    kappa_max
        Largest |κ| in the virtual partial-wave basis (default 4).
    n_virt_max
        Cap on virtuals per κ.
    e_virt_max
        Optional energy ceiling on virtual orbitals (Hartree).
    potential_mode
        ``"VN-1"`` (default) or ``"VN"`` for the virtual basis.
    Z_eff
        Effective nuclear charge for relativistic scaling.  Inferred
        from ``scf_result.Z`` if omitted.

    Returns
    -------
    SternheimerEDMResult
        Enhancement factor ``R_e`` and, if ``E_dm`` was supplied,
        the atomic EDM ``d_atom``.

    Raises
    ------
    ValueError
        If the SCF result is not converged or lacks the B-spline data.
    TypeError
        If ``atom_or_result`` is neither an :class:`~atomx.Atom` nor an
        ``SCFResult``.
    """
    scf_result = _to_scf_result(atom_or_result)
    return sternheimer_edm_enhancement(
        scf_result,
        target_subshell=target_subshell,
        kappa_max=kappa_max,
        n_virt_max=n_virt_max,
        e_virt_max=e_virt_max,
        potential_mode=potential_mode,
        Z_eff=Z_eff,
    )


def _to_scf_result(atom_or_result):
    """Coerce an :class:`~atomx.Atom` or ``SCFResult`` to an SCFResult.

    ``Atom`` is matched by duck-typing (it carries an ``_result``
    attribute, ``None`` until ``.solve()`` runs) so that this module
    never imports :mod:`atomx.atom` and stays free of circular imports.
    """
    # atomx.Atom high-level wrapper: solve on demand.
    if hasattr(atom_or_result, "_result"):
        if atom_or_result._result is None:
            if not hasattr(atom_or_result, "solve"):
                raise TypeError(
                    "compute_edm_enhancement() expects a solved atomx.Atom or an "
                    f"SCFResult, got {type(atom_or_result).__name__!r}."
                )
            atom_or_result.solve()
        return atom_or_result._result
    # Bare converged SCFResult.
    if hasattr(atom_or_result, "converged") and hasattr(atom_or_result, "_bsp_data"):
        return atom_or_result
    raise TypeError(
        "compute_edm_enhancement() expects a solved atomx.Atom or an SCFResult, "
        f"got {type(atom_or_result).__name__!r}."
    )
