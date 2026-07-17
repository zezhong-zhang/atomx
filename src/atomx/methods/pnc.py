"""High-level parity non-conservation (PNC) E1 amplitude API.

This module is the user-facing entry point for the spin-independent
PNC E1 amplitude between two s_{1/2} atomic states.  It adapts the
low-level sum-over-states calculation in :mod:`atomx.properties.pnc` to the
:class:`~atomx.Atom` high-level API: pass either a (possibly unsolved)
``Atom`` or a converged ``SCFResult`` and the weak-interaction E1
amplitude is returned as a :class:`~atomx.properties.pnc.PNCResult`.

The physics — weak Hamiltonian radial matrix elements, the
V^{N-1} p_{1/2} intermediate-state basis, and the Standard-Model weak
charge — lives in :mod:`atomx.properties.pnc`; see there for formulae and
references (Bouchiat & Bouchiat 1974; Dzuba, Flambaum & Kozlov 2002;
Ginges & Flambaum 2004).

Example
-------
    >>> from atomx import Atom
    >>> from atomx.methods.pnc import compute_pnc
    >>> cs = Atom(Z=55, config="1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 6s1")
    >>> cs.solve()
    >>> E = compute_pnc(cs, "6s1/2", "7s1/2", N=78)   # Cs-133
    >>> E.E_pnc_iea   # amplitude in 10^-11 i e a0 / Q_W
"""

from __future__ import annotations

from typing import Optional

from atomx.methods._result_adapters import to_scf_result
from atomx.properties.pnc import (
    PNCResult,
    pnc_amplitude_dirac_only,
    weak_charge_qw,
)

__all__ = ["compute_pnc", "PNCResult", "weak_charge_qw"]


def compute_pnc(
    atom_or_result,
    initial_label: str,
    final_label: str,
    *,
    N: Optional[int] = None,
    Q_W: Optional[float] = None,
    rho_nuc_func=None,
    n_intermediate_max: Optional[int] = None,
    e_intermediate_max: Optional[float] = None,
    potential_mode: str = "VN-1",
) -> PNCResult:
    """Spin-independent PNC E1 amplitude via sum-over-states.

    Computes

        E_PNC = Sum_n <f|D|n p_{1/2}><n p_{1/2}|H_W|i>/(E_i - E_n)
              + Sum_n <f|H_W|n p_{1/2}><n p_{1/2}|D|i>/(E_f - E_n)

    between the two s_{1/2} states ``initial_label`` and
    ``final_label`` of an already-converged SCF reference, summing over
    V^{N-1} p_{1/2} intermediate states.  The initial label must be an
    occupied s_{1/2} subshell.  If the final label is not occupied, the
    low-level property uses the lowest V^{N-1} virtual s_{1/2} state
    (for Cs ground-state references this is the requested 7s state).

    Parameters
    ----------
    atom_or_result
        Either a solved :class:`~atomx.Atom` (``.solve()`` is invoked
        automatically if it has not been solved yet) or a converged
        ``SCFResult``.
    initial_label, final_label
        Subshell labels of the two s_{1/2} states (e.g. ``"6s1/2"``,
        ``"7s1/2"``).  The weak Hamiltonian only mixes s_{1/2} <-> p_{1/2},
        so both endpoints must be s_{1/2}.
    N
        Neutron number, used to build the Standard-Model weak charge
        via :func:`~atomx.properties.pnc.weak_charge_qw`.  PNC is
        isotope-dependent, so either ``N`` or ``Q_W`` must be supplied.
    Q_W
        Nuclear weak charge, overriding the Standard-Model value.
    rho_nuc_func
        Nuclear charge density rho(r) (normalised to integral rho dr = 1).
        Default: a narrow Gaussian approximating the point-nucleus
        contact term used in :func:`~atomx.properties.pnc.pnc_amplitude_dirac_only`.
    n_intermediate_max
        Maximum number of p_{1/2} intermediate states.  ``None`` -> all
        physical V^{N-1} virtuals.
    e_intermediate_max
        Energy ceiling (Hartree) for intermediate states.  ``None`` ->
        no ceiling.
    potential_mode
        ``"VN-1"`` (default) or ``"VN"`` virtual-state Fock operator.

    Returns
    -------
    PNCResult
        Amplitude in atomic units (``E_pnc_au``) and in the standard
        convenience unit 10^-11 i e a_0 / Q_W (``E_pnc_iea``).

    Raises
    ------
    ValueError
        If the SCF result is not converged, lacks the B-spline data, the
        endpoints are not both s_{1/2}, or ``N`` and ``Q_W`` are both
        omitted.
    TypeError
        If ``atom_or_result`` is neither an :class:`~atomx.Atom` nor an
        ``SCFResult``.
    """
    scf_result = to_scf_result(atom_or_result, caller="compute_pnc()")
    return pnc_amplitude_dirac_only(
        scf_result,
        initial_label,
        final_label,
        rho_nuc_func=rho_nuc_func,
        N=N,
        Q_W=Q_W,
        n_intermediate_max=n_intermediate_max,
        e_intermediate_max=e_intermediate_max,
        potential_mode=potential_mode,
    )

