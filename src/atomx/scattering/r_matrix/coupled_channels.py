"""Public driver for the multi-channel close-coupling R-matrix.

Phase 2 of HYP-188 lands the close-coupling solver alongside the
phase-1 single-channel scaffolding. The top-level entry point

    compute_multichannel(targets, L_total, S_total, parity,
                         energies_au, ...)

assembles a Bloch-augmented inner-region Hamiltonian, projects the
boundary R-matrix onto the open-channel subspace, matches against
multi-channel Coulomb wave functions, and returns the K-matrix and
eigenphase sum at every requested total energy.

A second helper, :func:`extract_resonances`, runs an energy scan and
fits Breit-Wigner positions and widths to the rising-pi steps in the
eigenphase sum.

The function :func:`r_matrix_excitation_cross_section` is a thin
forward-compatibility hook that mirrors the
:func:`atomx.scattering.distorted_wave.dw_excitation_cross_section`
signature (Phase 3 will wire it into the full scattering API; phase 2
just exposes the entry point so consumers can dispatch on it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from atomx.constants import HARTREE_EV

from atomx.scattering.r_matrix.cc_inner import (
    assemble_cc_inner_region,
    CCInnerSolution,
)
from atomx.scattering.r_matrix.cc_outer import (
    OuterMatching,
    channel_R_matrix,
    eigenphase_sum_unwrapped,
    match_K_matrix,
)
from atomx.scattering.r_matrix.channel_table import (
    Channel,
    channel_labels,
    enumerate_channels,
)
from atomx.scattering.r_matrix.resonance import (
    ResonanceFit,
    find_resonance_seeds,
    fit_breit_wigner,
)
from atomx.scattering.r_matrix.target_states import TargetState


@dataclass
class CCResult:
    """Output of a multi-channel close-coupling R-matrix calculation."""

    energies_au: np.ndarray
    channels: List[Channel]
    L_total: int
    S_total: int
    parity: int
    a_au: float
    inner: CCInnerSolution
    matchings: List[OuterMatching]
    eigenphase_sum: np.ndarray
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute_multichannel(
    targets: Iterable[TargetState],
    L_total: int,
    S_total: int,
    parity: int,
    energies_au: np.ndarray,
    Z: float = 2.0,
    a_au: float = 25.0,
    n_basis: int = 60,
    k_spline_order: int = 6,
    n_gauss: int = 12,
    l_c_max: int = 2,
) -> CCResult:
    """Multi-channel R-matrix close-coupling solve for one total ``LSπ`` block.

    Parameters
    ----------
    targets :
        Iterable of :class:`TargetState`. The He+ defaults from
        :func:`heplus_target_states` cover {1s, 2s, 2p}.
    L_total, S_total, parity :
        Total orbital momentum, spin, and parity for the LSπ block.
        Parity is ``+1`` (even) or ``-1`` (odd).
    energies_au :
        Total energies (Hartree) at which to evaluate R, K, eigenphases.
    Z :
        Target nuclear charge (default He, Z=2).
    a_au :
        R-matrix boundary radius. Burke 1971 used ``a ~ 15`` a₀ with a
        tighter inner basis; ``a = 25`` with a denser B-spline grid is
        equivalent for the 1 %-level resonance test.
    n_basis :
        Number of clamped B-splines per channel on ``[0, a]``.
    k_spline_order :
        Polynomial order of the splines.
    n_gauss :
        Gauss-Legendre quadrature points per knot cell.
    l_c_max :
        Cutoff on the projectile partial wave (typically 2 for ¹S, ¹D
        and 2 for ¹P; raise to 3 to include εf in the ¹D block).
    """
    targets = list(targets)
    channels = enumerate_channels(targets, L_total, parity, l_c_max=l_c_max)
    if not channels:
        raise ValueError(
            f"No channels found for L={L_total}, parity={parity}, "
            f"l_c_max={l_c_max}"
        )

    inner = assemble_cc_inner_region(
        channels=channels,
        L_total=L_total,
        S_total=S_total,
        Z=Z,
        a_au=a_au,
        n_basis=n_basis,
        k_spline_order=k_spline_order,
        n_gauss=n_gauss,
    )

    energies_au = np.atleast_1d(np.asarray(energies_au, dtype=float))
    matchings: List[OuterMatching] = []
    eigsum = np.zeros(energies_au.size, dtype=float)
    for k, E in enumerate(energies_au):
        R_full = channel_R_matrix(
            energy_au=float(E),
            eigenvalues_E_n_au=inner.eigenvalues_au,
            surface_amplitudes=inner.surface_amplitudes,
            a_au=a_au,
        )
        match = match_K_matrix(
            energy_au=float(E), channels=channels, R_full=R_full, a_au=a_au
        )
        matchings.append(match)
        eigsum[k] = match.eigenphase_sum

    return CCResult(
        energies_au=energies_au,
        channels=channels,
        L_total=L_total,
        S_total=S_total,
        parity=parity,
        a_au=a_au,
        inner=inner,
        matchings=matchings,
        eigenphase_sum=eigsum,
        provenance={
            "module": "atomx.scattering.r_matrix.coupled_channels",
            "method": "close_coupling_R_matrix",
            "Z": Z,
            "n_basis": n_basis,
            "n_gauss": n_gauss,
            "k_spline_order": k_spline_order,
            "l_c_max": l_c_max,
            "n_channels": len(channels),
            "channel_labels": channel_labels(channels),
        },
    )


def extract_resonances(
    result: CCResult, window_au: float = 0.03
) -> List[ResonanceFit]:
    """Identify and Breit-Wigner-fit each resonance in a CC energy scan."""
    delta = eigenphase_sum_unwrapped(result.energies_au, result.eigenphase_sum)
    seeds = find_resonance_seeds(result.energies_au, delta, min_jump=0.5)
    fits: List[ResonanceFit] = []
    for E_seed in seeds:
        fit = fit_breit_wigner(
            result.energies_au,
            delta,
            E_guess=E_seed,
            Gamma_guess=window_au * 0.3,
            window_au=window_au,
        )
        fits.append(fit)
    return fits


def r_matrix_excitation_cross_section(
    atom_initial,
    atom_final,
    energies_eV,
    *,
    L_max: int = 4,
    kappa_max: int = 10,
    transition_energy_eV=None,
):
    """Mirror of :func:`dw_excitation_cross_section` — phase-3 production hook.

    The full production-side wiring (atom-object dispatch, channel
    enumeration from the SCF orbital set, K -> T -> partial-wave
    excitation cross-sections in MBarn) lands under HYP-189 (phase 3).
    The signature is exposed here in phase 2 so downstream consumers
    can already dispatch on the entry-point name; calling it raises
    a clear ``NotImplementedError`` pending phase 3.
    """
    raise NotImplementedError(
        "r_matrix_excitation_cross_section: phase-3 production hook is "
        "still pending. The phase-2 driver "
        "atomx.scattering.r_matrix.compute_multichannel returns the "
        "underlying K-matrix and eigenphase sum on which the cross "
        "section will be built."
    )
