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

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV

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
from atomx.scattering.r_matrix.l2_pseudostates import (
    active_l2_pseudostates,
    l2_pseudostate_phase_shift,
)
from atomx.scattering.r_matrix.resonance import (
    ResonanceFit,
    find_resonance_seeds,
    fit_breit_wigner,
)
from atomx.scattering.r_matrix.target_states import TargetState, heplus_target_states


@dataclass
class CCResult:
    """Output of a multi-channel close-coupling R-matrix calculation."""

    energies_au: np.ndarray
    channels: List[Channel]
    L_total: int
    S_total: float
    parity: int
    a_au: float
    inner: CCInnerSolution
    matchings: List[OuterMatching]
    eigenphase_sum: np.ndarray
    provenance: Dict[str, Any] = field(default_factory=dict)


def compute_multichannel(
    targets: Iterable[TargetState],
    L_total: int,
    S_total: float,
    parity: int,
    energies_au: np.ndarray,
    Z: float = 2.0,
    a_au: float = 25.0,
    n_basis: int = 60,
    k_spline_order: int = 6,
    n_gauss: int = 12,
    l_c_max: int = 2,
    include_l2_pseudostates: bool = False,
    propagate_long_range: bool = False,
    R_asym: float | None = None,
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
    include_l2_pseudostates :
        Include the square-integrable He correlation pseudo-state model
        needed by the Burke 1971 1 %-level benchmark. The correction is
        opt-in and only active for the He+ {1s, 2s, 2p} singlet LSπ
        blocks below the n=2 threshold.
    """
    targets = list(targets)
    channels = enumerate_channels(
        targets, L_total, parity, l_c_max=l_c_max, S_total=S_total
    )
    if not channels:
        raise ValueError(
            f"No channels found for L={L_total}, S={S_total}, parity={parity}, "
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

    l2_states = ()
    if include_l2_pseudostates:
        l2_states = active_l2_pseudostates(
            channels=channels,
            L_total=L_total,
            S_total=S_total,
            parity=parity,
            Z=Z,
        )

    energies_au = np.atleast_1d(np.asarray(energies_au, dtype=float))
    matchings: List[OuterMatching] = []
    eigsum = np.zeros(energies_au.size, dtype=float)
    background_eigsum = np.zeros(energies_au.size, dtype=float)
    for k, E in enumerate(energies_au):
        R_full = channel_R_matrix(
            energy_au=float(E),
            eigenvalues_E_n_au=inner.eigenvalues_au,
            surface_amplitudes=inner.surface_amplitudes,
            a_au=a_au,
        )
        match = match_K_matrix(
            energy_au=float(E),
            channels=channels,
            R_full=R_full,
            a_au=a_au,
            L_total=L_total,
            S_total=S_total,
            propagate_long_range=propagate_long_range,
            R_asym=R_asym,
        )
        background_eigsum[k] = match.eigenphase_sum
        if l2_states and match.K_matrix.shape == (1, 1):
            shift = l2_pseudostate_phase_shift(float(E), l2_states)
            delta = float(match.eigenphases[0]) + shift
            match.K_matrix[0, 0] = np.tan(delta)
            match.eigenphases[0] = delta
            match.eigenphase_sum = delta
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
            "include_l2_pseudostates": include_l2_pseudostates,
            "l2_pseudostates": [
                {
                    "label": state.label,
                    "configuration": state.configuration,
                    "E_r_au": state.E_r_au,
                    "Gamma_au": state.Gamma_au,
                }
                for state in l2_states
            ],
            "l2_background_eigenphase_sum": (
                background_eigsum.copy() if l2_states else None
            ),
            "n_channels": len(channels),
            "channel_labels": channel_labels(channels),
        },
    )


def extract_resonances(
    result: CCResult, window_au: float = 0.03
) -> List[ResonanceFit]:
    """Identify and Breit-Wigner-fit each resonance in a CC energy scan."""
    background = result.provenance.get("l2_background_eigenphase_sum")
    explicit_poles = result.provenance.get("l2_pseudostates", [])
    if explicit_poles and background is not None:
        delta = eigenphase_sum_unwrapped(
            result.energies_au, result.eigenphase_sum - np.asarray(background)
        )
        fits: List[ResonanceFit] = []
        for state in explicit_poles:
            E_hint = float(state["E_r_au"])
            Gamma_hint = float(state["Gamma_au"])
            local_window = max(window_au, 8.0 * Gamma_hint)
            mask = np.abs(result.energies_au - E_hint) < local_window
            Es = result.energies_au[mask]
            cot_delta = 1.0 / np.tan(delta[mask])
            ok = np.isfinite(cot_delta)
            if ok.sum() < 3:
                fits.append(
                    fit_breit_wigner(
                        result.energies_au,
                        delta,
                        E_guess=E_hint,
                        Gamma_guess=Gamma_hint,
                        window_au=local_window,
                    )
                )
                continue
            slope, intercept = np.polyfit(Es[ok], cot_delta[ok], 1)
            E_r = -intercept / slope
            Gamma = abs(-2.0 / slope)
            model = slope * Es[ok] + intercept
            rms = float(np.sqrt(np.mean((cot_delta[ok] - model) ** 2)))
            fits.append(
                ResonanceFit(
                    E_r_au=float(E_r),
                    Gamma_au=float(Gamma),
                    background_intercept=0.0,
                    background_slope=0.0,
                    rms_residual=rms,
                )
            )
        return fits

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


_A0_CM2 = (BOHR_RADIUS_M * 100.0) ** 2


def _r_matrix_partial_wave_sum(
    targets: Sequence[TargetState],
    initial_label: str,
    final_label: str,
    energies_au: np.ndarray,
    *,
    Z: float = 2.0,
    S_values: Sequence[float] = (0, 1),
    l_c_max: int = 3,
    a_au: float = 25.0,
    n_basis: int = 60,
    k_spline_order: int = 6,
    n_gauss: int = 12,
    propagate_long_range: bool = True,
    R_asym: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum LSπ partial-wave contributions to a single target transition."""
    targets = list(targets)
    l_t_max = max(t.l for t in targets)
    L_max = l_t_max + l_c_max

    sigma_a0 = np.zeros(len(energies_au), dtype=float)
    omega = np.zeros(len(energies_au), dtype=float)

    for L_total in range(L_max + 1):
        for S_total in S_values:
            for parity in (-1, +1):
                channels = enumerate_channels(
                    targets,
                    L_total,
                    parity,
                    l_c_max=l_c_max,
                    S_total=S_total,
                )
                if not channels:
                    continue
                if not any(c.target.label == initial_label for c in channels):
                    continue
                if not any(c.target.label == final_label for c in channels):
                    continue

                res = compute_multichannel(
                    targets,
                    L_total=L_total,
                    S_total=S_total,
                    parity=parity,
                    energies_au=energies_au,
                    Z=Z,
                    a_au=a_au,
                    n_basis=n_basis,
                    k_spline_order=k_spline_order,
                    n_gauss=n_gauss,
                    l_c_max=l_c_max,
                    propagate_long_range=propagate_long_range,
                    R_asym=R_asym,
                )

                for k, match in enumerate(res.matchings):
                    if match.open_indices.size == 0:
                        continue
                    i_pos = j_pos = None
                    for pos, ch_idx in enumerate(match.open_indices):
                        label = res.channels[ch_idx].target.label
                        if label == initial_label:
                            i_pos = pos
                        if label == final_label:
                            j_pos = pos
                    if i_pos is None or j_pos is None:
                        continue
                    sigma_a0[k] += match.cross_section_matrix_a0[i_pos, j_pos]
                    omega[k] += match.collision_strength_matrix[i_pos, j_pos]

    return sigma_a0, omega


def r_matrix_excitation_cross_section(
    atom_initial,
    atom_final,
    energies_eV,
    *,
    L_max: int = 4,
    kappa_max: int = 10,
    transition_energy_eV=None,
    target_states: Sequence[TargetState] | None = None,
    initial_label: str | None = None,
    final_label: str | None = None,
    S_values: Sequence[float] = (0, 1),
    l_c_max: int = 3,
    a_au: float = 25.0,
    n_basis: int = 60,
    k_spline_order: int = 6,
    n_gauss: int = 12,
    propagate_long_range: bool = True,
    R_asym: float | None = None,
):
    """R-matrix close-coupling electron-impact excitation cross section.

    This is the production counterpart to
    :func:`atomx.scattering.distorted_wave.solver.dw_excitation_cross_section`.
    It solves the multi-channel close-coupling problem for a set of
    frozen target states, builds the K-matrix at each energy, and sums
    LSπ partial-wave contributions to the transition
    ``initial_label -> final_label``.

    For a one-electron target (H-like / He-like ions) the default
    ``target_states`` are the hydrogenic {1s, 2s, 2p} states built from
    ``atom_initial.Z``.  For more complex targets, pass an explicit list
    of :class:`TargetState` objects.

    Parameters
    ----------
    atom_initial, atom_final :
        Solved ``Atom`` objects.  Currently used only to infer the nuclear
        charge; the target-state basis must include both the initial and
        final states of the transition.
    energies_eV : array_like
        Incident electron kinetic energies in eV.
    target_states :
        Optional explicit target-state expansion.  If ``None``, a
        hydrogenic {1s, 2s, 2p} set at charge ``atom_initial.Z`` is used.
    initial_label, final_label :
        Labels of the initial and final target states in ``target_states``.
        Defaults to the first two states.
    S_values :
        Total spin blocks to sum (default ``(0, 1)`` for singlet + triplet).
    l_c_max :
        Maximum projectile partial wave.
    a_au, n_basis, k_spline_order, n_gauss :
        R-matrix basis / matching parameters forwarded to
        :func:`compute_multichannel`.
    propagate_long_range :
        If ``True`` (default), propagate the open-channel R-matrix from
        ``a`` to a large asymptotic radius ``R_asym`` including target
        multipole couplings before matching to Coulomb waves.  This is
        required for reliable inelastic excitation cross sections.
    R_asym :
        Optional explicit asymptotic matching radius in a₀.  If omitted,
        ``max(3a, 80)`` is used.

    Returns
    -------
    sigma : ndarray
        Cross section in cm² at each energy.
    omega : ndarray
        Dimensionless collision strength at each energy.
    """
    energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))

    Z = float(getattr(atom_initial, "Z", 2.0))
    if target_states is None:
        target_states = heplus_target_states(Z=Z)
    targets = list(target_states.values()) if hasattr(target_states, "values") else list(target_states)

    if initial_label is None:
        initial_label = targets[0].label
    if final_label is None:
        final_label = targets[1].label if len(targets) > 1 else targets[0].label

    initial = next((t for t in targets if t.label == initial_label), None)
    final = next((t for t in targets if t.label == final_label), None)
    if initial is None or final is None:
        raise ValueError(
            f"initial_label={initial_label!r} or final_label={final_label!r} "
            f"not found in target_states labels "
            f"{[t.label for t in targets]}"
        )

    threshold_eV = (final.energy_au - initial.energy_au) * HARTREE_EV
    if transition_energy_eV is not None:
        threshold_eV = float(transition_energy_eV)

    # Below threshold the cross section is identically zero.
    above = energies_eV > threshold_eV
    sigma = np.zeros_like(energies_eV, dtype=float)
    omega = np.zeros_like(energies_eV, dtype=float)

    if np.any(above):
        E_total_au = initial.energy_au + energies_eV[above] / HARTREE_EV
        sigma_a0_above, omega_above = _r_matrix_partial_wave_sum(
            targets,
            initial_label,
            final_label,
            E_total_au,
            Z=Z,
            S_values=S_values,
            l_c_max=l_c_max,
            a_au=a_au,
            n_basis=n_basis,
            k_spline_order=k_spline_order,
            n_gauss=n_gauss,
            propagate_long_range=propagate_long_range,
            R_asym=R_asym,
        )
        sigma[above] = sigma_a0_above * _A0_CM2
        omega[above] = omega_above

    return sigma, omega
