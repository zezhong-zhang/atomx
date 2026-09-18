"""MBPT(2) transition-energy helper.

While :mod:`atomx.methods.mbpt` ships closed-shell second-order MBPT
for the *correlation energy* of the whole atom, transition-rate
work needs the second-order shift to *individual orbital energies*
so that excitation frequencies inherit the correlation contribution.

This module ships a thin, pragmatic wrapper:

* :func:`mbpt2_orbital_energy_shift(scf, orbital_label)` — returns
  the diagonal MBPT(2) self-energy estimate of orbital ``a`` in the
  closed-shell-frozen-core approximation.

* :func:`mbpt2_transition_energy(scf, alpha, beta)` — returns the
  correlation-corrected ω = (ε_β + ΔE_β) − (ε_α + ΔE_α).

The diagonal-MBPT estimate is

    ΔE_a ≈ ⟨a | Σ^(2) | a⟩

where Σ^(2) is the second-order self-energy operator.  In the
closed-shell DHF reference this equals the trace of the full
2nd-order amplitude tensor over occupied b ≠ a:

    ΔE_a = Σ_{b∈occ, b≠a} Σ_{r∈virt} Σ_{s∈virt}
              |V(a,b → r,s)|² / (ε_a + ε_b − ε_r − ε_s) · (1 / (2j_a+1))

For atoms with only one occupied subshell (He, He-like, hydrogenic)
the b ≠ a sum is empty and ΔE_a = 0 — the leading correlation
shift comes from the *off-diagonal* ⟨a | Σ^(2) | b⟩ matrix elements
which require an all-order treatment beyond MBPT(2).

A full TransitionMBPT solver — including off-diagonal matrix-element
corrections via RPA / core-polarization — is a separate effort
tracked as a remaining gap in REPORT_FAC_PARITY.md.

Reference
---------
* I. Lindgren & J. Morrison, *Atomic Many-Body Theory*, 2nd ed.
  (Springer, 1986), §13.5.
* M. F. Gu, *Astrophys. J.* **657**, 1172 (2007) — FAC
  ``TransitionMBPT``.
"""

from __future__ import annotations

from atomx.constants import HARTREE_EV


def mbpt2_orbital_energy_shift(
    scf_result,
    orbital_label: str,
    *,
    kappa_max: int = 2,
    energy_window_au: float = 50.0,
) -> float:
    """Diagonal second-order MBPT self-energy of one occupied orbital.

    Returns 0 when the reference has only the named orbital occupied
    (e.g. He 1s, He-like 1s²) — in those cases the leading shift
    comes from off-diagonal matrix elements that are not handled by
    this thin helper.  Use the full
    :func:`atomx.methods.mbpt.mbpt2_energy` for the closed-shell
    correlation energy or extend the function for transition-specific
    matrix elements.

    Parameters
    ----------
    scf_result
        Solved closed-shell SCFResult.
    orbital_label : str
        Subshell label (e.g. ``"2s1/2"``).
    kappa_max : int
        Largest |κ| for the virtual partial-wave sum (default 2).
    energy_window_au : float
        Cap on virtual energy in atomic units; states with ε > this
        are skipped to avoid integrable-but-noisy pseudo-continuum
        contributions (default 50 a.u.).

    Returns
    -------
    float
        ΔE_a in Hartree.
    """
    from atomx.angular.coefficients import exchange_lambda
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.numerics.bspline.convol import (
        ConvolCache,
        compute_slater_rk_cell,
    )

    sh_a = next((s for s in scf_result.subshells if s.label == orbital_label),
                 None)
    if sh_a is None:
        raise ValueError(
            f"orbital {orbital_label!r} not found among "
            f"{[s.label for s in scf_result.subshells]}"
        )
    bsp_data = scf_result._bsp_data
    cache = ConvolCache(bsp_data)
    eps_a = scf_result.orbital_energies[orbital_label]
    sh_a_subs = [s for s in scf_result.subshells if s.kappa == sh_a.kappa]
    P_a, Q_a, _ = scf_result.orbitals_raw[sh_a.kappa][sh_a_subs.index(sh_a)]
    kappa_a = sh_a.kappa

    # Find other occupied subshells b ≠ a
    other_occ = [s for s in scf_result.subshells if s.label != orbital_label]
    if not other_occ:
        return 0.0  # No b ≠ a — diagonal MBPT2 vanishes for this orbital

    delta_E = 0.0
    kappa_list_virt = list(range(-kappa_max, kappa_max + 1))
    kappa_list_virt = [k for k in kappa_list_virt if k != 0]

    tj_a = 2 * abs(kappa_a) - 1

    for sh_b in other_occ:
        eps_b = scf_result.orbital_energies[sh_b.label]
        kappa_b = sh_b.kappa
        sh_b_subs = [s for s in scf_result.subshells if s.kappa == kappa_b]
        P_b, Q_b, _ = scf_result.orbitals_raw[kappa_b][sh_b_subs.index(sh_b)]

        for kappa_r in kappa_list_virt:
            try:
                _, virt_r = build_virtual_basis(
                    scf_result, kappa_r, cache=cache,
                    potential_mode="VN-1",
                )
            except Exception:
                continue
            for (P_r, Q_r, eps_r) in virt_r:
                if eps_r > energy_window_au or eps_r < 0:
                    continue

                for kappa_s in kappa_list_virt:
                    try:
                        _, virt_s = build_virtual_basis(
                            scf_result, kappa_s, cache=cache,
                            potential_mode="VN-1",
                        )
                    except Exception:
                        continue
                    for (P_s, Q_s, eps_s) in virt_s:
                        if eps_s > energy_window_au or eps_s < 0:
                            continue
                        denom = eps_a + eps_b - eps_r - eps_s
                        if abs(denom) < 1e-6:
                            continue

                        # Allowed multipoles for direct (a,r)+(b,s)
                        ang_ar = exchange_lambda(kappa_a, kappa_r, 1)
                        ang_bs = exchange_lambda(kappa_b, kappa_s, 1)
                        if abs(ang_ar) < 1e-12 or abs(ang_bs) < 1e-12:
                            continue
                        # k = 1 dominant multipole only (efficient
                        # cap; for tighter calc add a k loop and
                        # appropriate Wigner reduction)
                        Rk = compute_slater_rk_cell(
                            P_a, Q_a, P_b, Q_b,
                            P_r, Q_r, P_s, Q_s,
                            bsp_data, 1, cache=cache,
                        )
                        V = ang_ar * ang_bs * Rk
                        delta_E += -(V * V) / (denom * (tj_a + 1))
    return float(delta_E)


def mbpt2_transition_energy(
    scf_result,
    alpha_label: str,
    beta_label: str,
    *,
    kappa_max: int = 2,
) -> dict:
    """MBPT(2)-corrected transition energy ω_corr = ε_β − ε_α + ΔE_β − ΔE_α.

    Returns a dict including the DHF reference, per-orbital diagonal
    self-energy shifts, and the corrected ω in Hartree and eV.

    For systems with a single occupied orbital both shifts are zero
    by construction (see :func:`mbpt2_orbital_energy_shift`); the
    dict still reports the DHF reference for downstream consistency.

    Parameters
    ----------
    scf_result : SCFResult
    alpha_label, beta_label : str
    kappa_max : int

    Returns
    -------
    dict
        {'omega_DHF_au', 'delta_E_alpha_au', 'delta_E_beta_au',
         'omega_corrected_au', 'omega_corrected_eV'}
    """
    eps_a = scf_result.orbital_energies[alpha_label]
    eps_b = scf_result.orbital_energies[beta_label]
    omega_DHF = eps_b - eps_a
    dE_a = mbpt2_orbital_energy_shift(scf_result, alpha_label,
                                       kappa_max=kappa_max)
    dE_b = mbpt2_orbital_energy_shift(scf_result, beta_label,
                                       kappa_max=kappa_max)
    omega_corr = omega_DHF + dE_b - dE_a
    return {
        "omega_DHF_au": float(omega_DHF),
        "delta_E_alpha_au": float(dE_a),
        "delta_E_beta_au": float(dE_b),
        "omega_corrected_au": float(omega_corr),
        "omega_corrected_eV": float(omega_corr * HARTREE_EV),
    }


__all__ = [
    "mbpt2_orbital_energy_shift",
    "mbpt2_transition_energy",
]
