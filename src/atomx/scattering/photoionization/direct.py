"""Single-channel electric-dipole photoionization on a B-spline continuum.

Scattering owns channel selection and cross-section normalization. Numerical
continuum eigenproblems and radial matrix elements remain in numerics.
"""
import numpy as np

from atomx.constants import ALPHA
from atomx.numerics.bspline.continuum import solve_continuum, dipole_matrix_element


def photoionization_cross_section(bound_P, bound_Q, bound_kappa, bound_energy,
                                   photon_energies, Z, bsp_data,
                                   V_scf_PP=None, V_scf_QQ=None):
    """Compute photoionization cross-section σ(ω) for a bound orbital.

    σ(ω) = (4π²αω/3) Σ_{κ_f} |<f|r|i>|² × angular_factor

    Parameters
    ----------
    bound_P, bound_Q : ndarray
        Bound orbital coefficients.
    bound_kappa : int
        Bound orbital kappa.
    bound_energy : float
        Bound orbital energy (negative, in Hartree).
    photon_energies : ndarray
        Photon energies in Hartree.
    Z : int
    bsp_data : dict
    V_scf_PP, V_scf_QQ : ndarray, optional

    Returns
    -------
    sigma : ndarray
        Cross-section in atomic units (a_0^2) at each photon energy.

    Notes
    -----
    The bound-free formula assumes continuum orbitals normalized to
    ``delta(E-E')``.  The B-spline diagonalization produces unit-box
    pseudo-states, so this public path must request ``energy_normalized=True``.
    Callers that need raw pseudo-states should use :func:`solve_continuum`
    directly.
    """
    photon_energies = np.asarray(photon_energies, dtype=float)
    if photon_energies.ndim != 1 or not np.all(np.isfinite(photon_energies)):
        raise ValueError('photon_energies must be a finite one-dimensional array')
    sigma = np.zeros_like(photon_energies)

    # Allowed final-state kappa values (E1 dipole selection rules: Δl = ±1)
    l_bound = bound_kappa if bound_kappa > 0 else -bound_kappa - 1

    # E1 dipole: final states with l_f = l_bound ± 1
    final_kappas = []
    for l_f in [l_bound - 1, l_bound + 1]:
        if l_f < 0:
            continue
        # j_f = l_f ± 1/2
        final_kappas.append(-(l_f + 1))  # kappa for j = l+1/2
        if l_f > 0:
            final_kappas.append(l_f)      # kappa for j = l-1/2

    from atomx.properties.transitions.angular import line_strength_factor

    for idx, omega in enumerate(photon_energies):
        electron_energy = omega + bound_energy  # kinetic energy of photoelectron
        if electron_energy <= 0:
            continue  # below threshold

        for kf in final_kappas:
            angular_factor = (
                line_strength_factor(bound_kappa, kf, 1, "E")
                / (2.0 * abs(bound_kappa))
            )
            if angular_factor == 0.0:
                continue

            Pf, Qf, E_actual = solve_continuum(
                kf, electron_energy, Z, bsp_data, V_scf_PP, V_scf_QQ,
                energy_normalized=True)

            dme = dipole_matrix_element(bound_P, bound_Q, Pf, Qf, bsp_data)

            # σ = 4π²αω/3 × |<f|r|i>|² × (2j_f+1)
            sigma[idx] += (4.0 * np.pi ** 2 * ALPHA * omega / 3.0
                          * dme ** 2 * angular_factor)

    return sigma
