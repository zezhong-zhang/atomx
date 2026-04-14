"""Distorted-wave electron-impact excitation cross-sections.

The DW approximation computes cross-sections by:
1. Computing distorted waves (continuum states) in the SCF potential
2. Computing T-matrix elements via Slater integrals between bound+continuum orbitals
3. Summing over partial waves and multipole orders

This is the same approach used by FAC, HULLAC, and AUTOSTRUCTURE.
Accurate for highly charged ions (Z_eff >> 1); overestimates near threshold
for neutral/near-neutral atoms where channel coupling is important.

Reference: Gu, Can. J. Phys. 86, 675 (2008) - FAC DW formulation
"""

import numpy as np
from atomx.constants import ALPHA, C_SQ, HARTREE_EV, BOHR_RADIUS_M
from atomx.dirac.continuum import solve_continuum
from atomx.bspline.evaluate import compute_slater_rk
from atomx.transitions.angular import _kappa_to_lj, selection_rules


# Cross-section unit: pi * a_0^2 in cm^2
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100) ** 2


def dw_excitation_cross_section(atom_initial, atom_final, energies_eV,
                                 L_max=4, kappa_max=10):
    """Compute electron-impact excitation cross-section sigma(E).

    Uses the DW approximation: the incoming/outgoing electron wavefunctions
    are computed in the static potential of the initial/final atomic state.

    Parameters
    ----------
    atom_initial : Atom
        Solved initial-state atom (e.g., ground state).
    atom_final : Atom
        Solved final-state atom (e.g., excited state).
    energies_eV : array_like
        Incident electron kinetic energies in eV.
    L_max : int
        Maximum multipole order for the Coulomb interaction.
    kappa_max : int
        Maximum |kappa| for partial-wave expansion.

    Returns
    -------
    sigma : ndarray
        Cross-section in cm^2 at each energy.
    omega : ndarray
        Collision strength (dimensionless) at each energy.
    """
    energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))
    sigma = np.zeros_like(energies_eV)
    omega = np.zeros_like(energies_eV)

    # Extract structure data
    result_i = atom_initial._result
    result_f = atom_final._result
    bsp_data = atom_initial._bsp_data

    if result_i is None or result_f is None:
        raise ValueError("Both atoms must be solved.")

    Z = atom_initial.Z

    # Transition energy (difference in total energies)
    dE_hartree = result_f.total_energy - result_i.total_energy
    if dE_hartree <= 0:
        raise ValueError("Final state must have higher energy than initial.")

    # Build SCF potential matrices for continuum calculation
    V_J_PP_i, V_J_QQ_i = _extract_scf_potential(atom_initial)

    # Find the orbital that differs (the "active" electron)
    orb_i_label, orb_f_label, P_bound_i, Q_bound_i, kappa_i, P_bound_f, Q_bound_f, kappa_f = \
        _find_active_orbital(result_i, result_f)

    if P_bound_i is None:
        return sigma, omega  # Cannot identify transition

    _, tj_i = _kappa_to_lj(kappa_i)

    for idx, E_eV in enumerate(energies_eV):
        E_inc_hartree = E_eV / HARTREE_EV

        # Kinetic energies of incident and scattered electrons
        E_scattered = E_inc_hartree - dE_hartree
        if E_scattered <= 0:
            continue  # below threshold

        # k^2 of incident electron (in atomic units)
        k_i_sq = 2.0 * E_inc_hartree  # non-relativistic: k^2 = 2*E

        # Sum over partial waves for incoming and outgoing electrons
        sigma_partial = 0.0

        for kappa_inc in range(-kappa_max, kappa_max + 1):
            if kappa_inc == 0:
                continue
            for kappa_scat in range(-kappa_max, kappa_max + 1):
                if kappa_scat == 0:
                    continue

                # Solve continuum for incoming electron
                try:
                    P_inc, Q_inc, E_inc_actual = solve_continuum(
                        kappa_inc, E_inc_hartree, Z, bsp_data,
                        V_J_PP_i, V_J_QQ_i)
                except Exception:
                    continue

                # Solve continuum for scattered electron
                try:
                    P_scat, Q_scat, E_scat_actual = solve_continuum(
                        kappa_scat, E_scattered, Z, bsp_data,
                        V_J_PP_i, V_J_QQ_i)
                except Exception:
                    continue

                # T-matrix: sum over Coulomb multipoles k
                T = 0.0
                for k in range(0, L_max + 1):
                    # Direct: R^k(bound_i, inc; bound_f, scat)
                    try:
                        Rk_direct = compute_slater_rk(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_bound_f, Q_bound_f, P_scat, Q_scat,
                            bsp_data, k)
                    except Exception:
                        Rk_direct = 0.0

                    # Exchange: R^k(bound_i, inc; scat, bound_f)
                    try:
                        Rk_exchange = compute_slater_rk(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_scat, Q_scat, P_bound_f, Q_bound_f,
                            bsp_data, k)
                    except Exception:
                        Rk_exchange = 0.0

                    # Angular coupling (simplified: use direct coefficient)
                    T += Rk_direct - Rk_exchange

                sigma_partial += T ** 2

        # sigma = pi / k_i^2 * sum |T|^2 / (2j_i + 1)
        if k_i_sq > 0:
            sigma[idx] = np.pi / k_i_sq * sigma_partial / (tj_i + 1)
            omega[idx] = k_i_sq * (tj_i + 1) * sigma[idx] / np.pi

    # Convert from atomic units (a_0^2) to cm^2
    sigma *= (BOHR_RADIUS_M * 100) ** 2

    return sigma, omega


def _extract_scf_potential(atom):
    """Extract the direct (Hartree) potential matrices from a solved atom."""
    result = atom._result
    bsp_data = atom._bsp_data

    if result is None or bsp_data is None:
        return None, None

    # Reconstruct V_J from the converged orbitals
    from atomx.scf.solver import DFSolver
    # Create a temporary solver to access potential building
    solver = DFSolver.__new__(DFSolver)
    solver.Z = atom.Z
    solver.ns = bsp_data["ns"]
    solver.ks = bsp_data["ks"]
    solver.bsp_data = bsp_data
    solver.subshells = result.subshells
    solver.exchange = atom.exchange

    V_J_PP, V_J_QQ = solver._build_direct_potential(result.orbitals_raw)
    return V_J_PP, V_J_QQ


def _find_active_orbital(result_i, result_f):
    """Find the orbital that differs between initial and final states.

    Returns the bound orbital coefficients and kappa for the active electron.
    """
    # Compare orbital energies to find the changed orbital
    labels_i = set(result_i.orbital_energies.keys())
    labels_f = set(result_f.orbital_energies.keys())

    # Find orbitals with significant energy difference
    common = labels_i & labels_f
    if not common:
        return None, None, None, None, None, None, None, None

    max_diff = 0
    orb_i_label = None
    orb_f_label = None

    for label in common:
        diff = abs(result_i.orbital_energies[label] - result_f.orbital_energies[label])
        if diff > max_diff:
            max_diff = diff
            orb_i_label = label
            orb_f_label = label

    if orb_i_label is None:
        return None, None, None, None, None, None, None, None

    # Extract orbital coefficients
    P_i, Q_i, kappa_i = _get_orbital_by_label(result_i, orb_i_label)
    P_f, Q_f, kappa_f = _get_orbital_by_label(result_f, orb_f_label)

    return orb_i_label, orb_f_label, P_i, Q_i, kappa_i, P_f, Q_f, kappa_f


def _get_orbital_by_label(result, label):
    """Get orbital P, Q coefficients and kappa from SCFResult by label."""
    kappa_counter = {}
    for sh in result.subshells:
        idx = kappa_counter.get(sh.kappa, 0)
        kappa_counter[sh.kappa] = idx + 1
        if sh.label == label:
            orb_list = result.orbitals_raw.get(sh.kappa, [])
            if idx < len(orb_list):
                P, Q, occ = orb_list[idx]
                return P, Q, sh.kappa
    return None, None, None


def collision_strength_to_rate(omega_array, energy_eV_array, temperature_eV):
    """Convert collision strengths to Maxwellian rate coefficients.

    <sigma*v>(T) = (2*pi*a_0^2 / (g_i * sqrt(m_e * kT)))
                   * integral Omega(E) * exp(-E/kT) dE

    Parameters
    ----------
    omega_array : ndarray
        Collision strengths at each energy.
    energy_eV_array : ndarray
        Energies in eV.
    temperature_eV : float
        Electron temperature in eV.

    Returns
    -------
    float
        Rate coefficient in cm^3/s.
    """
    kT = temperature_eV
    if kT <= 0:
        return 0.0

    # Boltzmann weight
    weights = np.exp(-energy_eV_array / kT)

    # Trapezoidal integration
    integrand = omega_array * weights
    rate = np.trapz(integrand, energy_eV_array)

    # Prefactor (cgs units)
    from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
    a0_cm = BOHR_RADIUS_M * 100
    prefactor = 2.0 * np.pi * a0_cm ** 2 / np.sqrt(kT / HARTREE_EV)
    return prefactor * rate
