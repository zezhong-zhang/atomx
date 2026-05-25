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
from atomx.numerics.bspline.continuum import solve_continuum
from atomx.methods.kohn_sham import compute_slater_rk
from atomx.prop.transitions.angular import _kappa_to_lj, selection_rules, angular_eL
from atomx.prop.transitions.multipole import radial_eL
from atomx.prop.transitions.rates import oscillator_strength
from atomx.scattering._jit_kernels import dw_excitation_sigma_sum


# Cross-section unit: pi * a_0^2 in cm^2
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100) ** 2


def dw_excitation_cross_section(atom_initial, atom_final, energies_eV,
                                 L_max=4, kappa_max=10,
                                 transition_energy_eV=None):
    """Compute electron-impact excitation cross-section sigma(E).

    Uses the DW approximation: the incoming/outgoing electron wavefunctions
    are computed in the static potential of the initial/final atomic state.

    The inner partial-wave sum is JIT-compiled via numba for performance.

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
    transition_energy_eV : float or array_like, optional
        Override transition threshold energy.

    Returns
    -------
    sigma : ndarray
        Cross-section in cm^2 at each energy.
    omega : ndarray
        Collision strength (dimensionless) at each energy.
    """
    if L_max < 0:
        raise ValueError("L_max must be non-negative")
    if kappa_max < 1:
        raise ValueError("kappa_max must be positive")

    energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))

    # Extract structure data
    result_i = atom_initial._result
    result_f = atom_final._result
    bsp_data = atom_initial._bsp_data

    if result_i is None or result_f is None:
        raise ValueError("Both atoms must be solved.")

    Z = atom_initial.Z

    # Transition energy (difference in total energies), optionally overridden
    dE_hartree = result_f.total_energy - result_i.total_energy
    if transition_energy_eV is not None:
        dE_override = np.asarray(transition_energy_eV, dtype=float)
        if dE_override.ndim == 0:
            dE_override = np.full_like(energies_eV, float(dE_override))
        else:
            dE_override = np.broadcast_to(dE_override, energies_eV.shape)
        dE_hartree_grid = dE_override / HARTREE_EV
    else:
        dE_hartree_grid = np.full_like(energies_eV, dE_hartree)
    if np.all(dE_hartree_grid <= 0):
        raise ValueError("Final state must have higher energy than initial.")

    # Build SCF potential matrices for continuum calculation
    V_J_PP_i, V_J_QQ_i = _extract_scf_potential(atom_initial)

    # Find the orbital that differs (the "active" electron)
    orb_i_label, orb_f_label, P_bound_i, Q_bound_i, kappa_i, P_bound_f, Q_bound_f, kappa_f = \
        _find_active_orbital(result_i, result_f)

    if P_bound_i is None:
        return np.zeros_like(energies_eV), np.zeros_like(energies_eV)

    _, tj_i = _kappa_to_lj(kappa_i)

    n_energies = len(energies_eV)
    E_inc_hartree = energies_eV / HARTREE_EV

    # Build kappa arrays (exclude 0)
    kappa_inc_arr = np.array([k for k in range(-kappa_max, kappa_max + 1) if k != 0],
                             dtype=np.int64)
    kappa_scat_arr = kappa_inc_arr  # same range
    k_arr = np.arange(0, L_max + 1, dtype=np.int64)

    n_ki = len(kappa_inc_arr)
    n_ks = len(kappa_scat_arr)
    n_k = len(k_arr)

    # Pre-solve all continuum waves and pre-compute Slater integrals.
    # The continuum solve is Python-level (uses bsp_data dicts), but we
    # batch it so the inner numba kernel gets pure arrays.
    slater_direct = np.full((n_energies, n_ki, n_ks, n_k), np.nan)
    slater_exchange = np.full((n_energies, n_ki, n_ks, n_k), np.nan)

    # Cache continuum solutions: (kappa, E_hartree) -> (P, Q)
    continuum_cache = {}

    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        E_scattered = E_inc - dE_hartree_grid[idx]
        if E_scattered <= 0:
            continue

        # Solve continuum for all kappa at this energy (with caching)
        for iki, kappa_inc in enumerate(kappa_inc_arr):
            key_in = (int(kappa_inc), idx)
            if key_in not in continuum_cache:
                try:
                    P_inc, Q_inc, _ = solve_continuum(
                        int(kappa_inc), E_inc, Z, bsp_data,
                        V_J_PP_i, V_J_QQ_i, energy_normalized=True)
                    continuum_cache[key_in] = (P_inc, Q_inc)
                except Exception:
                    continuum_cache[key_in] = None
            inc_result = continuum_cache[key_in]
            if inc_result is None:
                continue
            P_inc, Q_inc = inc_result

            for iks, kappa_scat in enumerate(kappa_scat_arr):
                key_scat = (int(kappa_scat), idx, 1)  # 1 = scattered
                # Use energy as scatter key
                key_scat = (int(kappa_scat), round(E_scattered, 12))
                if key_scat not in continuum_cache:
                    try:
                        P_scat, Q_scat, _ = solve_continuum(
                            int(kappa_scat), E_scattered, Z, bsp_data,
                            V_J_PP_i, V_J_QQ_i, energy_normalized=True)
                        continuum_cache[key_scat] = (P_scat, Q_scat)
                    except Exception:
                        continuum_cache[key_scat] = None
                scat_result = continuum_cache[key_scat]
                if scat_result is None:
                    continue
                P_scat, Q_scat = scat_result

                for ik, k in enumerate(k_arr):
                    # Direct: R^k(bound_i, inc; bound_f, scat)
                    try:
                        Rk_d = compute_slater_rk(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_bound_f, Q_bound_f, P_scat, Q_scat,
                            bsp_data, int(k))
                        slater_direct[idx, iki, iks, ik] = Rk_d
                    except Exception:
                        pass

                    # Exchange: R^k(bound_i, inc; scat, bound_f)
                    try:
                        Rk_e = compute_slater_rk(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_scat, Q_scat, P_bound_f, Q_bound_f,
                            bsp_data, int(k))
                        slater_exchange[idx, iki, iks, ik] = Rk_e
                    except Exception:
                        pass

    # JIT-compiled sigma assembly
    sigma_au, omega = dw_excitation_sigma_sum(
        n_energies, kappa_inc_arr, kappa_scat_arr, k_arr,
        slater_direct, slater_exchange, E_inc_hartree,
        dE_hartree_grid, tj_i,
    )

    # Convert from atomic units (a_0^2) to cm^2
    sigma = sigma_au * (BOHR_RADIUS_M * 100) ** 2

    # FAC blends/augments high partial waves with a Coulomb-Bethe limit for
    # optically allowed transitions.
    zero_mask = (sigma <= 0.0) & (energies_eV > dE_hartree_grid * HARTREE_EV)
    if np.any(zero_mask) and selection_rules(kappa_i, kappa_f, 1, "E"):
        omega0 = float(np.mean(dE_hartree_grid))
        if omega0 > 0.0:
            rad = radial_eL(P_bound_i, Q_bound_i, P_bound_f, Q_bound_f, bsp_data, L=1)
            ang = angular_eL(kappa_i, kappa_f, 1)
            S_optical = (ang * rad) ** 2
            f_optical, _gf = oscillator_strength(omega0, S_optical, 1, tj_i, _kappa_to_lj(kappa_f)[1])
            omega_bethe = coulomb_bethe_topup(
                f_optical,
                energies_eV,
                float(omega0 * HARTREE_EV),
                L_truncation=max(1, L_max),
            )
            k_i_sq_grid = 2.0 * energies_eV / HARTREE_EV
            sigma_bethe_au = np.divide(
                np.pi * omega_bethe,
                k_i_sq_grid * (tj_i + 1),
                out=np.zeros_like(omega_bethe),
                where=k_i_sq_grid > 0.0,
            )
            sigma_bethe = sigma_bethe_au * (BOHR_RADIUS_M * 100) ** 2
            sigma[zero_mask] = sigma_bethe[zero_mask]
            omega[zero_mask] = omega_bethe[zero_mask]

    return sigma, omega


def _extract_scf_potential(atom, include_local_exchange: bool = False):
    """Extract the direct (Hartree) potential matrices from a solved atom."""
    result = atom._result
    bsp_data = atom._bsp_data

    if result is None or bsp_data is None:
        return None, None

    from atomx.methods.dirac_fock import DFSolver
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
    """Find the orbital that differs between initial and final states."""
    occ_i = {(sh.n, sh.kappa): float(sh.occupation) for sh in result_i.subshells}
    occ_f = {(sh.n, sh.kappa): float(sh.occupation) for sh in result_f.subshells}
    sources = []
    destinations = []
    for key in sorted(set(occ_i) | set(occ_f)):
        delta = occ_f.get(key, 0.0) - occ_i.get(key, 0.0)
        if delta < -0.01:
            sources.append((abs(delta), key))
        elif delta > 0.01:
            destinations.append((abs(delta), key))

    if len(sources) != 1 or not destinations:
        return None, None, None, None, None, None, None, None

    _, source_key = max(sources)
    _, destination_key = max(destinations)
    orb_i_label = _label_for_key(result_i, source_key)
    orb_f_label = _label_for_key(result_f, destination_key)
    if orb_i_label is None or orb_f_label is None:
        return None, None, None, None, None, None, None, None

    P_i, Q_i, kappa_i = _get_orbital_by_key(result_i, source_key)
    P_f, Q_f, kappa_f = _get_orbital_by_key(result_f, destination_key)

    return orb_i_label, orb_f_label, P_i, Q_i, kappa_i, P_f, Q_f, kappa_f


def _label_for_key(result, key):
    for sh in result.subshells:
        if (sh.n, sh.kappa) == key:
            return sh.label
    return None


def _get_orbital_by_key(result, key):
    """Get orbital P, Q coefficients and kappa from SCFResult by (n, kappa)."""
    kappa_counter = {}
    for sh in result.subshells:
        idx = kappa_counter.get(sh.kappa, 0)
        kappa_counter[sh.kappa] = idx + 1
        if (sh.n, sh.kappa) == key:
            orb_list = result.orbitals_raw.get(sh.kappa, [])
            if idx < len(orb_list):
                P, Q, occ = orb_list[idx]
                return P, Q, sh.kappa
    return None, None, None


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

    weights = np.exp(-energy_eV_array / kT)
    integrand = omega_array * weights
    _trap = getattr(np, "trapezoid", None) or np.trapz
    rate = _trap(integrand, energy_eV_array)

    from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
    a0_cm = BOHR_RADIUS_M * 100
    prefactor = 2.0 * np.pi * a0_cm ** 2 / np.sqrt(kT / HARTREE_EV)
    return prefactor * rate


# ---------------------------------------------------------------------------
# Coulomb-Bethe high-l top-up for dipole-allowed transitions
# ---------------------------------------------------------------------------


def coulomb_bethe_topup(
    f_ab: float,
    energies_eV: np.ndarray,
    dE_eV: float,
    L_truncation: int,
) -> np.ndarray:
    """High-l Coulomb-Bethe (Burgess-Sheorey) top-up for the dipole-
    allowed collision strength.

    Parameters
    ----------
    f_ab
        Optical (dipole) oscillator strength of the i -> f transition.
    energies_eV
        Incident-electron energies, eV.
    dE_eV
        Excitation energy E_f - E_i, eV.
    L_truncation
        l value at which the DW sum was truncated.

    Returns
    -------
    np.ndarray
        Top-up Omega(E) per energy on the same grid as ``energies_eV``.
    """
    from atomx.constants import HARTREE_EV
    energies_au = np.asarray(energies_eV, dtype=float) / HARTREE_EV
    omega_au = dE_eV / HARTREE_EV
    out = np.zeros_like(energies_au)
    above = energies_au > omega_au + 1e-12
    if not np.any(above):
        return out
    k_i_sq = 2.0 * energies_au[above]
    k_f_sq = 2.0 * (energies_au[above] - omega_au)
    k_i = np.sqrt(k_i_sq)
    k_f = np.sqrt(k_f_sq)
    log_arg = (k_i + k_f) / np.maximum(k_i - k_f, 1e-30)
    out[above] = (4.0 * f_ab / omega_au) * np.log(log_arg)
    suppression = max(0.0, 1.0 - 1.0 / max(L_truncation + 1, 2))
    return out * suppression


__all__ = [
    "dw_excitation_cross_section",
    "collision_strength_to_rate",
    "coulomb_bethe_topup",
]
