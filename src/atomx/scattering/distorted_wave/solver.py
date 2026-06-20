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

import math

import numpy as np
from atomx.constants import ALPHA, C_SQ, HARTREE_EV, BOHR_RADIUS_M
from atomx.numerics.bspline.continuum import solve_continuum
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.prop.transitions.angular import _kappa_to_lj, selection_rules, angular_eL
from atomx.prop.transitions.multipole import radial_eL
from atomx.prop.transitions.rates import oscillator_strength
from atomx.angular.csf import csf_parity, csf_twice_j, one_body_csf_coupling
from atomx.angular.wigner import wigner_3j, wigner_6j
from atomx.methods.mcdhf.config import CSF
from atomx.scattering._jit_kernels import dw_excitation_angular_sum


def _l_of(kappa):
    """Orbital angular momentum l from kappa."""
    return kappa if kappa > 0 else -kappa - 1


def _tj_of(kappa):
    """Twice-j from kappa."""
    return 2 * abs(kappa) - 1


def _reduced_C(kappa_a, kappa_b, k):
    """Reduced matrix element ⟨κ_a‖C^k‖κ_b⟩.

    Follows Grant (2007) convention:
        (-1)^{j_a+1/2} · √((2j_a+1)(2j_b+1)) · 3j(j_a, k, j_b; -1/2, 0, 1/2)
    """
    la, lb = _l_of(kappa_a), _l_of(kappa_b)
    if (la + k + lb) % 2 != 0:
        return 0.0
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    tL = 2 * k
    if tL < abs(tja - tjb) or tL > tja + tjb:
        return 0.0
    w = wigner_3j(tja, tL, tjb, -1, 0, 1)
    if abs(w) < 1e-15:
        return 0.0
    phase_exp = (tja + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0
    return phase * math.sqrt((tja + 1) * (tjb + 1)) * w


def _allowed_k(kappa_a, kappa_b):
    """Allowed multipole k values for reduced C^k between two kappa states.

    Selection rules: |j_a - j_b| ≤ k ≤ j_a + j_b, l_a + l_b + k even.
    """
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    la, lb = _l_of(kappa_a), _l_of(kappa_b)
    kmin = abs(tja - tjb) // 2
    kmax = (tja + tjb) // 2
    return [k for k in range(kmin, kmax + 1) if (k + la + lb) % 2 == 0]


def _dominant_csf(result):
    """Return the dominant CSF for the lowest state, if present."""
    csfs = getattr(result, "csfs", None)
    if not csfs:
        return None
    coeffs = getattr(result, "mixing_coefficients", None)
    if coeffs is None or np.size(coeffs) == 0:
        return csfs[0]
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 2 or coeffs.shape[0] == 0:
        return csfs[0]
    idx = int(np.argmax(np.abs(coeffs[:, 0])))
    return csfs[idx] if idx < len(csfs) else csfs[0]


def _csf_from_result_subshells(result):
    """Build a lightweight CSF from a single-configuration result."""
    subshells = list(getattr(result, "subshells", []))
    parity = 1
    for sh in subshells:
        if int(round(float(sh.occupation))) % 2:
            parity *= -1 if _l_of(sh.kappa) % 2 else 1
    csf = CSF(subshells=subshells, total_J2=0, parity=parity, label="scf")
    tj = csf_twice_j(csf)
    return CSF(subshells=subshells, total_J2=tj, parity=parity, label=f"scf [J={tj}/2]")


def _target_state_metadata(result):
    """Return ``(2J, parity, csf)`` for the target state represented by result."""
    csf = _dominant_csf(result)
    if csf is not None:
        state_j2 = getattr(result, "state_J2", None)
        state_parity = getattr(result, "state_parity", None)
        if state_j2 is not None and len(state_j2) > 0:
            tj = int(state_j2[0])
        else:
            tj = csf_twice_j(csf)
        if state_parity is not None and len(state_parity) > 0:
            parity = int(state_parity[0])
        else:
            parity = csf_parity(csf)
        return tj, parity, csf

    csf = _csf_from_result_subshells(result)
    return csf_twice_j(csf), csf_parity(csf), csf


def _target_projectile_J_range(target_tj_i, target_tj_f, tj_in, tj_out):
    """Allowed total 2J range for target + projectile recoupling."""
    tJ_lo = max(abs(target_tj_i - tj_in), abs(target_tj_f - tj_out))
    tJ_hi = min(target_tj_i + tj_in, target_tj_f + tj_out)
    if tJ_lo > tJ_hi:
        return None
    return tJ_lo, tJ_hi


def _target_transition_population(csf_i, csf_f, k, kappa_i, kappa_f):
    """CSF population/selection factor for a one-electron target tensor."""
    coupling = one_body_csf_coupling(csf_f, csf_i, k, "E")
    if coupling is None:
        return 0.0
    if coupling.source[1] != kappa_i or coupling.destination[1] != kappa_f:
        return 0.0
    return coupling.population_factor


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

    target_tj_i, _target_parity_i, _target_csf_i = _target_state_metadata(result_i)
    target_tj_f, _target_parity_f, _target_csf_f = _target_state_metadata(result_f)
    l_i = _l_of(kappa_i)
    l_f = _l_of(kappa_f)

    n_energies = len(energies_eV)
    E_inc_hartree = energies_eV / HARTREE_EV

    # Build kappa arrays (exclude 0)
    kappa_inc_arr = np.array([k for k in range(-kappa_max, kappa_max + 1) if k != 0],
                             dtype=np.int64)
    kappa_scat_arr = kappa_inc_arr

    # Cache continuum solutions and Slater radial cells for this energy grid.
    continuum_cache = {}
    slater_cache = ConvolCache(bsp_data)

    sigma_au = np.zeros(n_energies)
    omega = np.zeros(n_energies)

    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        E_scattered = E_inc - dE_hartree_grid[idx]
        if E_scattered <= 0:
            continue
        k_i_sq = 2.0 * E_inc

        _ch_tj_in = []
        _ch_tj_out = []
        _ch_tJ_lo = []
        _ch_tJ_hi = []
        _ch_n_d = []
        _ch_n_e = []
        _direct_products = []
        _exchange_products = []
        _w6j_direct = []
        _w6j_exchange = []

        for iki, kappa_inc in enumerate(kappa_inc_arr):
            tj_in = _tj_of(kappa_inc)
            l_in = _l_of(kappa_inc)

            # Solve incoming continuum wave (with caching)
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
                tj_out = _tj_of(kappa_scat)
                l_out = _l_of(kappa_scat)

                # Parity conservation: l_i + l_in + l_f + l_out must be even
                if (l_i + l_in + l_f + l_out) % 2 != 0:
                    continue

                # Total J range for target + projectile channels.
                tJ_range = _target_projectile_J_range(
                    target_tj_i, target_tj_f, tj_in, tj_out)
                if tJ_range is None:
                    continue
                tJ_lo, tJ_hi = tJ_range

                # Solve scattered continuum wave
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

                # Allowed k: direct (i→f, in→out) and exchange (i→out, in→f)
                kd_pair = [k for k in _allowed_k(kappa_i, kappa_f)
                           if k in _allowed_k(kappa_inc, kappa_scat) and k <= L_max]
                ke_pair = [k for k in _allowed_k(kappa_i, kappa_scat)
                           if k in _allowed_k(kappa_inc, kappa_f) and k <= L_max]
                if not kd_pair and not ke_pair:
                    continue

                # Direct products and 6j
                d_prods = []
                d_w6j = []
                for k in kd_pair:
                    pop = _target_transition_population(
                        _target_csf_i, _target_csf_f, k, kappa_i, kappa_f)
                    C_if = pop * _reduced_C(kappa_i, kappa_f, k)
                    C_inout = _reduced_C(kappa_inc, kappa_scat, k)
                    if abs(C_if) < 1e-15 or abs(C_inout) < 1e-15:
                        continue
                    try:
                        Rk = compute_slater_rk_cell(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_bound_f, Q_bound_f, P_scat, Q_scat,
                            bsp_data, int(k), cache=slater_cache)
                    except Exception:
                        continue
                    d_prods.append(C_if * C_inout * Rk)
                    n_J = (tJ_hi - tJ_lo) // 2 + 1
                    for iJ in range(n_J):
                        tJ = tJ_lo + 2 * iJ
                        d_w6j.append(wigner_6j(target_tj_i, tj_in, tJ, tj_out, target_tj_f, 2 * k))

                # Exchange products and 6j
                e_prods = []
                e_w6j = []
                for k in ke_pair:
                    pop = _target_transition_population(
                        _target_csf_i, _target_csf_f, k, kappa_i, kappa_f)
                    C_iout = pop * _reduced_C(kappa_i, kappa_scat, k)
                    C_inf = _reduced_C(kappa_inc, kappa_f, k)
                    if abs(C_iout) < 1e-15 or abs(C_inf) < 1e-15:
                        continue
                    try:
                        Rk = compute_slater_rk_cell(
                            P_bound_i, Q_bound_i, P_inc, Q_inc,
                            P_scat, Q_scat, P_bound_f, Q_bound_f,
                            bsp_data, int(k), cache=slater_cache)
                    except Exception:
                        continue
                    e_prods.append(C_iout * C_inf * Rk)
                    n_J = (tJ_hi - tJ_lo) // 2 + 1
                    for iJ in range(n_J):
                        tJ = tJ_lo + 2 * iJ
                        e_w6j.append(wigner_6j(target_tj_i, tj_in, tJ, target_tj_f, tj_out, 2 * k))

                if not d_prods and not e_prods:
                    continue

                _ch_tj_in.append(tj_in)
                _ch_tj_out.append(tj_out)
                _ch_tJ_lo.append(tJ_lo)
                _ch_tJ_hi.append(tJ_hi)
                _ch_n_d.append(len(d_prods))
                _ch_n_e.append(len(e_prods))
                _direct_products.extend(d_prods)
                _exchange_products.extend(e_prods)
                _w6j_direct.extend(d_w6j)
                _w6j_exchange.extend(e_w6j)

        # JIT-compiled angular coupling sum
        n_channels = len(_ch_tj_in)
        if n_channels > 0:
            omega_sum = dw_excitation_angular_sum(
                n_channels,
                target_tj_i,
                target_tj_f,
                np.array(_ch_tj_in, dtype=np.int64),
                np.array(_ch_tj_out, dtype=np.int64),
                np.array(_ch_tJ_lo, dtype=np.int64),
                np.array(_ch_tJ_hi, dtype=np.int64),
                np.array(_ch_n_d, dtype=np.int64),
                np.array(_ch_n_e, dtype=np.int64),
                np.array(_direct_products, dtype=np.float64),
                np.array(_exchange_products, dtype=np.float64),
                np.array(_w6j_direct, dtype=np.float64),
                np.array(_w6j_exchange, dtype=np.float64),
            )
            if k_i_sq > 0.0:
                sigma_au[idx] = np.pi / k_i_sq * omega_sum / (target_tj_i + 1.0)
                omega[idx] = omega_sum

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
            f_optical, _gf = oscillator_strength(omega0, S_optical, 1, target_tj_i, target_tj_f)
            omega_bethe = coulomb_bethe_topup(
                f_optical,
                energies_eV,
                float(omega0 * HARTREE_EV),
                L_truncation=max(1, L_max),
            )
            k_i_sq_grid = 2.0 * energies_eV / HARTREE_EV
            sigma_bethe_au = np.divide(
                np.pi * omega_bethe,
                k_i_sq_grid * (target_tj_i + 1),
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
