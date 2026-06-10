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
from atomx.angular.wigner import wigner_3j, wigner_6j
from atomx.prop.transitions.angular import _kappa_to_lj, selection_rules, angular_eL
from atomx.prop.transitions.multipole import radial_eL
from atomx.prop.transitions.rates import oscillator_strength
from atomx.scattering._jit_kernels import dw_excitation_inner_sum


# Cross-section unit: pi * a_0^2 in cm^2
PI_A0_SQ_CM2 = np.pi * (BOHR_RADIUS_M * 100) ** 2


def _l_of(kappa):
    return kappa if kappa > 0 else -kappa - 1


def _tj_of(kappa):
    return 2 * abs(kappa) - 1


def _allowed_k(kappa_a, kappa_b):
    ja, jb = abs(kappa_a) - 0.5, abs(kappa_b) - 0.5
    kmin, kmax = int(abs(ja - jb)), int(ja + jb)
    la, lb = _l_of(kappa_a), _l_of(kappa_b)
    return [k for k in range(kmin, kmax + 1) if (k + la + lb) % 2 == 0]


def _reduced_C(kappa_a, kappa_b, k):
    la, lb = _l_of(kappa_a), _l_of(kappa_b)
    if (la + k + lb) % 2 != 0:
        return 0.0
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    w = wigner_3j(tja, 2 * k, tjb, -1, 0, 1)
    if abs(w) < 1e-15:
        return 0.0
    phase = (-1) ** ((tja + 1) // 2)
    return phase * math.sqrt((tja + 1) * (tjb + 1)) * w


def dw_excitation_cross_section(atom_initial, atom_final, energies_eV,
                                 L_max=4, kappa_max=10,
                                 transition_energy_eV=None):
    """Compute electron-impact excitation cross-section sigma(E).

    Uses the DW approximation: the incoming/outgoing electron wavefunctions
    are computed in the static potential of the initial/final atomic state.

    The inner partial-wave sum with proper j-j angular coupling is
    JIT-compiled via numba for performance, mirroring the EII module's
    channel-batch + continuum-cache pattern.

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
    _, tj_f = _kappa_to_lj(kappa_f)

    n_energies = len(energies_eV)
    E_inc_hartree = energies_eV / HARTREE_EV

    # Build kappa array (exclude 0, limit l <= kappa_max)
    kappa_list = []
    for k_abs in range(1, kappa_max + 2):
        kappa_list.append(-k_abs)
        kappa_list.append(+k_abs)
    kappa_list = [k for k in kappa_list if k != 0 and _l_of(k) <= kappa_max]

    cache = ConvolCache(bsp_data)

    # Pre-compute the channel structure (energy-independent angular data).
    # Each channel stores: (kappa_in, kappa_out, tj_in, tj_out, tJ_lo, tJ_hi,
    #   direct_k_info, exchange_k_info) where k_info = [(k, C_a_b, C_c_d, w6j_list), ...]
    _channels = []
    for kappa_in in kappa_list:
        tj_in = _tj_of(kappa_in)
        for kappa_out in kappa_list:
            tj_out = _tj_of(kappa_out)

            tJ_lo = max(abs(tj_i - tj_in), abs(tj_f - tj_out))
            tJ_hi = min(tj_i + tj_in, tj_f + tj_out)
            if tJ_lo > tJ_hi:
                continue

            l_i = _l_of(kappa_i); l_in = _l_of(kappa_in)
            l_f = _l_of(kappa_f); l_out = _l_of(kappa_out)
            if (l_i + l_in + l_f + l_out) % 2 != 0:
                continue

            kd = [k for k in _allowed_k(kappa_i, kappa_in)
                  if k in _allowed_k(kappa_out, kappa_f) and k <= L_max]
            ke = [k for k in _allowed_k(kappa_i, kappa_out)
                  if k in _allowed_k(kappa_in, kappa_f) and k <= L_max]
            if not kd and not ke:
                continue

            n_J = (tJ_hi - tJ_lo) // 2 + 1

            d_info = []
            for k in kd:
                c_i_in = _reduced_C(kappa_i, kappa_in, k)
                c_out_f = _reduced_C(kappa_out, kappa_f, k)
                if abs(c_i_in) < 1e-15 or abs(c_out_f) < 1e-15:
                    continue
                w6j_list = []
                for iJ in range(n_J):
                    tJ = tJ_lo + 2 * iJ
                    w6j_list.append(wigner_6j(tj_i, tj_in, tJ, tj_out, tj_f, 2 * k))
                d_info.append((k, c_i_in * c_out_f, w6j_list))

            e_info = []
            for k in ke:
                c_i_out = _reduced_C(kappa_i, kappa_out, k)
                c_in_f = _reduced_C(kappa_in, kappa_f, k)
                if abs(c_i_out) < 1e-15 or abs(c_in_f) < 1e-15:
                    continue
                w6j_list = []
                for iJ in range(n_J):
                    tJ = tJ_lo + 2 * iJ
                    w6j_list.append(wigner_6j(tj_i, tj_in, tJ, tj_f, tj_out, 2 * k))
                e_info.append((k, c_i_out * c_in_f, w6j_list))

            if not d_info and not e_info:
                continue

            _channels.append((kappa_in, kappa_out, tj_in, tj_out,
                              tJ_lo, tJ_hi, d_info, e_info))

    # Continuum-wave cache: (kappa, round(energy, precision)) -> (P, Q)
    continuum_cache = {}

    def _get_continuum(kappa, energy_au):
        key = (int(kappa), round(energy_au, 10))
        if key in continuum_cache:
            return continuum_cache[key]
        try:
            P, Q, _ = solve_continuum(
                int(kappa), energy_au, Z, bsp_data,
                V_J_PP_i, V_J_QQ_i, energy_normalized=True)
            continuum_cache[key] = (P, Q)
            return P, Q
        except Exception:
            continuum_cache[key] = None
            return None, None

    # Pre-solve all incoming continuum waves (one per kappa per energy)
    # and collect all scattered energies needed.
    incoming_waves = {}  # (idx, kappa_in) -> (P, Q)
    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        for ch in _channels:
            kappa_in = ch[0]
            key = (idx, kappa_in)
            if key not in incoming_waves:
                P, Q = _get_continuum(kappa_in, E_inc)
                if P is not None:
                    incoming_waves[key] = (P, Q)

    sigma = np.zeros(n_energies)
    omega = np.zeros(n_energies)

    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        dE = dE_hartree_grid[idx]
        E_scattered = E_inc - dE
        if E_scattered <= 0.0:
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

        for ch in _channels:
            kappa_in, kappa_out, tj_in, tj_out, tJ_lo, tJ_hi, d_info, e_info = ch

            inc = incoming_waves.get((idx, kappa_in))
            if inc is None:
                continue
            P_in, Q_in = inc

            P_out, Q_out = _get_continuum(kappa_out, E_scattered)
            if P_out is None:
                continue

            d_prods = []
            d_w6j = []
            for k, ang_prod, w6j_list in d_info:
                Rk = compute_slater_rk_cell(
                    P_bound_i, Q_bound_i, P_in, Q_in,
                    P_bound_f, Q_bound_f, P_out, Q_out,
                    bsp_data, k, cache=cache,
                )
                d_prods.append(ang_prod * Rk)
                d_w6j.extend(w6j_list)

            e_prods = []
            e_w6j = []
            for k, ang_prod, w6j_list in e_info:
                Rk = compute_slater_rk_cell(
                    P_bound_i, Q_bound_i, P_in, Q_in,
                    P_out, Q_out, P_bound_f, Q_bound_f,
                    bsp_data, k, cache=cache,
                )
                e_prods.append(ang_prod * Rk)
                e_w6j.extend(w6j_list)

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

        n_channels = len(_ch_tj_in)
        if n_channels > 0:
            t_sq = dw_excitation_inner_sum(
                n_channels,
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
                sigma[idx] = np.pi / k_i_sq * t_sq / (tj_i + 1.0)
                omega[idx] = k_i_sq * (tj_i + 1.0) * sigma[idx] / np.pi

    # Convert from atomic units (a_0^2) to cm^2
    sigma *= (BOHR_RADIUS_M * 100) ** 2

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
