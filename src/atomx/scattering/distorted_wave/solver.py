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
from functools import lru_cache

import numba as nb
import numpy as np
from atomx.constants import ALPHA, C_SQ, HARTREE_EV, BOHR_RADIUS_M
from atomx.numerics.bspline.continuum import solve_continuum
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.numerics.bspline.evaluate import eval_P, eval_Q, compute_yk
from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_many,
    solve_dirac_continuum_shooting,
)
from atomx.properties.transitions.angular import _kappa_to_lj, selection_rules, angular_eL
from atomx.properties.transitions.multipole import radial_eL
from atomx.properties.transitions.rates import oscillator_strength
from atomx.angular.csf import csf_parity, csf_twice_j, one_body_csf_coupling
from atomx.angular.coefficients import reduced_C, reduced_cl_j2
from atomx.angular.wigner import wigner_3j, wigner_6j
from atomx.methods.mcdhf.config import CSF
from atomx.scattering._jit_kernels import dw_excitation_angular_sum


def _l_of(kappa):
    """Orbital angular momentum l from kappa."""
    return kappa if kappa > 0 else -kappa - 1


def _tj_of(kappa):
    """Twice-j from kappa."""
    return 2 * abs(kappa) - 1


# Reduced matrix element ⟨κ_a‖C^k‖κ_b⟩, Grant (2007) convention.
@lru_cache(maxsize=None)
def _allowed_k(kappa_a, kappa_b):
    """Allowed multipole k values for reduced C^k between two kappa states.

    Selection rules: |j_a - j_b| ≤ k ≤ j_a + j_b, l_a + l_b + k even.
    (lru_cached; callers must not mutate the returned list.)
    """
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    la, lb = _l_of(kappa_a), _l_of(kappa_b)
    kmin = abs(tja - tjb) // 2
    kmax = (tja + tjb) // 2
    return [k for k in range(kmin, kmax + 1) if (k + la + lb) % 2 == 0]


@lru_cache(maxsize=None)
def _cg2(tj1, tm1, tj2, tm2, tJ, tM):
    """Clebsch-Gordan <j1 m1 j2 m2 | J M> (all arguments doubled)."""
    w = wigner_3j(tj1, tj2, tJ, tm1, tm2, -tM)
    if abs(w) < 1e-15:
        return 0.0
    phase = 1.0 if ((tj1 - tj2 + tM) // 2) % 2 == 0 else -1.0
    return phase * math.sqrt(tJ + 1.0) * w


@lru_cache(maxsize=None)
def _ck_m(kappa_c, tmc, kappa_a, tma, k, tq):
    """<j_c m_c | C^k_q | j_a m_a> via Wigner-Eckart with Grant's <||C^k||>."""
    red = reduced_C(kappa_c, kappa_a, k)
    if red == 0.0:
        return 0.0
    tjc, tja = _tj_of(kappa_c), _tj_of(kappa_a)
    w = wigner_3j(tjc, 2 * k, tja, -tmc, tq, tma)
    if abs(w) < 1e-15:
        return 0.0
    phase = 1.0 if ((tjc - tmc) // 2) % 2 == 0 else -1.0
    return phase * w * red


@lru_cache(maxsize=None)
def _cc_direct(kappa_c, kappa_d, kappa_a, kappa_b, tJ, k):
    """<(c d) J | C^k(1)·C^k(2) | (a b) J> by explicit m-summation.

    Slot pairing (c<-a, d<-b): the direct channel of the scattering
    matrix element (a=target_i, b=projectile_in, c=target_f,
    d=projectile_out).  Radial factor excluded.
    """
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    tjc, tjd = _tj_of(kappa_c), _tj_of(kappa_d)
    tM = min(tJ, tja + tjb, tjc + tjd)
    tot = 0.0
    for tma in range(-tja, tja + 1, 2):
        tmb = tM - tma
        if abs(tmb) > tjb:
            continue
        c1 = _cg2(tja, tma, tjb, tmb, tJ, tM)
        if abs(c1) < 1e-15:
            continue
        for tmc in range(-tjc, tjc + 1, 2):
            tmd = tM - tmc
            if abs(tmd) > tjd:
                continue
            c2 = _cg2(tjc, tmc, tjd, tmd, tJ, tM)
            if abs(c2) < 1e-15:
                continue
            tq = tmc - tma
            if abs(tq) > 2 * k:
                continue
            e1 = _ck_m(kappa_c, tmc, kappa_a, tma, k, tq)
            if e1 == 0.0:
                continue
            e2 = _ck_m(kappa_d, tmd, kappa_b, tmb, k, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            tot += c1 * c2 * phase * e1 * e2
    return tot


@lru_cache(maxsize=None)
def _cc_exchange(kappa_c, kappa_d, kappa_a, kappa_b, tJ, t):
    """Exchange coupled matrix element by explicit m-summation.

    Same bra/ket couplings as :func:`_cc_direct` but with the one-electron
    operators pairing (d<-a, c<-b) — the antisymmetrization term
    <(c d) J | C^t·C^t | (a b) J>_exch.  The antisymmetrized amplitude is
    ``T_J = D_J - E_J`` (radial factors excluded here).
    """
    tja, tjb = _tj_of(kappa_a), _tj_of(kappa_b)
    tjc, tjd = _tj_of(kappa_c), _tj_of(kappa_d)
    tM = min(tJ, tja + tjb, tjc + tjd)
    tot = 0.0
    for tma in range(-tja, tja + 1, 2):
        tmb = tM - tma
        if abs(tmb) > tjb:
            continue
        c1 = _cg2(tja, tma, tjb, tmb, tJ, tM)
        if abs(c1) < 1e-15:
            continue
        for tmc in range(-tjc, tjc + 1, 2):
            tmd = tM - tmc
            if abs(tmd) > tjd:
                continue
            c2 = _cg2(tjc, tmc, tjd, tmd, tJ, tM)
            if abs(c2) < 1e-15:
                continue
            tq = tmd - tma  # slot 1 transfers a -> d
            if abs(tq) > 2 * t:
                continue
            e1 = _ck_m(kappa_d, tmd, kappa_a, tma, t, tq)
            if e1 == 0.0:
                continue
            e2 = _ck_m(kappa_c, tmc, kappa_b, tmb, t, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            tot += c1 * c2 * phase * e1 * e2
    return tot


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


def _distorting_potential_fn(result, bsp_data, Z, screening="fac"):
    """Local distorting potential of the target for the projectile electron.

    ``screening="fac"`` (default) reproduces FAC's convention: the continuum
    electron moves in the same mean central potential as the bound electrons,
    V(r) = -Z/r + ((N-1)/N)·V_Hartree(r), asymptotic charge Z-N+1.  For a
    one-electron target this is the bare -Z/r, exactly as FAC.
    ``screening="full"`` uses the full N-electron Hartree potential
    (asymptotic charge Z-N).
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    nv = bsp_data["nv"]
    ng = gr.shape[1]
    rho = np.zeros_like(gr)
    n_bound = 0.0
    # orbitals_raw: kappa -> [(P, Q, occ), ...]
    for _kappa, orb_list in result.orbitals_raw.items():
        for P, Q, occ in orb_list:
            if occ <= 0.0:
                continue
            Pg = eval_P(P, bsp_data)
            Qg = eval_Q(Q, bsp_data)
            rho += float(occ) * (Pg * Pg + Qg * Qg)
            n_bound += float(occ)
    scale = 1.0
    if screening == "fac" and n_bound > 0.0:
        scale = (n_bound - 1.0) / n_bound
    V_H = compute_yk(rho, gr, grw, nv, ng, 0)  # Hartree Y^0(r) potential
    V_gr = -float(Z) / np.maximum(gr, 1e-30) + scale * V_H
    r_flat = gr.ravel()
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    V_sorted = V_gr.ravel()[order]
    # Coulombic extrapolation beyond the B-spline box: V -> -Z_asym/r.
    r_last = float(r_sorted[-1])
    z_asym = -float(V_sorted[-1]) * r_last

    def V_func(r_query):
        r_query = np.asarray(r_query, dtype=float)
        v = np.interp(r_query, r_sorted, V_sorted)
        beyond = r_query > r_last
        if np.any(beyond):
            v = np.where(beyond, -z_asym / np.maximum(r_query, 1e-30), v)
        return v

    return V_func


from atomx.numerics.quadrature import (   # canonical kernels (E5)
    RadialGridResourceError, grid_slater_rk, make_fine_grid, p_rel)


SHOOTING_POINT_CAP = 300000


def _shooting_points(r_max, p_max):
    """Shooting-solver grid size for <1% amplitude accuracy: h·p <= 0.1."""
    r = float(r_max)
    p = float(p_max)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"_shooting_points: r_max must be finite and > 0, got {r_max!r}")
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"_shooting_points: p_max must be finite and > 0, got {p_max!r}")
    required = int(np.ceil(10.0 * r * max(p, 1.0)))
    if required > SHOOTING_POINT_CAP:
        raise RadialGridResourceError(
            "_shooting_points would exceed shooting-grid point cap: "
            f"required_points={required}, cap={SHOOTING_POINT_CAP}, "
            f"p_max={p:.12g}, r_max={r:.12g}"
        )
    return int(max(required, 3000))


def _geometric_tail(prof_l, l_last):
    """Geometric-ratio estimate of the l > l_last partial-wave tail.

    FAC's TopUpQk scheme: tail = q(l_last) * r / (1 - r) with
    r = q(l_last)/q(l_last - 1).  When the profile has not yet entered its
    geometric decay (r -> 1) the raw formula diverges; the ratio is capped
    at 0.85 (tail <= 5.7 q_last), which keeps the estimate a lower bound
    instead of a blow-up.  Callers needing better accuracy should raise
    kappa_max.
    """
    if l_last < 2:
        return 0.0
    q1 = float(prof_l[l_last])
    q0 = float(prof_l[l_last - 1])
    if q0 <= 0.0 or q1 <= 0.0 or q1 >= q0:
        return 0.0
    ratio = min(q1 / q0, 0.85)
    return q1 * ratio / (1.0 - ratio)


def _continuum_on_grid(kappa, energy, V_func, Z_res, r_grid, r_max, n_grid):
    """δ(E)-normalized shooting continuum (P, Q) sampled on ``r_grid``."""
    r_s, P_s, Q_s, _phase = solve_dirac_continuum_shooting(
        float(energy), int(kappa), V_func,
        r_max=r_max, n_grid=int(n_grid), Z_residual=float(Z_res),
    )
    P_g = np.interp(r_grid, r_s, P_s)
    Q_g = np.interp(r_grid, r_s, Q_s)
    return P_g, Q_g


def dw_excitation_cross_section(atom_initial, atom_final, energies_eV,
                                 L_max=6, kappa_max=12,
                                 transition_energy_eV=None,
                                 continuum_method="shooting",
                                 shooting_n_grid=None,
                                 exchange=True,
                                 topup=True):
    """Electron-impact excitation cross-section sigma(E) and collision strength.

    Relativistic distorted-wave implementation following FAC exactly
    (Gu 2008, Can. J. Phys. 86, 675; faclib/excitation.c + radial.c):

    For a one-electron target transition a -> b the collision strength is

        Omega = 2*pi^2 * Sum_pairs pop_ab * Sum_{k, kap_in, kap_out}
                (sd + se)^2 / (2k+1)                                     (*)

        sd = R^k(a,in; b,out) <j_a||C^k||j_b> <j_in||C^k||j_out>
        se = Sum_t (2k+1) W6j(j_a j_b k; j_in j_out t)
                 R^t(a,in; out,b) <j_a||C^t||j_out> <j_in||C^t||j_b>

    with delta(E)-normalized continuum waves (asymptotic amplitude
    sqrt(2/(pi*k))).  The 2*pi^2 constant is the S-matrix factor
    |S-1|^2 = 4*pi^2 |<f|V|i>|^2 combined with the electron-spin average
    1/2; it is algebraically identical to FAC's ``8.0 * qk / (2k+1)``
    given FAC's 1/sqrt(k)-amplitude waves (faclib/orbital.c:1988).
    Verified channel-by-channel against FAC's CE tables (H-like 1s->2p,
    1s->3d; He-like 1s2->1s2p).

    The target factor treats the transition in the average-configuration
    approximation: ``pop_ab = q_a * (g_b - q_b) / g_b`` summed incoherently
    over all relativistic subshell pairs (a -> b) of the configuration
    change.  Omega is therefore the *multiplet-summed* collision strength
    from the initial level (sum over all fine-structure final levels).

    Parameters
    ----------
    atom_initial, atom_final : Atom
        Solved initial/final-state atoms.
    energies_eV : array_like
        Incident electron kinetic energies in eV.
    L_max : int
        Maximum multipole rank for the Coulomb interaction (direct and
        exchange).
    kappa_max : int
        Maximum |kappa| of the projectile partial waves computed
        explicitly.  For dipole-allowed transitions the l > kappa_max
        tail is estimated with a geometric-ratio top-up (FAC's TopUpQk
        scheme) when ``topup`` is true.
    transition_energy_eV : float or array_like, optional
        Override transition threshold energy.
    continuum_method : {"shooting"}, optional
        Continuum backend.  The delta(E)-normalized adaptive-Numerov
        shooting solver in the target's FAC-convention mean potential
        (-Z/r + (N-1)/N V_Hartree) is the only supported backend; the
        legacy B-spline box continuum mis-normalizes at keV energies and
        was removed from this path.
    shooting_n_grid : int, optional
        Radial points for the shooting solver.  Default (None) scales
        with energy so that h*p <= 0.1 (sub-1% amplitude accuracy).
    exchange : bool
        Include the exchange amplitude (FAC's ``se``).  Default True.
    topup : bool
        Add the geometric high-l tail per multipole rank.  Default True.

    Returns
    -------
    sigma : ndarray
        Cross-section in cm^2 at each energy.
    omega : ndarray
        Multiplet-summed collision strength at each energy.
    """
    if L_max < 0:
        raise ValueError("L_max must be non-negative")
    if kappa_max < 1:
        raise ValueError("kappa_max must be positive")

    energies_eV = np.atleast_1d(np.asarray(energies_eV, dtype=float))

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

    pairs = _active_transition_pairs(result_i, result_f)
    if not pairs:
        return np.zeros_like(energies_eV), np.zeros_like(energies_eV)

    target_tj_i, _p, _c = _target_state_metadata(result_i)
    g_i = float(target_tj_i + 1)

    n_energies = len(energies_eV)
    E_inc_hartree = energies_eV / HARTREE_EV

    # FAC-convention distorting potential and residual charge for the
    # shooting continuum solver.
    N_bound = sum(float(sh.occupation) for sh in result_i.subshells)
    Z_res = max(float(Z) - N_bound + 1.0, 1.0)
    V_func = _distorting_potential_fn(result_i, bsp_data, Z, screening="fac")
    r_max_box = float(bsp_data["t"][-1])

    # Fine integration grid sized to the fastest wave on the panel.
    E_max = float(np.max(E_inc_hartree))
    p_max = p_rel(max(E_max, 1.0))
    r_fine = make_fine_grid(r_max_box, p_max, z_inner=float(Z))
    dr_fine = np.gradient(r_fine)
    n_shoot = (int(shooting_n_grid) if shooting_n_grid
               else _shooting_points(r_max_box, p_max))

    # Bound orbitals interpolated onto the fine grid (smooth: safe).
    gr_flat = bsp_data["gr"].ravel()
    _o = np.argsort(gr_flat)
    gr_sorted = gr_flat[_o]

    def _bound_to_fine(coef_P, coef_Q, src_bsp):
        Pg = eval_P(coef_P, src_bsp).ravel()[_o]
        Qg = eval_Q(coef_Q, src_bsp).ravel()[_o]
        return (np.interp(r_fine, gr_sorted, Pg),
                np.interp(r_fine, gr_sorted, Qg))

    bsp_f = atom_final._bsp_data
    pair_data = []
    for (ka, kb, pop, PQa, PQb) in pairs:
        Pa, Qa = _bound_to_fine(PQa[0], PQa[1], bsp_data)
        Pb, Qb = _bound_to_fine(PQb[0], PQb[1], bsp_f)
        pair_data.append((ka, kb, float(pop), Pa, Qa, Pb, Qb))
    if not pair_data:
        return np.zeros_like(energies_eV), np.zeros_like(energies_eV)

    kappa_list = [k for k in range(-kappa_max - 1, kappa_max + 1)
                  if k != 0 and _l_of(k) <= kappa_max]

    E_OUT_MIN = 1.0 / HARTREE_EV  # evaluate the threshold point at E_out = 1 eV

    # Exact plane-wave Born (GOS) for the high-energy completion: FAC's
    # _topup=2 blends the DW value toward Born as the truncated-tail
    # fraction of the partial-wave sum grows (excitation.c:1330-1337).
    born_tr = None
    if topup:
        from atomx.scattering.distorted_wave.born import (
            BornTransition, born_blend_weights)
        pos = dE_hartree_grid[dE_hartree_grid > 0]
        try:
            born_tr = BornTransition(pair_data, r_fine, dr_fine,
                                     float(np.median(pos)), E_max)
        except Exception:
            born_tr = None

    sigma_au = np.zeros(n_energies)
    omega = np.zeros(n_energies)
    tail_omega = np.zeros(n_energies)

    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        E_out = E_inc - dE_hartree_grid[idx]
        if E_out <= 0.0:
            if E_inc >= dE_hartree_grid[idx] * (1.0 - 1e-9):
                E_out = E_OUT_MIN
            else:
                continue
        E_out = max(E_out, E_OUT_MIN)

        cont = {}
        # Thread-parallel prefill of all partial waves at both energies
        # (doutw is nogil; each solve is ms-scale — the right thread grain).
        try:
            reqs = ([(k, E_inc) for k in kappa_list]
                    + [(k, E_out) for k in kappa_list])
            _r_s, raw = solve_dirac_continuum_many(
                reqs, V_func, r_max=r_max_box, n_grid=n_shoot,
                Z_residual=Z_res, interp_grid=r_fine)
            for (kap, E), pq in raw.items():
                cont[(int(kap), round(float(E), 12))] = pq
        except Exception:
            cont = {}

        def _wave(kappa, E):
            key = (int(kappa), round(float(E), 12))
            if key not in cont:
                try:
                    cont[key] = _continuum_on_grid(
                        int(kappa), float(E), V_func, Z_res,
                        r_fine, r_max_box, n_shoot)
                except Exception:
                    cont[key] = None
            return cont[key]

        # Per-partial-wave collision-strength profile, indexed by l of the
        # incident electron (FAC's qk(l) profile, used for the high-l tail).
        prof_l = np.zeros(kappa_max + 1)

        for kappa_in in kappa_list:
            w_in = _wave(kappa_in, E_inc)
            if w_in is None:
                continue
            tj_in = _tj_of(kappa_in)
            l_in = _l_of(kappa_in)
            for kappa_out in kappa_list:
                tj_out = _tj_of(kappa_out)
                l_out = _l_of(kappa_out)
                w_out = None
                for (ka, kb, pop, Pa, Qa, Pb, Qb) in pair_data:
                    la, lb = _l_of(ka), _l_of(kb)
                    if (la + l_in + lb + l_out) % 2 != 0:
                        continue
                    tja, tjb = _tj_of(ka), _tj_of(kb)
                    ks = [k for k in _allowed_k(ka, kb)
                          if k in _allowed_k(kappa_in, kappa_out)
                          and k <= L_max]
                    ts = []
                    if exchange:
                        ts = [t for t in _allowed_k(ka, kappa_out)
                              if t in _allowed_k(kappa_in, kb) and t <= L_max]
                    if not ks and not ts:
                        continue
                    if w_out is None:
                        w_out = _wave(kappa_out, E_out)
                        if w_out is None:
                            break
                    Rd = {k: grid_slater_rk(
                        Pa, Qa, w_in[0], w_in[1],
                        Pb, Qb, w_out[0], w_out[1],
                        r_fine, k) for k in ks}
                    Re = {t: grid_slater_rk(
                        Pa, Qa, w_in[0], w_in[1],
                        w_out[0], w_out[1], Pb, Qb,
                        r_fine, t) for t in ts}
                    # Explicit total-J sum: T_J = D_J - E_J with the coupled
                    # scalar-product matrix elements evaluated by exact
                    # m-summation (no 6j phase-convention risk).
                    tJ_lo = max(abs(tja - tj_in), abs(tjb - tj_out))
                    tJ_hi = min(tja + tj_in, tjb + tj_out)
                    acc = 0.0
                    for tJ in range(tJ_lo, tJ_hi + 1, 2):
                        D = 0.0
                        for k in ks:
                            ang = _cc_direct(kb, kappa_out, ka, kappa_in,
                                             tJ, k)
                            if ang != 0.0:
                                D += Rd[k] * ang
                        E_ = 0.0
                        for t in ts:
                            ang = _cc_exchange(kb, kappa_out, ka, kappa_in,
                                               tJ, t)
                            if ang != 0.0:
                                E_ += Re[t] * ang
                        T = D - E_
                        if T != 0.0:
                            acc += (tJ + 1) * T * T
                    prof_l[l_in] += pop * acc

        om = float(np.sum(prof_l))
        tail = _geometric_tail(prof_l, kappa_max) if topup else 0.0
        om = (om + tail) * 2.0 * math.pi ** 2

        p_i_sq = 2.0 * E_inc + (E_inc * E_inc) / C_SQ
        omega[idx] = om
        tail_omega[idx] = tail * 2.0 * math.pi ** 2
        sigma_au[idx] = math.pi / p_i_sq * om / g_i

    # Born completion: blend each energy toward the exact plane-wave Born
    # collision strength with weight w = (tail/Omega)^{1/2}, clamped
    # monotone non-decreasing in energy (FAC _topup=2 default).
    if born_tr is not None:
        order = np.argsort(E_inc_hartree, kind="stable")
        w_sorted = born_blend_weights(omega[order], tail_omega[order])
        w = np.empty(n_energies)
        w[order] = w_sorted
        for idx in range(n_energies):
            if w[idx] <= 0.0 or omega[idx] <= 0.0:
                continue
            om_b = born_tr.omega(E_inc_hartree[idx],
                                 dE_au=float(dE_hartree_grid[idx]))
            if om_b <= 0.0:
                continue
            om = omega[idx] * (1.0 - w[idx]) + om_b * w[idx]
            E_inc = E_inc_hartree[idx]
            p_i_sq = 2.0 * E_inc + (E_inc * E_inc) / C_SQ
            omega[idx] = om
            sigma_au[idx] = math.pi / p_i_sq * om / g_i

    sigma = sigma_au * (BOHR_RADIUS_M * 100) ** 2
    return sigma, omega


def _active_transition_pairs(result_i, result_f):
    """All relativistic subshell pairs (a -> b) of the config change.

    The non-relativistic source and destination shells (n, l) are identified
    from the occupation change between the two solves; the pair list then
    expands over ALL jj components of both shells (a jj-averaged or
    lowest-level solve may put the excited electron into only one j
    component, but the multiplet-summed collision strength needs both).

    Returns ``(kappa_a, kappa_b, pop, (P_a, Q_a), (P_b, Q_b))`` with the
    average-configuration population factor
    ``pop = q_a * (g_b - q_b) / g_b`` built from initial-state occupations.
    Source orbitals come from the initial solve, destination orbitals from
    the final (relaxed) solve.
    """
    occ_i = {(sh.n, sh.kappa): float(sh.occupation) for sh in result_i.subshells}
    occ_f = {(sh.n, sh.kappa): float(sh.occupation) for sh in result_f.subshells}
    src_nl = set()
    dst_nl = set()
    for key in sorted(set(occ_i) | set(occ_f)):
        delta = occ_f.get(key, 0.0) - occ_i.get(key, 0.0)
        n, kappa = key
        if delta < -0.01:
            src_nl.add((n, _l_of(kappa)))
        elif delta > 0.01:
            dst_nl.add((n, _l_of(kappa)))
    dst_nl -= src_nl
    if not src_nl or not dst_nl:
        return []

    def _kappas_of_l(l):
        return [-1] if l == 0 else [l, -(l + 1)]

    def _orb(result, key):
        n, kappa = key
        same = [sh for sh in result.subshells if sh.kappa == kappa]
        for i, sh in enumerate(same):
            if sh.n == n:
                orb_list = result.orbitals_raw.get(kappa, [])
                if i < len(orb_list):
                    P, Q, _q = orb_list[i]
                    return (P, Q)
        return None

    out = []
    for (n_a, l_a) in sorted(src_nl):
        for kappa_a in _kappas_of_l(l_a):
            skey = (n_a, kappa_a)
            q_a = occ_i.get(skey, 0.0)
            if q_a <= 1e-8:
                continue
            PQa = _orb(result_i, skey)
            if PQa is None:
                continue
            for (n_b, l_b) in sorted(dst_nl):
                for kappa_b in _kappas_of_l(l_b):
                    dkey = (n_b, kappa_b)
                    g_b = 2.0 * abs(kappa_b)
                    q_b = occ_i.get(dkey, 0.0)
                    # The coupled-channel J-sum already counts the full
                    # (2j_a+1) source line (all M_a); the population factor
                    # is therefore just the destination vacancy fraction.
                    # Exact for a closed source subshell and for a source
                    # subshell that defines the initial level (H-like);
                    # multi-open-shell config averages need level-resolved
                    # CI amplitudes (future work).
                    pop = max(g_b - q_b, 0.0) / g_b
                    if pop <= 1e-10:
                        continue
                    PQb = _orb(result_f, dkey)
                    if PQb is None:
                        continue
                    out.append((kappa_a, kappa_b, float(pop), PQa, PQb))
    return out


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
