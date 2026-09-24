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
import warnings
from functools import lru_cache

import numpy as np

from atomx.angular.coefficients import cg, ckm
from atomx.angular.csf import csf_parity, csf_twice_j, one_body_csf_coupling
from atomx.constants import BOHR_RADIUS_M, C_SQ, HARTREE_EV
from atomx.methods.mcdhf.config import CSF
from atomx.numerics.bspline.evaluate import compute_yk, eval_P, eval_Q
from atomx.numerics.quadrature import (  # canonical kernels (E5)
    grid_slater_rk,
    make_fine_grid,
    p_rel,
)
from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_many,
    solve_dirac_continuum_shooting,
)


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


# The m-resolved angular coefficient and Clebsch-Gordan helper live in the
# angular layer (single implementation, shared with the CI matrix elements in
# atomx.methods.mcdhf.matrix); these aliases keep the local call sites and the
# ``_ck_m`` / ``_cg2`` names used by atomx.scattering.autoionization.mscheme_ci.
_cg2 = cg
_ck_m = ckm


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


def _shooting_points(r_max, p_max, *, max_points=300000):
    """Shooting-solver grid size for <1% amplitude accuracy: h·p <= 0.1."""
    if not (math.isfinite(float(r_max)) and float(r_max) > 0.0):
        raise ValueError(f"_shooting_points: r_max must be finite and > 0, got {r_max!r}")
    if not (math.isfinite(float(p_max)) and float(p_max) > 0.0):
        raise ValueError(f"_shooting_points: p_max must be finite and > 0, got {p_max!r}")
    if int(max_points) < 2:
        raise ValueError("_shooting_points: max_points must be >= 2")
    required = int(max(10.0 * float(r_max) * max(float(p_max), 1.0), 3000))
    if required > int(max_points):
        from atomx.numerics.quadrature import GridResourceError
        raise GridResourceError(
            "_shooting_points exceeded radial grid cap: "
            f"required_points={required}, cap={int(max_points)}, "
            f"p={float(p_max):.12g} a.u., r_max={float(r_max):.12g} a0"
        )
    return required


_TAIL_RATIO_CAP = 0.85


def _geometric_tail_estimate(prof_l, l_last):
    """Geometric-ratio estimate of the l > l_last partial-wave tail.

    FAC's TopUpQk scheme: tail = q(l_last) * r / (1 - r) with
    r = q(l_last)/q(l_last - 1).  Returns ``(tail, converged)``.

    When the profile has not yet entered its geometric decay (r >= 0.85,
    including a still-rising profile at ``l_last``, where the raw formula
    diverges) the ratio is capped at 0.85, which keeps ``tail`` a finite
    floor (<= 5.7 q_last) instead of a blow-up, and ``converged`` is False:
    the value is a *floor* on the missing tail, not a measurement.  Callers
    that blend toward a high-energy limit must read the flag — returning
    exactly 0.0 there makes the Born completion read "no tail" and never
    engage (review C6, 2026-09-19).  Callers needing better accuracy should
    raise kappa_max.
    """
    if l_last < 2:
        return 0.0, True
    q1 = float(prof_l[l_last])
    q0 = float(prof_l[l_last - 1])
    if not (math.isfinite(q1) and math.isfinite(q0)) or q1 <= 0.0 or q0 <= 0.0:
        # No decay signal to extrapolate from (zero or non-finite top of the
        # profile): there is nothing to add, and nothing to flag.
        return 0.0, True
    ratio = q1 / q0
    if ratio >= _TAIL_RATIO_CAP:
        return q1 * _TAIL_RATIO_CAP / (1.0 - _TAIL_RATIO_CAP), False
    return q1 * ratio / (1.0 - ratio), True


def _geometric_tail(prof_l, l_last):
    """Tail value only, historical contract: 0.0 while the profile is rising.

    The EII integrator *adds* this value to its partial-wave sum, so it keeps
    the conservative pre-C6 behaviour exactly (0.0 when no decay ratio is
    available at ``l_last``); both functions return the same number whenever
    the profile has begun to decay.  Callers that blend toward a high-energy
    limit — where a still-rising profile means an unresolved truncation, not
    "no tail" — must use :func:`_geometric_tail_estimate` and read its flag.
    """
    if l_last < 2:
        return 0.0
    q1 = float(prof_l[l_last])
    q0 = float(prof_l[l_last - 1])
    if q0 <= 0.0 or q1 <= 0.0 or q1 >= q0:
        return 0.0
    return _geometric_tail_estimate(prof_l, l_last)[0]


def _continuum_on_grid(kappa, energy, V_func, Z_res, r_grid, r_max, n_grid):
    """δ(E)-normalized shooting continuum (P, Q) sampled on ``r_grid``."""
    r_s, P_s, Q_s, _phase = solve_dirac_continuum_shooting(
        float(energy), int(kappa), V_func,
        r_max=r_max, n_grid=int(n_grid), Z_residual=float(Z_res),
        match_phase=False,
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
        Add the geometric high-l tail per multipole rank and blend each
        energy toward the exact plane-wave Born value with weight
        w = (tail/Omega)^0.5 (FAC ``_topup=2``).  An energy whose profile
        has not decayed at ``kappa_max`` has an unresolved truncation: its
        geometric tail is only a floor, and the floor is raised to the
        computed partial-wave sum so that w >= 1/sqrt(2) there instead of
        falling to 0.  Default True.

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
    def _bound_to_fine(coef_P, coef_Q, src_bsp):
        source_r = src_bsp["gr"].ravel()
        order = np.argsort(source_r)
        Pg = eval_P(coef_P, src_bsp).ravel()[order]
        Qg = eval_Q(coef_Q, src_bsp).ravel()[order]
        return (np.interp(r_fine, source_r[order], Pg, left=0.0, right=0.0),
                np.interp(r_fine, source_r[order], Qg, left=0.0, right=0.0))

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

    # At E_inc == threshold the outgoing electron has zero kinetic energy and
    # the DW integral is singular; FAC evaluates the threshold point at a
    # small finite E_out.  This is a deliberate, documented regularisation —
    # it was previously an unexplained literal, so the resulting
    # threshold value cannot be compared with a code that evaluates exactly
    # at threshold.  See docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md M17.
    E_OUT_MIN = 1.0 / HARTREE_EV  # evaluate the threshold point at E_out = 1 eV

    # Exact plane-wave Born (GOS) for the high-energy completion: FAC's
    # _topup=2 blends the DW value toward Born as the truncated-tail
    # fraction of the partial-wave sum grows (excitation.c:1330-1337).
    born_tr = None
    if topup:
        from atomx.scattering.distorted_wave.born import (
            BornTransition,
            born_blend_weights,
        )
        pos = dE_hartree_grid[dE_hartree_grid > 0]
        try:
            born_tr = BornTransition(pair_data, r_fine, dr_fine,
                                     float(np.median(pos)), E_max)
        except Exception as exc:  # noqa: BLE001 - degraded, but never silent
            # The completion is an accuracy improvement, not a correctness
            # requirement, so degrade instead of raising — but say so, and
            # say why.  Previously this was a bare `except Exception` that
            # silently disabled the high-energy completion, which is exactly
            # how the "w = 0 at every energy" defect (finding C6) stayed
            # invisible.
            warnings.warn(
                f"DW excitation: plane-wave Born completion unavailable "
                f"({type(exc).__name__}: {exc}); collision strengths at high "
                f"incident energy will be underestimated.",
                RuntimeWarning, stacklevel=2,
            )
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
        except Exception as exc:  # noqa: BLE001 - degraded, but never silent
            warnings.warn(
                f"DW excitation: batched continuum solve failed "
                f"({type(exc).__name__}: {exc}); falling back to per-wave "
                f"solves, which is much slower.",
                RuntimeWarning, stacklevel=2,
            )
            cont = {}

        def _wave(kappa, E, cont=cont):
            key = (int(kappa), round(float(E), 12))
            if key not in cont:
                try:
                    cont[key] = _continuum_on_grid(
                        int(kappa), float(E), V_func, Z_res,
                        r_fine, r_max_box, n_shoot)
                except Exception as exc:  # noqa: BLE001 - degraded, not silent
                    # A None wave contributes zero to the partial-wave sum,
                    # i.e. this (kappa, E) channel is *dropped* from the
                    # collision strength.  Record it once per channel.
                    warnings.warn(
                        f"DW excitation: continuum wave (kappa={int(kappa)}, "
                        f"E={float(E):.6f} Ha) failed "
                        f"({type(exc).__name__}: {exc}); that partial wave "
                        f"is omitted from the sum.",
                        RuntimeWarning, stacklevel=2,
                    )
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

        om_partial = float(np.sum(prof_l))
        tail, tail_converged = (_geometric_tail_estimate(prof_l, kappa_max)
                                if topup else (0.0, True))
        if not tail_converged:
            # The profile had not decayed at kappa_max, so the geometric
            # ratio was capped: `tail` is only a floor, and if the truncation
            # is unresolved the l > kappa_max sum is at least as large as
            # everything computed.  Saturating the floor at the computed sum
            # keeps the Born blend weight at or above 1/sqrt(2) even when the
            # floor itself is small — an unconverged partial-wave sum must not
            # be reported as the DW answer (review C6).
            tail = max(tail, om_partial)
        om = (om_partial + tail) * 2.0 * math.pi ** 2

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


def collision_strength_to_rate(omega_array, energy_eV_array, temperature_eV,
                               *, g_i=1):
    """Convert collision strengths to Maxwellian rate coefficients.

    In atomic units (a_0 = m_e = hbar = 1, energies in Hartree)

        <sigma*v>(T) = a_0^2 * sqrt(2*pi/m_e) / g_i * (kT)^{-3/2}
                       * Integral Omega(E) exp(-E/kT) dE

    which, in the same units, is just

        <sigma*v>(T) = sqrt(2*pi)/g_i * (kT)^{-3/2}
                       * Integral Omega(E) exp(-E/kT) dE      [a_0^3/tau]

    and is converted to cm^3/s with a_0^3/tau = (BOHR_RADIUS_M*100)^3 /
    AU_TIME_S.  In cgs-consistent form (sigma and m_e in cgs, kT and E in
    erg) the identical expression is

        <sigma*v> = sqrt(2*pi) * hbar^2 / (m_e^{3/2} * g_i) * (kT)^{-3/2}
                    * Integral Omega(E) exp(-E/kT) dE

    i.e. the cross-section relation sigma(E) = pi*hbar^2*Omega(E)/
    (2*m_e*E*g_i) used by :func:`dw_excitation_cross_section` (there
    ``sigma_au = pi/p_i^2 * Omega / g_i``).  For Omega = 1, g_i = 1,
    kT = 1 eV both forms give 8.011e-8 cm^3/s, in agreement with the
    8.629e-6*Upsilon/(g_l*sqrt(T_K)) convention of
    :mod:`atomx.plasma.crm` and with
    :func:`atomx.plasma.rates.maxwellian_rate`.  The historical prefactor
    ``2*pi*a_0^2/sqrt(m_e*kT)`` carried cm^2/eV^{1/2} and was ~2e10 too
    small here (review H0, 2026-09-19).

    Parameters
    ----------
    omega_array : ndarray
        Collision strengths at each energy.
    energy_eV_array : ndarray
        Energies in eV; the Maxwellian average is taken over this grid.
    temperature_eV : float
        Electron temperature in eV.
    g_i : int
        Statistical weight of the initial state.  Default 1.

    Returns
    -------
    float
        Rate coefficient in cm^3/s.
    """
    if g_i <= 0:
        raise ValueError("collision_strength_to_rate: g_i must be > 0")
    kT_eV = float(temperature_eV)
    if kT_eV <= 0.0:
        return 0.0

    from atomx.constants import AU_TIME_S, BOHR_RADIUS_M, HARTREE_EV
    kT_au = kT_eV / HARTREE_EV
    E_au = np.asarray(energy_eV_array, dtype=float) / HARTREE_EV
    integrand = np.asarray(omega_array, dtype=float) * np.exp(-E_au / kT_au)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    integral = _trap(integrand, E_au)

    a0_cm = BOHR_RADIUS_M * 100.0
    prefactor = (a0_cm ** 3 / AU_TIME_S) * math.sqrt(2.0 * math.pi) / float(g_i)
    return prefactor * kT_au ** -1.5 * integral


# ---------------------------------------------------------------------------
# Coulomb-Bethe high-l top-up for dipole-allowed transitions
# ---------------------------------------------------------------------------


def coulomb_bethe_topup(
    f_ab: float,
    energies_eV: np.ndarray,
    dE_eV: float,
    L_truncation: int,
    *,
    g_i: float = 1.0,
) -> np.ndarray:
    """High-l Coulomb-Bethe (Burgess-Sheorey) top-up for the dipole-
    allowed collision strength.

    ``Omega_topup = (4 g_i f / dE) ln[(k_i + k_f)/(k_i - k_f)]`` with
    ``dE`` in the same units as the incident energy.  This is the exact
    Bethe limit of the dipole contribution: for a transition with
    oscillator strength ``f`` and initial statistical weight
    ``g_i = 2 J_i + 1`` it is the high-energy behaviour the DW partial-wave
    sum must approach, so it is a closed-form-checkable quantity (see
    ``tests/test_dw_born_completion.py``).

    Parameters
    ----------
    f_ab
        Optical (dipole) oscillator strength of the i -> f transition.
    energies_eV
        Incident-electron energies, eV.
    dE_eV
        Excitation energy E_f - E_i, eV.
    L_truncation
        ``l`` value at which the DW sum was truncated.  Unused by the
        closed form; accepted so callers can record it with the result.
    g_i
        Statistical weight ``2 J_i + 1`` of the *initial* level.  This was
        previously omitted, making the returned top-up low by exactly
        ``g_i`` (a factor 2 for the He-like 1s2 1S0 -> 1s2p 1P1 anchor).
        See ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding M16.

    Returns
    -------
    np.ndarray
        Top-up Omega(E) per energy on the same grid as ``energies_eV``.

    Notes
    -----
    The previous implementation also multiplied by an undocumented
    ``1 - 1/(L_truncation + 1)`` "suppression" factor (0.857 at L = 6),
    which silently broke the closed-form limit.  A caller that wants to
    model the residual tail beyond ``L_truncation`` should do so
    explicitly rather than have it folded in here.
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
    out[above] = (4.0 * float(g_i) * f_ab / omega_au) * np.log(log_arg)
    _ = L_truncation  # accepted for provenance only; the closed form needs it not
    return out


__all__ = [
    "dw_excitation_cross_section",
    "collision_strength_to_rate",
    "coulomb_bethe_topup",
]
