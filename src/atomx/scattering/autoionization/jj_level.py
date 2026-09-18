"""Level-resolved (jj-coupled) Auger rates with exact spectator recoupling.

The configuration-average entry point ``auger_rate_csf_average`` uses the
average-of-configuration angular factor ``exchange_lambda``, which
over-counts magnetic substates for a *specific* fine-structure resonance
(by 1.2-15x on the Fe XVII KLL panel).  This module computes the exact
per-level rate for a doubly-excited resonance with one core hole:

    |Res> = |( core-hole j_h  [x]  excited pair ) J_res M>

decaying to a closed-shell ion + free electron.  The amplitude is built by
explicit m-summation (no 6j/9j phase-convention risk — the same bulletproof
strategy validated against FAC for DW excitation):

    A(M) = Sum_{M_h m1 m2} C_scheme * (-1)^{j_h - M_h}
           * [ D(nu=-M_h, m1, m2) - X(nu=-M_h, m1, m2) ]

where C_scheme are the Clebsch-Gordan factors of the level's jj coupling
scheme, the hole operator is h+_{J M} = (-1)^{j-M} c_{j,-M} (standard
particle-hole conjugation), and D/X are the direct/exchange two-electron
Coulomb matrix elements filling the hole and emitting the continuum
electron.  Fermi's golden rule with delta(E)-normalized continuum waves
gives Gamma = 2*pi*|A|^2 (Hartree/hbar units) — identical content to FAC's
``AutoionizeRate`` prefactor ``4 * sum p^2 / (2J+1)`` given FAC's
1/sqrt(k)-amplitude waves (faclib/recombination.c:2168,
faclib/orbital.c:1988).

Supported coupling schemes (matching FAC level-name conventions):
- ``"sequential"``: ((j_h [x] j_1) J_a [x] j_2) J_res, for two excited
  electrons in different subshells, with ``two_J_inter`` = 2*J_a.
- ``"pair"``: (j_h [x] (j^2) J_p) J_res, for two equivalent excited
  electrons, with ``two_J_inter`` = 2*J_p (even).

Currently the final ion must be a closed shell (J_ion = 0), which covers
KLL-type resonances of closed-shell ions (He-like, Ne-like, ...); the
continuum electron then carries j_f = J_res exactly.
"""

from __future__ import annotations

import math

import numpy as np

from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.numerics.quadrature import grid_slater_rk, make_fine_grid, p_rel
from atomx.scattering.distorted_wave.solver import (
    _allowed_k,
    _cg2,
    _ck_m,
    _continuum_on_grid,
    _distorting_potential_fn,
    _l_of,
    _shooting_points,
    _tj_of,
)

_L_SYMBOLS = "spdfghiklmno"


def _kappa_from_label(label: str) -> tuple[int, int]:
    """Parse '3p1/2' -> (n, kappa)."""
    import re

    m = re.fullmatch(r"(\d+)([spdfghiklmno])(\d+)/2", label)
    if m is None:
        raise ValueError(f"cannot parse orbital label {label!r}")
    n = int(m.group(1))
    l = _L_SYMBOLS.index(m.group(2))
    tj = int(m.group(3))
    kappa = l if tj == 2 * l - 1 else -(l + 1)
    return n, kappa


def _orbital_PQ(result, label: str):
    n, kappa = _kappa_from_label(label)
    same = [sh for sh in result.subshells if sh.kappa == kappa]
    for i, sh in enumerate(same):
        if sh.n == n:
            orb_list = result.orbitals_raw.get(kappa, [])
            if i < len(orb_list):
                P, Q, _q = orb_list[i]
                return kappa, (P, Q)
    raise ValueError(f"orbital {label!r} not found in result")


def auger_rate_jj_level(
    result,
    *,
    hole_label: str,
    e1_label: str,
    e2_label: str,
    two_J_inter: int,
    two_J_res: int,
    resonance_energy_eV: float,
    scheme: str = "sequential",
    L_max: int = 6,
    shooting_n_grid: int | None = None,
) -> float:
    """Exact jj-level Auger rate (s^-1) for a one-hole/two-electron resonance.

    Parameters
    ----------
    result
        Solved structure result carrying the bound orbitals of the
        resonance configuration (hole subshell + both excited orbitals).
    hole_label, e1_label, e2_label
        Relativistic subshell labels, e.g. ``"2p1/2"``, ``"3s1/2"``.
        For ``scheme="sequential"`` the intermediate coupling is
        (hole [x] e1) J_a; for ``scheme="pair"`` e1 and e2 must be the
        same subshell coupled to J_p.
    two_J_inter
        2*J_a (sequential) or 2*J_p (pair).
    two_J_res
        2*J of the resonance level.
    resonance_energy_eV
        Auger electron energy (resonance above the ion ground state).
    """
    kappa_h, PQh = _orbital_PQ(result, hole_label)
    kappa_1, PQ1 = _orbital_PQ(result, e1_label)
    kappa_2, PQ2 = _orbital_PQ(result, e2_label)
    tj_h, tj_1, tj_2 = _tj_of(kappa_h), _tj_of(kappa_1), _tj_of(kappa_2)
    tJa = int(two_J_inter)
    tJ = int(two_J_res)

    E_res_au = float(resonance_energy_eV) / HARTREE_EV
    if E_res_au <= 0.0:
        return 0.0

    bsp_data = getattr(result, "bsp_data", None) or getattr(result, "_bsp_data", None)
    if bsp_data is None:
        raise ValueError("result carries no B-spline workspace")
    Z = int(getattr(result, "Z", 0))

    # Continuum: closed-shell final ion => j_f = J_res; parity fixes l_f.
    par_res = 1
    for kap in (kappa_h, kappa_1, kappa_2):
        # hole parity contributes like the orbital parity
        par_res *= -1 if _l_of(kap) % 2 else 1
    tj_f = tJ
    l_opts = [(tj_f + 1) // 2, (tj_f - 1) // 2]
    l_f = next((l for l in l_opts if l >= 0 and ((-1 if l % 2 else 1) == par_res)), None)
    if l_f is None:
        return 0.0
    kappa_f = l_f if tj_f == 2 * l_f - 1 else -(l_f + 1)

    # delta(E)-normalized shooting continuum in the target mean potential.
    V_func = _distorting_potential_fn(result, bsp_data, Z, screening="fac")
    N_bound = sum(float(sh.occupation) for sh in result.subshells)
    Z_res = max(float(Z) - N_bound + 1.0, 1.0)
    r_max_box = float(bsp_data["t"][-1])
    p_max = p_rel(max(E_res_au, 1.0))
    r_fine = make_fine_grid(r_max_box, p_max)
    n_shoot = (int(shooting_n_grid) if shooting_n_grid
               else _shooting_points(r_max_box, p_max))

    gr_flat = bsp_data["gr"].ravel()
    order = np.argsort(gr_flat)
    gr_sorted = gr_flat[order]

    def _to_fine(PQ):
        Pg = eval_P(PQ[0], bsp_data).ravel()[order]
        Qg = eval_Q(PQ[1], bsp_data).ravel()[order]
        return (np.interp(r_fine, gr_sorted, Pg),
                np.interp(r_fine, gr_sorted, Qg))

    Ph, Qh = _to_fine(PQh)
    P1, Q1 = _to_fine(PQ1)
    P2, Q2 = _to_fine(PQ2)
    try:
        Pf, Qf = _continuum_on_grid(kappa_f, E_res_au, V_func, Z_res,
                                    r_fine, r_max_box, n_shoot)
    except Exception:
        return 0.0

    # Radial integrals per rank: direct (h<-e1, f<-e2), exchange (h<-e2, f<-e1).
    ks_D = [k for k in _allowed_k(kappa_h, kappa_1)
            if k in _allowed_k(kappa_f, kappa_2) and k <= L_max]
    ks_X = [k for k in _allowed_k(kappa_h, kappa_2)
            if k in _allowed_k(kappa_f, kappa_1) and k <= L_max]
    RD = {k: grid_slater_rk(Ph, Qh, Pf, Qf, P1, Q1, P2, Q2,
                            r_fine, k) for k in ks_D}
    RX = {k: grid_slater_rk(Ph, Qh, Pf, Qf, P2, Q2, P1, Q1,
                            r_fine, k) for k in ks_X}
    if not RD and not RX:
        return 0.0

    # Explicit m-summation of the amplitude at M = J_res (M-independent).
    tM = tJ
    tm_f = tM  # closed-shell ion: m_f = M
    amp = 0.0
    for tM_h in range(-tj_h, tj_h + 1, 2):
        # hole conjugation: h+_{J_h M_h} = (-1)^{j_h - M_h} c_{j_h, -M_h}
        tnu = -tM_h
        hole_phase = 1.0 if ((tj_h - tM_h) // 2) % 2 == 0 else -1.0
        for tm_1 in range(-tj_1, tj_1 + 1, 2):
            tm_2 = None
            if scheme == "sequential":
                tMa = tM_h + tm_1
                if abs(tMa) > tJa:
                    continue
                c_a = _cg2(tj_h, tM_h, tj_1, tm_1, tJa, tMa)
                if abs(c_a) < 1e-15:
                    continue
                tm_2 = tM - tMa
                if abs(tm_2) > tj_2:
                    continue
                c_b = _cg2(tJa, tMa, tj_2, tm_2, tJ, tM)
                if abs(c_b) < 1e-15:
                    continue
                coeffs = [(tm_1, tm_2, c_a * c_b)]
            elif scheme == "pair":
                # (j^2) J_p pair, then (h [x] pair) J.  Normalized
                # antisymmetric pair state: (1/sqrt(2)) sum CG c+ c+ |0>.
                coeffs = []
                tMp = tM - tM_h
                if abs(tMp) > tJa:
                    continue
                c_hp = _cg2(tj_h, tM_h, tJa, tMp, tJ, tM)
                if abs(c_hp) < 1e-15:
                    continue
                tm_2 = tMp - tm_1
                if abs(tm_2) > tj_2:
                    continue
                c_p = _cg2(tj_1, tm_1, tj_2, tm_2, tJa, tMp)
                if abs(c_p) < 1e-15:
                    continue
                # 1/sqrt(2) pair normalization; the two c+ orderings of the
                # determinant give a factor 2 against the two-body operator,
                # net sqrt(2) relative to the distinct-orbital case, which
                # is exactly reproduced by keeping the full CG sum here and
                # dividing by sqrt(2).
                coeffs = [(tm_1, tm_2, c_hp * c_p / math.sqrt(2.0))]
            else:
                raise ValueError(f"unknown scheme {scheme!r}")

            for (tm1, tm2, cw) in coeffs:
                # D: (h nu <- e1 m1) x (f m_f <- e2 m2)
                term = 0.0
                tq = tnu - tm1
                for k in ks_D:
                    if abs(tq) > 2 * k:
                        continue
                    e1 = _ck_m(kappa_h, tnu, kappa_1, tm1, k, tq)
                    if e1 == 0.0:
                        continue
                    e2 = _ck_m(kappa_f, tm_f, kappa_2, tm2, k, -tq)
                    if e2 == 0.0:
                        continue
                    phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
                    term += RD[k] * phase * e1 * e2
                # X: (h nu <- e2 m2) x (f m_f <- e1 m1)
                tqx = tnu - tm2
                for k in ks_X:
                    if abs(tqx) > 2 * k:
                        continue
                    e1 = _ck_m(kappa_h, tnu, kappa_2, tm2, k, tqx)
                    if e1 == 0.0:
                        continue
                    e2 = _ck_m(kappa_f, tm_f, kappa_1, tm1, k, -tqx)
                    if e2 == 0.0:
                        continue
                    phase = 1.0 if (tqx // 2) % 2 == 0 else -1.0
                    term -= RX[k] * phase * e1 * e2
                amp += cw * hole_phase * term

    gamma_au = 2.0 * math.pi * amp * amp
    return float(gamma_au * AU_RATE_PER_S)


__all__ = ["auger_rate_jj_level"]
