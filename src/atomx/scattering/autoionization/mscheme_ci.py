"""m-scheme CI for autoionizing resonance manifolds + exact Auger widths.

Small determinant-basis CI over the active space of a one-hole /
two-excited-electron resonance configuration (e.g. Fe XVII KLL:
2p^5 3l 3l').  Everything — Hamiltonian, J^2 classification, and the
Auger decay amplitude to the closed-shell ion + free electron — is
evaluated over the SAME bitmask determinants with Slater-Condon rules,
so all fermionic signs are consistent by construction (no coupled-scheme
phase conventions anywhere).

Physics:
- H restricted to the manifold: one-body orbital energies (from the
  solved ground-config result; excited orbitals from the frozen-core
  V^{N-1} eigenbasis, matching FAC's fixed central potential) plus all
  active-active two-electron Coulomb interactions.  Core-active
  interactions are inside the orbital energies; cross-configuration
  mixing (e.g. with 2s-hole configs) is outside the manifold and
  neglected.
- Eigenvectors of H at fixed M carry sharp J (H is rotationally
  invariant); J is identified via <J^2>.
- Auger width of an eigenlevel |L> = sum_d w_d |det_d>:
      Gamma = 2*pi * |<ion(closed) + eps_f | V | L>|^2   (a.u.)
  with the delta(E)-normalized shooting continuum electron; the
  determinant matrix elements are two-orbital-difference Slater-Condon
  terms.  This equals FAC's ``4 sum p^2/(2J+1)`` under FAC's
  1/sqrt(k)-amplitude wave convention.

Validated against FAC AITable/level tables for the Fe XVII KLL panel
(see tests/test_auger_kll_vs_fac.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.scattering.distorted_wave.solver import (
    _allowed_k,
    _ck_m,
    _continuum_on_grid,
    _distorting_potential_fn,
    _grid_slater_rk,
    _l_of,
    _make_fine_grid,
    _p_rel,
    _shooting_points,
    _tj_of,
)


@dataclass
class ActiveOrbital:
    """One relativistic spin-orbital of the active space."""

    shell: int        # index of the active subshell
    n: int
    kappa: int
    tm: int           # 2*m
    P: np.ndarray     # fine-grid large component
    Q: np.ndarray


@dataclass
class ManifoldLevel:
    two_J: int
    energy_au: float          # relative to the manifold's lowest level
    gamma_auger_per_s: float
    weights: np.ndarray       # eigenvector over determinants
    dominant: str             # label of the dominant determinant
    E_res_au: float = 0.0     # absolute energy above the closed-shell ion
    a_radiative_per_s: float = 0.0


class ResonanceManifold:
    """Determinant CI over one resonance configuration.

    Parameters
    ----------
    subshells : list of (n, kappa, n_electrons_in_nl_group?)  — see
        ``groups``: the active space is specified as nl groups, each a
        list of (n, kappa) subshells sharing the electron count.
    """

    def __init__(self, result, groups, *, r_fine, dr_fine, orbital_map,
                 orbital_energies, configs=None):
        """``groups``: list of (subshell_list, n_electrons) with
        subshell_list = [(n, kappa), ...] (the jj components of one nl
        shell); ``orbital_map[(n, kappa)] = (P_fine, Q_fine)``;
        ``orbital_energies[(n, kappa)]`` in Hartree.

        ``configs``: optional list of per-group electron-count tuples for
        a multi-configuration manifold (union of determinant sets, e.g.
        [(5,1,1,0), (5,0,1,1)] for 2p^5{3s3p + 3p3d}).  When omitted the
        single configuration from ``groups`` is used."""
        self.result = result
        self.r_fine = r_fine
        self.dr_fine = dr_fine
        self.groups = groups
        self.orbitals: list[ActiveOrbital] = []
        self.group_bits: list[int] = []
        self.group_ne: list[int] = []
        self.eps: list[float] = []
        shell_idx = 0
        for subshell_list, ne in groups:
            mask = 0
            for (n, kappa) in subshell_list:
                tj = _tj_of(kappa)
                P, Q = orbital_map[(n, kappa)]
                for tm in range(-tj, tj + 1, 2):
                    mask |= 1 << len(self.orbitals)
                    self.orbitals.append(
                        ActiveOrbital(shell_idx, n, kappa, tm, P, Q))
                    self.eps.append(float(orbital_energies[(n, kappa)]))
                shell_idx += 1
            self.group_bits.append(mask)
            self.group_ne.append(int(ne))
        self.n_orb = len(self.orbitals)
        self.configs = ([tuple(self.group_ne)] if configs is None
                        else [tuple(c) for c in configs])
        self._rk_cache: dict = {}
        self._v_cache: dict = {}

    # ---- radial + m-resolved two-body elements -------------------------
    def _rk(self, a, b, c, d, k):
        """R^k(a,b;c,d) with orbitals given as ActiveOrbital or (P,Q)."""
        def _key(o):
            return (o.n, o.kappa) if isinstance(o, ActiveOrbital) else id(o)
        key = (_key(a), _key(b), _key(c), _key(d), k)
        val = self._rk_cache.get(key)
        if val is None:
            def _pq(o):
                return (o.P, o.Q) if isinstance(o, ActiveOrbital) else o
            Pa, Qa = _pq(a); Pb, Qb = _pq(b)
            Pc, Qc = _pq(c); Pd, Qd = _pq(d)
            val = _grid_slater_rk(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd,
                                  self.r_fine, self.dr_fine, k)
            self._rk_cache[key] = val
        return val

    def _v(self, a, b, c, d):
        """<a(1) b(2)| 1/r12 |c(1) d(2)> for ActiveOrbitals (m-resolved)."""
        key = ((a.n, a.kappa, a.tm), (b.n, b.kappa, b.tm),
               (c.n, c.kappa, c.tm), (d.n, d.kappa, d.tm))
        val = self._v_cache.get(key)
        if val is not None:
            return val
        tq = a.tm - c.tm
        if tq != d.tm - b.tm:
            self._v_cache[key] = 0.0
            return 0.0
        tot = 0.0
        for k in _allowed_k(a.kappa, c.kappa):
            if k not in _allowed_k(b.kappa, d.kappa):
                continue
            if abs(tq) > 2 * k:
                continue
            e1 = _ck_m(a.kappa, a.tm, c.kappa, c.tm, k, tq)
            if e1 == 0.0:
                continue
            e2 = _ck_m(b.kappa, b.tm, d.kappa, d.tm, k, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            tot += self._rk(a, b, c, d, k) * phase * e1 * e2
        self._v_cache[key] = tot
        return tot

    def _v_as(self, a, b, c, d):
        return self._v(a, b, c, d) - self._v(a, b, d, c)

    # ---- determinant basis ---------------------------------------------
    def determinants(self, tM: int) -> list[int]:
        """All configuration-restricted determinants with total 2M = tM,
        unioned over ``self.configs``."""
        dets: set[int] = set()
        for config in self.configs:
            per_group_choices = []
            for mask, ne in zip(self.group_bits, config):
                bits = [i for i in range(self.n_orb) if mask & (1 << i)]
                per_group_choices.append(list(combinations(bits, ne)))
            def _recurse(gi, occ_bits):
                if gi == len(per_group_choices):
                    tm = sum(self.orbitals[i].tm for i in occ_bits)
                    if tm == tM:
                        m = 0
                        for i in occ_bits:
                            m |= 1 << i
                        dets.add(m)
                    return
                for choice in per_group_choices[gi]:
                    _recurse(gi + 1, occ_bits + list(choice))
            _recurse(0, [])
        return sorted(dets)

    @staticmethod
    def _occ_list(mask):
        out = []
        i = 0
        while mask >> i:
            if mask & (1 << i):
                out.append(i)
            i += 1
        return out

    @staticmethod
    def _phase_remove(mask, i):
        return -1.0 if bin(mask & ((1 << i) - 1)).count("1") % 2 else 1.0

    # ---- Hamiltonian ----------------------------------------------------
    def hamiltonian(self, dets):
        n = len(dets)
        H = np.zeros((n, n))
        orb = self.orbitals
        for a_idx in range(n):
            Da = dets[a_idx]
            occ_a = self._occ_list(Da)
            # diagonal
            e = sum(self.eps[i] for i in occ_a)
            for x in range(len(occ_a)):
                for y in range(x + 1, len(occ_a)):
                    i, j = occ_a[x], occ_a[y]
                    e += self._v_as(orb[i], orb[j], orb[i], orb[j])
            H[a_idx, a_idx] = e
            for b_idx in range(a_idx + 1, n):
                Db = dets[b_idx]
                diff = Da ^ Db
                nd = bin(diff).count("1")
                if nd == 2:
                    # single replacement r (in Da) -> s (in Db)
                    r = (diff & Da).bit_length() - 1
                    s = (diff & Db).bit_length() - 1
                    common = Da & ~(1 << r)
                    ph = (self._phase_remove(Da, r)
                          * self._phase_remove(Db, s))
                    val = 0.0
                    for i in self._occ_list(common):
                        val += self._v_as(orb[r], orb[i], orb[s], orb[i])
                    val *= ph
                    H[a_idx, b_idx] = H[b_idx, a_idx] = val
                elif nd == 4:
                    ra = self._occ_list(diff & Da)   # r1 < r2 in Da
                    sb = self._occ_list(diff & Db)   # s1 < s2 in Db
                    r1, r2 = ra
                    s1, s2 = sb
                    ph = (self._phase_remove(Da, r2)
                          * self._phase_remove(Da & ~(1 << r2), r1)
                          * self._phase_remove(Db, s2)
                          * self._phase_remove(Db & ~(1 << s2), s1))
                    val = ph * self._v_as(orb[r1], orb[r2], orb[s1], orb[s2])
                    H[a_idx, b_idx] = H[b_idx, a_idx] = val
        return H

    # ---- J^2 ------------------------------------------------------------
    def j2_matrix(self, dets):
        """<det'|J^2|det> = <J-J+ + Jz(Jz+2)/... in doubled units.

        J^2 |M> with 2M fixed: J^2 = J-J+ + Jz^2 + Jz (hbar=1;
        using doubled m: Jz = tM/2)."""
        n = len(dets)
        index = {d: i for i, d in enumerate(dets)}
        orbs = self.orbitals
        # J+ applied to each det: sum over occupied i, raise m by 1 within
        # the same (n, kappa) subshell.
        jplus = {}
        for d in dets:
            terms = {}
            for i in self._occ_list(d):
                o = orbs[i]
                tj = _tj_of(o.kappa)
                if o.tm + 2 > tj:
                    continue
                # target orbital: same subshell, tm+2 (adjacent index)
                tgt = i + 1
                ot = orbs[tgt]
                assert (ot.n, ot.kappa, ot.tm) == (o.n, o.kappa, o.tm + 2)
                if d & (1 << tgt):
                    continue
                coeff = math.sqrt((tj - o.tm) * (tj + o.tm + 2)) / 2.0
                ph = self._phase_remove(d, i)
                mid = d & ~(1 << i)
                ph *= self._phase_remove(mid, tgt)
                nd = mid | (1 << tgt)
                terms[nd] = terms.get(nd, 0.0) + ph * coeff
            jplus[d] = terms
        J2 = np.zeros((n, n))
        tM = sum(orbs[i].tm for i in self._occ_list(dets[0]))
        mz = tM / 2.0
        for a_idx, d in enumerate(dets):
            J2[a_idx, a_idx] += mz * (mz + 1.0)
            ta = jplus[d]
            for b_idx, dp in enumerate(dets):
                tb = jplus[dp]
                # <dp| J- J+ |d> = sum_x <J+ dp|x><x|J+ d>
                v = 0.0
                for x, ca in ta.items():
                    cb = tb.get(x)
                    if cb is not None:
                        v += ca * cb
                J2[b_idx, a_idx] += v
        return J2

    # ---- Auger amplitude -------------------------------------------------
    def auger_widths(self, tM: int, resonance_energy_eV: float,
                     *, Z, bsp_data, shooting_n_grid=None,
                     L_max: int = 6, _only_level=None, _precomputed=None):
        """Diagonalize the manifold at 2M = tM and return ManifoldLevels
        with exact Auger widths for levels with 2J = tM.

        The final ion is the closed shell obtained by filling the hole
        group; the free electron carries j_f = J and the parity-allowed l.
        ``_only_level``/``_precomputed`` are internal hooks used by
        :meth:`dr_levels` to reuse a diagonalization and evaluate a single
        eigenvector at its own resonance energy.
        """
        if _precomputed is not None:
            dets, H, J2, evals, evecs = _precomputed
        else:
            dets = self.determinants(tM)
            if not dets:
                return []
            H = self.hamiltonian(dets)
            J2 = self.j2_matrix(dets)
            evals, evecs = np.linalg.eigh(H)

        # continuum electron
        E_res_au = float(resonance_energy_eV) / HARTREE_EV
        par = 1
        d0 = dets[0]
        for i in self._occ_list(d0):
            par *= -1 if _l_of(self.orbitals[i].kappa) % 2 else 1
        # hole parity: closed core is even; the det parity IS the resonance
        # parity relative to the closed shell times the hole-orbital parity
        # squared — i.e. par computed over occupied active orbitals times
        # the parity of the full hole group (even, since the closed group
        # has even parity iff capacity*l even; for a full jj group parity
        # is (+1) only if l*capacity even — compute explicitly):
        hole_group_par = 1
        mask = self.group_bits[0]
        for i in range(self.n_orb):
            if mask & (1 << i):
                hole_group_par *= -1 if _l_of(self.orbitals[i].kappa) % 2 else 1
        par_res = par * hole_group_par  # det holes: closed/det ratio
        tj_f = tM
        l_opts = [(tj_f + 1) // 2, (tj_f - 1) // 2]
        l_f = next((l for l in l_opts
                    if l >= 0 and ((-1 if l % 2 else 1) == par_res)), None)
        levels = []
        if l_f is None:
            return levels
        kappa_f = l_f if tj_f == 2 * l_f - 1 else -(l_f + 1)

        V_func = _distorting_potential_fn(self.result, bsp_data,
                                          Z, screening="fac")
        N_bound = sum(float(sh.occupation) for sh in self.result.subshells)
        Z_res = max(float(Z) - N_bound + 1.0, 1.0)
        r_max_box = float(bsp_data["t"][-1])
        n_shoot = (int(shooting_n_grid) if shooting_n_grid
                   else _shooting_points(r_max_box, _p_rel(max(E_res_au, 1.0))))
        try:
            Pf, Qf = _continuum_on_grid(kappa_f, E_res_au, V_func, Z_res,
                                        self.r_fine, r_max_box, n_shoot)
        except Exception:
            return levels
        cont = (Pf, Qf)
        tm_f = tM  # closed-shell ion: m_f = M

        # Amplitude per determinant: bra = closed hole-group + eps_f;
        # ket = det.  Two-orbital difference: ket-only = {e1, e2} (the
        # excited electrons), bra-only = {hole nu, eps_f}.
        hole_mask = self.group_bits[0]
        full_hole = hole_mask

        def _amp(det):
            # holes nu = hole-group orbitals absent in det
            missing = full_hole & ~det
            occ_exc = [i for i in self._occ_list(det)
                       if not (full_hole & (1 << i))]
            if bin(missing).count("1") != 1 or len(occ_exc) != 2:
                return 0.0
            nu = missing.bit_length() - 1
            e1, e2 = occ_exc
            o_nu, o1, o2 = (self.orbitals[nu], self.orbitals[e1],
                            self.orbitals[e2])
            if o_nu.tm + tm_f != o1.tm + o2.tm:
                return 0.0
            # Slater-Condon two-difference element with explicit alignment:
            # <bra|V|ket> = ph * [ v(nu f; e1 e2) - v(nu f; e2 e1) ]
            # bra = det with e1,e2 removed, nu added, f appended (last bit).
            ph = self._phase_remove(det, e2)
            mid = det & ~(1 << e2)
            ph *= self._phase_remove(mid, e1)
            mid &= ~(1 << e1)
            ph *= self._phase_remove(mid, nu)   # creation phase at nu
            # eps_f is appended beyond all active bits: creation phase =
            # (-1)^{N_occ(mid|nu)} with the electron count above... appended
            # LAST => phase from bits below = all occupied => (-1)^{count}:
            nocc = bin(mid | (1 << nu)).count("1")
            if nocc % 2:
                ph = -ph
            # antisymmetrized: direct (nu<-e1)(f<-e2) minus exchange
            tot = self._v_cont(o_nu, cont, kappa_f, tm_f, o1, o2)
            tot -= self._v_cont(o_nu, cont, kappa_f, tm_f, o2, o1)
            return ph * tot

        levels_out = []
        e_min = float(np.min(evals))
        level_range = (range(len(evals)) if _only_level is None
                       else [_only_level])
        for li in level_range:
            w = evecs[:, li]
            j2_exp = float(w @ J2 @ w)
            tJ_val = int(round(-1 + math.sqrt(1.0 + 4.0 * j2_exp)))
            if tJ_val != tM:
                continue
            A = 0.0
            for d_idx, det in enumerate(dets):
                wi = w[d_idx]
                if abs(wi) < 1e-12:
                    continue
                A += wi * _amp(det)
            gam = 2.0 * math.pi * A * A * AU_RATE_PER_S
            imax = int(np.argmax(np.abs(w)))
            levels_out.append(ManifoldLevel(
                two_J=tM,
                energy_au=float(evals[li] - e_min),
                gamma_auger_per_s=float(gam),
                weights=w,
                dominant=self._det_label(dets[imax]),
            ))
        return levels_out

    def _v_cont(self, o_nu, cont, kappa_f, tm_f, oc, od):
        """<nu(1) f(2)| 1/r12 |c(1) d(2)> with f the continuum orbital."""
        tq = o_nu.tm - oc.tm
        if tq != od.tm - tm_f:
            return 0.0
        tot = 0.0
        Pf, Qf = cont
        for k in _allowed_k(o_nu.kappa, oc.kappa):
            if k not in _allowed_k(kappa_f, od.kappa):
                continue
            if abs(tq) > 2 * k:
                continue
            e1 = _ck_m(o_nu.kappa, o_nu.tm, oc.kappa, oc.tm, k, tq)
            if e1 == 0.0:
                continue
            e2 = _ck_m(kappa_f, tm_f, od.kappa, od.tm, k, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            key = ("cont", (o_nu.n, o_nu.kappa), (oc.n, oc.kappa),
                   (od.n, od.kappa), kappa_f, k)
            val = self._rk_cache.get(key)
            if val is None:
                val = _grid_slater_rk(o_nu.P, o_nu.Q, Pf, Qf,
                                      oc.P, oc.Q, od.P, od.Q,
                                      self.r_fine, self.dr_fine, k)
                self._rk_cache[key] = val
            tot += val * phase * e1 * e2
        return tot

    def ion_energy(self) -> float:
        """Energy of the closed-shell ion (hole group filled, no excited
        electrons) in the same frozen-orbital accounting as the manifold
        Hamiltonian — differences give resonance / transition energies."""
        full = self.group_bits[0]
        return float(self.hamiltonian([full])[0, 0])

    def _rdip(self, o_a, o_b):
        """Length-gauge radial dipole integral (P_a P_b + Q_a Q_b) r dr."""
        key = ("dip", (o_a.n, o_a.kappa), (o_b.n, o_b.kappa))
        val = self._rk_cache.get(key)
        if val is None:
            rho = o_a.P * o_b.P + o_a.Q * o_b.Q
            val = float(np.sum(rho * self.r_fine * self.dr_fine))
            self._rk_cache[key] = val
        return val

    def _radiative_width(self, dets, w, E_level: float) -> float:
        """Total E1 stabilizing width of a CI level (fills the hole).

        Final states: closed hole group + one spectator electron e'.
        One-orbital-difference Slater-Condon: the dipole moves one excited
        electron into the single hole; the other must match e'.

            A = (4/3) alpha^3 sum_f w_f^3 sum_q |<f| d_q |level>|^2
        """
        from atomx.constants import ALPHA

        full = self.group_bits[0]
        # coherent amplitudes keyed by (spectator orbital index, 2q)
        amps: dict = {}
        efinal: dict = {}
        for d_idx, det in enumerate(dets):
            wd = w[d_idx]
            if abs(wd) < 1e-12:
                continue
            missing = full & ~det
            if bin(missing).count("1") != 1:
                continue
            nu = missing.bit_length() - 1
            occ_exc = [i for i in self._occ_list(det)
                       if not (full & (1 << i))]
            if len(occ_exc) != 2:
                continue
            o_nu = self.orbitals[nu]
            for e_i, spect in ((occ_exc[0], occ_exc[1]),
                               (occ_exc[1], occ_exc[0])):
                o_e = self.orbitals[e_i]
                tq = o_nu.tm - o_e.tm
                if abs(tq) > 2:
                    continue
                ang = _ck_m(o_nu.kappa, o_nu.tm, o_e.kappa, o_e.tm, 1, tq)
                if ang == 0.0:
                    continue
                # SC one-body sign: remove e_i, create at nu.
                ph = self._phase_remove(det, e_i)
                mid = det & ~(1 << e_i)
                ph *= self._phase_remove(mid, nu)
                amp = wd * ph * ang * self._rdip(o_nu, o_e)
                key = (spect, tq)
                amps[key] = amps.get(key, 0.0) + amp
                if spect not in efinal:
                    fdet = full | (1 << spect)
                    efinal[spect] = float(
                        self.hamiltonian([fdet])[0, 0])
        a_r = 0.0
        for (spect, tq), amp in amps.items():
            omega = E_level - efinal[spect]
            if omega <= 0.0:
                continue
            a_r += (4.0 / 3.0) * (ALPHA ** 3) * omega ** 3 * amp * amp
        return a_r * AU_RATE_PER_S

    def dr_levels(self, tM: int, *, Z, bsp_data, shooting_n_grid=None,
                  L_max: int = 6, e_ion: float | None = None):
        """Manifold levels with ab-initio resonance energies, Auger and
        E1 stabilizing widths — everything a DR calculation needs.

        Like :meth:`auger_widths` but the Auger-electron energy is the
        level's own ``E_level - E_ion`` (no external resonance energy),
        and each level also carries its total radiative width.
        """
        dets = self.determinants(tM)
        if not dets:
            return []
        H = self.hamiltonian(dets)
        J2 = self.j2_matrix(dets)
        evals, evecs = np.linalg.eigh(H)
        if e_ion is None:
            e_ion = self.ion_energy()

        out = []
        e_min = float(np.min(evals))
        for li in range(len(evals)):
            wvec = evecs[:, li]
            j2_exp = float(wvec @ J2 @ wvec)
            tJ_val = int(round(-1 + math.sqrt(1.0 + 4.0 * j2_exp)))
            if tJ_val != tM:
                continue
            E_res = float(evals[li] - e_ion)
            if E_res <= 0.0:
                continue
            lv = self.auger_widths(
                tM, E_res * HARTREE_EV, Z=Z, bsp_data=bsp_data,
                shooting_n_grid=shooting_n_grid, L_max=L_max,
                _only_level=li, _precomputed=(dets, H, J2, evals, evecs))
            gam_a = lv[0].gamma_auger_per_s if lv else 0.0
            a_r = self._radiative_width(dets, wvec, float(evals[li]))
            imax = int(np.argmax(np.abs(wvec)))
            out.append(ManifoldLevel(
                two_J=tM,
                energy_au=float(evals[li] - e_min),
                gamma_auger_per_s=float(gam_a),
                weights=wvec,
                dominant=self._det_label(dets[imax]),
                E_res_au=E_res,
                a_radiative_per_s=float(a_r),
            ))
        return out

    def _det_label(self, det):
        parts = []
        _L = "spdfghiklmno"
        for i in self._occ_list(det):
            o = self.orbitals[i]
            if not (self.group_bits[0] & (1 << i)):
                parts.append(f"{o.n}{_L[_l_of(o.kappa)]}{_tj_of(o.kappa)}/2"
                             f"[m={o.tm}/2]")
        missing = self.group_bits[0] & ~det
        for i in self._occ_list(missing):
            o = self.orbitals[i]
            parts.insert(0, f"{o.n}{_L[_l_of(o.kappa)]}{_tj_of(o.kappa)}/2^-1"
                            f"[m={o.tm}/2]")
        return " ".join(parts)


__all__ = ["ResonanceManifold", "ManifoldLevel"]
