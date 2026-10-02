"""m-scheme CI and coherent Auger widths for autoionizing manifolds.

Small determinant-basis CI over the active space of a one-hole /
two-excited-electron resonance configuration (e.g. Ne-like Fe LMM capture:
2p^5 3l 3l').  Everything — Hamiltonian, J^2 classification, and the
Auger decay amplitude to the closed-shell ion + free electron — is
evaluated over the SAME bitmask determinants with Slater-Condon rules,
so all fermionic signs are consistent by construction (no coupled-scheme
phase conventions anywhere).

Physics:
- H restricted to the manifold: bare nuclear/environment Dirac matrix
  elements plus the direct and exchange contraction of the inactive closed
  core, plus all active-active Coulomb terms. Bound integrals use the native
  radial representation; continuum integrals use the scattering grid.
  A direct low-level construction without one_body explicitly retains the
  approximate dressed-eigenvalue model used by historical diagnostics.
- Eigenvectors of H at fixed M carry sharp J (H is rotationally
  invariant); J is identified via <J^2>.
- Auger width of an eigenlevel |L> = sum_d w_d |det_d>:
      Gamma = 2*pi * |<ion(closed) + eps_f | V | L>|^2   (a.u.)
  with the delta(E)-normalized shooting continuum electron; the
  determinant matrix elements are two-orbital-difference Slater-Condon
  terms.  This equals FAC's ``4 sum p^2/(2J+1)`` under FAC's
  1/sqrt(k)-amplitude wave convention.

The matched Ne-like Fe dr33 panel compares complete level inventories,
aggregate widths and thermal rates with FAC (tests/test_dr_manifold_e2e.py).
Individual widths remain sensitive to the orbital and mixing model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from atomx.angular.coefficients import ckm
from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.numerics.quadrature import grid_slater_rk, p_rel
from atomx.numerics.radial_metric import RadialMetric
from atomx.potentials.static_field import static_hartree_potential
from atomx.scattering.distorted_wave.solver import (
    _allowed_k,
    _continuum_on_grid,
    _l_of,
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
    parity: int = 0           # +1 / -1 relative to the closed-shell ion
    radiative_channels: list[RadiativeChannel] = field(default_factory=list)


@dataclass(frozen=True)
class RadiativeChannel:
    """E1 decay to one bound final CI level, summed over its magnetic substates."""

    final_kappa: int
    final_index: int
    final_energy_au: float
    omega_au: float
    A_per_s: float
    dominant: tuple[int, int]
    photon_approximation: str


class ResonanceManifold:
    """Determinant CI over one resonance configuration.

    Parameters
    ----------
    subshells : list of (n, kappa, n_electrons_in_nl_group?)  — see
        ``groups``: the active space is specified as nl groups, each a
        list of (n, kappa) subshells sharing the electron count.
    """

    def __init__(self, result, groups, *, r_fine, dr_fine, orbital_map,
                 orbital_energies, configs=None, V_func=None,
                 one_body=None, bound_radial=None, bound_orbitals=None,
                 bound_workspace=None):
        """``groups``: list of (subshell_list, n_electrons) with
        subshell_list = [(n, kappa), ...] (the jj components of one nl
        shell); ``orbital_map[(n, kappa)] = (P_fine, Q_fine)``;
        ``orbital_energies[(n, kappa)]`` in Hartree.

        ``configs``: optional list of per-group electron-count tuples for
        a multi-configuration manifold (union of determinant sets, e.g.
        [(5,1,1,0), (5,0,1,1)] for 2p^5{3s3p + 3p3d}).  When omitted the
        single configuration from ``groups`` is used.

        ``bound_radial`` evaluates the full-CI radial-key convention.
        ``bound_orbitals`` and ``bound_workspace`` provide the native bound
        representation for photon integrals. High-level constructors supply
        these; ``orbital_map`` remains the separate continuum-grid sampling.
        """
        self.result = result
        self.r_fine = r_fine
        self.dr_fine = dr_fine
        self.groups = groups
        self.bound_radial = bound_radial
        if (bound_orbitals is None) != (bound_workspace is None):
            raise ValueError('bound orbitals and their native workspace must be supplied together')
        if bound_orbitals is None:
            from atomx.numerics.radial_workspace import make_grid_workspace
            bound_orbitals = orbital_map
            bound_workspace = make_grid_workspace(r_fine, dr_fine)
        self.bound_orbitals = bound_orbitals
        self.bound_workspace = bound_workspace
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
        #: Optional local potential override for the Auger continuum
        #: electron (e.g. the converged exchange='gu' DFS potential —
        #: FAC's own convention).  Default: rebuilt from ``result``.
        self.V_func = V_func
        #: Optional core-folded one-body matrix t[(n,kappa),(n',kappa)]
        #: = <a| h + V_core |b> (same kappa; off-diagonals included).
        #: High-level constructors supply the core-folded matrix. Direct
        #: construction without one_body uses dressed eigenvalues and is
        #: an explicitly approximate active-space Hamiltonian.
        self.one_body = one_body
        if one_body is not None:
            # per-orbital-index matrix (m-independent, same-kappa only)
            self._t = np.zeros((self.n_orb, self.n_orb))
            for i, oi in enumerate(self.orbitals):
                for j, oj in enumerate(self.orbitals):
                    if oi.kappa != oj.kappa or oi.tm != oj.tm:
                        continue
                    val = one_body.get(((oi.n, oi.kappa), (oj.n, oj.kappa)))
                    if val is not None:
                        self._t[i, j] = float(val)
        else:
            self._t = None
        self._rk_cache: dict = {}
        self._v_cache: dict = {}
        self._radiative_final_cache = None

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
            Pa, Qa = _pq(a)
            Pb, Qb = _pq(b)
            Pc, Qc = _pq(c)
            Pd, Qd = _pq(d)
            if self.bound_radial is not None:
                val = self.bound_radial(("r", k, _key(a), _key(c), _key(b), _key(d)))
            else:
                val = grid_slater_rk(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd,
                                     self.r_fine, k)
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
            e1 = ckm(a.kappa, a.tm, c.kappa, c.tm, k, tq)
            if e1 == 0.0:
                continue
            e2 = ckm(b.kappa, b.tm, d.kappa, d.tm, k, -tq)
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
        """Configuration determinants with total 2M, pruning impossible branches.

        The orbital combinations remain exact. Bounds on the remaining
        magnetic projections avoid materializing the full Cartesian product.
        """
        dets: set[int] = set()
        for config in self.configs:
            groups = []
            lower = []
            upper = []
            for mask, ne in zip(self.group_bits, config, strict=False):
                bits = [i for i in range(self.n_orb) if mask & (1 << i)]
                groups.append((bits, ne))
                values = sorted(self.orbitals[i].tm for i in bits)
                lower.append(sum(values[:ne]))
                upper.append(sum(values[len(values)-ne:]) if ne else 0)
            remain_lo = [0]*(len(groups)+1)
            remain_hi = [0]*(len(groups)+1)
            for i in range(len(groups)-1, -1, -1):
                remain_lo[i] = remain_lo[i+1]+lower[i]
                remain_hi[i] = remain_hi[i+1]+upper[i]

            def visit(gi, mask, tm, groups=groups,
                      remain_lo=remain_lo, remain_hi=remain_hi):
                if tm+remain_lo[gi] > tM or tm+remain_hi[gi] < tM:
                    return
                if gi == len(groups):
                    dets.add(mask)
                    return
                bits, ne = groups[gi]
                for choice in combinations(bits, ne):
                    cmask = mask
                    ctm = tm
                    for i in choice:
                        cmask |= 1 << i
                        ctm += self.orbitals[i].tm
                    visit(gi+1, cmask, ctm)

            visit(0, 0, 0)
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

    def parity_of_det(self, det: int) -> int:
        """Parity of a determinant relative to the closed-shell ion.

        Product of (-1)^l over occupied active orbitals times the parity
        of the full (closed) hole group — i.e. the parity the continuum
        electron must carry away in the Auger decay.  H conserves parity,
        so all determinants mixed into one eigenlevel share this value;
        a multi-parity determinant basis (e.g. a full 3l3l' complex)
        block-diagonalizes and each level inherits the parity of its own
        block."""
        par = 1
        for i in self._occ_list(det):
            par *= -1 if _l_of(self.orbitals[i].kappa) % 2 else 1
        mask = self.group_bits[0]
        for i in range(self.n_orb):
            if mask & (1 << i):
                par *= -1 if _l_of(self.orbitals[i].kappa) % 2 else 1
        return par

    # ---- Hamiltonian ----------------------------------------------------
    def hamiltonian(self, dets):
        n = len(dets)
        H = np.zeros((n, n))
        orb = self.orbitals
        t = self._t
        for a_idx in range(n):
            Da = dets[a_idx]
            occ_a = self._occ_list(Da)
            # diagonal
            if t is None:
                e = sum(self.eps[i] for i in occ_a)
            else:
                e = sum(t[i, i] for i in occ_a)
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
                    if t is not None:
                        val += t[r, s]
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

        EXCITED-channel partial widths (resonance -> excited ion + e',
        needed for ab-initio resonant excitation) are the documented
        generalization: the final ion is then open-shell (hole group with
        one active spectator), its levels come from diagonalizing the
        ion manifold with this same machinery, and the decay amplitude
        needs the ONE-difference two-body Slater-Condon rule (a sum over
        the shared core) in addition to the two-difference rule used
        here; the continuum orbital can ride the existing fine-grid
        v-machinery as an extra ActiveOrbital (n=0 key).  Validation
        target: clean-room FAC AITable partial rates to excited ion
        configurations (open channels appear for KLM / dr34-class
        resonances).  Until then
        :func:`~atomx.scattering.dielectronic_recombination.solver.resonant_excitation_rate`
        takes the partial widths as input.
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

        # continuum electron: one partial wave per parity block (a
        # multi-configuration determinant basis may span both parities;
        # H is parity-block-diagonal, so each eigenlevel carries the
        # parity of its dominant determinant and decays into the l_f
        # that parity allows).
        E_res_au = float(resonance_energy_eV) / HARTREE_EV
        V_func = self.V_func
        if V_func is None:
            V_func = static_hartree_potential(self.result, screening="mean")
        N_bound = sum(float(sh.occupation) for sh in self.result.subshells)
        Z_res = max(float(Z) - N_bound + 1.0, 1.0)
        r_max_box = float(bsp_data["t"][-1])
        n_shoot = (int(shooting_n_grid) if shooting_n_grid
                   else _shooting_points(r_max_box, p_rel(max(E_res_au, 1.0))))
        tm_f = tM  # closed-shell ion: m_f = M
        tj_f = tM

        _cont_cache: dict = {}
        # All levels in this call use one continuum energy and potential.
        # Keep its radial integrals local: the manifold's persistent bound
        # cache must never retain integrals from another electron energy.
        continuum_radial: dict = {}

        def _cont_for_parity(par_res):
            """(kappa_f, (Pf, Qf)) for this parity, or None if forbidden."""
            if par_res in _cont_cache:
                return _cont_cache[par_res]
            l_opts = [(tj_f + 1) // 2, (tj_f - 1) // 2]
            l_f = next((l for l in l_opts
                        if l >= 0 and ((-1 if l % 2 else 1) == par_res)),
                       None)
            out = None
            if l_f is not None:
                kappa_f = l_f if tj_f == 2 * l_f - 1 else -(l_f + 1)
                try:
                    Pf, Qf = _continuum_on_grid(
                        kappa_f, E_res_au, V_func, Z_res,
                        self.r_fine, r_max_box, n_shoot)
                    out = (kappa_f, (Pf, Qf))
                except Exception as exc:
                    raise RuntimeError(
                        f"Auger continuum failed: kappa={kappa_f}, "
                        f"E={E_res_au} Ha, Z_res={Z_res}") from exc
            _cont_cache[par_res] = out
            return out

        # Amplitude per determinant: bra = closed hole-group + eps_f;
        # ket = det.  Two-orbital difference: ket-only = {e1, e2} (the
        # excited electrons), bra-only = {hole nu, eps_f}.
        hole_mask = self.group_bits[0]
        full_hole = hole_mask

        def _amp(det, kappa_f, cont):
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
            tot = self._v_cont(o_nu, cont, kappa_f, tm_f, o1, o2, continuum_radial)
            tot -= self._v_cont(o_nu, cont, kappa_f, tm_f, o2, o1, continuum_radial)
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
            imax = int(np.argmax(np.abs(w)))
            par_res = self.parity_of_det(dets[imax])
            chan = _cont_for_parity(par_res)
            gam = 0.0
            if chan is not None:
                kappa_f, cont = chan
                A = 0.0
                for d_idx, det in enumerate(dets):
                    wi = w[d_idx]
                    if abs(wi) < 1e-12:
                        continue
                    A += wi * _amp(det, kappa_f, cont)
                gam = 2.0 * math.pi * A * A * AU_RATE_PER_S
            levels_out.append(ManifoldLevel(
                two_J=tM,
                energy_au=float(evals[li] - e_min),
                gamma_auger_per_s=float(gam),
                weights=w,
                dominant=self._det_label(dets[imax]),
                parity=par_res,
            ))
        return levels_out

    def _v_cont(self, o_nu, cont, kappa_f, tm_f, oc, od, radial_cache):
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
            e1 = ckm(o_nu.kappa, o_nu.tm, oc.kappa, oc.tm, k, tq)
            if e1 == 0.0:
                continue
            e2 = ckm(kappa_f, tm_f, od.kappa, od.tm, k, -tq)
            if e2 == 0.0:
                continue
            phase = 1.0 if (tq // 2) % 2 == 0 else -1.0
            key = ((o_nu.n, o_nu.kappa), (oc.n, oc.kappa),
                   (od.n, od.kappa), kappa_f, k)
            val = radial_cache.get(key)
            if val is None:
                val = grid_slater_rk(o_nu.P, o_nu.Q, Pf, Qf,
                                     oc.P, oc.Q, od.P, od.Q,
                                     self.r_fine, k)
                radial_cache[key] = val
            tot += val * phase * e1 * e2
        return tot

    def ion_energy(self) -> float:
        """Energy of the closed-shell ion (hole group filled, no excited
        electrons) in the same frozen-orbital accounting as the manifold
        Hamiltonian — differences give resonance / transition energies."""
        full = self.group_bits[0]
        return float(self.hamiltonian([full])[0, 0])

    def electric_radial(self, bra, ket, rank, omega=None):
        """Bound electric multipole on the native radial representation.

        ``omega`` is the signed many-electron E_bra - E_ket in Hartree.
        Omit it only for an explicit static-operator diagnostic.
        """
        from atomx.properties.transitions.multipole import radial_eL

        a, b = (bra.n, bra.kappa), (ket.n, ket.kappa)
        key = ("electric", rank, omega, a, b)
        value = self._rk_cache.get(key)
        if value is None:
            value = radial_eL(*self.bound_orbitals[b], *self.bound_orbitals[a],
                              self.bound_workspace, rank,
                              kappa_i=b[1], kappa_f=a[1], omega=omega)
            self._rk_cache[key] = value
        return value

    def _rdip(self, o_a, o_b):
        """Static dipole for the mixing-invariant trace diagnostic."""
        return self.electric_radial(o_a, o_b, 1)

    def _radiative_final_blocks(self):
        """Closed hole shell plus one electron, including same-kappa radial CI.

        Each block has J=j_spectator and identical radial mixing for all M.
        Diagonalize it once at M=-J, then reuse that mixing for each magnetic
        projection in the photon sum. The inactive-core constant cancels.
        """
        if self.one_body is None:
            raise ValueError('radiative final-state CI requires the bare plus frozen-core '
                             'one-body Hamiltonian; dressed orbital energies are insufficient')
        if self._radiative_final_cache is not None:
            return self._radiative_final_cache
        full = self.group_bits[0]
        spectators = {}
        for i, orb in enumerate(self.orbitals):
            if not full & (1 << i):
                spectators.setdefault(orb.kappa, {}).setdefault(orb.tm, []).append(i)
        blocks = []
        for kappa, by_m in sorted(spectators.items()):
            tm = min(by_m)
            indices = by_m[tm]
            dets = [full | (1 << i) for i in indices]
            energies, mixing = np.linalg.eigh(self.hamiltonian(dets))
            blocks.append((kappa, by_m, energies, mixing))
        self._radiative_final_cache = blocks
        return blocks

    def radiative_channels(self, dets, weights, energy_au, *,
                           photon_approximation="finite_wavelength"):
        """E1 stabilization into closed-hole plus spectator CI eigenstates.

        Initial and final energies come from the same frozen-core Hamiltonian.
        Every orbital pair in a line uses its signed many-electron photon
        frequency. Amplitudes are contracted coherently before summing final
        magnetic substates. The initial vector may have any fixed M; no extra
        statistical division is needed for this sum over photon polarizations.
        Only final levels below the closed-shell ion threshold stabilize.
        """
        from atomx.angular.mscheme import one_body_action
        from atomx.constants import ALPHA

        if photon_approximation not in ('finite_wavelength', 'long_wavelength'):
            raise ValueError('unknown photon approximation')
        full = self.group_bits[0]
        if any((full & ~d).bit_count() != 1 or (d & ~full).bit_count() != 2 for d in dets):
            raise ValueError('E1 stabilization requires one hole and two excited electrons')
        projections = {sum(self.orbitals[i].tm for i in self._occ_list(d)) for d in dets}
        if len(projections) != 1:
            raise ValueError('initial CI vector must have a single magnetic projection')
        tm_initial = projections.pop()
        threshold = self.ion_energy()
        angular = {}
        for a in self._occ_list(full):
            oa = self.orbitals[a]
            for b, ob in enumerate(self.orbitals):
                if full & (1 << b):
                    continue
                q = oa.tm-ob.tm
                if abs(q) <= 2:
                    value = ckm(oa.kappa, oa.tm, ob.kappa, ob.tm, 1, q)
                    if value:
                        angular[a, b] = value
        channels = []
        for kappa, by_m, energies, mixing in self._radiative_final_blocks():
            for level, final_energy in enumerate(energies):
                omega = float(energy_au-final_energy)
                if omega <= 0 or final_energy >= threshold:
                    continue
                frequency = -omega if photon_approximation == 'finite_wavelength' else None
                elements = {pair: coefficient*self.electric_radial(
                    self.orbitals[pair[0]], self.orbitals[pair[1]], 1, frequency)
                    for pair, coefficient in angular.items()}
                strength = 0.
                for tm, indices in by_m.items():
                    if abs(tm-tm_initial) > 2:
                        continue
                    final_dets = [full | (1 << i) for i in indices]
                    amplitude = np.vdot(mixing[:, level],
                        one_body_action(final_dets, dets, weights, elements))
                    strength += float(abs(amplitude)**2)
                dominant_index = by_m[min(by_m)][int(np.argmax(abs(mixing[:, level])))]
                dominant = self.orbitals[dominant_index]
                channels.append(RadiativeChannel(kappa, level, float(final_energy), omega,
                    (4./3.)*ALPHA**3*omega**3*strength*AU_RATE_PER_S,
                    (dominant.n, dominant.kappa), photon_approximation))
        return channels

    def dr_levels(self, tM: int, *, Z, bsp_data, shooting_n_grid=None,
                  L_max: int = 6, e_ion: float | None = None,
                  photon_approximation="finite_wavelength"):
        """Manifold levels with ab-initio resonance energies, Auger and
        E1 stabilizing widths — everything a DR calculation needs.

        Like :meth:`auger_widths` but the Auger-electron energy is the
        level's own ``E_level - E_ion`` (no external resonance energy),
        and each level also carries its E1 stabilization channels and sum.
        The final bound states use the same CI Hamiltonian. An explicit
        ``e_ion`` changes the Auger threshold only, not photon frequencies.
        ``photon_approximation='long_wavelength'`` is an explicit diagnostic;
        the default keeps the finite-frequency photon operator.
        """
        if photon_approximation not in ('finite_wavelength', 'long_wavelength'):
            raise ValueError('unknown photon approximation')
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
            radiative = self.radiative_channels(dets, wvec, float(evals[li]),
                                                photon_approximation=photon_approximation)
            a_r = sum(channel.A_per_s for channel in radiative)
            imax = int(np.argmax(np.abs(wvec)))
            out.append(ManifoldLevel(
                two_J=tM,
                energy_au=float(evals[li] - e_min),
                gamma_auger_per_s=float(gam_a),
                weights=wvec,
                dominant=self._det_label(dets[imax]),
                E_res_au=E_res,
                a_radiative_per_s=float(a_r),
                radiative_channels=radiative,
                parity=(lv[0].parity if lv
                        else self.parity_of_det(dets[imax])),
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


def bspline_bound_integrals(result, active_coeffs):
    """Native Coulomb callback and complete coefficient map for a bound space.

    Occupied core spinors and virtuals retain their original B-spline
    representation. The radial evaluator owns its nuclear Dirac blocks, so
    subsequent manifolds or CI builds cannot change this Hamiltonian.
    """
    from atomx.physics.csf.matrix import _radial_evaluator

    bsp = getattr(result, "_bsp_data", None) or result.bsp_data
    ns = max(bsp["nsp"], bsp["nsq"])
    orbitals = {}
    counts = {}
    for sh in result.subshells:
        index = counts.get(sh.kappa, 0)
        orbitals[sh.n, sh.kappa] = result.orbitals_raw[sh.kappa][index][:2]
        counts[sh.kappa] = index + 1
    orbitals.update(active_coeffs)
    orbitals = {key: tuple(np.pad(v, (0, ns-len(v))) for v in pair)
                for key, pair in orbitals.items()}
    radial = _radial_evaluator(orbitals, bsp, int(result.Z), ns,
                              nuclear_potential=getattr(result, "_nuclear_potential", None))
    return radial, orbitals


# ---------------------------------------------------------------------------
# High-level entry point: config strings -> J-resolved CI levels
# ---------------------------------------------------------------------------

_L_CHARS = "spdfghiklmno"


def _parse_config_tokens(config: str) -> dict:
    """'2p5 3s1 3p1' -> {(n, l): occ}."""
    import re

    out: dict = {}
    for tok in config.split():
        m = re.fullmatch(r"(\d+)([a-z])(\d*)", tok)
        if m is None:
            raise ValueError(f"cannot parse config token {tok!r}")
        n, l_char, occ = int(m.group(1)), m.group(2), m.group(3)
        l = _L_CHARS.index(l_char)
        out[(n, l)] = out.get((n, l), 0) + (int(occ) if occ else 1)
    return out


def _kappas_of_l(l: int) -> list[int]:
    return [-1] if l == 0 else [l, -(l + 1)]


def _orbital_label(n: int, kappa: int) -> str:
    return f"{n}{_L_CHARS[_l_of(kappa)]}{_tj_of(kappa)}/2"


def _subshell_meta(result):
    """[(n, kappa, occupation)] for a solved closed-shell result.

    B-spline results carry ``subshells``; shooting results are
    reconstructed from the ``orbital_energies`` labels (the ground
    configuration is closed-shell, so each subshell holds 2j+1)."""
    import re

    subs = getattr(result, "subshells", None)
    if subs is not None:
        return [(sh.n, sh.kappa, float(sh.occupation)) for sh in subs]
    out = []
    for label in result.orbital_energies:
        m = re.fullmatch(r"(\d+)([a-z])(\d+)/2", label)
        if m is None:
            raise ValueError(f"cannot parse orbital label {label!r}")
        n = int(m.group(1))
        l = _L_CHARS.index(m.group(2))
        tj = int(m.group(3))
        kappa = l if tj == 2 * l - 1 else -(l + 1)
        out.append((n, kappa, float(tj + 1)))
    return out


class _ClosedShellTarget:
    """Minimal result adapter the manifold needs: Z + subshell occupations."""

    class _Sub:
        __slots__ = ("n", "kappa", "occupation")

        def __init__(self, n, kappa, occupation):
            self.n, self.kappa, self.occupation = n, kappa, occupation

    def __init__(self, Z, meta):
        self.Z = Z
        self.subshells = [self._Sub(*m) for m in meta]


def manifold_from_configs(result, configs, *, e_max_eV: float = 1000.0,
                          one_body=None, V_func=None):
    """Build a :class:`ResonanceManifold` from non-relativistic config strings.

    ``result`` is the solved closed-shell ground configuration of the
    recombining ion; ``configs`` are strings like ``"2p5 3s1 3p1"`` naming
    one or more resonance configurations of the SAME hole subshell
    (closed-core tokens such as ``1s2 2s2`` may be included and are
    ignored).  All configurations are unioned into a single CI manifold,
    so levels mix across configurations within each (J, parity) block —
    the physics the configuration-average proxy misses.

    Two orbital-basis pipelines, selected by the solve that produced
    ``result``:

    * **FAC-like local-DFS basis** (``engine="shooting"``,
      ``exchange="gu"`` — the result carries the converged ``_rv``
      potential): every orbital, occupied or virtual, is an eigenstate
      of the one optimized central potential (FAC's ``OptimizeRadial``
      scheme, via :func:`~atomx.numerics.shooting.local.dfs_frozen_orbital`),
      the exact core-folded one-body matrix
      (:func:`atomx.physics.csf.frozen_core.closed_core_one_body`) makes the manifold a proper
      active-space CI, and the Auger continuum runs in the same
      potential.  This is the configuration validated against FAC
      AITable per-level widths.
    * **B-spline V^N basis** (results with native coefficients): hole orbitals from the
      solved result and virtuals from the frozen-core basis. The one-body
      matrix is reconstructed from the bare Dirac Hamiltonian and frozen
      core direct/exchange integrals, so active interactions are counted
      exactly once. The FAC-like DFS basis is still the matched reference
      for quantitative comparisons.

    Returns ``(manifold, bsp_data)`` ready for
    :meth:`ResonanceManifold.dr_levels` / :meth:`~ResonanceManifold.auger_widths`.
    """
    from atomx.physics.csf.frozen_core import closed_core_one_body
    from atomx.radial_state import radial_workspace_of

    if not np.isfinite(e_max_eV) or e_max_eV <= 0:
        raise ValueError('continuum grid energy must be finite and positive')
    if getattr(result, '_equation', 'dirac') != 'dirac':
        raise NotImplementedError('resonance CI requires Dirac spinors')
    if (getattr(result, '_grid', None) is not None
            and getattr(result, '_rv', None) is None):
        raise NotImplementedError(
            'resonance virtuals for nonlocal shooting HF are not implemented; '
            'supply a solved local-potential shooting result or a native '
            'B-spline result. The orbital model is not converted automatically.')
    if (getattr(result, '_rv', None) is None
            and not (getattr(result, '_bsp_data', None) or getattr(result, 'bsp_data', None))):
        raise NotImplementedError('resonance manifold requires solved native radial data')
    meta = _subshell_meta(result)

    # ---- classify tokens against the ground configuration ---------------
    ground_occ: dict = {}
    for (n, kappa, occ) in meta:
        key = (n, _l_of(kappa))
        ground_occ[key] = ground_occ.get(key, 0.0) + occ

    parsed = [_parse_config_tokens(c) for c in configs]
    hole_nl = None
    excited_nls: list = []
    for p in parsed:
        for (n, l), occ in p.items():
            cap = 2 * (2 * l + 1)
            g_occ = ground_occ.get((n, l), 0.0)
            if g_occ >= cap - 0.5:                      # filled in ground
                if occ == cap:
                    continue                            # closed core token
                if occ != cap - 1:
                    raise ValueError(
                        f"hole subshell {n}{_L_CHARS[l]}{occ}: only "
                        f"single-hole resonances are supported")
                if hole_nl is not None and hole_nl != (n, l):
                    raise ValueError(
                        f"all configs must share one hole subshell; got "
                        f"{hole_nl} and {(n, l)}")
                hole_nl = (n, l)
            elif g_occ == 0.0:
                if (n, l) not in excited_nls:
                    excited_nls.append((n, l))
            elif occ < cap:
                # partially occupied in the ground result but the CONFIG
                # token marks it excited: allowed — this is how a
                # resonance-average solve hands RELAXED virtual
                # orbitals to the manifold (their fractional occupation
                # only relaxed the orbitals; the active electrons are
                # handled entirely by the determinant CI below and are
                # excluded from N_bound / the core contraction)
                if (n, l) not in excited_nls:
                    excited_nls.append((n, l))
            else:
                raise ValueError(
                    f"subshell {n}{_L_CHARS[l]} is partially occupied in "
                    f"the ground result — not a closed-shell core")
    if hole_nl is None:
        raise ValueError("no hole subshell found in configs")
    excited_nls.sort()

    # ---- fine grid + orbital basis ---------------------------------------
    from atomx.numerics.quadrature import make_fine_grid

    n_h, l_h = hole_nl
    hole_kappas = _kappas_of_l(l_h)
    active_nks = ([(n_h, k) for k in hole_kappas]
                  + [(n_e, k) for (n_e, l_e) in excited_nls
                     for k in _kappas_of_l(l_e)])

    orbital_map: dict = {}
    orbital_eps: dict = {}
    rv = getattr(result, "_rv", None)

    if rv is not None:
        # ---- FAC-like basis: converged local-DFS potential ---------------
        from atomx.numerics.shooting.local import dfs_frozen_orbital

        grid = result._grid
        Z = int(result.Z)
        r_max = float(grid[-1])
        r_fine = make_fine_grid(r_max, p_rel(float(e_max_eV) / HARTREE_EV), z_inner=int(result.Z))
        dr_fine = RadialMetric.trapezoid(r_fine).weights
        # closed-core electron count: excited-nl subshells (empty
        # virtuals or fractional resonance-average occupations) do not
        # contribute — the Auger residual is the closed core alone
        _excited = set(excited_nls)
        N_bound = sum(occ for (_n, _k, occ) in meta
                      if (_n, _l_of(_k)) not in _excited)
        Z_res = max(Z - N_bound + 1.0, 1.0)

        def _tofine(P, Q):
            return (np.interp(r_fine, grid, P),
                    np.interp(r_fine, grid, Q))

        active_set = set(active_nks)
        core = {}
        native_orbitals = {}
        occupied_fine = {}
        for i, (n, kappa, occ) in enumerate(meta):
            PQ = _tofine(result._pa[i], result._qa[i])
            occupied_fine[(n, kappa)] = PQ
            native_orbitals[n, kappa] = (result._pa[i], result._qa[i])
            if (n, kappa) not in active_set and occ:
                core[n, kappa] = occ

        for (n, kappa) in active_nks:
            if (n, kappa) in occupied_fine:
                P, Q = occupied_fine[(n, kappa)]
                eps = float(result.orbital_energies[_orbital_label(n, kappa)])
            else:
                e_guess = -0.5 * ((Z_res + 1.0) / n) ** 2
                P_g = Q_g = eps = None
                for scale in (1.0, 1.6, 0.6):
                    try:
                        P_g, Q_g, eps = dfs_frozen_orbital(
                            result, n, kappa, e_guess * scale)
                        break
                    except RuntimeError:
                        continue
                if eps is None:
                    raise RuntimeError(
                        f"dfs_frozen_orbital failed for "
                        f"{_orbital_label(n, kappa)}")
                P, Q = _tofine(P_g, Q_g)
                native_orbitals[n, kappa] = (P_g, Q_g)
            orbital_map[(n, kappa)] = (P, Q)
            orbital_eps[(n, kappa)] = float(eps)

        from atomx.numerics.shooting.quadrature import slag6_weights
        from atomx.physics.csf.matrix import GridCIWorkspace, _radial_evaluator

        rv_nuc = getattr(result, "_rv_nuc", None)
        if rv_nuc is None:
            raise ValueError('bound CI requires the nuclear/environment field used by '
                             'the shooting solve; solve again to retain _rv_nuc')
        workspace = GridCIWorkspace(grid, result._difr, rv_nuc)
        bound_radial = _radial_evaluator(native_orbitals, workspace, Z, len(grid))
        property_workspace, offset = radial_workspace_of(result)
        bound_orbitals = {key: (p[offset:], q[offset:])
                          for key, (p, q) in native_orbitals.items()}
        if one_body is None:
            weights = slag6_weights(len(grid))*result._difr
            screen = (rv-rv_nuc)*workspace.inv_r
            bare = {}
            for a in active_nks:
                for b in active_nks:
                    if a[1] != b[1]:
                        continue
                    pa, qa = native_orbitals[a]
                    pb, qb = native_orbitals[b]
                    density = pa*pb+qa*qb
                    # Symmetric eigenstate identity in the common local field.
                    bare[a, b] = float(np.sum(weights*density*(
                        .5*(orbital_eps[a]+orbital_eps[b])-screen)))
            one_body = closed_core_one_body(active_nks, core, bound_radial,
                                             bare_one_body=bare)
        if V_func is None:
            def V_func(r, _grid=grid, _rv=rv):
                r = np.asarray(r, dtype=float)
                return np.interp(r, _grid, _rv) / np.maximum(r, 1e-30)
        target = _ClosedShellTarget(Z, meta)
        bsp = {"t": np.asarray(grid)}
    else:
        # ---- B-spline basis: solved orbitals + V^N virtuals ----------
        from atomx.methods.mbpt import build_virtual_basis
        from atomx.numerics.bspline.evaluate import eval_P, eval_Q

        bsp = getattr(result, "_bsp_data", None) or result.bsp_data
        target = result
        r_max = float(bsp["t"][-1])
        r_fine = make_fine_grid(r_max, p_rel(float(e_max_eV) / HARTREE_EV), z_inner=int(result.Z))
        dr_fine = RadialMetric.trapezoid(r_fine).weights
        gr = bsp["gr"].ravel()
        order = np.argsort(gr)
        grs = gr[order]

        def _tofine(P, Q):
            Pg = eval_P(P, bsp).ravel()[order]
            Qg = eval_Q(Q, bsp).ravel()[order]
            return np.interp(r_fine, grs, Pg), np.interp(r_fine, grs, Qg)

        orbital_coeffs = {}
        for kappa in hole_kappas:
            same = [sh for sh in result.subshells if sh.kappa == kappa]
            idx = next(i for i, sh in enumerate(same) if sh.n == n_h)
            P, Q = result.orbitals_raw[kappa][idx][:2]
            orbital_map[(n_h, kappa)] = _tofine(P, Q)
            orbital_coeffs[(n_h, kappa)] = (P, Q)
            orbital_eps[(n_h, kappa)] = float(
                result.orbital_energies[_orbital_label(n_h, kappa)])

        for (n_e, l_e) in excited_nls:
            for kappa in _kappas_of_l(l_e):
                present = next((sh for sh in result.subshells
                                if sh.kappa == kappa and sh.n == n_e), None)
                if present is not None:
                    # RELAXED VIRTUAL: the resonance-average solve
                    # supplies the excited orbital directly (fractional
                    # occupation) — use it instead of a frozen-core
                    # V^N virtual
                    same = [sh for sh in result.subshells
                            if sh.kappa == kappa]
                    idx = same.index(present)
                    P, Q = result.orbitals_raw[kappa][idx][:2]
                    orbital_map[(n_e, kappa)] = _tofine(P, Q)
                    orbital_coeffs[(n_e, kappa)] = (P, Q)
                    orbital_eps[(n_e, kappa)] = float(
                        result.orbital_energies[
                            _orbital_label(n_e, kappa)])
                    continue
                occ_ns = [sh.n for sh in result.subshells
                          if sh.kappa == kappa]
                n_first_virt = max([_l_of(kappa)] + occ_ns) + 1
                _occ, virt = build_virtual_basis(result, kappa)
                vi = n_e - n_first_virt
                if vi < 0 or vi >= len(virt):
                    raise ValueError(
                        f"virtual orbital {_orbital_label(n_e, kappa)} not "
                        f"in the V^N basis (index {vi} of {len(virt)})")
                P, Q, eps = virt[vi]
                orbital_map[(n_e, kappa)] = _tofine(P, Q)
                orbital_coeffs[(n_e, kappa)] = (P, Q)
                orbital_eps[(n_e, kappa)] = float(eps)

        bound_radial, bound_orbitals = bspline_bound_integrals(result, orbital_coeffs)
        property_workspace = bsp
        if one_body is None:
            core = {(n, k): occ for n, k, occ in meta
                    if (n, k) not in active_nks and occ}
            one_body = closed_core_one_body(active_nks, core, bound_radial)

    groups = [([(n_h, k) for k in hole_kappas],
               2 * (2 * l_h + 1) - 1)]
    for (n_e, l_e) in excited_nls:
        groups.append(([(n_e, k) for k in _kappas_of_l(l_e)],
                       parsed[0].get((n_e, l_e), 0)))

    config_tuples = []
    for p in parsed:
        tup = [2 * (2 * l_h + 1) - 1]
        tup += [p.get(nl, 0) for nl in excited_nls]
        config_tuples.append(tuple(tup))

    manifold = ResonanceManifold(
        target, groups, r_fine=r_fine, dr_fine=dr_fine,
        orbital_map=orbital_map, orbital_energies=orbital_eps,
        configs=config_tuples, V_func=V_func, one_body=one_body,
        bound_radial=bound_radial, bound_orbitals=bound_orbitals,
        bound_workspace=property_workspace)
    return manifold, bsp


def auger_levels(result, configs, *, two_J=None, e_max_eV: float = 1000.0,
                 shooting_n_grid=None, L_max: int = 6,
                 one_body=None, V_func=None):
    """J-resolved, CI-mixed autoionization levels of a resonance complex.

    The proper-physics replacement for the configuration-average
    ``auger_rate_csf_average`` proxy (which over-counts magnetic
    substates by 1.2-15x on the Fe XVII KLL panel): a determinant-CI
    diagonalization of the full complex (union of ``configs``) with
    exact Slater-Condon Auger amplitudes coherently summed over each
    CI eigenvector — see :class:`ResonanceManifold`.

    Parameters
    ----------
    result
        Solved closed-shell structure result of the recombined ion
        (``Atom`` instances are accepted and unwrapped).
    configs : sequence of str
        Resonance configurations, e.g.
        ``["2p5 3s2", "2p5 3s1 3p1", "2p5 3p2", ...]``.
    two_J : int or sequence of int, optional
        Restrict to these 2J values; default = every J the complex
        supports.

    Returns
    -------
    list of :class:`ManifoldLevel`
        Sorted by resonance energy; each carries the ab-initio
        ``E_res_au`` (above the closed-shell ion), ``gamma_auger_per_s``,
        ``a_radiative_per_s``, ``two_J`` and the dominant-determinant
        label.
    """
    if hasattr(result, "_result") and result._result is not None:
        result = result._result
    manifold, bsp = manifold_from_configs(
        result, configs, e_max_eV=e_max_eV,
        one_body=one_body, V_func=V_func)
    Z = int(getattr(result, "Z", 0))

    if two_J is None:
        n_act = sum(manifold.configs[0])
        tMs = []
        tM = 1 if n_act % 2 else 0
        while manifold.determinants(tM):
            tMs.append(tM)
            tM += 2
    else:
        tMs = [int(two_J)] if np.isscalar(two_J) else [int(t) for t in two_J]

    levels: list[ManifoldLevel] = []
    for tM in tMs:
        levels.extend(manifold.dr_levels(
            tM, Z=Z, bsp_data=bsp,
            shooting_n_grid=shooting_n_grid, L_max=L_max))
    levels.sort(key=lambda L: L.E_res_au)
    return levels


__all__ = ["ResonanceManifold", "ManifoldLevel", "RadiativeChannel", "auger_levels",
           "bspline_bound_integrals",
           "manifold_from_configs"]
