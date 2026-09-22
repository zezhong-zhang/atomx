"""Exact J-resolved radiative line strengths via determinant CI.

The configuration-average transition path (``one_body_csf_coupling`` +
FAC's UTA population factor) gets multiplet *totals* right but cannot
split them into individual lines when a multiplet has several levels of
the same J (e.g. the three J=1 levels of Ne-like 2p5 3d — the famous
Fe XVII 3C/3D pair).  This module computes the exact many-electron
reduced matrix elements with the same FAC-verified m-scheme machinery
used for Auger/DR widths (:mod:`atomx.scattering.autoionization.mscheme_ci`):

1. Enumerate bitmask determinants for BOTH configurations over one
   shared orbital basis (bound orbitals from the solved closed-shell
   result + virtuals in the same frozen potential, via
   :func:`~atomx.scattering.autoionization.mscheme_ci.manifold_from_configs`).
2. Diagonalize each (J, parity) block of the active-space Hamiltonian
   per configuration (H is rotationally invariant on the determinant
   space, so eigenvectors at fixed M carry sharp J, identified via J^2;
   degenerate clusters are J-purified by re-diagonalizing J^2 inside
   the cluster).
3. Between an upper eigenstate fixed at M_i = J_i and the lower
   eigenstates at every reachable M_f, accumulate the one-body
   transition amplitude by Slater-Condon one-orbital-difference rules,
   coherently over determinants and CI weights.  For each (M_i, M_f)
   only one spherical component q = M_f - M_i contributes, so by
   Wigner-Eckart / 3j orthogonality

       S = |<f || d^L || i>|^2 = (2J_i + 1) * sum_{M_f} |amp(M_f)|^2

   — the line strength directly, no phase matching across M blocks
   needed (each squared amplitude is invariant under the arbitrary
   sign of its eigenvector).

One-body operator (electric 2^L-pole, length gauge):

    <a| r^L C^L_q |b> = _ck_m(kappa_a, m_a, kappa_b, m_b, L, q)
                        * Int (P_a P_b + Q_a Q_b) r^L dr

on the same fine grid as the Slater integrals.  Magnetic multipoles are
NOT implemented here (the M-pole one-body operator has a different
radial/angular structure); callers get ``NotImplementedError``.

Exact sum rule (mixing-invariant, tested to 1e-10): the sum of S over
all line pairs equals the determinant trace
``sum_{det_i, det_f, q} |<det_f| d_q |det_i>|^2`` — the exact
configuration-pair total that the UTA path distributes statistically.

Validation (2026-07-17, live clean-room FAC 1.1.5, SetSE/VP/MS/Breit
off, OptimizeRadial on the ground config — the exact counterpart of the
``engine='shooting', exchange='gu'`` pipeline): Fe XVII 2p5 3s -> 2p6
and 2p5 3d -> 2p6 per-line A ratios 0.99-1.03, in-multiplet relative
line strengths within 3.2% (3C/3D ratio 0.962 of FAC), transition
energies +2.4 to +2.9 eV of ~730-830 eV.  Reference pinned in
``tests/data/fe17_tr_fac_cleanroom.json`` / ``tests/test_ci_line_strengths.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atomx.constants import HARTREE_INV_ANGSTROM
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength
from atomx.scattering.autoionization.mscheme_ci import (
    _parse_config_tokens,
    _subshell_meta,
    manifold_from_configs,
)
from atomx.scattering.distorted_wave.solver import _ck_m, _l_of, _tj_of

_L_CHARS = "spdfghiklmno"


class CISizeError(ValueError):
    """Determinant block exceeds the configured size cap."""


@dataclass(frozen=True)
class CILevel:
    """One J-resolved CI eigenlevel of a configuration."""

    config: str
    index: int            # id within this configuration's level list
    two_J: int
    parity: int           # +1 / -1 relative to the closed core
    energy_au: float      # absolute, in the manifold's frozen-orbital accounting
    label: str            # dominant jj determinant, e.g. "2p3/2^3 3d5/2 [J=1,-]"


@dataclass(frozen=True)
class CILine:
    """One radiative line between two CI eigenlevels."""

    upper: CILevel
    lower: CILevel
    multipole: str
    omega_au: float       # E_upper - E_lower  (> 0)
    S: float              # line strength |<f||T^L||i>|^2 (a.u.)
    A_per_s: float        # Einstein A, length gauge
    gf: float             # weighted absorption oscillator strength
    f_value: float        # absorption oscillator strength (from the lower level)

    @property
    def wavelength_angstrom(self):
        if self.omega_au <= 0.0:
            return np.inf
        return 1.0 / (HARTREE_INV_ANGSTROM * self.omega_au)


@dataclass
class CILineSet:
    """All lines of one configuration pair plus the exact sum-rule data."""

    config_i: str
    config_f: str
    multipole: str
    lines: list[CILine]
    levels_i: list[CILevel]
    levels_f: list[CILevel]
    S_sum: float          # sum of S over every allowed level pair
    trace: float          # determinant trace = exact configuration-pair total
    manifold: object = field(default=None, repr=False)  # diagnostics access


# ---------------------------------------------------------------------------
# configuration bookkeeping
# ---------------------------------------------------------------------------

def _shell_cap(l: int) -> int:
    return 2 * (2 * l + 1)


def ground_config_of_pair(config_a: str, config_b: str) -> str:
    """Closed-shell ground configuration string underlying a config pair.

    Supported pairs are single-hole excitations of a closed-shell ion
    (possibly with the closed ground itself as one member).  The two
    configs are classified *jointly* — a lone ``3s1`` token is a hole
    or an excited electron depending on the partner config — into:

    * core subshells: filled (occ = cap) in both configs;
    * THE hole subshell: occ = cap - 1 in at least one config and
      cap or cap - 1 in the other;
    * excited subshells: everything else (must total <= 1 electron per
      config, and both configs must carry the same electron count).

    Returns the core + filled-hole string, i.e. what to solve to feed
    :func:`ci_line_strengths`.
    """
    pa = _parse_config_tokens(config_a)
    pb = _parse_config_tokens(config_b)
    if sum(pa.values()) != sum(pb.values()):
        raise ValueError(
            f"configs {config_a!r} / {config_b!r} carry different electron "
            f"counts — write both with the same core tokens")
    closed = []
    candidates = []
    others = []
    for (n, l) in sorted(set(pa) | set(pb)):
        cap = _shell_cap(l)
        oa = pa.get((n, l), 0)
        ob = pb.get((n, l), 0)
        if oa == cap and ob == cap:
            closed.append((n, l, cap))
        elif oa in (cap, cap - 1) and ob in (cap, cap - 1) \
                and min(oa, ob) == cap - 1 and (n, l) in pa and (n, l) in pb:
            candidates.append((n, l))
        else:
            others.append((n, l))
    if not candidates:
        raise ValueError(
            f"configs {config_a!r} / {config_b!r} share no single-hole "
            f"subshell")
    # Ambiguity (e.g. a lone 3s1 in both configs looks like a hole of a
    # 3s2 ground): the physical hole is the innermost candidate; the
    # rest are excited electrons.
    hole = min(candidates)
    closed.append((hole[0], hole[1], _shell_cap(hole[1])))
    exc_a = exc_b = 0
    for (n, l) in others + [c for c in candidates if c != hole]:
        exc_a += pa.get((n, l), 0)
        exc_b += pb.get((n, l), 0)
    if exc_a > 1 or exc_b > 1:
        raise ValueError(
            f"configs {config_a!r} / {config_b!r}: more than one excited "
            f"electron — outside the single-hole CI model")
    closed.sort()
    return " ".join(f"{n}{_L_CHARS[l]}{occ}" for (n, l, occ) in closed)


def _config_parity(tup, groups) -> int:
    """Parity (-1)^(sum l * occ) over the active groups."""
    s = 0
    for (subshells, _ne), occ in zip(groups, tup, strict=False):
        l = _l_of(subshells[0][1])
        s += l * int(occ)
    return -1 if s % 2 else 1


class _PairCI:
    """Shared determinant-CI workspace for one configuration pair."""

    def __init__(self, result, config_i, config_f, *, e_max_eV=1000.0,
                 det_cap=2000):
        if hasattr(result, "_result") and result._result is not None:
            result = result._result
        self.result = result
        self.config_i = config_i
        self.config_f = config_f
        self.det_cap = int(det_cap)

        # ---- classify the pair against the ground occupations ------------
        meta = _subshell_meta(result)
        ground_occ: dict = {}
        for (n, kappa, occ) in meta:
            key = (n, _l_of(kappa))
            ground_occ[key] = ground_occ.get(key, 0.0) + occ

        parsed = {cfg: _parse_config_tokens(cfg) for cfg in (config_i, config_f)}

        def _is_open(cfg):
            for (n, l), occ in parsed[cfg].items():
                cap = _shell_cap(l)
                g = ground_occ.get((n, l), 0.0)
                if g >= cap - 0.5 and occ == cap - 1:
                    return True
            return False

        open_cfgs = [c for c in (config_i, config_f) if _is_open(c)]
        if not open_cfgs:
            raise ValueError(
                f"neither {config_i!r} nor {config_f!r} carries a hole "
                f"relative to the solved ground — nothing to promote")
        # de-duplicate while preserving order (config_i may equal config_f)
        seen = set()
        open_cfgs = [c for c in open_cfgs if not (c in seen or seen.add(c))]

        self.manifold, self.bsp = manifold_from_configs(
            result, open_cfgs, e_max_eV=e_max_eV)
        man = self.manifold

        # group layout: groups[0] is the hole group, then excited nl groups
        hole_sub = man.groups[0][0][0]
        self.hole_nl = (hole_sub[0], _l_of(hole_sub[1]))
        self.cap_h = _shell_cap(self.hole_nl[1])
        self.excited_nls = [(g[0][0][0], _l_of(g[0][0][1]))
                            for g in man.groups[1:]]

        def _tuple_of(cfg):
            p = parsed[cfg]
            occ_h = p.get(self.hole_nl)
            if occ_h is None:
                raise ValueError(
                    f"{cfg!r} must state the {self.hole_nl} subshell occupation")
            if occ_h not in (self.cap_h, self.cap_h - 1):
                raise ValueError(
                    f"{cfg!r}: hole subshell occupation {occ_h} is neither "
                    f"closed nor single-hole")
            tup = [int(occ_h)]
            for nl in self.excited_nls:
                tup.append(int(p.get(nl, 0)))
            # every remaining token must be closed in the ground
            for (n, l), occ in p.items():
                if (n, l) == self.hole_nl or (n, l) in self.excited_nls:
                    continue
                if ground_occ.get((n, l), 0.0) < _shell_cap(l) - 0.5 or \
                        occ != _shell_cap(l):
                    raise ValueError(
                        f"{cfg!r}: token {n}{_L_CHARS[l]}{occ} is neither a "
                        f"closed-core shell nor part of the active space")
            return tuple(tup)

        self.tup_i = _tuple_of(config_i)
        self.tup_f = _tuple_of(config_f)
        if sum(self.tup_i) != sum(self.tup_f):
            raise ValueError(
                f"configs {config_i!r} / {config_f!r} carry different "
                f"electron counts ({sum(self.tup_i)} vs {sum(self.tup_f)})")
        self.n_active = sum(self.tup_i)
        self.parity_i = _config_parity(self.tup_i, man.groups)
        self.parity_f = _config_parity(self.tup_f, man.groups)
        self._eig_cache: dict = {}
        self._elem_cache: dict = {}

    # ---- determinant enumeration ----------------------------------------
    def _dets(self, tup, tM):
        man = self.manifold
        old = man.configs
        man.configs = [tuple(tup)]
        try:
            return man.determinants(tM)
        finally:
            man.configs = old

    def _all_dets(self, tup):
        """Every determinant of a configuration (all M)."""
        from itertools import combinations

        man = self.manifold
        per_group = []
        for mask, ne in zip(man.group_bits, tup, strict=False):
            bits = [i for i in range(man.n_orb) if mask & (1 << i)]
            per_group.append(list(combinations(bits, int(ne))))
        dets = []

        def _rec(gi, acc):
            if gi == len(per_group):
                m = 0
                for i in acc:
                    m |= 1 << i
                dets.append(m)
                return
            for choice in per_group[gi]:
                _rec(gi + 1, acc + list(choice))

        _rec(0, [])
        return dets

    # ---- (J-purified) eigenblocks ----------------------------------------
    def eig_block(self, tup, tM):
        """(dets, evals, evecs, tJs) for one M block; J-purified.

        Eigenvectors inside a degenerate eigenvalue cluster are rotated
        to J^2 eigenstates (H commutes with J^2, so clusters that mix
        different J are resolvable exactly)."""
        key = (tuple(tup), int(tM))
        hit = self._eig_cache.get(key)
        if hit is not None:
            return hit
        man = self.manifold
        dets = self._dets(tup, tM)
        if not dets:
            self._eig_cache[key] = None
            return None
        if len(dets) > self.det_cap:
            raise CISizeError(
                f"{len(dets)} determinants at 2M={tM} exceeds the CI cap "
                f"({self.det_cap})")
        H = man.hamiltonian(dets)
        J2 = man.j2_matrix(dets)
        evals, evecs = np.linalg.eigh(H)
        n = len(evals)
        tJs = np.empty(n, dtype=int)
        i = 0
        while i < n:
            j = i + 1
            tol = 1e-9 * max(1.0, abs(evals[i]))
            while j < n and evals[j] - evals[i] < tol:
                j += 1
            if j - i > 1:
                sub = evecs[:, i:j]
                _w2, v2 = np.linalg.eigh(sub.T @ J2 @ sub)
                evecs[:, i:j] = sub @ v2
            for k in range(i, j):
                w = evecs[:, k]
                j2e = float(w @ J2 @ w)
                tj = int(round(-1.0 + math.sqrt(1.0 + 4.0 * j2e)))
                if abs(j2e - 0.25 * tj * (tj + 2)) > 0.05:
                    raise RuntimeError(
                        f"eigenvector at 2M={tM} is not a J eigenstate "
                        f"(<J^2>={j2e:.6f})")
                tJs[k] = tj
            i = j
        out = (dets, evals, evecs, tJs)
        self._eig_cache[key] = out
        return out

    # ---- one-body operator table -----------------------------------------
    def elem_table(self, L: int):
        """{(a, b): <a| r^L C^L_q |b>} over active orbital indices.

        ``a`` is the created (bra-side) orbital, ``b`` the annihilated
        (ket-side) one; q is fixed by m_a - m_b."""
        tab = self._elem_cache.get(L)
        if tab is not None:
            return tab
        man = self.manifold
        orbs = man.orbitals
        tab = {}
        for a, oa in enumerate(orbs):
            for b, ob in enumerate(orbs):
                tq = oa.tm - ob.tm
                if abs(tq) > 2 * L:
                    continue
                ang = _ck_m(oa.kappa, oa.tm, ob.kappa, ob.tm, L, tq)
                if ang == 0.0:
                    continue
                tab[(a, b)] = ang * self._radial_L(oa, ob, L)
        self._elem_cache[L] = tab
        return tab

    def _radial_L(self, oa, ob, L):
        """Length-gauge radial integral Int (P_a P_b + Q_a Q_b) r^L dr."""
        man = self.manifold
        if L == 1:
            return man._rdip(oa, ob)
        key = ("dipL", L, (oa.n, oa.kappa), (ob.n, ob.kappa))
        val = man._rk_cache.get(key)
        if val is None:
            rho = oa.P * ob.P + oa.Q * ob.Q
            val = float(np.sum(rho * man.r_fine ** L * man.dr_fine))
            man._rk_cache[key] = val
        return val

    # ---- coherent amplitude between two CI vectors -------------------------
    def amplitude(self, dets_bra, w_bra, dets_ket, w_ket, elem):
        """<bra| sum_ab elem[a,b] a+_a a_b |ket> by Slater-Condon rules."""
        man = self.manifold
        occ_list = man._occ_list
        ph_rm = man._phase_remove
        idx_bra = {d: i for i, d in enumerate(dets_bra)}
        partners: dict = {}
        for (a, b) in elem:
            partners.setdefault(b, []).append(a)
        tot = 0.0
        for ik, dk in enumerate(dets_ket):
            wk = w_ket[ik]
            if abs(wk) < 1e-14:
                continue
            for b in occ_list(dk):
                for a in partners.get(b, ()):
                    if a == b:
                        ib = idx_bra.get(dk)
                        if ib is not None:
                            tot += wk * w_bra[ib] * elem[(a, b)]
                        continue
                    if (dk >> a) & 1:
                        continue
                    db = (dk & ~(1 << b)) | (1 << a)
                    ib = idx_bra.get(db)
                    if ib is None:
                        continue
                    wb = w_bra[ib]
                    if abs(wb) < 1e-14:
                        continue
                    tot += (wk * wb * ph_rm(dk, b) * ph_rm(db, a)
                            * elem[(a, b)])
        return tot

    # ---- exact configuration-pair total (mixing invariant) ----------------
    def det_trace(self, L: int) -> float:
        """sum_{det_i, det_f, q} |<det_f| T^L_q |det_i>|^2 over all M.

        Equals the sum of line strengths over every level pair (tested
        to 1e-10) — the exact total the UTA path distributes."""
        man = self.manifold
        occ_list = man._occ_list
        ph_rm = man._phase_remove
        elem = self.elem_table(L)
        partners: dict = {}
        for (a, b) in elem:
            partners.setdefault(b, []).append(a)
        dets_f = set(self._all_dets(self.tup_f))
        trace = 0.0
        for dk in self._all_dets(self.tup_i):
            acc: dict = {}
            for b in occ_list(dk):
                for a in partners.get(b, ()):
                    if a == b:
                        if dk in dets_f:
                            acc[dk] = acc.get(dk, 0.0) + elem[(a, b)]
                        continue
                    if (dk >> a) & 1:
                        continue
                    db = (dk & ~(1 << b)) | (1 << a)
                    if db not in dets_f:
                        continue
                    acc[db] = (acc.get(db, 0.0)
                               + ph_rm(dk, b) * ph_rm(db, a) * elem[(a, b)])
            trace += sum(v * v for v in acc.values())
        return trace

    # ---- level enumeration -------------------------------------------------
    def canonical_levels(self, tup, cfg_string, parity):
        """One CILevel per physical level (eigenvector taken at M = J)."""
        levels = []
        vectors = []
        tM = self.n_active % 2
        while True:
            blk = self.eig_block(tup, tM)
            if blk is None:
                break
            dets, evals, evecs, tJs = blk
            for li in range(len(evals)):
                if tJs[li] != tM:
                    continue
                w = evecs[:, li]
                imax = int(np.argmax(np.abs(w)))
                levels.append(CILevel(
                    config=cfg_string, index=len(levels), two_J=int(tM),
                    parity=int(parity), energy_au=float(evals[li]),
                    label=self._level_label(dets[imax], int(tM), parity)))
                vectors.append((int(tM), w, dets))
            tM += 2
        return levels, vectors

    def _level_label(self, det, tJ, parity):
        man = self.manifold
        counts: dict = {}
        for i in man._occ_list(det):
            o = man.orbitals[i]
            key = (o.n, o.kappa)
            counts[key] = counts.get(key, 0) + 1
        parts = []
        for (n, kappa), c in sorted(counts.items()):
            lab = f"{n}{_L_CHARS[_l_of(kappa)]}{_tj_of(kappa)}/2"
            parts.append(lab if c == 1 else f"{lab}^{c}")
        sign = "+" if parity > 0 else "-"
        return " ".join(parts) + f" [J={tJ}/2,{sign}]"


def ci_line_strengths(result, config_i, config_f, *, multipole="E1",
                      e_max_eV: float = 1000.0, det_cap: int = 2000):
    """Exact J-resolved line strengths between two configurations.

    Parameters
    ----------
    result
        Solved closed-shell ground result of the ion (``Atom`` accepted
        and unwrapped).  The ``engine='shooting', exchange='gu'``
        pipeline gives the FAC-equivalent orbital basis (single
        optimized DFS potential, frozen virtuals) — the configuration
        validated against clean-room FAC; B-spline results fall back to
        the V^{N-1} virtual basis.
    config_i, config_f : str
        The two configurations, e.g. ``"2p5 3d1"`` and ``"2p6"``
        (closed-core tokens optional).  Both must be single-hole
        excitations of the solved ground (or the ground itself).
    multipole : str
        ``"E1"`` (default) or any electric ``"EL"``.  Magnetic
        multipoles raise ``NotImplementedError``.
    det_cap : int
        Maximum determinants per (M) block; :class:`CISizeError` above.

    Returns
    -------
    CILineSet
        Lines for every allowed level pair with ``E_i > E_f`` treated as
        emission (upper = higher level regardless of argument order),
        plus the exact determinant-trace sum rule.
    """
    mp = str(multipole).upper()
    if not mp or mp[0] not in ("E", "M"):
        raise ValueError(f"multipole must look like 'E1'/'E2'/... (got {multipole!r})")
    if mp[0] == "M":
        raise NotImplementedError(
            "magnetic multipoles are not implemented in the CI line engine "
            "yet (the M-pole one-body operator has a different "
            "radial/angular structure); use the UTA path")
    L = int(mp[1:])

    pair = _PairCI(result, config_i, config_f,
                   e_max_eV=e_max_eV, det_cap=det_cap)

    lines: list[CILine] = []
    op_parity = 1 if L % 2 == 0 else -1

    levels_i, vecs_i = pair.canonical_levels(pair.tup_i, config_i, pair.parity_i)
    same_config = pair.tup_i == pair.tup_f
    if same_config:
        levels_f, _vecs_f = levels_i, vecs_i
    else:
        levels_f, _vecs_f = pair.canonical_levels(pair.tup_f, config_f,
                                                  pair.parity_f)

    S_sum = 0.0
    if pair.parity_i * pair.parity_f == op_parity:
        elem = pair.elem_table(L)
        # canonical (energy, index) lookup for the f side, per 2J
        by_tj: dict = {}
        for lv in levels_f:
            by_tj.setdefault(lv.two_J, []).append(lv)

        for lv_i, (tM_i, w_i, dets_i) in zip(levels_i, vecs_i, strict=False):
            tJ_i = lv_i.two_J
            # accumulated |amp|^2 per f-level index
            acc: dict = {}
            for tM_f in range(tM_i - 2 * L, tM_i + 2 * L + 2, 2):
                blk = pair.eig_block(pair.tup_f, tM_f)
                if blk is None:
                    continue
                dets_f, evals_f, evecs_f, tJs_f = blk
                for lf in range(len(evals_f)):
                    tJ_f = int(tJs_f[lf])
                    if abs(tJ_f - tJ_i) > 2 * L or tJ_f + tJ_i < 2 * L:
                        continue
                    cands = by_tj.get(tJ_f)
                    if not cands:
                        continue
                    E_f = float(evals_f[lf])
                    lv_f = min(cands, key=lambda c: abs(c.energy_au - E_f))
                    if abs(lv_f.energy_au - E_f) > 1e-6 * (1.0 + abs(E_f)):
                        raise RuntimeError(
                            f"cannot identify level at E={E_f} (2J={tJ_f}, "
                            f"2M={tM_f}) with a canonical level — "
                            f"rotational-invariance violation?")
                    a = pair.amplitude(dets_f, evecs_f[:, lf],
                                       dets_i, w_i, elem)
                    if a == 0.0:
                        continue
                    acc[lv_f.index] = acc.get(lv_f.index, 0.0) + a * a
            for idx_f, s2 in acc.items():
                lv_f = levels_f[idx_f]
                S = (tJ_i + 1) * s2
                S_sum += S
                if same_config and idx_f >= lv_i.index:
                    # ordered pairs both enter S_sum (matching the ordered
                    # determinant trace) but each physical line only once
                    continue
                # emission direction: upper = higher energy
                if lv_i.energy_au >= lv_f.energy_au:
                    up, lo = lv_i, lv_f
                else:
                    up, lo = lv_f, lv_i
                omega = up.energy_au - lo.energy_au
                if omega <= 0.0:
                    continue
                A = einstein_A_eL(omega, S, L, up.two_J)
                f_val, gf = oscillator_strength(omega, S, L, lo.two_J, up.two_J)
                lines.append(CILine(
                    upper=up, lower=lo, multipole=mp, omega_au=float(omega),
                    S=float(S), A_per_s=float(A), gf=float(gf),
                    f_value=float(f_val)))

    trace = pair.det_trace(L) if pair.parity_i * pair.parity_f == op_parity \
        else 0.0
    lines.sort(key=lambda ln: ln.omega_au)
    return CILineSet(
        config_i=config_i, config_f=config_f, multipole=mp, lines=lines,
        levels_i=levels_i, levels_f=levels_f, S_sum=float(S_sum),
        trace=float(trace), manifold=pair.manifold)


__all__ = ["CILine", "CILineSet", "CILevel", "CISizeError",
           "ci_line_strengths", "ground_config_of_pair"]
