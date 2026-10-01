"""Sigma(2) two-body screening: RPA bubble in J-coupled pair space.

First CI+MBPT increment per ``docs/CI_MBPT_SIGMA2_SPEC.md``: the
core-polarisation screening of the valence-valence Coulomb interaction,

    delta G^{(J)}[(ab),(cd)] = - S^{(J)} . D^{-1} . S^{(J)T}
    S^{(J)}[(ab),(mr)] = <ab; J|v|mr; J>_A          (valence pair x core-virtual)
    D[(mr)] = eps_r - eps_m                          (floored away from zero)

evaluated per total-pair-J channel as one dense product over the
(m, r) intermediate pairs (the high-performance architecture: the only
radial work is the S-block build; the channel products are dgemms).
``V_J`` values come from :func:`atomx.methods.mbpt._pair_v_J` — the same
kernel the tested closed-shell E^(2) path uses.

Scope of this first increment (documented honestly):

* SAME-J-channel screening only — both bubble vertices carry the pair's
  total J.  The full Dzuba Sigma(2) additionally mixes channels at the
  two vertices and adds hole-line and particle-particle families; those
  are follow-ups gated on the Eu3+ anchor below.
* The screening is delivered as a RAW (un-antisymmetrised) additive
  correction delta_v(a, b, c, d) to the m-scheme two-body element,
  obtained from the channel tables by Clebsch-Gordan recoupling; the
  sparse assembler's existing antisymmetrisation (v - v2) then applies
  unchanged.

Validation gates (spec): sign — the direct screening must be attractive
(delta G_dd < 0); magnitude anchor — on Eu3+ 4f6 the 5D0-7F0 gap must
move DOWN toward NIST 17270 from the unscreened 20465; convergence —
virtual window sweeps must be stable.
"""
from __future__ import annotations

from functools import lru_cache
from math import sqrt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from atomx.angular.wigner import wigner_3j
from atomx.methods.mbpt import _pair_v_J, _twoj_from_kappa  # same kernel as E^(2)
from atomx.numerics.bspline.convol import ConvolCache

__all__ = ["Sigma2Bubble", "build_sigma2_bubble"]


def _twoj(kappa: int) -> int:
    return _twoj_from_kappa(kappa)


def _cg(j2_a, m2_a, j2_b, m2_b, j2_tot, m2_tot) -> float:
    """Clebsch-Gordan <j_a m_a, j_b m_b | J M> from the Wigner 3j.

    All quantum numbers in doubled units (2j, 2m, 2J, 2M).
    """
    # CG = (-1)^{j_a - j_b + M} sqrt(2J+1) * 3j(j_a j_b J; m_a m_b -M)
    phase = (-1.0) ** ((j2_a - j2_b + m2_tot) // 2)
    w = wigner_3j(j2_a, j2_b, j2_tot, m2_a, m2_b, -m2_tot)
    if w == 0.0:
        return 0.0
    return phase * sqrt(j2_tot + 1) * w


class Sigma2Bubble:
    """Screened-interaction corrections for one core/virtual reference.

    Holds the per-J channel products delta G^{(J)} over the valence pair
    list and answers raw m-scheme quadruple corrections by CG recoupling.
    """

    def __init__(self, delta_g: Dict[int, np.ndarray],
                 pair_index: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
                 pair_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                 n_clipped: int):
        # delta_g[2J] : (n_pairs, n_pairs) screening matrix in pair space
        self._delta_g = delta_g
        self._pair_index = pair_index
        self._pair_list = pair_list
        self.n_clipped = n_clipped
        self.n_calls = 0

    @property
    def channels(self):
        return sorted(self._delta_g)

    def summary(self) -> str:
        nz = {j2: float(np.min(M[M != 0])) if np.any(M) else 0.0
              for j2, M in self._delta_g.items()}
        return (f"Sigma2Bubble: channels={self.channels}, "
                f"pairs={len(self._pair_list)}, clipped_denoms={self.n_clipped}, "
                f"min_diag_screening={nz}")

    def delta_raw(self, ka, m2a, kb, m2b, kc, m2c, kd, m2d,
                  a: Tuple[int, int], b: Tuple[int, int],
                  c: Tuple[int, int], d: Tuple[int, int]) -> float:
        """Antisymmetrised m-scheme correction <ab|dv_A|cd>.

        ``a..d`` are (n, kappa) orbital keys; ``m2*`` doubled m values;
        ``ka..kd`` kappas.  The channel tables are already
        pair-antisymmetrised (``_pair_v_J`` returns <ab;J|v|cd;J>_A), so
        the CG recoupling lands directly on the antisymmetrised
        m-scheme element — the caller must ADD it to the assembled
        (v - v2) combination, never to the raw pieces.
        """
        key_ab = (a, b) if (a, b) in self._pair_index else None
        key_cd = (c, d) if (c, d) in self._pair_index else None
        if key_ab is None or key_cd is None:
            return 0.0
        i_ab = self._pair_index[key_ab]
        i_cd = self._pair_index[key_cd]
        j2a, j2b, j2c, j2d = _twoj(ka), _twoj(kb), _twoj(kc), _twoj(kd)
        j2_min = max(abs(j2a - j2b), abs(j2c - j2d))
        j2_max = min(j2a + j2b, j2c + j2d)
        m2 = m2a + m2b  # = m2c + m2d for m-conserving elements
        total = 0.0
        for j2 in range(j2_min, j2_max + 1, 2):
            M = self._delta_g.get(j2)
            if M is None:
                continue
            # sum over the single M fixed by the m values
            cg_bra = _cg(j2a, m2a, j2b, m2b, j2, m2)
            cg_ket = _cg(j2c, m2c, j2d, m2d, j2, m2)
            if cg_bra == 0.0 or cg_ket == 0.0:
                continue
            total += cg_bra * cg_ket * M[i_ab, i_cd]
        self.n_calls += 1
        return total


def build_sigma2_bubble(orbitals_dict, bsp_data, Z, ns,
                        core_keys: Sequence[Tuple[int, int]],
                        virt_keys: Sequence[Tuple[int, int]],
                        valence_pairs: Sequence[Tuple[Tuple[int, int],
                                                       Tuple[int, int]]],
                        eps: Dict[Tuple[int, int], float],
                        denom_floor: float = 0.05,
                        j2_max: int | None = None,
                        kappa_max_virt: int = 4):
    """Build the per-J screening matrices over ``valence_pairs``.

    Parameters
    ----------
    orbitals_dict, bsp_data, Z, ns
        The CI run's radial workspace (the same one feeding the sparse
        assembler) — the core is the closed [Xe] carried in every
        determinant, the virtuals supply the intermediate states.
    core_keys
        (n, kappa) of the closed core subshells.
    virt_keys
        (n, kappa) of the intermediate-state virtuals.
    valence_pairs
        Ordered ((n,kappa), (n,kappa)) valence pairs spanning the CI
        open shells (both orderings the CI will ask for must be present).
    eps
        Orbital energies (Hartree) for the denominator, keyed like the
        orbitals.
    denom_floor
        Minimum |eps_m - eps_r| kept (Hartree); smaller denominators are
        clipped and COUNTED — quasi-degenerate intermediates must not
        blow up the screening silently.
    """
    cache = ConvolCache(bsp_data)
    valence_pairs = list(valence_pairs)
    pair_index = {p: i for i, p in enumerate(valence_pairs)}
    channels: Dict[int, np.ndarray] = {}
    n_clipped = 0

    core = [k for k in core_keys if k in orbitals_dict]
    virt = [k for k in virt_keys if k in orbitals_dict]

    for j2 in range(0, (j2_max or 16) + 1, 2):
        rows = []
        diag = []
        # S blocks: valence-pair x (core, virt) channels
        any_pair = False
        S = np.zeros((len(valence_pairs),
                      sum(1 for m in core for _ in virt)))
        col = 0
        denom = np.zeros(S.shape[1])
        for m in core:
            Pm, Qm = orbitals_dict[m]
            for r in virt:
                Pr, Qr = orbitals_dict[r]
                d = eps[r] - eps[m]  # excitation cost; > 0 for bound core
                if abs(d) < denom_floor:
                    d = denom_floor * (1.0 if d >= 0 else -1.0)
                    n_clipped += 1
                denom[col] = d
                for i, (a, b) in enumerate(valence_pairs):
                    Pa, Qa = orbitals_dict[a]
                    Pb, Qb = orbitals_dict[b]
                    v = _pair_v_J(Pa, Qa, a[1], Pb, Qb, b[1],
                                  Pm, Qm, m[1], Pr, Qr, r[1],
                                  j2, bsp_data, cache)
                    # normalisation of coupled product pairs
                    v /= sqrt((1 + (a == b)) * (1 + (m == r)))
                    if v != 0.0:
                        any_pair = True
                    S[i, col] = v
                col += 1
        if not any_pair:
            continue
        # delta G^{(J)} = - S D^{-1} S^T
        Dinv = 1.0 / denom
        G = -(S * Dinv) @ S.T
        channels[j2] = 0.5 * (G + G.T)
    return Sigma2Bubble(channels, pair_index, valence_pairs, n_clipped)


def build_sigma2_mscheme(orbitals_dict, bsp_data, Z, ns,
                         core_keys, virt_keys, valence_keys, eps,
                         denom_floor: float = 0.05):
    """Complete second-order screening in the m-scheme (all J channels).

    The bubble in m-scheme needs no J-channel reconstruction — the
    angular-momentum sums live inside the m-scheme Coulomb elements
    themselves, so this construction includes the cross-channel (multi-J)
    physics the same-J pair-space version structurally could not see:

        d(a,b,c,d) = -sum_{m in core, r in virt} [
              g(a, m, c, r) g(b, r, d, m)
            + g(b, m, d, r) g(a, r, c, m) ] / (eps_r - eps_m)

    with raw (un-antisymmetrised) m-scheme Coulomb elements g.  Returns a
    callable keyed by global spin-orbital indices over an EXTENDED table
    (valence + core + virtual spinors); the caller adapts it to the
    assembler's pattern-restricted spinor indices through the orbital
    (n, kappa, 2m) labels.
    """
    from atomx.angular.csf_hamiltonian import _m2_values, two_body_angular, twoj
    import atomx.physics.csf.matrix as MAT
    from atomx.numerics.bspline.convol import ConvolCache

    ext = sorted(set(valence_keys) | set(core_keys) | set(virt_keys))
    spinors = [(n, k, m2) for (n, k) in ext
               for m2 in _m2_values(twoj(k))]
    index = {s: i for i, s in enumerate(spinors)}
    core_idx = [index[(n, k, m2)] for (n, k) in core_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    virt_idx = [index[(n, k, m2)] for (n, k) in virt_keys
                if (n, k) in orbitals_dict
                for m2 in _m2_values(twoj(k))]
    denom = []
    mr_pairs = []
    for mi in core_idx:
        for ri in virt_idx:
            d = eps[spinors[ri][:2]] - eps[spinors[mi][:2]]
            if abs(d) < denom_floor:
                continue  # clipped; counted by omission from the window
            denom.append(d)
            mr_pairs.append((mi, ri))
    cache = ConvolCache(bsp_data)

    @lru_cache(maxsize=None)
    def g(ia, ic, im, ir_):
        a, c, m, r_ = spinors[ia], spinors[ic], spinors[im], spinors[ir_]
        terms = two_body_angular(a[1], a[2], m[1], m[2],
                                 c[1], c[2], r_[1], r_[2])
        if not terms:
            return 0.0
        Pa, Qa = orbitals_dict[(a[0], a[1])]
        Pc, Qc = orbitals_dict[(c[0], c[1])]
        Pm, Qm = orbitals_dict[(m[0], m[1])]
        Pr, Qr = orbitals_dict[(r_[0], r_[1])]
        total = 0.0
        for k, coef in terms:
            total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pm, Qm, Pr, Qr,
                                    bsp_data, k)
        return total

    def delta(ia, ib, ic, id_):
        """Antisymmetrised m-scheme screening <ab|dV_A|cd>, spinor indices.

        direct bubble:  vertex 1 on the (a, c) lines, vertex 2 on (b, d);
        exchange bubble: vertex 1 on the (a, d) lines, vertex 2 on (b, c)
        (the valence-exchange counterpart subtracted, matching the (v - v2)
        convention the assembler applies).
        """
        total = 0.0
        for n_, (mi, ri) in enumerate(mr_pairs):
            g1 = g(ia, ic, mi, ri)    # <a m|V|c r>
            g3 = g(ia, id_, mi, ri)   # <a m|V|d r>
            if g1 == 0.0 and g3 == 0.0:
                continue
            g2 = g(ib, id_, ri, mi)   # <b r|V|d m>
            g4 = g(ib, ic, ri, mi)    # <b r|V|c m>
            term = g1 * g2 - g3 * g4
            if term != 0.0:
                total -= term / denom[n_]
        return total

    delta.n_intermediate = len(mr_pairs)
    delta.spinors = spinors
    delta.index = index
    return delta
