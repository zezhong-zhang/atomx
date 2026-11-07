"""Discrete-bath cluster solver (design E2): impurity + bath = cluster superset.

The E1 impurity generalizes to bath spin-orbital pairs hybridizing with the
correlated d shell, plus an optional filled core shell with its own
impurity-core Coulomb. This makes the embedding solver a strict superset of
ligand/cluster models: the hyperx NiO charge-transfer cluster (Ni 3d +
effective ligand shell + 2p core) is a *configuration* of this module, and
the cross-repo anchor pins its ground-state spectrum to the hyperx
cluster_ct assembly.

Scope (v1, honest): frozen-bath Coulomb tier — no two-body terms involving
bath orbitals; bath enters through one-body levels and hopping only.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from atomx.embedding.impurity import _cf_matrix, _soc_matrix

# ------------------------------------------------------------------ tables --
_TABLES = None


def _gaunt_tables():
    """Quadrature Gaunt tables: DD (same-shell d), PP (same-shell p),
    DX (cross p* k d). Built once (lazily)."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    nth, nph = 160, 320
    x, w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(nph) / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W2 = w[:, None] * (2 * np.pi / nph)

    def Y(l_, m_):
        from scipy.special import sph_harm_y
        return sph_harm_y(l_, m_, TH, PH)

    Yd = {m: Y(2, m) for m in range(-2, 3)}
    Yp = {m: Y(1, m) for m in range(-1, 2)}
    DD, PP, DX = {}, {}, {}
    for k in (0, 2, 4):
        norm = math.sqrt(4 * math.pi / (2 * k + 1))
        Yk = {q: norm * Y(k, q) for q in range(-k, k + 1)}
        for m in range(-2, 3):
            for mp in range(-2, 3):
                for q in range(-k, k + 1):
                    if q != m - mp:
                        continue
                    DD[(k, m, mp, q)] = np.sum(
                        W2 * np.conj(Yd[m]) * Yk[q] * Yd[mp]
                    )
    for k in (0, 2):
        norm = math.sqrt(4 * math.pi / (2 * k + 1))
        Yk = {q: norm * Y(k, q) for q in range(-k, k + 1)}
        for m in range(-1, 2):
            for mp in range(-1, 2):
                for q in range(-k, k + 1):
                    if q != m - mp:
                        continue
                    PP[(k, m, mp, q)] = np.sum(
                        W2 * np.conj(Yp[m]) * Yk[q] * Yp[mp]
                    )
    for k in (1, 2, 3):
        norm = math.sqrt(4 * math.pi / (2 * k + 1))
        Yk = {q: norm * Y(k, q) for q in range(-k, k + 1)}
        for mp in range(-1, 2):      # p index first (bra side)
            for m in range(-2, 3):   # d index second (ket side)
                for q in range(-k, k + 1):
                    if q != mp - m:
                        continue
                    DX[(k, mp, m, q)] = np.sum(
                        W2 * np.conj(Yp[mp]) * Yk[q] * Yd[m]
                    )
    _TABLES = (DD, PP, DX)
    return _TABLES


def _dd_element(i, j, k, l, f0, f2, f4):
    DD = _gaunt_tables()[0]
    m = lambda o: o // 2 - 2  # noqa: E731
    v = 0.0 + 0.0j
    for kk, FF in {0: f0, 2: f2, 4: f4}.items():
        for q in range(-kk, kk + 1):
            v += FF * DD.get((kk, m(k), m(i), q), 0j).conjugate() * DD.get(
                (kk, m(j), m(l), q), 0j
            )
    return v


def _dp_element(i, j, k, l, f2dp, g1, g3):
    """Impurity(d, 0-9) x core(p, 10-15) Coulomb in the p-slot layout.

    direct <d p | V | d p> : k = 0, 2 (F0_dp absorbed, F2_dp kept)
    exchange <d p | V | p d> : k = 1, 3 (G1_dp, G3_dp)
    """
    _DD, PP, DX = _gaunt_tables()
    md = lambda o: o // 2 - 2  # noqa: E731
    mp = lambda o: (o - 10) // 2 - 1  # noqa: E731
    d_idx = lambda o: o < 10  # noqa: E731
    if d_idx(i) and not d_idx(j) and d_idx(k) and not d_idx(l):
        v = 0.0 + 0.0j
        for kk, FF in {0: 0.0, 2: f2dp}.items():
            for q in range(-kk, kk + 1):
                v += FF * _DD.get((kk, md(k), md(i), q), 0j).conjugate() * PP.get(
                    (kk, mp(j), mp(l), q), 0j
                )
        return v
    if d_idx(i) and not d_idx(j) and not d_idx(k) and d_idx(l):
        v = 0.0 + 0.0j
        for kk, GG in {1: g1, 3: g3}.items():
            for q in range(-kk, kk + 1):
                v += GG * DX.get((kk, mp(k), md(i), q), 0j).conjugate() * DX.get(
                    (kk, mp(j), md(l), q), 0j
                )
        return v
    return 0.0 + 0.0j


# -------------------------------------------------------------------- spec --
@dataclass(frozen=True)
class ClusterSpec:
    """d impurity + discrete bath (+ optional filled core shell) cluster."""

    n_valence: int = 14                    # electrons in (d + bath)
    bath_levels_eV: tuple = ()              # one level per bath spin-pair
    hopping_eV: object = None               # (10, 2*nb) d-bath one-body
    dd_slater_eV: dict = field(default_factory=dict)
    zeta_d_eV: float = 0.0
    cf: dict = field(default_factory=dict)
    core_slater_eV: dict = field(default_factory=dict)  # F2_dp, G1_dp, G3_dp


def _layout(spec: ClusterSpec):
    nb = len(spec.bath_levels_eV)
    bath = list(range(10, 10 + 2 * nb))
    core = list(range(10 + 2 * nb, 16 + 2 * nb)) if spec.core_slater_eV else []
    return bath, core


def _one_body(spec: ClusterSpec, bath, core):
    ntot = 10 + len(bath) + len(core)
    h1 = np.zeros((ntot, ntot), dtype=complex)
    h1[0:10, 0:10] += _soc_matrix(spec.zeta_d_eV)
    if spec.cf:
        # cf keys carry 10Dq (physical splitting); _cf_matrix takes dq = 10Dq/10
        h1[0:10, 0:10] += _cf_matrix(
            spec.cf.get("Dq10", 0.0) / 10.0,
            spec.cf.get("ds", 0.0),
            spec.cf.get("dt", 0.0),
        )
    for i, b in enumerate(bath):
        h1[b, b] = spec.bath_levels_eV[i // 2]
    if spec.hopping_eV is not None and len(bath):
        V = np.asarray(spec.hopping_eV, dtype=complex)
        h1[np.ix_(list(range(10)), bath)] += V
        h1[np.ix_(bath, list(range(10)))] += V.conj().T
    return h1


def _sgn_one(B, remove, add):
    work = list(B)
    sgn = (-1) ** work.index(remove)
    work.remove(remove)
    return sgn * (-1) ** sum(1 for x in work if x < add)


def _sgn_two(B, remove, add):
    work = list(B)
    sgn = 1
    for r in (remove[0], remove[1]):
        idx = work.index(r)
        sgn *= (-1) ** idx
        work.pop(idx)
    for aa in (add[1], add[0]):
        sgn *= (-1) ** sum(1 for x in work if x < aa)
        work.append(aa)
        work.sort()
    return sgn


def _element(spec, bath, core, i, j, k, l):
    """Two-body element dispatch: dd within impurity, impurity-core dp,
    zero for anything involving bath (frozen-bath tier). Core orbitals are
    translated to the p-slot layout for the dp tables."""
    if {i, j, k, l} & set(bath):
        return 0.0 + 0.0j
    if not core or max(i, j, k, l) < 10:
        f = spec.dd_slater_eV
        return _dd_element(
            i, j, k, l, f.get("F0", 0.0), f.get("F2", 0.0), f.get("F4", 0.0)
        )
    nb_pairs = len(bath) // 2  # bath entries are spin-pair levels
    tr = lambda o: o if o < 10 else 10 + (o - 10 - 2 * nb_pairs)  # noqa: E731
    cs = spec.core_slater_eV
    # mirrored orientations (p d | p d), (p d | d p)
    ii, jj, kk, ll = tr(i), tr(j), tr(k), tr(l)
    d_idx = lambda o: o < 10  # noqa: E731
    ori = (d_idx(ii), d_idx(jj), d_idx(kk), d_idx(ll))
    if ori == (True, False, True, False):
        return _dp_element(ii, jj, kk, ll, cs.get("F2_dp", 0.0),
                           cs.get("G1_dp", 0.0), cs.get("G3_dp", 0.0))
    if ori == (False, True, False, True):
        # <p d | V | p d> = <d_j p_i | V | d_l p_k> (swap within bra/ket)
        return _dp_element(jj, ii, ll, kk, cs.get("F2_dp", 0.0),
                           cs.get("G1_dp", 0.0), cs.get("G3_dp", 0.0))
    if ori == (True, False, False, True):
        # <d p | V | p d> : exchange branch
        return _dp_element(ii, jj, kk, ll, cs.get("F2_dp", 0.0),
                           cs.get("G1_dp", 0.0), cs.get("G3_dp", 0.0))
    if ori == (False, True, True, False):
        # <p d | V | d p> : mirrored exchange = <d j | p i | V | p l d k>
        return _dp_element(jj, ii, ll, kk, cs.get("F2_dp", 0.0),
                           cs.get("G1_dp", 0.0), cs.get("G3_dp", 0.0))
    return 0.0 + 0.0j


def build_ground_hamiltonian(spec: ClusterSpec):
    """Fixed-n_valence sector basis and Hamiltonian."""
    bath, core = _layout(spec)
    valence = list(range(10)) + bath
    basis = [
        tuple(sorted(c + tuple(core)))
        for c in itertools.combinations(valence, spec.n_valence)
    ]
    h1 = _one_body(spec, bath, core)
    sets = [frozenset(b) for b in basis]
    H = np.zeros((len(basis), len(basis)), dtype=complex)
    for a in range(len(basis)):
        for b in range(a, len(basis)):
            B = basis[b]
            el = 0.0 + 0.0j
            common = sets[a] & sets[b]
            onlyA = sorted(sets[a] - common)
            onlyB = sorted(sets[b] - common)
            if len(onlyA) == 1 and len(onlyB) == 1:
                i, j = onlyA[0], onlyB[0]
                # h1 carries its own spin structure (SOC spin-flip); no
                # spin check — same fix as every prior assembly copy
                el += h1[i, j] * _sgn_one(B, j, i)
            elif not onlyA and not onlyB:
                for i in B:
                    el += h1[i, i]
                for u, v in itertools.combinations(B, 2):
                    for i, j in ((u, v), (v, u)):
                        for k, l in ((u, v), (v, u)):
                            if i % 2 != k % 2 or j % 2 != l % 2:
                                continue
                            vv = _element(spec, bath, core, i, j, k, l)
                            if vv != 0:
                                el += 0.5 * vv * _sgn_two(B, (k, l), (i, j))
            elif len(onlyA) == 2 and len(onlyB) == 2:
                for i, j in itertools.permutations(onlyA):
                    for k, l in itertools.permutations(onlyB):
                        if i % 2 != k % 2 or j % 2 != l % 2:
                            continue
                        vv = _element(spec, bath, core, i, j, k, l)
                        if vv == 0:
                            continue
                        el += 0.5 * vv * _sgn_two(B, (k, l), (i, j))
            if abs(el) > 0:
                H[a, b] += el
                if b != a:
                    H[b, a] += np.conj(el)
    return basis, H


def solve_cluster_ground(spec: ClusterSpec) -> dict:
    basis, H = build_ground_hamiltonian(spec)
    ev, vec = np.linalg.eigh(H)
    return {"basis": basis, "evals": ev, "evecs": vec, "dim": len(basis)}


def nio_hyperx_configuration(*, delta_eV=4.5, u_baricenter=0.409231746031746,
                             t_pd=1.3, dq10=0.85, f2=7.9521, f4=4.9387,
                             zeta_d=0.083, f2dp=7.33495, g1=4.0509, g3=2.3037,
                             eps_L=-5.212880):
    """The hyperx cluster_ct NiO configuration as a ClusterSpec.

    The hopping matrix reproduces hyperx's single-amplitude structure: the
    dipole-angular x-component scaled to a max element of t_pd. eps_L is the
    hyperx-calibrated ligand level (T=0 bisected to the 4.5 eV CT gap).
    """
    # single-amplitude hopping on the dipole angular structure (x component)
    nth, nph = 120, 240
    x, w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(nph) / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    from scipy.special import sph_harm_y

    def Y(l_, m_):
        return sph_harm_y(l_, m_, TH, PH)

    W2 = w[:, None] * (2 * np.pi / nph)
    sx = np.sin(TH) * np.cos(PH)
    M = np.zeros((16, 16), dtype=complex)
    for mp in range(-1, 2):
        for m in range(-2, 3):
            val = np.sum(W2 * np.conj(Y(1, mp)) * sx * Y(2, m))
            M[2 * (m + 2), 10 + 2 * (mp + 1)] = val
            M[2 * (m + 2) + 1, 11 + 2 * (mp + 1)] = val
    M = M / np.abs(M).max()
    V = t_pd * M[np.ix_(range(10), range(10, 16))]
    return ClusterSpec(
        n_valence=14,
        bath_levels_eV=(eps_L, eps_L, eps_L),
        hopping_eV=V,
        dd_slater_eV={"F0": u_baricenter, "F2": f2, "F4": f4},
        zeta_d_eV=zeta_d,
        cf={"Dq10": dq10},
        core_slater_eV={"F2_dp": f2dp, "G1_dp": g1, "G3_dp": g3},
    )
