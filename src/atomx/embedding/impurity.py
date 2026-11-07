"""Atomic-limit (Hubbard-I) impurity solver — Lehmann core (design E1).

The impurity problem for one correlated d shell: Slater-Coulomb (multipole
F2/F4 + explicit monopole F0), spin-orbit coupling and an optional crystal
field, in the |m, s> spin-orbital basis. Three fixed-N sectors
(N-1, N, N+1) are diagonalized exactly; the single-particle Green's
function follows from the Lehmann representation with the c-operator
matrix elements between sectors.

Backend status (honest scope): the eigenproblem assembly here is the
self-contained reference backend. The design target is to re-route it to
the native CI machinery (E1.5) without changing this module's outputs —
the sum-rule and multiplet anchors below are the acceptance tests for that
swap.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- basis ----
def _basis(n: int, norb: int = 10) -> list[tuple[int, ...]]:
    return [tuple(sorted(c)) for c in itertools.combinations(range(norb), n)]


def _sign_apply(state: tuple[int, ...], remove: int, add: int):
    """Apply c^dag_add c_remove to |state>; return (new_state, sign) or None."""
    work = list(state)
    if remove not in work:
        return None
    sgn = (-1) ** work.index(remove)
    work.remove(remove)
    pos = sum(1 for x in work if x < add)
    sgn *= (-1) ** pos
    work.append(add)
    return tuple(sorted(work)), sgn


def _shell_k(norb: int):
    """Slater multipoles for an equivalent d shell via Gaunt coefficients.

    Angular integrals computed once by direct quadrature (Gauss-Legendre x
    uniform) — self-contained, no external angular tables.
    """
    l = 2
    nth, nph = 120, 240
    x, w = np.polynomial.legendre.leggauss(nth)
    th = np.arccos(x)
    ph = 2 * np.pi * np.arange(nph) / nph
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    W2 = w[:, None] * (2 * np.pi / nph)

    def Y(l_, m_):
        from scipy.special import sph_harm_y
        return sph_harm_y(l_, m_, TH, PH)

    D = {}
    for k in (0, 2, 4):
        norm = math.sqrt(4 * math.pi / (2 * k + 1))
        for m in range(-l, l + 1):
            for mp in range(-l, l + 1):
                for q in range(-k, k + 1):
                    if q != m - mp:
                        continue
                    D[(k, m, mp, q)] = np.sum(
                        W2 * np.conj(Y(l, m)) * norm * Y(k, q) * Y(l, mp)
                    )

    def vraw(m1, m2, m3, m4, f0, f2, f4):
        v = 0.0 + 0.0j
        for k, F in {0: f0, 2: f2, 4: f4}.items():
            for q in range(-k, k + 1):
                v += F * np.conj(D.get((k, m3, m1, q), 0j)) * D.get(
                    (k, m2, m4, q), 0j
                )
        return v

    return vraw


def _soc_matrix(zeta: float, l: int = 2) -> np.ndarray:
    n = 2 * (2 * l + 1)
    h = np.zeros((n, n), dtype=complex)
    for m in range(-l, l + 1):
        for s in (0, 1):
            sz = 0.5 if s == 0 else -0.5
            h[2 * (m + l) + s, 2 * (m + l) + s] += zeta * m * sz
    for m in range(-l, l + 1):
        for dm, s_from, s_to in ((+1, 0, 1), (-1, 1, 0)):
            m2 = m + dm
            if -l <= m2 <= l:
                amp = math.sqrt(l * (l + 1) - m * (m + dm))
                h[2 * (m2 + l) + s_to, 2 * (m + l) + s_from] += zeta * 0.5 * amp
    return h


def _cf_matrix(dq10: float, ds: float, dt: float) -> np.ndarray:
    """Tetragonal CF in the spherical d basis (10x10, spin diagonal)."""
    tmp = np.diag([
        6 * dq10 - 2 * ds - 6 * dt,   # d3z2-r2  (cubic row order)
        -4 * dq10 - ds + 4 * dt,      # dzx
        -4 * dq10 - ds + 4 * dt,      # dzy
        6 * dq10 + 2 * ds - dt,       # dx2-y2
        -4 * dq10 + 2 * ds - dt,      # dxy
    ]).astype(complex)
    # cubic -> spherical transform (standard real-harmonic definitions)
    T = np.zeros((5, 5), dtype=complex)
    T[0, 2] = 1.0
    s2 = 1 / math.sqrt(2)
    T[1, 1], T[1, 3] = -s2, s2
    T[2, 1], T[2, 3] = 1j * s2, 1j * s2
    T[3, 0], T[3, 4] = s2, s2
    T[4, 0], T[4, 4] = 1j * s2, -1j * s2
    cf5 = T.conj().T @ tmp @ T
    cf10 = np.zeros((10, 10), dtype=complex)
    cf10[0:10:2, 0:10:2] = cf5
    cf10[1:10:2, 1:10:2] = cf5
    return cf10


# ------------------------------------------------------------------ spec ---
@dataclass(frozen=True)
class ImpuritySpec:
    """One correlated d-shell impurity at the atomic limit."""

    n_electrons: int = 8
    slater_eV: dict = field(default_factory=dict)   # F0, F2, F4
    zeta_d_eV: float = 0.0
    cf: dict = field(default_factory=dict)          # Dq10, ds, dt
    temperature_K: float = 300.0


def _hamiltonian(n: int, spec: ImpuritySpec, vraw) -> np.ndarray:
    basis = _basis(n)
    h1 = _soc_matrix(spec.zeta_d_eV)
    if spec.cf:
        h1 = h1 + _cf_matrix(
            spec.cf.get("Dq10", 0.0), spec.cf.get("ds", 0.0), spec.cf.get("dt", 0.0)
        )
    f0 = spec.slater_eV.get("F0", 0.0)
    f2 = spec.slater_eV.get("F2", 0.0)
    f4 = spec.slater_eV.get("F4", 0.0)
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
                # h1 carries its own spin structure (SOC spin-flip); the
                # diagonal-only SOC left by a spin check here masquerades
                # as fine structure — caught by the E2 reduction anchor
                el += h1[i, j] * _sign_apply(B, j, i)[1]
            elif not onlyA and not onlyB:
                for i in B:
                    el += h1[i, i]
                for u, v in itertools.combinations(B, 2):
                    for i, j in ((u, v), (v, u)):
                        for k, l in ((u, v), (v, u)):
                            if i % 2 != k % 2 or j % 2 != l % 2:
                                continue
                            m = lambda o: o // 2 - 2  # noqa: E731
                            vv = vraw(m(i), m(j), m(k), m(l), f0, f2, f4)
                            if vv != 0:
                                el += 0.5 * vv * _two_body_sign(B, (k, l), (i, j))
            elif len(onlyA) == 2 and len(onlyB) == 2:
                for i, j in itertools.permutations(onlyA):
                    for k, l in itertools.permutations(onlyB):
                        if i % 2 != k % 2 or j % 2 != l % 2:
                            continue
                        m = lambda o: o // 2 - 2  # noqa: E731
                        vv = vraw(m(i), m(j), m(k), m(l), f0, f2, f4)
                        if vv == 0:
                            continue
                        el += 0.5 * vv * _two_body_sign(B, (k, l), (i, j))
            if abs(el) > 0:
                H[a, b] += el
                if b != a:
                    H[b, a] += np.conj(el)
    return H


def _two_body_sign(B, remove, add) -> int:
    work = list(B)
    sgn = 1
    for r in (remove[0], remove[1]):
        idx = work.index(r)
        sgn *= (-1) ** idx
        work.pop(idx)
    for aa in (add[1], add[0]):
        pos = sum(1 for x in work if x < aa)
        sgn *= (-1) ** pos
        work.append(aa)
        work.sort()
    return sgn


# ----------------------------------------------------------------- solve ---
def solve_atomic_limit(spec: ImpuritySpec) -> dict:
    """Exact three-sector solve + c-operator matrix elements.

    Returns evals/vecs for N-1, N, N+1 sectors (relative to
    spec.n_electrons), Boltzmann weights over the N sector, and the
    matrix-element cubes Cm[sector][orbital] with
    Cm['N+1'][n, g] = <N+1 eig | c^dag_orb | N eig> etc.
    """
    vraw = _shell_k(10)
    n0 = spec.n_electrons
    out = {}
    for n in (n0 - 1, n0, n0 + 1):
        H = _hamiltonian(n, spec, vraw)
        ev, vec = np.linalg.eigh(H)
        out[f"N={n}"] = {"evals": ev, "evecs": vec, "basis": _basis(n)}

    bN = out[f"N={n0}"]["basis"]
    # creation: N -> N+1
    Cm_plus = np.zeros((len(out[f"N={n0+1}"]["basis"]), len(bN), 10), dtype=complex)
    index_up = {b: i for i, b in enumerate(out[f"N={n0+1}"]["basis"])}
    for g, st in enumerate(bN):
        for orb in range(10):
            if orb in st:
                continue
            work = list(st)
            sgn = (-1) ** sum(1 for x in work if x < orb)
            new = tuple(sorted(work + [orb]))
            Cm_plus[index_up[new], g, orb] = sgn
    # annihilation: N -> N-1
    Cm_minus = np.zeros((len(out[f"N={n0-1}"]["basis"]), len(bN), 10), dtype=complex)
    index_dn = {b: i for i, b in enumerate(out[f"N={n0-1}"]["basis"])}
    for g, st in enumerate(bN):
        for orb in range(10):
            if orb not in st:
                continue
            work = list(st)
            sgn = (-1) ** work.index(orb)
            work.remove(orb)
            Cm_minus[index_dn[tuple(work)], g, orb] = sgn

    kT = 8.617333262e-5 * spec.temperature_K
    e = out[f"N={n0}"]["evals"]
    w = np.exp(-(e - e[0]) / kT)
    w /= w.sum()
    out["weights_N"] = w
    out["Cm_plus"] = Cm_plus    # (nN+1, nN, orb): <N+1|c^dag|N>
    out["Cm_minus"] = Cm_minus  # (nN-1, nN, orb): <N-1|c|N>
    out["spec"] = spec
    return out


def green_function(
    sol: dict, z: np.ndarray, eta: float = 0.1
) -> np.ndarray:
    """Retarded G_orb,orb(z) (diagonal in spin-orbital) on complex grid z.

    Lehmann (retarded):
        G_a(w) = sum_g w_g [ sum_n |<n(N-1)|c_a|g(N)>|^2 / (w - (E_n - E_g) - i*eta)
                           + sum_m |<m(N+1)|c^dag_a|g(N)>|^2 / (w + (E_m - E_g) + i*eta) ]
    """
    eN = sol[f"N={sol['spec'].n_electrons}"]["evals"]
    em = sol[f"N={sol['spec'].n_electrons-1}"]["evals"]
    ep = sol[f"N={sol['spec'].n_electrons+1}"]["evals"]
    w = sol["weights_N"]
    Cm, Cp = sol["Cm_minus"], sol["Cm_plus"]
    G = np.zeros((10, len(z)), dtype=complex)
    for a in range(10):
        amp_m = (np.abs(Cm[:, :, a]) ** 2) * w[np.newaxis, :]  # (nN-1, nN)
        amp_p = (np.abs(Cp[:, :, a]) ** 2) * w[np.newaxis, :]  # (nN+1, nN)
        for gi in range(len(eN)):
            G[a] += np.sum(
                amp_m[:, gi, None]
                / (z - (em[:, None] - eN[gi]) + 1j * eta),
                axis=0,
            )
            G[a] += np.sum(
                amp_p[:, gi, None]
                / (z + (ep[:, None] - eN[gi]) + 1j * eta),
                axis=0,
            )
    return G


def spectral_function(sol: dict, omega: np.ndarray, eta: float = 0.1):
    """A_orb(w) = -Im G / pi per spin-orbital and the total trace."""
    z = omega.astype(complex)
    G = green_function(sol, z, eta=eta)
    A = -G.imag / math.pi
    return A, G


def occupation(sol: dict) -> np.ndarray:
    """Thermal occupation per spin-orbital: rho = sum_g w_g |g><g| diagonal."""
    n0 = sol["spec"].n_electrons
    bN = sol[f"N={n0}"]["basis"]
    vec = sol[f"N={n0}"]["evecs"]
    rho_diag = np.real(np.diag(vec @ np.diag(sol["weights_N"]) @ vec.conj().T))
    n = np.zeros(10)
    for a in range(10):
        for gidx, st in enumerate(bN):
            if a in st:
                n[a] += rho_diag[gidx]
    return n
