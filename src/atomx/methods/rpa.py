"""Closed-shell TDHF in normalized particle-hole tensor states.

The A and B kernels contract spin-orbital Coulomb integrals with the
Clebsch-Gordan tensor of a particle and a time-reversed hole. Coupled
particle-particle matrix elements cannot be substituted for these tensors.
Virtual states must form an orthonormal complement to the occupied space
of the same reference Fock operator. Finite-box, energy and partial-wave
cutoffs require convergence checks before interpreting absolute response.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.csf_hamiltonian import two_body_angular, twoj
from atomx.numerics.bspline.convol import (
    ConvolCache,
    build_slater_rk_bilinear,
    eval_slater_rk_bilinear,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL

__all__ = [
    "RPAResult",
    "solve_rpa",
    "solve_symmetric_rpa",
    "ph_pairs_for",
    "reduced_dipole",
    "dipole_polarizability",
    "trk_sum",
    "dense_continuum_virtuals",
]


@dataclass
class RPAResult:
    """Poles and amplitudes of one J^pi TDHF block."""

    twoJ: int
    omega_au: np.ndarray  # positive poles, sorted
    X: np.ndarray  # (n_ph, n_poles)
    Y: np.ndarray
    ph_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    A: np.ndarray | None = None
    B: np.ndarray | None = None


def _parity(kappa: int) -> int:
    l = kappa if kappa > 0 else -kappa - 1
    return -1 if l % 2 else 1


def ph_pairs_for(
    occupied: Sequence[Tuple[int, int]],
    virtuals: Sequence[Tuple[int, int]],
    twoJ: int,
    exc_parity: int,
) -> List:
    """(p, h) pairs coupling to twoJ with excitation parity pi_p*pi_h
    (electric multipoles flip parity: pass exc_parity = (-1)**J)."""
    out = []
    for p in virtuals:
        for h in occupied:
            if _parity(p[1]) * _parity(h[1]) != exc_parity:
                continue
            jp, jh = twoj(p[1]), twoj(h[1])
            if twoJ < abs(jp - jh) or twoJ > jp + jh:
                continue
            out.append((p, h))
    return out


def solve_symmetric_rpa(A, B):
    """Solve the TDHF eigenproblem.

    Direct route on the doubled non-Hermitian matrix
    [[A, B], [-B, -A]] z = omega z: take the eigenvalues with
    Re(omega) > 0 (they come in +- pairs), normalise X^T X - Y^T Y = 1
    per mode.  For the small ph blocks of atomic RPA this is cheaper
    than arguing about symmetric square roots — the earlier
    chol-based route silently used S = L^T whose square is NOT A - B
    (Cholesky factors do not self-multiply), caught by the
    columnwise-residual gate G2.
    """
    from scipy.linalg import eig

    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if (
        A.ndim != 2
        or A.shape[0] != A.shape[1]
        or B.shape != A.shape
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(B))
    ):
        raise ValueError("RPA requires finite square A and B of equal shape")
    if not np.allclose(A, A.T, atol=1e-10, rtol=0) or not np.allclose(
        B, B.T, atol=1e-10, rtol=0
    ):
        raise ValueError("RPA requires symmetric A and B")
    n = A.shape[0]
    R = np.zeros((2 * n, 2 * n))
    R[:n, :n] = A
    R[:n, n:] = B
    R[n:, :n] = -B
    R[n:, n:] = -A
    wv, Z = eig(R)
    sel = np.where((wv.real > 1e-9) & (np.abs(wv.imag) < 1e-9))[0]
    if len(sel) != n:
        raise ValueError(
            f"TDHF block unstable or defective: {len(sel)} of {2 * n} "
            "eigenvalues are positive-real"
        )
    order = np.argsort(wv.real[sel])
    sel = sel[order]
    w = wv.real[sel]
    X = Z[:n, sel].real
    Y = Z[n:, sel].real
    nrm = np.einsum("in,in->n", X, X) - np.einsum("in,in->n", Y, Y)
    if np.any(nrm <= 0):
        raise ValueError("anomalous normalisation (X^T X - Y^T Y <= 0)")
    scale = 1.0 / np.sqrt(nrm)
    X = X * scale[None, :]
    Y = Y * scale[None, :]
    return w, X, Y


@lru_cache(maxsize=4096)
def _ph_tensor(kp, kh, twoJ):
    """Normalized J,M=0 tensor a†_p times the time-reversed hole."""
    jp, jh = twoj(kp), twoj(kh)
    return tuple(
        (m, (-1.0) ** ((jh - m) // 2) * clebsch_gordan(jp, m, jh, -m, twoJ, 0))
        for m in range(-min(jp, jh), min(jp, jh) + 1, 2)
    )


@lru_cache(maxsize=4096)
def _ph_angular(kp, kh, kp2, kh2, twoJ):
    """Rank coefficients of the four spin-orbital TDHF contractions.

    A = g(p,h';h,p') - g(p,h';p',h)
    B = g(p,p';h,h') - g(p,p';h',h).
    These are particle-hole tensors, not coupled two-particle states.
    """
    out = [dict() for _ in range(4)]
    for m, c in _ph_tensor(kp, kh, twoJ):
        for m2, c2 in _ph_tensor(kp2, kh2, twoJ):
            arguments = (
                (kp, m, kh2, m2, kh, m, kp2, m2),
                (kp, m, kh2, m2, kp2, m2, kh, m),
                (kp, m, kp2, m2, kh, m, kh2, m2),
                (kp, m, kp2, m2, kh2, m2, kh, m),
            )
            for table, args in zip(out, arguments, strict=True):
                for rank, coefficient in two_body_angular(*args):
                    table[rank] = table.get(rank, 0.0) + c * c2 * coefficient
    return tuple(
        tuple((k, v) for k, v in sorted(table.items()) if abs(v) > 1e-14)
        for table in out
    )


def _check_ph_metric(orbitals_dict, bsp_data, Z, occupied, virtuals):
    """Canonical particle-hole equations require a common orthonormal basis."""
    keys = list(occupied) + list(virtuals)
    for kap in {k[1] for k in keys}:
        same = [key for key in keys if key[1] == kap]
        _, S = build_dirac_matrices(bsp_data["ns"], kap, Z, bsp_data)
        C = np.column_stack([np.concatenate(orbitals_dict[key]) for key in same])
        if not np.allclose(C.T @ S @ C, np.eye(len(same)), atol=1e-7, rtol=0):
            raise ValueError("RPA orbitals must be orthonormal in the radial metric")


def solve_rpa(
    orbitals_dict, bsp_data, eps, occupied, virtuals, twoJ, exc_parity
) -> RPAResult:
    """Assemble TDHF by contracting normalized particle-hole tensors.

    The reference must use canonical occupied and virtual orbitals of one
    Fock operator. Orthonormality is checked; the caller owns its energies.
    """
    if twoJ < 0 or twoJ % 2:
        raise ValueError("particle-hole total J must be a nonnegative integer")
    if set(occupied) & set(virtuals):
        raise ValueError("orbitals both occupied and virtual")
    _check_ph_metric(orbitals_dict, bsp_data, 0, occupied, virtuals)
    pairs = ph_pairs_for(occupied, virtuals, twoJ, exc_parity)
    if not pairs:
        raise ValueError("empty ph basis")
    cache = ConvolCache(bsp_data)
    A = np.zeros((len(pairs), len(pairs)))
    B = np.zeros_like(A)

    @lru_cache(maxsize=128)
    def bilinear(a, b, rank):
        Pa, Qa = orbitals_dict[a]
        Pb, Qb = orbitals_dict[b]
        return build_slater_rk_bilinear(Pa, Qa, Pb, Qb, bsp_data, rank, cache=cache)

    @lru_cache(maxsize=None)
    def radial(a, c, b, d, rank):
        Pc, Qc = orbitals_dict[c]
        Pd, Qd = orbitals_dict[d]
        return eval_slater_rk_bilinear(bilinear(a, b, rank), Pc, Qc, Pd, Qd)

    for i, (p, h) in enumerate(pairs):
        for j, (p2, h2) in enumerate(pairs):
            ad, ax, bd, bx = _ph_angular(p[1], h[1], p2[1], h2[1], twoJ)
            A[i, j] = sum(c * radial(p, h, h2, p2, k) for k, c in ad) - sum(
                c * radial(p, p2, h2, h, k) for k, c in ax
            )
            B[i, j] = sum(c * radial(p, h, p2, h2, k) for k, c in bd) - sum(
                c * radial(p, h2, p2, h, k) for k, c in bx
            )
        A[i, i] += eps[p] - eps[h]
    w, X, Y = solve_symmetric_rpa(A, B)
    return RPAResult(twoJ=twoJ, omega_au=w, X=X, Y=Y, ph_pairs=pairs, A=A, B=B)


def dense_continuum_virtuals(result, kappas=(-1, 1, -2, 2, -3), e_cap=8.0):
    """Canonical V^N DHF virtuals, including bound and box continuum roots.

    Exchange, occupied-space projection, and the reference nuclear field
    are retained. Synthetic n labels describe eigenstates, not quantum n.
    The finite box and energy ceiling still require convergence checks.
    """
    from atomx.methods.mbpt import build_virtual_basis

    data = result._bsp_data
    cache = ConvolCache(data)
    orbitals, eps = {}, {}
    for kap in kappas:
        _, states = build_virtual_basis(
            result,
            kap,
            cache=cache,
            e_max=e_cap,
            potential_mode="VN",
            orthogonal_to_occupied=True,
        )
        for count, (P, Q, e) in enumerate(states):
            key = (100 + count, kap)
            orbitals[key] = (P.copy(), Q.copy())
            eps[key] = e
    return orbitals, eps


def reduced_dipole(orbitals_dict, bsp_data, p, h) -> float:
    """Static length-form reduced E1 element <p|| r ||h> (a0)."""
    Pp, Qp = orbitals_dict[p]
    Ph, Qh = orbitals_dict[h]
    return angular_eL(h[1], p[1], 1) * radial_eL(Ph, Qh, Pp, Qp, bsp_data, L=1)


def trk_sum(res: RPAResult, orbitals_dict, bsp_data, eps) -> float:
    """DF-only Thomas-Reiche-Kuhn sum  S = (2/3) sum_ph de |d|^2.

    For a complete one-electron basis S = N_electron (length gauge,
    up to relativistic corrections).  A value far below N quantifies
    the basis-completeness deficit of the virtual set."""
    d = np.array(
        [reduced_dipole(orbitals_dict, bsp_data, p, h) for p, h in res.ph_pairs]
    )
    de = np.array([eps[p] - eps[h] for p, h in res.ph_pairs])
    return float(np.sum(2.0 / 3.0 * de * d**2))


def dipole_polarizability(res: RPAResult, orbitals_dict, bsp_data) -> float:
    """alpha_D = (2/3) sum_nu |(X+Y).d|^2 / omega_nu (length form).

    E1 response only: asserts the block is twoJ = 2 (dipole)."""
    if res.twoJ != 2:
        raise ValueError("dipole polarizability needs the J=1 block")
    d = np.array(
        [reduced_dipole(orbitals_dict, bsp_data, p, h) for p, h in res.ph_pairs]
    )
    alpha = 0.0
    for k in range(len(res.omega_au)):
        amp = float((res.X[:, k] + res.Y[:, k]) @ d)
        alpha += 2.0 / 3.0 * amp**2 / res.omega_au[k]
    return alpha
