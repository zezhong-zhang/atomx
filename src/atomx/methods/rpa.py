"""Random-phase approximation / TDHF linear response for closed shells.

For a closed-shell Dirac-Fock reference the neutral-excitation
spectrum of multipole J^pi solves the TDHF (RPA) problem on the
particle-hole (ph) basis — pairs (p in virtuals, h in occupied) that
couple to total J with the excitation parity.  The two-body kernels
are the J-coupled antisymmetrised pair elements from
:func:`atomx.methods.mbpt._pair_v_J`:

    A^J_{ph,p'h'} = (eps_p - eps_h) delta + <p h'; J|v|p' h; J>_A
    B^J_{ph,p'h'} = <p p'; J|v|h h'; J>_A

The eigenproblem is solved via the symmetric route
(:func:`solve_symmetric_rpa`), pinned against the analytic 1x1 case:
omega^2 = (a+b)(a-b) and the single-pair polarisability
alpha = (2/3) d^2/(a+b) — i.e. positive B SCREENS the DF response.

MEASURED STATUS (Ne, scripts/rpa_ne_validation.py): machinery
validated — RPA/DF = 0.78 (correct screening direction and magnitude
class), poles split correctly; the ABSOLUTE alpha (0.11 vs experiment
2.669 a0^3) is limited by the 1e-6-occupation frozen-virtual orbital
quality (3d lands at <r> = 24 a0, quasi-free; the same orbital-layer
pathology measured in the R-matrix arc).  Quantitative response needs
term-relaxed virtuals — recorded as the follow-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from atomx.angular.csf_hamiltonian import twoj
from atomx.methods.mbpt import _pair_v_J
from atomx.numerics.bspline.convol import ConvolCache
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL

__all__ = ["RPAResult", "solve_rpa", "solve_symmetric_rpa",
           "ph_pairs_for", "reduced_dipole", "dipole_polarizability"]


@dataclass
class RPAResult:
    """Poles and amplitudes of one J^pi TDHF block."""

    twoJ: int
    omega_au: np.ndarray                       # positive poles, sorted
    X: np.ndarray                              # (n_ph, n_poles)
    Y: np.ndarray
    ph_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    A: np.ndarray | None = None
    B: np.ndarray | None = None


def _parity(kappa: int) -> int:
    l = kappa if kappa > 0 else -kappa - 1
    return -1 if l % 2 else 1


def ph_pairs_for(occupied: Sequence[Tuple[int, int]],
                 virtuals: Sequence[Tuple[int, int]],
                 twoJ: int, exc_parity: int) -> List:
    """(p, h) pairs coupling to twoJ with excitation parity pi_p*pi_h
    (electric multipoles flip parity: pass exc_parity = -(-1)**J)."""
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
            f"TDHF block unstable or defective: {len(sel)} of {2*n} "
            "eigenvalues are positive-real")
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


def solve_rpa(orbitals_dict, bsp_data, eps, occupied, virtuals, twoJ,
              exc_parity) -> RPAResult:
    """Assemble and solve one J^pi TDHF block."""
    pairs = ph_pairs_for(occupied, virtuals, twoJ, exc_parity)
    if not pairs:
        raise ValueError("empty ph basis")
    n = len(pairs)
    cache = ConvolCache(bsp_data)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    de = np.array([eps[p] - eps[h] for p, h in pairs])
    for i, (p, h) in enumerate(pairs):
        Pp, Qp = orbitals_dict[p]
        Ph, Qh = orbitals_dict[h]
        for j, (p2, h2) in enumerate(pairs):
            Pp2, Qp2 = orbitals_dict[p2]
            Ph2, Qh2 = orbitals_dict[h2]
            # A: ph-ph coupling <p h'; J|v|p' h; J>_A
            A[i, j] = _pair_v_J(Pp, Qp, p[1], Ph2, Qh2, h2[1],
                                Pp2, Qp2, p2[1], Ph, Qh, h[1],
                                twoJ, bsp_data, cache)
            # B: backward pp-hh coupling <p p'; J|v|h h'; J>_A
            B[i, j] = _pair_v_J(Pp, Qp, p[1], Pp2, Qp2, p2[1],
                                Ph, Qh, h[1], Ph2, Qh2, h2[1],
                                twoJ, bsp_data, cache)
    np.fill_diagonal(A, A.diagonal() + de)
    w, X, Y = solve_symmetric_rpa(A, B)
    return RPAResult(twoJ=twoJ, omega_au=w, X=X, Y=Y, ph_pairs=pairs,
                     A=A, B=B)


def reduced_dipole(orbitals_dict, bsp_data, p, h) -> float:
    """Static length-form reduced E1 element <p|| r ||h> (a0)."""
    Pp, Qp = orbitals_dict[p]
    Ph, Qh = orbitals_dict[h]
    return angular_eL(h[1], p[1], 1) * radial_eL(Ph, Qh, Pp, Qp,
                                                  bsp_data, L=1)


def dipole_polarizability(res: RPAResult, orbitals_dict, bsp_data) -> float:
    """alpha_D = (2/3) sum_nu |(X+Y).d|^2 / omega_nu (length form)."""
    d = np.array([reduced_dipole(orbitals_dict, bsp_data, p, h)
                  for p, h in res.ph_pairs])
    alpha = 0.0
    for k in range(len(res.omega_au)):
        amp = float((res.X[:, k] + res.Y[:, k]) @ d)
        alpha += 2.0 / 3.0 * amp ** 2 / res.omega_au[k]
    return alpha
