"""Multi-channel close-coupling inner-region Hamiltonian assembly.

Phase 2 of HYP-188 extends the single-channel Wigner R-matrix to a
``len(channels)``-block coupled radial problem on [0, a] in atomic
units. Each channel ``i`` carries a frozen target orbital
``P_{t,i}(r)`` with energy ``ε_{t,i}``; the projectile (electron N+1)
moves with partial-wave ``l_{c,i}`` in the static + exchange potential
of that target electron, plus inter-channel coupling.

The radial close-coupling equation for total LSπ block is

    [ -1/2 d²/dr² + l_{c,i}(l_{c,i}+1)/(2 r²) - Z/r + ε_{t,i} ] F_i(r)
    + sum_j [ V^d_ij(r) F_j(r) + integral W^x_ij(r, r') F_j(r') dr' ]
    = E F_i(r)

with direct + exchange potentials assembled from
:mod:`atomx.scattering.r_matrix.angular_cc`. The eigenpairs of the
Bloch-augmented matrix on [0, a] feed the surface-matching layer.

Conventions match the phase-1 single-channel scaffold: clamped B-splines
on [0, a] with the first spline dropped to enforce ``u(0) = 0``,
``b = 0`` Bloch constant (symmetric kinetic form, no surface term),
generalised Hermitian eigenvalue problem on ``(n_basis - 1) * n_channels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

from atomx.scattering.r_matrix.angular_cc import direct_terms, exchange_terms
from atomx.scattering.r_matrix.cc_potentials import cumulative_y_k
from atomx.scattering.r_matrix.channel_table import Channel, TargetState
from atomx.scattering.two_electron.angular import (
    _electron_contribution,
    direct_coefficient,
)


@dataclass
class CCInnerSolution:
    """Eigenpairs of the multi-channel Bloch-augmented Hamiltonian."""

    eigenvalues_au: np.ndarray            # (n_states,)
    surface_amplitudes: np.ndarray         # (n_states, n_channels)
    a_au: float
    n_channels: int
    n_basis_effective: int                 # n_basis - 1 (after dropping the
    # first clamped spline that enforces u(0) = 0)
    k_spline_order: int
    H_au: np.ndarray | None = None        # full Bloch-augmented Hamiltonian
    S_au: np.ndarray | None = None        # overlap matrix


def _make_clamped_knots(a: float, n_basis: int, k_order: int) -> np.ndarray:
    n_interior = n_basis - k_order
    if n_interior < 0:
        raise ValueError(
            f"n_basis={n_basis} too small for spline order={k_order}"
        )
    if n_interior == 0:
        interior = np.empty(0)
    else:
        interior = np.linspace(0.0, a, n_interior + 2)[1:-1]
    return np.concatenate(
        [np.zeros(k_order), interior, np.full(k_order, a)]
    )


def _bspline_design(
    r: np.ndarray, knots: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    n_basis = len(knots) - degree - 1
    coef = np.eye(n_basis)
    spline = BSpline(knots, coef, degree, extrapolate=False)
    B = spline(r)
    dB = spline.derivative(1)(r)
    nan_mask = np.isnan(B).any(axis=1)
    if nan_mask.any():
        r_fixed = np.where(np.isclose(r, knots[-1]), knots[-1] * (1 - 1e-15), r)
        B = spline(r_fixed)
        dB = spline.derivative(1)(r_fixed)
    return B, dB


def _gauss_legendre_per_cell(
    knots: np.ndarray, n_gauss: int
) -> tuple[np.ndarray, np.ndarray]:
    breakpoints = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    pts = []
    wts = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=False):
        if right <= left:
            continue
        half = 0.5 * (right - left)
        mid = 0.5 * (right + left)
        pts.append(half * nodes + mid)
        wts.append(half * weights)
    return np.concatenate(pts), np.concatenate(wts)


def assemble_cc_inner_region(
    channels: Sequence[Channel],
    L_total: int,
    S_total: float,
    Z: float,
    a_au: float,
    n_basis: int,
    k_spline_order: int = 6,
    n_gauss: int = 10,
    pair_targets: Sequence[tuple] = (),
) -> CCInnerSolution:
    """Assemble + diagonalise the multi-channel inner-region Hamiltonian.

    Returns the eigenpairs ``(E_n, ψ_n)`` of the generalised eigenvalue
    problem ``H ψ = E S ψ`` together with the surface amplitudes
    ``w_n^i = ψ_n[i, :] . B(a)`` in each channel ``i``.

    ``pair_targets`` — optional NON-ORTHOGONAL correlation pseudo-state
    channels (Burke's chi_j layer): list of ``(label, P_callable,
    energy_au)`` with l = 0, appended as (target, eps s) channels whose
    target may OVERLAP the regular s-targets.  This adds non-diagonal S
    and one-body cross blocks (singlet 1S, all-s only — the angular
    orthogonality zeroes the cross overlap against p-channels).  The
    (phi, phi) dielectronic pair function lies inside the new channel's
    span; an ORTHOGONAL phi cannot bind it (variational bound: any
    state perpendicular to (1s, 2s-bar) sits above E(3s)).
    """
    if pair_targets:
        if L_total != 0 or S_total != 0.0:
            raise NotImplementedError(
                "pair_targets cross-blocks derived for singlet 1S only")
    channels = list(channels) + [
        Channel(index=len(channels) + i,
                target=TargetState(label=lab, n_electrons=1, l=0,
                                   energy_au=e, P=P, n=2, Z=Z),
                l_c=0, threshold_au=e)
        for i, (lab, P, e) in enumerate(pair_targets)]
    n_ch = len(channels)
    if n_ch == 0:
        raise ValueError("must pass at least one channel")
    has_composite = any(ch.target.n_electrons > 1 for ch in channels)
    if a_au <= 0:
        raise ValueError(f"a_au must be positive, got {a_au}")
    if n_basis < k_spline_order:
        raise ValueError(
            f"n_basis ({n_basis}) must be >= spline order ({k_spline_order})"
        )

    knots = _make_clamped_knots(a_au, n_basis, k_spline_order)
    degree = k_spline_order - 1
    r_q, w_q = _gauss_legendre_per_cell(knots, n_gauss)
    B_full, dB_full = _bspline_design(r_q, knots, degree)
    # Drop the first spline (u(0) = 0).
    B = B_full[:, 1:]
    dB = dB_full[:, 1:]
    n_eff = B.shape[1]

    inv_r = 1.0 / r_q
    inv_r2 = inv_r * inv_r
    Bw = B * w_q[:, None]

    # ------------------------------------------------------------------
    # Common single-channel one-body operators.
    # ------------------------------------------------------------------
    S_one = Bw.T @ B
    T_one = 0.5 * (dB * w_q[:, None]).T @ dB
    V_nuc_one = -Z * (Bw * inv_r[:, None]).T @ B
    Centrifugal = [
        (0.5 * l * (l + 1)) * ((Bw * inv_r2[:, None]).T @ B)
        for l in range(max(ch.l_c for ch in channels) + 1)
    ]

    # Target-orbital sample tables on the quadrature grid (we treat the
    # target orbitals as analytic, so they extend correctly through the
    # full integration support).
    target_labels = []
    P_target_table: dict[str, np.ndarray] = {}
    for ch in channels:
        if ch.target.label not in P_target_table:
            P_target_table[ch.target.label] = np.asarray(ch.target.P(r_q))
            target_labels.append(ch.target.label)

    # Per-electron radial samples for composite (two-electron) targets.
    target_orbitals: dict[tuple[str, int], np.ndarray] = {}
    for ch in channels:
        term = ch.target.two_electron_term
        if term is not None:
            for idx, P in enumerate((term.P1, term.P2), start=1):
                key = (ch.target.label, idx)
                if key not in target_orbitals:
                    target_orbitals[key] = np.asarray(P(r_q))

    # Direct-multipole y_k integrals between target orbital pairs.
    target_y_k: dict[tuple[str, str, int], np.ndarray] = {}

    def y_k_target(label_a: str, label_b: str, k: int) -> np.ndarray:
        key = (label_a, label_b, k) if label_a <= label_b else (label_b, label_a, k)
        if key not in target_y_k:
            f_q = P_target_table[label_a] * P_target_table[label_b]
            target_y_k[key] = cumulative_y_k(r_q, w_q, f_q, k)
        return target_y_k[key]

    # y_k for one active electron pair in a composite target.
    two_e_y_k: dict[tuple[str, int, str, int, int], np.ndarray] = {}

    def y_k_two_electron(label_a: str, a: int, label_b: str, b: int, k: int) -> np.ndarray:
        if (label_a, a) < (label_b, b):
            key = (label_a, a, label_b, b, k)
        else:
            key = (label_b, b, label_a, a, k)
        if key not in two_e_y_k:
            f_q = target_orbitals[(label_a, a)] * target_orbitals[(label_b, b)]
            two_e_y_k[key] = cumulative_y_k(r_q, w_q, f_q, k)
        return two_e_y_k[key]

    # B-spline-times-target multipole moments y_k^{target, B_alpha}(r).
    # Shape (n_q, n_eff) — one column per B-spline.
    target_bsp_y_k: dict[tuple[str, int], np.ndarray] = {}

    def y_k_target_bsp(label_t: str, k: int) -> np.ndarray:
        key = (label_t, k)
        if key not in target_bsp_y_k:
            Pt = P_target_table[label_t]
            out = np.empty((r_q.size, n_eff), dtype=float)
            for beta in range(n_eff):
                f_q = Pt * B[:, beta]
                out[:, beta] = cumulative_y_k(r_q, w_q, f_q, k)
            target_bsp_y_k[key] = out
        return target_bsp_y_k[key]

    # ------------------------------------------------------------------
    # Assemble the (n_ch * n_eff) generalized eigenvalue problem.
    # ------------------------------------------------------------------
    N = n_ch * n_eff
    H = np.zeros((N, N), dtype=float)
    S = np.zeros((N, N), dtype=float)

    def block(j: int, i: int) -> tuple[slice, slice]:
        return (slice(j * n_eff, (j + 1) * n_eff),
                slice(i * n_eff, (i + 1) * n_eff))

    for j_idx, ch_j in enumerate(channels):
        # Channel-diagonal one-body + overlap.
        sl_jj = block(j_idx, j_idx)
        S[sl_jj] = S_one
        H[sl_jj] += T_one
        H[sl_jj] += Centrifugal[ch_j.l_c]
        H[sl_jj] += V_nuc_one
        H[sl_jj] += ch_j.target.energy_au * S_one

        for i_idx, ch_i in enumerate(channels):
            sl_ji = block(j_idx, i_idx)

            # ----------------------------------------------------------
            # Direct multipole couplings (always present, diagonal+offdiag).
            # ----------------------------------------------------------
            if ch_j.target.n_electrons == 1:
                d_terms = direct_terms(
                    l_tj=ch_j.target.l,
                    l_cj=ch_j.l_c,
                    l_ti=ch_i.target.l,
                    l_ci=ch_i.l_c,
                    L=L_total,
                )
                for term in d_terms:
                    y = y_k_target(
                        ch_j.target.label, ch_i.target.label, term.k
                    )
                    V_loc = term.coefficient * y
                    H[sl_ji] += (Bw * V_loc[:, None]).T @ B
            else:
                term_j = ch_j.target.two_electron_term
                term_i = ch_i.target.two_electron_term
                if term_j is None or term_i is None:
                    raise ValueError(
                        "Composite target missing two_electron_term"
                    )
                for a_idx, (l_aj, l_ai) in enumerate(
                    zip((term_j.l1, term_j.l2), (term_i.l1, term_i.l2), strict=False),
                    start=1,
                ):
                    l_bj = term_j.l2 if a_idx == 1 else term_j.l1
                    l_bi = term_i.l2 if a_idx == 1 else term_i.l1
                    if l_bj != l_bi:
                        continue
                    k_min = max(
                        abs(l_aj - l_ai),
                        abs(ch_j.l_c - ch_i.l_c),
                    )
                    k_max = min(
                        l_aj + l_ai,
                        ch_j.l_c + ch_i.l_c,
                    )
                    for k in range(k_min, k_max + 1):
                        rm_active = _electron_contribution(
                            l_aj, l_ai,
                            ch_j.target.l, ch_i.target.l,
                            l_bj, k,
                        )
                        if rm_active == 0.0:
                            continue
                        coef = direct_coefficient(
                            l_aj, ch_j.l_c, l_ai, ch_i.l_c,
                            L_total, k, rm_active,
                        )
                        if coef == 0.0:
                            continue
                        y = y_k_two_electron(
                            ch_j.target.label, a_idx,
                            ch_i.target.label, a_idx, k,
                        )
                        V_loc = coef * y
                        H[sl_ji] += (Bw * V_loc[:, None]).T @ B

            # ----------------------------------------------------------
            # Exchange (non-local) couplings.
            # ----------------------------------------------------------
            if ch_j.target.n_electrons == 1 and ch_i.target.n_electrons == 1:
                x_terms = exchange_terms(
                    l_tj=ch_j.target.l,
                    l_cj=ch_j.l_c,
                    l_ti=ch_i.target.l,
                    l_ci=ch_i.l_c,
                    L=L_total,
                    S=S_total,
                )
                for term in x_terms:
                    # [W F_i](r) = v_k^x P_{t_i}(r) * y_k(P_{t_j}, F_i)(r)
                    # Matrix element <B_alpha | W | B_beta> with F_i = B_beta:
                    #   M_{alpha,beta} = v_k * sum_q w_q B_alpha(r_q) P_{t_i}(r_q)
                    #                                  y_k^{t_j, B_beta}(r_q)
                    Pt_i = P_target_table[ch_i.target.label]
                    y_tj_B = y_k_target_bsp(ch_j.target.label, term.k)
                    left = Bw * Pt_i[:, None]              # (n_q, n_eff)
                    H[sl_ji] += term.coefficient * (left.T @ y_tj_B)
            elif has_composite:
                # Composite-target exchange requires three-electron
                # recoupling; it is not yet implemented.
                pass

    # ------------------------------------------------------------------
    # NON-ORTHOGONAL cross blocks (pair_targets vs s-channels), 1S all-s:
    # bilinear forms of h = T + V_nuc between grid-sampled targets and
    # the spline columns.  (S exchange term and the angular factor are
    # +1 for spatial-symmetric singlet products.)
    # ------------------------------------------------------------------
    if pair_targets:
        s_channels = [(idx, ch) for idx, ch in enumerate(channels)
                      if ch.target.l == 0 and ch.l_c == 0]
        grids = {}
        for idx, ch in s_channels:
            g = np.asarray(ch.target.P(r_q))
            assert np.all(np.isfinite(g)), ch.target.label
            grids[idx] = g

        def h_bilinear(f_g, g_g):
            """<f|h|g> row/column pieces vs splines or scalars."""
            df = np.gradient(f_g, r_q)
            dg = np.gradient(g_g, r_q)
            t_scalar = 0.5 * float(np.sum(w_q * df * dg))
            v_scalar = -Z * float(np.sum(w_q * f_g * g_g * inv_r))
            return t_scalar + v_scalar

        def h_row(f_g):
            """<f|h|B_b> for every spline column b."""
            df = np.gradient(f_g, r_q)
            t_row = 0.5 * (dB.T @ (w_q * df))
            v_row = -Z * (B.T @ (w_q * f_g * inv_r))
            return t_row + v_row

        def ov_row(f_g):
            return B.T @ (w_q * f_g)          # <B_b|f>

        for x_idx, ch_x in s_channels:
            phi_g = grids[x_idx]
            phB = ov_row(phi_g)
            hphiB = h_row(phi_g)
            for i_idx, ch_i in s_channels:
                if i_idx == x_idx:
                    continue
                Phi_g = grids[i_idx]
                ov = float(np.sum(w_q * phi_g * Phi_g))
                if abs(ov) < 1e-12 and ch_x.target.label == \
                        ch_i.target.label:
                    continue
                PhiB = ov_row(Phi_g)
                hPhiB = h_row(Phi_g)
                hphiPhi = h_bilinear(phi_g, Phi_g)
                cross_S = ov * S_one + np.outer(phB, PhiB)
                sx = slice(x_idx * n_eff, (x_idx + 1) * n_eff)
                si = slice(i_idx * n_eff, (i_idx + 1) * n_eff)
                S[sx, si] += cross_S
                H[sx, si] += (ov * (T_one + V_nuc_one)
                              + hphiPhi * S_one
                              + np.outer(hphiB, PhiB)
                              + np.outer(phB, hPhiB)
                              + ch_i.target.energy_au * cross_S)

    # Symmetrise to undo accumulated round-off before eigensolve.
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)

    eigvals, eigvecs = eigh(H, S)

    # Surface amplitudes at r = a_au: each B-spline's value at the boundary.
    B_at_a_full, _ = _bspline_design(np.array([a_au]), knots, degree)
    b_at_a = B_at_a_full[0, 1:]    # (n_eff,)

    # eigvecs shape (N, n_states); reshape per channel to (n_ch, n_eff, n_states)
    n_states = eigvecs.shape[1]
    psi = eigvecs.reshape(n_ch, n_eff, n_states)
    # Surface amp w_n^i = sum_alpha psi[i, alpha, n] * b_at_a[alpha]
    w = np.einsum("ian,a->ni", psi, b_at_a)

    return CCInnerSolution(
        eigenvalues_au=eigvals,
        surface_amplitudes=w,
        a_au=a_au,
        n_channels=n_ch,
        n_basis_effective=n_eff,
        k_spline_order=k_spline_order,
        H_au=H,
        S_au=S,
    )
