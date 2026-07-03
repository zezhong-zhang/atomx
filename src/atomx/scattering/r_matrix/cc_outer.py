"""Multi-channel outer-region matching: R-matrix -> K-matrix -> eigenphases.

For a total energy ``E`` (Hartree) the channels split into open
(``E > epsilon_i``) and closed (``E < epsilon_i``). The closed channels
decay exponentially over the outer region; at the Burke-1971 boundary
radius ``a ~ 25 a_0`` they are negligible at ``r = a`` and we project
the inner-region R-matrix onto the open-channel sub-block before
matching to Coulomb wave functions.

For the open block, with Bloch constant ``b = 0`` the matching condition
at ``r = a`` is

    u(a) = a R_oo(E) u'(a)

with the channel-radial functions taking the asymptotic form
``u_i(r) = A_i F_l_i(eta_i, k_i r) + B_i G_l_i(eta_i, k_i r)``
(``k_i = sqrt(2 (E - epsilon_i))``, ``eta_i = -1 / k_i`` for an
electron in the residual +1 ion). Imposing ``A = I, B = K`` yields the
real symmetric K-matrix

    K = ( F - R_oo  D_k  F' )^{-1} ( R_oo D_k G' - G )

where ``F, G, F', G'`` are diagonal channel-wise and
``D_k = diag(a * k_i)``. Eigenphases ``delta_alpha = arctan(K_alpha)``
follow by diagonalising ``K``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from atomx.scattering.r_matrix.angular_cc import direct_terms
from atomx.scattering.r_matrix.channel_table import Channel
from atomx.scattering.r_matrix.coulomb_fn import coulomb_FG
from atomx.scattering.two_electron.angular import (
    _electron_contribution,
    direct_coefficient,
)


@dataclass
class OuterMatching:
    """Result of one outer-region matching at a single energy."""

    energy_au: float
    open_indices: np.ndarray             # int array, indices into channels list
    K_matrix: np.ndarray                  # (n_open, n_open) real symmetric
    eigenphases: np.ndarray              # (n_open,) sorted; arctan eigenvalues
    eigenphase_sum: float                 # sum of eigenphases (mod pi handled)
    R_open: np.ndarray
    k_diag: np.ndarray = field(default_factory=lambda: np.empty(0))
    S_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=complex))
    T_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=complex))
    cross_section_matrix_a0: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    collision_strength_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def channel_R_matrix(
    energy_au: float,
    eigenvalues_E_n_au: np.ndarray,
    surface_amplitudes: np.ndarray,
    a_au: float,
) -> np.ndarray:
    """Multi-channel R-matrix ``R_ji(E)`` at the boundary.

    Parameters
    ----------
    energy_au :
        Total energy ``E`` (Hartree).
    eigenvalues_E_n_au :
        Inner-region eigenvalues, shape ``(n_states,)``.
    surface_amplitudes :
        Surface amplitudes ``w_n^i``, shape ``(n_states, n_channels)``.
    a_au :
        Boundary radius (Bohr).
    """
    diff = eigenvalues_E_n_au - energy_au       # (n_states,)
    # Skip resonance-poles: small denominators get clipped; the pole at
    # an embedded discrete eigenpair is the resonance itself.
    safe = np.where(np.abs(diff) > 1e-14, diff, np.copysign(1e-14, diff))
    weights = surface_amplitudes / safe[:, None]   # (n_states, n_ch)
    R = (1.0 / (2.0 * a_au)) * (surface_amplitudes.T @ weights)
    return 0.5 * (R + R.T)


def S_matrix_from_K(K: np.ndarray) -> np.ndarray:
    """Unitary S-matrix from real symmetric K-matrix.

    ``S = (I + iK) (I - iK)^{-1}``; for real symmetric K the result is
    symmetric and unitary.
    """
    K = np.asarray(K, dtype=complex)
    n = K.shape[0]
    I = np.eye(n, dtype=complex)
    return np.linalg.solve(I - 1j * K, I + 1j * K)


def T_matrix_from_K(K: np.ndarray) -> np.ndarray:
    """Transition matrix ``T = S - I`` from the K-matrix."""
    return S_matrix_from_K(K) - np.eye(K.shape[0], dtype=complex)


def _partial_wave_cross_section_factor(
    channel: Channel, L_total: int, S_total: float, target_spin: float
) -> float:
    """Angular/statistical factor for the σ_{i→j}^{LSπ} partial wave.

    ``σ = (π / k_i²) * factor * |T_ij|²`` in atomic units.

    For electron + target (L_t, S_t) → (L_t', S_t') with projectile
    partial wave l_i, the LS-coupling partial-wave cross section is

        σ_{ij}^{LSπ} = (π / k_i²) * (2L+1)(2S+1)
                       / [(2L_t+1)(2S_t+1)(2l_i+1)] * |T_ij^{LSπ}|²

    (Burke 2011, eq. 5.2.8).  The target spin is assumed the same for
    all channels (e.g. S_t = 1/2 for a one-electron target).
    """
    two_L_t = 2 * channel.target.l + 1
    # For composite targets the target spin is stored on the channel; the
    # legacy target_spin argument is ignored.
    two_S_t = int(round(2 * channel.target.S_t + 1))
    two_l_i = 2 * channel.l_c + 1
    return (
        (2 * L_total + 1) * (2 * S_total + 1)
        / (two_L_t * two_S_t * two_l_i)
    )


def scattering_matrices_and_cross_sections(
    channels: Sequence[Channel],
    open_indices: np.ndarray,
    k_diag: np.ndarray,
    K: np.ndarray,
    L_total: int,
    S_total: float,
    target_spin: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build S, T, cross-section and collision-strength matrices.

    Returns
    -------
    S, T, sigma_a0, Omega
        Matrices are ``(n_open, n_open)``.  ``sigma_a0`` is in units of
        ``a₀²``; ``Omega`` is the dimensionless collision strength.
    """
    n_open = K.shape[0]
    S = S_matrix_from_K(K)
    T = S - np.eye(n_open, dtype=complex)
    sigma = np.zeros((n_open, n_open), dtype=float)
    omega = np.zeros((n_open, n_open), dtype=float)

    for i_idx, i_ch_idx in enumerate(open_indices):
        ch_i = channels[i_ch_idx]
        k_i = k_diag[i_idx]
        if k_i <= 0.0:
            continue
        factor = _partial_wave_cross_section_factor(
            ch_i, L_total, S_total, target_spin
        )
        prefactor = math.pi / (k_i * k_i) * factor
        for j_idx in range(n_open):
            Tij2 = abs(T[i_idx, j_idx]) ** 2
            sigma[i_idx, j_idx] = prefactor * Tij2
            # Collision strength contribution for this partial wave:
            # Ω_{ij}^{LSπ} = (2L+1)(2S+1) / (2l_i+1) |T_{ij}|².
            two_l_i = 2 * ch_i.l_c + 1
            omega[i_idx, j_idx] = (
                (2 * L_total + 1) * (2 * S_total + 1) / two_l_i * Tij2
            )
    return S, T, sigma, omega


def _target_multipole_moment(
    P_a: np.ndarray,
    P_b: np.ndarray,
    k: int,
    r_grid: np.ndarray,
) -> float:
    """Moment ``∫ P_a(r) P_b(r) r^k dr`` used for the asymptotic tail."""
    return float(np.trapezoid(P_a * P_b * (r_grid ** k), r_grid))


def _build_long_range_coupling_terms(
    channels: Sequence[Channel],
    open_indices: np.ndarray,
    L_total: int,
    r_grid: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    """Precompute asymptotic direct-multipole couplings for each open pair.

    Returns ``{(i_pos, j_pos): [(k, coeff * moment_k), ...]}`` so the
    outer-region potential matrix at radius ``r`` is
    ``W_ij(r) = 2 * Σ (coeff * moment_k) / r^{k+1}``.
    Diagonal k=0 monopole terms are omitted because the net asymptotic
    Coulomb tail is already fixed to ``-Z_res/r``.
    """
    P_cache = {ch.target.label: ch.target.P(r_grid) for ch in channels}

    # Per-electron radial samples for composite (two-electron) targets.
    target_orbitals: dict[tuple[str, int], np.ndarray] = {}
    for ch in channels:
        term = ch.target.two_electron_term
        if term is not None:
            for idx, P in enumerate((term.P1, term.P2), start=1):
                key = (ch.target.label, idx)
                if key not in target_orbitals:
                    target_orbitals[key] = np.asarray(P(r_grid))

    moment_cache: dict[tuple[str, str, int], float] = {}
    two_e_moment_cache: dict[tuple[str, int, str, int, int], float] = {}

    def _moment(label_a: str, label_b: str, k: int) -> float:
        key = (label_a, label_b, k) if label_a <= label_b else (label_b, label_a, k)
        if key not in moment_cache:
            moment_cache[key] = _target_multipole_moment(
                P_cache[key[0]], P_cache[key[1]], k, r_grid
            )
        return moment_cache[key]

    def _two_e_moment(label_a: str, a: int, label_b: str, b: int, k: int) -> float:
        if (label_a, a) < (label_b, b):
            key = (label_a, a, label_b, b, k)
        else:
            key = (label_b, b, label_a, a, k)
        if key not in two_e_moment_cache:
            f_q = target_orbitals[(label_a, a)] * target_orbitals[(label_b, b)]
            two_e_moment_cache[key] = float(
                np.trapezoid(f_q * (r_grid ** k), r_grid)
            )
        return two_e_moment_cache[key]

    couplings: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for i_pos, i_ch_idx in enumerate(open_indices):
        ch_i = channels[i_ch_idx]
        for j_pos, j_ch_idx in enumerate(open_indices):
            ch_j = channels[j_ch_idx]
            entries: list[tuple[int, float]] = []

            if ch_j.target.n_electrons == 1:
                terms = direct_terms(
                    ch_j.target.l,
                    ch_j.l_c,
                    ch_i.target.l,
                    ch_i.l_c,
                    L_total,
                )
                for term in terms:
                    # Skip the diagonal monopole: already in the Coulomb tail.
                    if i_pos == j_pos and term.k == 0:
                        continue
                    moment = _moment(ch_j.target.label, ch_i.target.label, term.k)
                    entries.append((term.k, term.coefficient * moment))
            else:
                term_j = ch_j.target.two_electron_term
                term_i = ch_i.target.two_electron_term
                if term_j is None or term_i is None:
                    raise ValueError(
                        "Composite target missing two_electron_term"
                    )
                for a_idx, (l_aj, l_ai) in enumerate(
                    zip((term_j.l1, term_j.l2), (term_i.l1, term_i.l2)),
                    start=1,
                ):
                    b_idx = 2 if a_idx == 1 else 1
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
                        if i_pos == j_pos and k == 0:
                            continue
                        moment = _two_e_moment(
                            ch_j.target.label, a_idx,
                            ch_i.target.label, a_idx, k,
                        )
                        entries.append((k, coef * moment))

            if entries:
                couplings[(i_pos, j_pos)] = entries
    return couplings


def _outer_region_W_matrix(
    r: float,
    channels: Sequence[Channel],
    open_indices: np.ndarray,
    energy_au: float,
    couplings: dict[tuple[int, int], list[tuple[int, float]]],
    Z_res: float = 1.0,
) -> np.ndarray:
    """Local + asymptotic-coupling matrix ``W(r)`` for the log-derivative ODE.

    The radial equation is ``u'' = W(r) u`` with ``W = 2(V - E)``; the
    net asymptotic Coulomb tail is ``-Z_res/r`` where ``Z_res`` is the
    residual charge of the target ion/atom.
    """
    n = open_indices.size
    W = np.zeros((n, n), dtype=float)
    for pos, ch_idx in enumerate(open_indices):
        ch = channels[ch_idx]
        W[pos, pos] = (
            ch.l_c * (ch.l_c + 1) / (r * r)
            - 2.0 * Z_res / r
            + 2.0 * (ch.threshold_au - energy_au)
        )
    for (i_pos, j_pos), entries in couplings.items():
        val = 0.0
        for k, coeff_moment in entries:
            val += coeff_moment / (r ** (k + 1))
        W[i_pos, j_pos] += 2.0 * val
    return W


def propagate_R_matrix_long_range(
    R_open: np.ndarray,
    channels: Sequence[Channel],
    open_indices: np.ndarray,
    energy_au: float,
    a_au: float,
    L_total: int,
    R_asym: float | None = None,
    Z_res: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Propagate the open-channel R-matrix from ``a`` to ``R_asym``.

    We integrate the linear close-coupling equations
    ``U' = V``, ``V' = W(r) U`` for the regular solution matrix with the
    R-matrix boundary condition at ``r = a``:
    ``U(a) = a R_open`` and ``V(a) = I``.  Long-range direct multipoles
    between target states are included via their asymptotic
    ``moment / r^{k+1}`` tails.  At ``R_asym`` the propagated R-matrix is
    ``R(R_asym) = U(R_asym) V(R_asym)^{-1}`` and is matched to Coulomb
    waves.  The linear form is stable even when the R-matrix itself has
    negative eigenvalues, whereas the Riccati form has movable poles.
    """
    n = R_open.shape[0]
    if R_asym is None:
        R_asym = max(a_au * 3.0, 80.0)

    # Grid for target-state multipole moments; the inner region [0, a]
    # captures essentially all of the target charge for He-like ions.
    n_grid = 300
    r_grid = np.linspace(0.0, a_au, n_grid)
    r_grid[0] = max(r_grid[0], 1e-8)
    couplings = _build_long_range_coupling_terms(
        channels, open_indices, L_total, r_grid
    )

    # Fixed-step RK4 for the linear system.  Step size is set to ~30
    # points per shortest open-channel wavelength.
    k_max = 0.0
    for idx in open_indices:
        E_k = energy_au - channels[idx].threshold_au
        if E_k > 0.0:
            k_max = max(k_max, np.sqrt(2.0 * E_k))
    if k_max <= 0.0:
        k_max = 1.0
    n_steps = max(int((R_asym - a_au) * k_max * 30.0) + 1, 500)
    h = (R_asym - a_au) / n_steps

    U = a_au * R_open.copy()
    V = np.eye(n)
    r = a_au

    def _step(U: np.ndarray, V: np.ndarray, r0: float, h_step: float) -> tuple[np.ndarray, np.ndarray]:
        W0 = _outer_region_W_matrix(r0, channels, open_indices, energy_au, couplings, Z_res=Z_res)
        k1U = V
        k1V = W0 @ U

        Wm = _outer_region_W_matrix(r0 + 0.5 * h_step, channels, open_indices, energy_au, couplings, Z_res=Z_res)
        U2 = U + 0.5 * h_step * k1U
        V2 = V + 0.5 * h_step * k1V
        k2U = V2
        k2V = Wm @ U2

        U3 = U + 0.5 * h_step * k2U
        V3 = V + 0.5 * h_step * k2V
        k3U = V3
        k3V = Wm @ U3

        W1 = _outer_region_W_matrix(r0 + h_step, channels, open_indices, energy_au, couplings, Z_res=Z_res)
        U4 = U + h_step * k3U
        V4 = V + h_step * k3V
        k4U = V4
        k4V = W1 @ U4

        U_new = U + (h_step / 6.0) * (k1U + 2.0 * k2U + 2.0 * k3U + k4U)
        V_new = V + (h_step / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
        return U_new, V_new

    for _ in range(n_steps):
        U, V = _step(U, V, r, h)
        r += h

    # The propagated solution satisfies ``u(R) = U c`` and ``u'(R) = V c``.
    # The dimensionless R-matrix is defined by ``u(R) = R \mathcal{R} u'(R)``,
    # hence ``\mathcal{R} = U V^{-1} / R``.
    R_at_R = (U @ np.linalg.inv(V)) / R_asym
    # Keep the R-matrix manifestly symmetric to avoid K-matrix drift.
    R_at_R = 0.5 * (R_at_R + R_at_R.T)
    return R_at_R, R_asym


def _closed_channel_log_derivative(
    l_c: int, kappa_c: float, a_au: float, Z_res: float = 1.0
) -> float:
    """Log-derivative of the decaying closed-channel solution at ``r = a``.

    For a residual Coulomb charge ``Z_res`` the decaying solution at large
    ``r`` is proportional to ``r^{Z_res/kappa_c} exp(-kappa_c r)``.  We
    integrate the radial closed-channel Schrödinger equation inward from
    ``r_far >> a`` and take the ratio ``u'(a) / u(a)``.
    """
    from atomx.scattering.r_matrix.coulomb_fn import _numerov_outward
    # Inward integration: change variable s = r_far - r so the grid is
    # uniformly increasing in s while r decreases. Numerov in s for the
    # equation u_ss = +d²u/dr² (sign flips because (d/ds)² = (d/dr)²).
    r_far = max(a_au + 60.0 / kappa_c, a_au * 2.5)
    n_steps = max(int((r_far - a_au) / 0.005) + 2, 800)
    s_grid = np.linspace(0.0, r_far - a_au, n_steps)
    # u(r) = u(r_far - s). Initial conditions at s = 0 (r = r_far):
    # u_far = r_far^{Z_res/kappa_c} exp(-kappa_c r_far),
    # du/dr at r_far = (Z_res/(kappa_c r_far) - kappa_c) * u_far
    # du/ds = -du/dr
    u_far = (r_far ** (Z_res / kappa_c)) * math.exp(-kappa_c * r_far)
    du_dr_far = (Z_res / (kappa_c * r_far) - kappa_c) * u_far
    du_ds_far = -du_dr_far

    def f_func(s):
        r = r_far - s
        return (
            l_c * (l_c + 1) / (r * r)
            - 2.0 * Z_res / r
            + kappa_c * kappa_c
        )

    u_grid = _numerov_outward(f_func, s_grid, u_far, du_ds_far)
    h = s_grid[1] - s_grid[0]
    u_at_a = u_grid[-1]
    # du/ds at s_end ≈ (u[-1] - u[-3]) / (2h); du/dr = -du/ds.
    du_ds_a = (u_grid[-1] - u_grid[-3]) / (2.0 * h)
    du_dr_a = -du_ds_a
    return du_dr_a / u_at_a


def match_K_matrix(
    energy_au: float,
    channels: Sequence[Channel],
    R_full: np.ndarray,
    a_au: float,
    closed_channel_treatment: str = "log_derivative",
    *,
    L_total: int = 0,
    S_total: float = 0,
    target_spin: float = 0.5,
    compute_cross_sections: bool = True,
    propagate_long_range: bool = False,
    R_asym: float | None = None,
) -> OuterMatching:
    """Match R-matrix at the boundary to open-channel Coulomb wave functions.

    ``closed_channel_treatment``:

    * ``"log_derivative"``: closed channels eliminated via the decaying-
      Whittaker boundary condition ``u'_c(a) = beta_c u_c(a)`` with
      ``beta_c`` from inward Numerov of the radial Coulomb equation.
      The closed-channel block is then folded into an effective R-matrix
      for the open block:
      ``R_eff = R_oo + R_oc beta (I - a R_cc beta)^{-1} a R_co``.
      This carries the resonance signature (closed channels are the
      bound configurations that mix into the open continuum).
    * ``"drop"``: legacy / debug path that throws closed channels away
      entirely. Used by single-open-channel sanity checks.

    ``propagate_long_range``:

    * If ``True``, the open-channel R-matrix is first propagated from the
      R-matrix boundary ``r = a`` out to a large asymptotic radius
      ``R_asym`` using the log-derivative propagator, including the
      long-range target multipole couplings.  The Coulomb matching is
      then performed at ``R_asym``.  This is essential for inelastic
      excitation cross sections because the target dipole terms only
      become weak far from the nucleus.
    * If ``False`` (default), matching occurs directly at ``r = a``.
    """
    open_mask = np.array(
        [energy_au > ch.threshold_au for ch in channels], dtype=bool
    )
    open_idx = np.where(open_mask)[0]
    closed_idx = np.where(~open_mask)[0]
    n_open = open_idx.size
    n_closed = closed_idx.size

    residual_charges = {float(ch.target.residual_charge) for ch in channels}
    if len(residual_charges) != 1:
        raise ValueError(
            "All channels must share the same target residual charge, "
            f"got {residual_charges}"
        )
    Z_res = residual_charges.pop()

    if n_open == 0:
        return OuterMatching(
            energy_au=energy_au,
            open_indices=open_idx,
            K_matrix=np.empty((0, 0)),
            eigenphases=np.empty(0),
            eigenphase_sum=0.0,
            R_open=np.empty((0, 0)),
            k_diag=np.empty(0),
        )

    R_oo = R_full[np.ix_(open_idx, open_idx)]
    if n_closed > 0 and closed_channel_treatment == "log_derivative":
        R_oc = R_full[np.ix_(open_idx, closed_idx)]
        R_co = R_full[np.ix_(closed_idx, open_idx)]
        R_cc = R_full[np.ix_(closed_idx, closed_idx)]
        beta = np.zeros(n_closed)
        for j, i_ch in enumerate(closed_idx):
            ch = channels[i_ch]
            kappa = np.sqrt(2.0 * (ch.threshold_au - energy_au))
            beta[j] = _closed_channel_log_derivative(ch.l_c, kappa, a_au, Z_res=Z_res)
        B = np.diag(beta)
        # u_c'(a) = beta u_c(a) closes the closed-channel block:
        #   u_c(a) = (I - a R_cc B)^{-1} a R_co u_o'(a)
        # Effective R-matrix for open block:
        M = np.eye(n_closed) - a_au * R_cc @ B
        Y = np.linalg.solve(M, R_co)
        R_open = R_oo + a_au * R_oc @ B @ Y
    else:
        R_open = R_oo

    r_match = a_au
    if propagate_long_range:
        R_open, r_match = propagate_R_matrix_long_range(
            R_open,
            channels,
            open_idx,
            energy_au,
            a_au,
            L_total,
            R_asym=R_asym,
            Z_res=Z_res,
        )

    F_diag = np.zeros(n_open)
    G_diag = np.zeros(n_open)
    Fp_diag = np.zeros(n_open)
    Gp_diag = np.zeros(n_open)
    k_diag = np.zeros(n_open)
    for j, i_ch in enumerate(open_idx):
        ch = channels[i_ch]
        E_k = energy_au - ch.threshold_au
        k = np.sqrt(2.0 * E_k)
        eta = -Z_res / k
        rho = k * r_match
        F, dF, G, dG = coulomb_FG(ch.l_c, eta, rho)
        F_diag[j] = F
        G_diag[j] = G
        Fp_diag[j] = dF
        Gp_diag[j] = dG
        k_diag[j] = k

    aK = r_match * k_diag                       # diag(r_match k_i)

    # K = ( F - R_oo diag(a k) F' )^{-1} ( R_oo diag(a k) G' - G )
    F_mat = np.diag(F_diag)
    G_mat = np.diag(G_diag)
    Fp_mat = np.diag(Fp_diag)
    Gp_mat = np.diag(Gp_diag)
    aK_mat = np.diag(aK)

    M_left = F_mat - R_open @ aK_mat @ Fp_mat
    M_right = R_open @ aK_mat @ Gp_mat - G_mat

    K = np.linalg.solve(M_left, M_right)
    K = 0.5 * (K + K.T)

    eigvals_K = np.linalg.eigvalsh(K)
    eigenphases = np.arctan(eigvals_K)
    eigenphase_sum = float(eigenphases.sum())

    S_mat = np.empty((0, 0), dtype=complex)
    T_mat = np.empty((0, 0), dtype=complex)
    sigma = np.empty((0, 0))
    omega = np.empty((0, 0))
    if compute_cross_sections and n_open > 0:
        S_mat, T_mat, sigma, omega = scattering_matrices_and_cross_sections(
            channels=channels,
            open_indices=open_idx,
            k_diag=k_diag,
            K=K,
            L_total=L_total,
            S_total=S_total,
            target_spin=target_spin,
        )

    return OuterMatching(
        energy_au=energy_au,
        open_indices=open_idx,
        K_matrix=K,
        eigenphases=eigenphases,
        eigenphase_sum=eigenphase_sum,
        R_open=R_open,
        k_diag=k_diag,
        S_matrix=S_mat,
        T_matrix=T_mat,
        cross_section_matrix_a0=sigma,
        collision_strength_matrix=omega,
    )


def eigenphase_sum_unwrapped(
    energies_au: np.ndarray, eigenphase_sums: np.ndarray
) -> np.ndarray:
    """Unwrap pi-jumps in the eigenphase sum across an energy scan."""
    return np.unwrap(eigenphase_sums, period=np.pi)
