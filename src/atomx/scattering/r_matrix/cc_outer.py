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
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from atomx.scattering.r_matrix.channel_table import Channel
from atomx.scattering.r_matrix.coulomb_fn import coulomb_FG


@dataclass
class OuterMatching:
    """Result of one outer-region matching at a single energy."""

    energy_au: float
    open_indices: np.ndarray             # int array, indices into channels list
    K_matrix: np.ndarray                  # (n_open, n_open) real symmetric
    eigenphases: np.ndarray              # (n_open,) sorted; arctan eigenvalues
    eigenphase_sum: float                 # sum of eigenphases (mod pi handled)
    R_open: np.ndarray


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


def _closed_channel_log_derivative(
    l_c: int, kappa_c: float, a_au: float
) -> float:
    """Log-derivative of the decaying closed-channel Whittaker at ``r = a``.

    For the residual +1 Coulomb tail the decaying solution is
    ``W_{1/kappa_c, l_c + 1/2}(2 kappa_c r)``. We integrate the radial
    closed-channel Schrödinger equation inward from ``r_far >> a``
    (anchored by the exponential asymptotic ``u ~ r^{1/kappa_c}
    exp(-kappa_c r)``), then take the ratio ``u'(a) / u(a)``.
    """
    from atomx.scattering.r_matrix.coulomb_fn import _numerov_outward
    # Inward integration: change variable s = r_far - r so the grid is
    # uniformly increasing in s while r decreases. Numerov in s for the
    # equation u_ss = +d²u/dr² (sign flips because (d/ds)² = (d/dr)²).
    r_far = max(a_au + 60.0 / kappa_c, a_au * 2.5)
    n_steps = max(int((r_far - a_au) / 0.005) + 2, 800)
    s_grid = np.linspace(0.0, r_far - a_au, n_steps)
    # u(r) = u(r_far - s). Initial conditions at s = 0 (r = r_far):
    # u_far = r_far^{1/kappa_c} exp(-kappa_c r_far),
    # du/dr at r_far = (1/kappa_c r_far - kappa_c) * u_far
    # du/ds = -du/dr
    u_far = r_far ** (1.0 / kappa_c) * math.exp(-kappa_c * r_far)
    du_dr_far = (1.0 / (kappa_c * r_far) - kappa_c) * u_far
    du_ds_far = -du_dr_far

    def f_func(s):
        r = r_far - s
        return (
            l_c * (l_c + 1) / (r * r) - 2.0 / r + kappa_c * kappa_c
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
    """
    open_mask = np.array(
        [energy_au > ch.threshold_au for ch in channels], dtype=bool
    )
    open_idx = np.where(open_mask)[0]
    closed_idx = np.where(~open_mask)[0]
    n_open = open_idx.size
    n_closed = closed_idx.size
    if n_open == 0:
        return OuterMatching(
            energy_au=energy_au,
            open_indices=open_idx,
            K_matrix=np.empty((0, 0)),
            eigenphases=np.empty(0),
            eigenphase_sum=0.0,
            R_open=np.empty((0, 0)),
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
            beta[j] = _closed_channel_log_derivative(ch.l_c, kappa, a_au)
        B = np.diag(beta)
        # u_c'(a) = beta u_c(a) closes the closed-channel block:
        #   u_c(a) = (I - a R_cc B)^{-1} a R_co u_o'(a)
        # Effective R-matrix for open block:
        M = np.eye(n_closed) - a_au * R_cc @ B
        Y = np.linalg.solve(M, R_co)
        R_open = R_oo + a_au * R_oc @ B @ Y
    else:
        R_open = R_oo

    F_diag = np.zeros(n_open)
    G_diag = np.zeros(n_open)
    Fp_diag = np.zeros(n_open)
    Gp_diag = np.zeros(n_open)
    k_diag = np.zeros(n_open)
    for j, i_ch in enumerate(open_idx):
        ch = channels[i_ch]
        E_k = energy_au - ch.threshold_au
        k = np.sqrt(2.0 * E_k)
        eta = -1.0 / k
        rho_a = k * a_au
        F, dF, G, dG = coulomb_FG(ch.l_c, eta, rho_a)
        F_diag[j] = F
        G_diag[j] = G
        Fp_diag[j] = dF
        Gp_diag[j] = dG
        k_diag[j] = k

    aK = a_au * k_diag                          # diag(a k_i)

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

    return OuterMatching(
        energy_au=energy_au,
        open_indices=open_idx,
        K_matrix=K,
        eigenphases=eigenphases,
        eigenphase_sum=eigenphase_sum,
        R_open=R_open,
    )


def eigenphase_sum_unwrapped(
    energies_au: np.ndarray, eigenphase_sums: np.ndarray
) -> np.ndarray:
    """Unwrap pi-jumps in the eigenphase sum across an energy scan."""
    return np.unwrap(eigenphase_sums, period=np.pi)
