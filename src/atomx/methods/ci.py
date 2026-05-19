"""Configuration-interaction diagonalisation on frozen orbitals.

This is the *CI half* of an MCDHF calculation — build the CI matrix
for a list of configuration-state functions (CSFs) that share a
common orbital basis, then diagonalise to obtain state energies and
mixing coefficients.  Call it on converged DHF orbitals to get a
pure-CI result (no orbital re-optimisation); MCDHF's outer loop
composes this with the Guerra average-level orbital-opt iteration.

Decoupling the CI step from orbital optimisation makes two things
possible:

1. **Pure CI on converged DHF orbitals** — much cheaper than a full
   MCDHF, useful for trial CSF spaces and for benchmarking the
   orbital-optimisation gain.
2. **Perturbative correlation methods** (MBPT, CC) built on the same
   infrastructure, calling :func:`ci_solve` with a truncated CSF
   list corresponding to their amplitude sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy import linalg


@dataclass
class CIResult:
    """Outcome of a CI diagonalisation.

    Attributes
    ----------
    state_energies
        Eigenvalues of the CI matrix, sorted in ascending order.
    mixing_coefficients
        CSF mixing coefficients, shape ``(n_csf, n_states)``;
        ``mixing[:, k]`` are the coefficients of the k-th state.
    ground_energy
        Shortcut for ``state_energies[0]``.
    provenance
        Free-form notes on the basis, CSF count, and whether Breit/QED
        were folded into the CI matrix.
    """

    state_energies: np.ndarray
    mixing_coefficients: np.ndarray
    ground_energy: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state_energies.size:
            self.ground_energy = float(self.state_energies[0])


def ci_solve(
    csfs,
    orbitals_dict: Dict[tuple, tuple],
    orbital_basis,
    bsp_data: Dict[str, Any],
    Z: int,
    ns: int,
    *,
    n_states: int | None = None,
    solver: str = "auto",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> CIResult:
    """Build the CI matrix and diagonalise it.

    Parameters
    ----------
    csfs
        List of :class:`~atomx.methods.mcdhf.config.CSF` objects.
    orbitals_dict
        ``{(n, kappa): (P_coeffs, Q_coeffs)}`` — converged radial
        orbitals on the shared B-spline basis.
    orbital_basis
        Shared orbital basis descriptor.
    bsp_data
        B-spline workspace (grid, weights, basis).
    Z
        Nuclear charge.
    ns
        B-spline dimension.
    n_states
        Number of low-lying eigenpairs to compute.  ``None`` (default)
        diagonalises the full CI matrix; otherwise the iterative
        solver is used to extract the lowest ``n_states`` only.
    solver
        ``"dense"`` (LAPACK ``eigh``), ``"davidson"`` (Davidson-Liu
        block-iterative for large CI spaces), or ``"auto"`` (default
        — picks dense if n_csf ≤ 1500, Davidson otherwise).
    tol
        Convergence tolerance on the residual norm (Davidson only).
    max_iter
        Hard iteration cap (Davidson only).

    Returns
    -------
    :class:`CIResult`
    """
    from atomx.methods.mcdhf.matrix import build_ci_matrix

    H_ci = build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns)
    n = H_ci.shape[0]
    # FAC diagonalises independent J/pi symmetry blocks.  Keep the dense
    # matrix representation for simplicity, but remove numerically generated
    # cross-block couplings so CI eigenvectors cannot mix incompatible CSFs.
    for i in range(n):
        Ji = getattr(csfs[i], "total_J2", 0)
        pi = getattr(csfs[i], "parity", 1)
        for j in range(i + 1, n):
            if Ji != getattr(csfs[j], "total_J2", 0) or pi != getattr(csfs[j], "parity", 1):
                H_ci[i, j] = 0.0
                H_ci[j, i] = 0.0

    if solver == "auto":
        solver = "dense" if (n <= 1500 or n_states is None) else "davidson"
    if solver not in ("dense", "davidson"):
        raise ValueError(
            f"Unknown solver {solver!r}; expected 'dense', 'davidson', "
            f"or 'auto'."
        )

    # Block-diagonalise by (total_J2, parity) when the CSF list carries
    # those labels (jj-CSFs from enumerate_jj_csfs). Total angular
    # momentum and parity are exact symmetries of the Dirac–Coulomb
    # Hamiltonian, so off-block matrix elements must vanish; zeroing
    # them eliminates spurious cross-J mixing introduced by the
    # average-of-jj-configurations matrix-element formulas.
    blocks = _group_csfs_by_J_parity(csfs)
    if blocks is not None and len(blocks) > 1:
        state_energies, mixing = _solve_blocked(H_ci, blocks)
        if n_states is not None:
            state_energies = state_energies[:n_states]
            mixing = mixing[:, :n_states]
        provenance = {
            "n_csf": n, "n_orbitals": len(orbitals_dict),
            "solver": "dense-blocked",
            "n_blocks": len(blocks),
            "block_sizes": [len(idx) for _, idx in blocks],
        }
    elif solver == "dense":
        state_energies, mixing = linalg.eigh(H_ci)
        if n_states is not None:
            state_energies = state_energies[:n_states]
            mixing = mixing[:, :n_states]
        provenance = {"n_csf": n, "n_orbitals": len(orbitals_dict),
                       "solver": "dense"}
    else:
        k = n_states if n_states is not None else min(8, n)
        state_energies, mixing, n_iter = davidson_eigh(
            H_ci, n_states=k, tol=tol, max_iter=max_iter
        )
        provenance = {"n_csf": n, "n_orbitals": len(orbitals_dict),
                       "solver": "davidson", "n_iter": int(n_iter),
                       "tol": tol}

    return CIResult(
        state_energies=np.asarray(state_energies),
        mixing_coefficients=np.asarray(mixing),
        provenance=provenance,
    )


def _group_csfs_by_J_parity(csfs):
    """Group CSF indices by (total_J2, parity); return None if labels missing.

    Returns
    -------
    blocks : list of ((J2, parity), [csf_idx, ...]) or None
        Sorted so that (J2=0, +1) comes first.  Returns ``None`` when no
        CSF carries a non-default J/parity label (legacy AL-flat path).
    """
    labels = []
    seen_non_default = False
    for csf in csfs:
        j2 = int(getattr(csf, "total_J2", 0))
        par = int(getattr(csf, "parity", 1))
        if j2 != 0 or par != 1:
            seen_non_default = True
        labels.append((j2, par))
    if not seen_non_default:
        return None
    groups: dict = {}
    for i, key in enumerate(labels):
        groups.setdefault(key, []).append(i)
    # Stable order: lowest J2 first, even parity before odd
    return sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1]))


def _solve_blocked(H_ci, blocks):
    """Block-diagonalise H_ci according to (J2, parity) groupings.

    Each block is diagonalised independently with LAPACK ``eigh``; the
    eigenvectors are scattered back into a full-size mixing matrix and
    the eigenvalues are concatenated in ascending order.  Off-block
    matrix elements are dropped because they must vanish by angular
    selection rules.
    """
    n = H_ci.shape[0]
    all_eigvals = []
    all_eigvecs = []
    for (_J2, _par), idx in blocks:
        sub = H_ci[np.ix_(idx, idx)]
        # Symmetrise residual numerical noise before eigh
        sub = 0.5 * (sub + sub.T)
        w, V = linalg.eigh(sub)
        for k in range(len(w)):
            vec = np.zeros(n)
            vec[idx] = V[:, k]
            all_eigvals.append(float(w[k]))
            all_eigvecs.append(vec)
    order = np.argsort(all_eigvals)
    state_energies = np.asarray([all_eigvals[i] for i in order])
    mixing = np.column_stack([all_eigvecs[i] for i in order])
    return state_energies, mixing


# ---------------------------------------------------------------------------
# Davidson-Liu block iterative eigensolver
# ---------------------------------------------------------------------------


def davidson_eigh(
    H: np.ndarray,
    n_states: int,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    block_size: int | None = None,
):
    """Davidson-Liu block-iterative diagonalisation of a real symmetric
    matrix, returning the lowest ``n_states`` eigenpairs.

    Designed for sparse-like dense CI matrices where ``n_states ≪ n``
    so the LAPACK ``eigh`` cost (O(n³)) is wasteful.  Davidson scales
    as O(n_iter · k · n²) with k = subspace size capped at ~5
    n_states.

    Parameters
    ----------
    H
        Real symmetric matrix (n × n).
    n_states
        Number of low-lying eigenpairs.
    tol
        Residual-norm convergence threshold per state.
    max_iter
        Hard iteration cap; raises ``RuntimeError`` if exceeded.
    block_size
        Vectors added to the subspace per iteration; default = n_states.

    Returns
    -------
    eigenvalues : ndarray, shape (n_states,)
    eigenvectors : ndarray, shape (n, n_states)
    n_iter : int

    References
    ----------
    - E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975).
    - B. Liu, *Lawrence Berkeley Lab. Tech. Rep.* **8158** (1978).
    """
    n = H.shape[0]
    if H.shape != (n, n):
        raise ValueError(f"H must be square; got shape {H.shape}.")
    if n_states > n:
        raise ValueError(f"n_states={n_states} exceeds matrix dim {n}.")
    if max_iter < 1:
        raise ValueError("max_iter must be ≥ 1.")
    if block_size is None:
        block_size = n_states
    if block_size < 1:
        raise ValueError("block_size must be ≥ 1.")
    rnorms = np.full(n_states, np.inf)  # for the RuntimeError diagnostics

    # Build initial guess: lowest-diagonal CSFs (cheap and effective
    # for diagonally dominant CI matrices).
    diag = np.diag(H).real
    initial_indices = np.argsort(diag)[:n_states]
    V = np.zeros((n, n_states))
    for i, idx in enumerate(initial_indices):
        V[idx, i] = 1.0
    V, _ = np.linalg.qr(V)
    HV = H @ V

    max_subspace = min(n, max(8 * n_states, n_states + 20))

    for it in range(1, max_iter + 1):
        # Rayleigh-Ritz projection
        T = V.T @ HV
        T = 0.5 * (T + T.T)  # enforce symmetry against drift
        eigvals, eigvecs = linalg.eigh(T)
        eigvals = eigvals[:n_states]
        eigvecs = eigvecs[:, :n_states]

        ritz_vec = V @ eigvecs
        ritz_HV = HV @ eigvecs
        residuals = ritz_HV - ritz_vec * eigvals[None, :]
        rnorms = np.linalg.norm(residuals, axis=0)

        if np.all(rnorms < tol):
            return eigvals, ritz_vec, it

        # Davidson preconditioner: t_i = -r_i / (D - λ_i)
        if V.shape[1] + block_size > max_subspace:
            # Restart: shrink subspace to current Ritz vectors
            V = ritz_vec.copy()
            HV = ritz_HV.copy()
            continue

        new_dirs: list[np.ndarray] = []
        for i in range(min(block_size, n_states)):
            if rnorms[i] < tol:
                continue
            denom = diag - eigvals[i]
            denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
            t = -residuals[:, i] / denom
            # Orthogonalise against V
            t = t - V @ (V.T @ t)
            nrm = np.linalg.norm(t)
            if nrm < 1e-14:
                continue
            t /= nrm
            # Orthogonalise against pending new directions
            for nd in new_dirs:
                t = t - nd * (nd @ t)
                nrm = np.linalg.norm(t)
                if nrm < 1e-14:
                    break
                t /= nrm
            if nrm > 1e-14:
                new_dirs.append(t)

        if not new_dirs:
            raise RuntimeError(
                "Davidson stalled before convergence; no independent "
                f"correction vectors could be generated. residuals = {rnorms}."
            )

        T_new = np.column_stack(new_dirs)
        V = np.column_stack([V, T_new])
        HV = np.column_stack([HV, H @ T_new])

    raise RuntimeError(
        f"Davidson did not converge in {max_iter} iterations; "
        f"residuals = {rnorms}."
    )


__all__ = ["CIResult", "ci_solve", "davidson_eigh"]
