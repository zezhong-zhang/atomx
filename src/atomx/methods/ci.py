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
    nuclear_potential=None,
    convol_cache=None,
    include_breit=False,
    breit_orbital_energies=None,
) -> CIResult:
    """Build the CI matrix and diagonalise it.

    Parameters
    ----------
    csfs
        List of :class:`~atomx.config.csf.CSF` objects.
    orbitals_dict
        ``{(n, kappa): (P, Q)}`` — converged radial orbitals in the
        representation matching ``bsp_data``: B-spline coefficient
        vectors for the B-spline workspace, or grid-tabulated arrays
        (e.g. ``ShootingResult._pa/_qa`` rows) for the grid workspace.
    orbital_basis
        Shared orbital basis descriptor.
    bsp_data
        B-spline workspace (grid, weights, basis) or a
        :class:`~atomx.physics.csf.matrix.GridCIWorkspace` wrapping
        the shooting grid ``(r, difr, r·V_nuc)``.  The grid flavour
        evaluates every matrix element with the shooting engine's own
        numerics (SLAG6 Slater integrals, the solver's nuclear
        potential array) so single-CSF CI totals reproduce
        ``ShootingResult.total_energy`` — same consistency principle
        as the cell integrator on the B-spline side.
    Z
        Nuclear charge.
    ns
        B-spline dimension (grid flavour: the grid size ``len(r)``).
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
    nuclear_potential
        ``V(r)`` callable for the one-electron Dirac blocks (finite
        nucleus); ``None`` = point nucleus.  Pass the SCF solver's
        potential so CI and SCF energies share the same nuclear model.
        B-spline flavour only — the grid workspace carries ``rv_nuc``.
    convol_cache
        Optional pre-built cell-integral cache on ``bsp_data`` (see
        :func:`atomx.physics.csf.matrix.build_ci_matrix`).  B-spline
        flavour only.
    include_breit
        Add the static Breit interaction (Gaunt plus retardation) to every
        ``(2J, parity)`` block of the CI matrix, so it shifts levels
        differentially. Off by default.
    breit_orbital_energies
        Optional explicit ``{(n, kappa): energy_in_Hartree}`` mapping.
        With ``include_breit=True``, replace static Breit by the real
        finite-frequency transverse operator, averaging the two vertex
        frequencies. These orbital energies stay fixed during CI;
        assigning them to correlation orbitals is a model choice.

    Returns
    -------
    :class:`CIResult`
    """
    from atomx.physics.csf.matrix import build_ci_matrix

    n = len(csfs)
    if solver not in ("dense", "davidson", "auto"):
        raise ValueError(f"Unknown solver {solver!r}; expected 'dense', 'davidson', or 'auto'.")
    if n_states is not None and (not isinstance(n_states, (int, np.integer))
                                 or not 1 <= n_states <= n):
        raise ValueError(f"n_states must be between 1 and {n}")
    if solver == "auto":
        solver = "dense" if (n <= 1500 or n_states is None) else "davidson"
    H_ci = build_ci_matrix(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns,
                           nuclear_potential=nuclear_potential,
                           convol_cache=convol_cache,
                           include_breit=include_breit,
                           breit_orbital_energies=breit_orbital_energies,
                           as_operator=(solver == "davidson"))
    groups = _group_csfs_by_J_parity(csfs) or [((0, 1), list(range(n)))]
    if hasattr(H_ci, "blocks"):
        blocks = H_ci.blocks
    else:
        blocks = [(np.asarray(idx), H_ci[np.ix_(idx, idx)]) for _, idx in groups]
    values, vectors, iterations = [], [], []
    for idx, block in blocks:
        size = len(idx)
        if size == 0:
            continue
        # No block can contribute more than k states to the global lowest k.
        k = size if n_states is None else min(n_states, size)
        if solver == "davidson":
            w, v, n_iter = davidson_eigh(block, n_states=k, tol=tol, max_iter=max_iter)
            iterations.append(n_iter)
        else:
            subset = (0, k - 1) if k < size else None
            w, v = linalg.eigh(block, subset_by_index=subset)
        full = np.zeros((n, len(w)))
        full[idx] = v
        values.extend(w)
        vectors.append(full)
    order = np.argsort(values, kind="stable")[:n_states]
    state_energies = np.asarray(values)[order]
    mixing = (np.column_stack(vectors)[:, order] if vectors else np.zeros((n, 0)))
    provenance = {
        "n_csf": n, "n_orbitals": len(orbitals_dict),
        "solver": solver + ("-blocked" if len(blocks) > 1 else ""),
        "n_blocks": len(blocks), "block_sizes": [len(idx) for idx, _ in blocks],
        "matrix_free": solver == "davidson",
    }
    if iterations:
        provenance.update(n_iter=max(iterations), block_iterations=iterations, tol=tol)
    if include_breit:
        if breit_orbital_energies is None:
            from atomx.physics.csf.breit_ci import RETARDATION_NOTE
            provenance.update(breit="static-breit-gaunt-retardation-in-blocks", breit_note=RETARDATION_NOTE)
        else:
            from atomx.physics.csf.transverse import frequency_provenance
            frequency = frequency_provenance({key: breit_orbital_energies[key]
                                               for key in orbitals_dict})
            provenance.update(
                breit="finite-frequency-transverse-in-blocks",
                **{'breit_'+key: value for key, value in frequency.items()})

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
    if n_states < 1:
        raise ValueError("n_states must be positive.")
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
    diag = np.asarray(H.diagonal() if hasattr(H, "diagonal") else np.diag(H)).real
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
