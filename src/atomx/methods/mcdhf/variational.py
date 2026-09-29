"""Correlated Dirac–Coulomb orbital energy and analytic radial derivatives.

The angular coefficient table used for the Hamiltonian is contracted with
CI determinant amplitudes to form one-/two-body radial integral weights.
Differentiating those same cell integrals gives the orbital gradient,
including configuration interference absent from occupation-only SCF.
"""
from __future__ import annotations

import numpy as np

from atomx.methods.ci import ci_solve
from atomx.numerics.bspline.convol import (
    ConvolCache,
    _banded_to_full_add,
    _convol_case1_t,
    _density_sym,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices


class CorrelatedOrbitalFunctional:
    """Ground-state CI functional on orthonormal B-spline radial spinors.

    The gradient is Euclidean in the P/Q coefficient vectors. Project it
    into an orthogonality-preserving tangent space before an update. At an
    exact ground-state degeneracy we return the equal-weight ensemble
    gradient. Breit/QED are subsequent RCI/perturbative corrections.
    """

    def __init__(self, csfs, bsp_data, Z, nuclear_potential=None, cache=None):
        self.csfs = csfs
        self.data = bsp_data
        self.Z = Z
        self.ns = bsp_data["ns"]
        self.nuclear_potential = nuclear_potential
        self.cache = cache if cache is not None else ConvolCache(bsp_data)
        self.keys = tuple(sorted({(sh.n, sh.kappa) for c in csfs for sh in c.subshells}))
        self.one_body = {}
        self.metric = {}
        # Populations from the most recent gradient evaluation; the
        # preconditioner scaling reads this before the first evaluate()
        # (init builds blocks first), so it must default to empty and
        # scale_blocks floors missing keys at 0.05.
        self.last_occupations = {}
        for kappa in {key[1] for key in self.keys}:
            h, s = build_dirac_matrices(self.ns, kappa, Z, bsp_data, nuclear_potential)
            self.one_body[kappa] = np.tril(h) + np.tril(h, -1).T
            self.metric[kappa] = s
        from atomx.methods.mcdhf.radial_density import CIRadialDensity
        self.radial_density = CIRadialDensity(csfs)
        self.blocks = self.radial_density.blocks

    def evaluate(self, orbitals, *, gradient=True, ci_result=None):
        result = ci_result
        if result is None:
            result = ci_solve(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                              nuclear_potential=self.nuclear_potential, convol_cache=self.cache)
        energy = float(result.ground_energy)
        if not gradient:
            return energy, result
        weights = self.radial_density.contract(result)
        gradient = {key: np.zeros(2 * self.ns) for key in self.keys}
        self.last_occupations = {key: max(0., weights.get(("h", key[0], key[1], key[0]), 0.))
                                 for key in self.keys}
        vectors = {key: np.concatenate(orbitals[key]) for key in self.keys}
        gradient_terms = {}

        for key, weight in weights.items():
            if key[0] == "h":
                _, na, kappa, nb = key
                a, b = (na, kappa), (nb, kappa)
                h = self.one_body[kappa]
                gradient[a] += weight * (h @ vectors[b])
                gradient[b] += weight * (h @ vectors[a])
            else:
                _, rank, a, b, c, d = key
                for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                    pair = (rank, *sorted((source_a, source_b)))
                    gradient_terms.setdefault(pair, []).append((left, right, weight))
        # Contract each source pair once and discard its radial matrices
        # immediately: memory does not grow with the two-particle RDM size.
        for (rank, source_a, source_b), terms in gradient_terms.items():
            pp, qq = self._pair_potential(rank, orbitals[source_a], orbitals[source_b])
            for left, right, weight in terms:
                pl, ql = orbitals[left]
                pr, qr = orbitals[right]
                gradient[left][:self.ns] += weight * (pp @ pr)
                gradient[left][self.ns:] += weight * (qq @ qr)
                gradient[right][:self.ns] += weight * (pp @ pl)
                gradient[right][self.ns:] += weight * (qq @ ql)
        self.last_gradient = gradient
        return energy, gradient, result

    def _pair_potential(self, rank, orbital_a, orbital_b):
        """Differentiate the exact four-component cell Slater integral."""
        data, ns = self.data, self.ns
        orders = (data.get("ksp", data["ks"]), data.get("ksq", data["ks"]))
        output = [np.zeros((ns, ns)), np.zeros((ns, ns))]
        for left, label_left in enumerate(("p", "q")):
            for right, label_right in enumerate(("p", "q")):
                bank, extent, _ = self.cache.get_rkb_t(rank, label_left + label_right + label_left + label_right)
                density = _density_sym(orbital_a[right], orbital_b[right], extent, orders[right])
                band = _convol_case1_t(bank, density, extent, orders[left], orders[right])
                _banded_to_full_add(output[left], band, extent, orders[left], 1.0)
        return tuple(output)


def optimize_orbitals(functional, orbitals, *, max_iter=160, tolerance=1e-6,
                      energy_tolerance=1e-10, preconditioner_matrices=None,
                      preconditioner_factory=None, refresh_patience=15,
                      refresh_min_ratio=0.25, max_refreshes=4,
                      projector_source="nuclear", occupation_floor=0.05,
                      progress_callback=None, orbital_preconditioner="spectral"):
    """Minimize the correlated energy in a fixed positive-energy subspace.

    A nuclear Dirac projector fixes the finite-basis no-pair Hamiltonian.
    Every trial uses a metric-orthonormal set (polar retraction); an Armijo
    line search accepts only energy-lowering steps. The derivative is the
    full CI Hellmann–Feynman derivative, not a configuration average.
    Limited-memory secant updates and orbital-energy preconditioning avoid
    forming the orbital Hessian.

    ``orbital_preconditioner='shooting'`` uses a two-sided positive radial
    Green solve for update directions. The Hamiltonian, no-pair space,
    orthogonality constraints, and spectral convergence diagnostic are
    unchanged. A failed search or less than twofold residual reduction over
    20 iterations switches back to the spectral inverse and clears history.

    Robustness extensions over the plain Riemannian L-BFGS:

    * Line-search trials that raise ``ValueError`` (rank-deficient polar
      retraction) or ``FloatingPointError`` (non-finite trial CI) are
      treated as rejections and the step is halved, instead of aborting
      the whole solve.
    * When the backtracking budget is exhausted the search falls back to
      a preconditioned steepest-descent step (secant history reset); the
      solve only stops when the fallback also fails.
    * With ``preconditioner_factory`` supplied, the spectral Fock
      preconditioner is rebuilt at the current orbitals whenever the
      preconditioned-gradient residual stalls (less than
      ``1/refresh_min_ratio``-fold reduction over ``refresh_patience``
      consecutive iterations, at most ``max_refreshes`` times).  The
      secant history is cleared on refresh because the changed metric
      invalidates stored curvature pairs — that history reset is what
      makes refresh stable (refresh on top of stale pairs was measured
      to diverge on stiff open-shell landscapes).
    * The first trial step of each line search reuses the previously
      accepted step length when it was shorter than unity, avoiding a
      wasted full-step trial in the backtracking regime.

    ``projector_source`` selects the positive-energy (no-pair) subspace the
    orbital update is constrained to.  ``"nuclear"`` (default) fixes the
    projector to the nuclear Dirac spectrum — exact for hydrogenic
    shells, but a deliberately crude positive-energy boundary for a
    screened heavy-atom valence shell: valence spinors see a much
    weaker potential than Z/r, so their small components and continuum
    mixing differ from the nuclear eigenbasis, and the SCF can carry
    small negative-energy leakage the nuclear projector never removes
    (critical review F3).  ``"fock"`` rebuilds the projector from the
    SCF-averaged Fock fields (``preconditioner_matrices``) at the
    current orbitals — the same construction GRASP-style no-pair
    Hamiltonians use — and requires ``preconditioner_matrices``.
    Either way the returned diagnostics carry
    ``no_pair_projector_leakage``, the worst S-norm weight any active
    orbital carries outside the *alternative* projector (nuclear vs
    Fock); a value near 1 means the two no-pair boundaries disagree on
    that shell and the projector choice matters.
    """
    from atomx.numerics.bspline.dirac_assembly import solve_dirac

    if orbital_preconditioner not in ("spectral", "shooting"):
        raise ValueError("orbital_preconditioner must be 'spectral' or 'shooting'")
    if not np.isfinite(occupation_floor) or occupation_floor <= 0:
        raise ValueError("occupation_floor must be finite and positive")
    if projector_source not in ("nuclear", "fock"):
        raise ValueError(
            f"projector_source must be 'nuclear' or 'fock', got {projector_source!r}")
    if projector_source == "fock" and preconditioner_matrices is None:
        raise ValueError(
            "projector_source='fock' needs the SCF Fock fields as "
            "preconditioner_matrices to build the no-pair space from")

    groups = {}
    for key in functional.keys:
        groups.setdefault(key[1], []).append(key)
    one_electron = sum(sh.occupation for sh in functional.csfs[0].subshells) == 1
    layout = []
    inverse_blocks = []
    initial, inverse_diagonal = [], []
    cursor = 0
    # Occupation weights for averaging per-kappa Fock fields (projector
    # diagnostics and the 'fock' no-pair space).  Subshell occupations from
    # the leading CSF; missing entries default to equal sharing.
    occ_weights = {}
    for csf0 in functional.csfs[:1]:
        for sh in csf0.subshells:
            occ_weights[(sh.n, sh.kappa)] = max(float(sh.occupation), 1.0)

    no_pair_leakage = {}
    for kappa, keys in sorted(groups.items()):
        metric = functional.metric[kappa]
        result = solve_dirac(functional.one_body[kappa], metric, functional.ns, kappa,
                             nsp=functional.data.get("nsp"), nsq=functional.data.get("nsq"))
        # Preserve an explicitly excited active shell when lower shells of
        # this symmetry are absent (e.g. a hydrogenic 2s/2p calculation).
        # Otherwise unrestricted minimization would silently replace 2s by
        # 1s. Missing lower reference roots are frozen outside the active
        # space, within the same stated no-pair projector model.
        l = kappa if kappa > 0 else -kappa - 1
        has_kb = functional.data.get("ksp", functional.data["ks"]) != functional.data.get("ksq", functional.data["ks"])
        first = min(key[0] for key in keys) - l - 1 + int(kappa > 0 and not has_kb)
        proj = result.coefficients[:, first:]
        eps_source = result.eigenvalues[first:]
        if projector_source == "fock" or preconditioner_matrices is not None:
            # The alternative no-pair boundary: a screened Fock field
            # (occupation-averaged over this symmetry's active shells)
            # instead of the bare nuclear one.  Diagonalising it against
            # the same metric and keeping the branch above the sea edge is
            # the GRASP-style construction.  Used either as THE projector
            # (projector_source="fock") or as the leakage yardstick for the
            # nuclear projector.
            fock_k = np.zeros_like(functional.one_body[kappa])
            total = 0.0
            for key in keys:
                if preconditioner_matrices is not None and key in preconditioner_matrices:
                    w = occ_weights.get(key, 1.0)
                    fock_k += w * preconditioner_matrices[key]
                    total += w
            if total > 0.0:
                fock_k = functional.one_body[kappa] + fock_k / total
                alt = solve_dirac(fock_k, metric, functional.ns, kappa,
                                  nsp=functional.data.get("nsp"),
                                  nsq=functional.data.get("nsq"))
                alt_projector = alt.coefficients[:, first:]
                if projector_source == "fock":
                    proj = alt_projector
                    eps_source = alt.eigenvalues[first:]
                # Worst S-norm weight the active orbitals carry outside the
                # no-pair space actually in use: for the nuclear projector
                # the yardstick is the screened Fock space (how much of the
                # shell a GRASP-style boundary would reclassify), and vice
                # versa.  Far from 0: the two no-pair models disagree on
                # this symmetry's shell and the projector choice (F3) is a
                # physical decision, not a numerical nicety.
                yardstick = (result.coefficients[:, first:]
                             if projector_source == "fock" else alt_projector)
                coefficients = np.column_stack(
                    [np.concatenate(orbitals[key]) for key in keys])
                overlap = yardstick.T @ metric @ coefficients
                norms = np.einsum("ij,ij->j", overlap, overlap)
                seed_norms = np.einsum("ij,ij->j", coefficients,
                                       metric @ coefficients)
                with np.errstate(invalid="ignore", divide="ignore"):
                    leak = 1.0 - norms / np.maximum(seed_norms, 1e-30)
                no_pair_leakage[kappa] = float(np.clip(np.max(leak), 0.0, 1.0))
        coefficients = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
        y = proj.T @ metric @ coefficients
        if one_electron:
            # Complete one-electron CI is invariant under active-orbital
            # rotations. Fix that gauge to actual nuclear eigenfunctions,
            # so principal labels and unoccupied orbital energies remain
            # meaningful instead of retaining an arbitrary SCF rotation.
            y[:] = 0.
            for column, (n, _) in enumerate(keys):
                root = n - l - 1 + int(kappa > 0 and not has_kb) - first
                y[root, column] = 1.
        gram_values, gram_vectors = np.linalg.eigh(y.T @ y)
        if gram_values.min() < 1e-10:
            raise ValueError("orbital seed has no independent positive-energy projection")
        y = y @ ((gram_vectors / np.sqrt(gram_values)) @ gram_vectors.T)
        extent = y.size
        layout.append((keys, proj, y.shape, slice(cursor, cursor + extent)))
        cursor += extent
        initial.append(y.ravel())
        eps = eps_source
        orbital_eps = np.sum(y * (eps[:, None] * y), axis=0)
        inverse_diagonal.append((1.0 / (2 * np.maximum(abs(eps[:, None] - orbital_eps), 0.5))).ravel())
    def build_blocks(mats, point):
        """Spectral inverse-Hessian blocks from full-space Fock matrices.

        Factored out of the initialisation so the stagnation-triggered
        refresh can rebuild the preconditioner at the current orbitals
        (``point``) — the Rayleigh-quotient reference in each column's
        spectral gap is recomputed against the current ``y`` column, and
        the population scaling is reapplied, so a refresh is exactly the
        initial construction evaluated at the current iterate.
        """
        blocks_out = []
        for keys, projector, shape, where in layout:
            y = point[where].reshape(shape)
            blocks = []
            for column, key in enumerate(keys):
                fock = projector.T @ mats[key] @ projector
                fock = 0.5 * (fock + fock.T)
                values, eigenvectors = np.linalg.eigh(fock)
                reference = float(y[:, column] @ fock @ y[:, column])
                inverse = 1. / (2 * np.maximum(abs(values - reference), 0.5))
                blocks.append((eigenvectors * inverse) @ eigenvectors.T)
            blocks_out.append(blocks)
        return blocks_out

    x = np.concatenate(initial)
    inverse_diagonal = np.concatenate(inverse_diagonal)

    def scale_blocks(blocks_out):
        """Apply the occupation-weight scaling shared by init and refresh."""
        for group, (keys, _, _shape, _where) in enumerate(layout):
            populations = np.array([max(functional.last_occupations.get(key, occupation_floor), occupation_floor)
                                    for key in keys])
            blocks_out[group] = [block / population for block, population
                                 in zip(blocks_out[group], populations, strict=True)]
        return blocks_out

    def retract(point):
        result = point.copy()
        for _, _, shape, where in layout:
            y = point[where].reshape(shape)
            values, vectors = np.linalg.eigh(y.T @ y)
            if values.min() <= 1e-14:
                raise ValueError("rank-deficient orbital trial")
            result[where] = (y @ ((vectors / np.sqrt(values)) @ vectors.T)).ravel()
        return result

    def tangent(point, vector):
        result = vector.copy()
        for _, _, shape, where in layout:
            y = point[where].reshape(shape)
            g = vector[where].reshape(shape)
            multiplier = y.T @ g
            result[where] = (g - y @ (0.5 * (multiplier + multiplier.T))).ravel()
        return result

    def unpack(point):
        orbitals = {}
        for keys, projector, shape, where in layout:
            coefficients = projector @ point[where].reshape(shape)
            for i, key in enumerate(keys):
                orbitals[key] = (coefficients[:functional.ns, i].copy(),
                                 coefficients[functional.ns:, i].copy())
        return orbitals

    def evaluate(point, gradient=True, ci_result=None):
        if not gradient:
            return functional.evaluate(unpack(point), gradient=False)
        energy, radial_gradient, ci = functional.evaluate(unpack(point), ci_result=ci_result)
        g = np.zeros_like(point)
        for keys, projector, _shape, where in layout:
            derivative = np.column_stack([radial_gradient[key] for key in keys])
            g[where] = (projector.T @ derivative).ravel()
        return energy, tangent(point, g), ci

    energy, g, ci = evaluate(x)
    # The orbital Hessian scales with its occupation. Omitting this factor
    # makes a nearly empty correlation orbital converge orders of magnitude
    # more slowly than the closed core. Keep a finite floor for null-density
    # directions, whose radial form is not defined by a ground-state target.
    for _group, (keys, _, shape, where) in enumerate(layout):
        populations = np.array([max(functional.last_occupations.get(key, occupation_floor), occupation_floor) for key in keys])
        inverse_diagonal[where] /= np.broadcast_to(populations, shape).ravel()
    if preconditioner_matrices is not None:
        inverse_blocks = scale_blocks(build_blocks(preconditioner_matrices, x))

    def spectral_precondition(vector):
        if not inverse_blocks:
            return inverse_diagonal * vector
        result = np.zeros_like(vector)
        for (_, _, shape, where), blocks in zip(layout, inverse_blocks, strict=True):
            matrix = vector[where].reshape(shape)
            result[where] = np.column_stack([block @ matrix[:, i]
                                            for i, block in enumerate(blocks)]).ravel()
        return result
    precondition = spectral_precondition
    if orbital_preconditioner == "shooting":
        from atomx.numerics.shooting.preconditioner import ShootingPreconditioner
        precondition = ShootingPreconditioner(functional, layout, occupation_floor,
                                              x, preconditioner_matrices)
    energies = [energy]
    history = []
    converged = False
    termination_reason = "maximum_iterations"
    line_search_fallbacks = 0
    shooting_active = orbital_preconditioner == "shooting"
    shooting_iterations = 0
    shooting_fallback_reason = None
    checkpoint_residual = None
    for iteration in range(max_iter + 1):
        # Convergence must not depend on how small the proposed update is.
        # Keep exactly the same residual yardstick for both numerical paths.
        preconditioned_gradient = tangent(x, spectral_precondition(g))
        residual = float(np.linalg.norm(preconditioned_gradient))
        energy_error = 0.5 * abs(float(g @ preconditioned_gradient))
        if progress_callback is not None:
            progress_callback(iteration, unpack(x), ci, {"energy_ha": energy,
                "orbital_residual": residual, "estimated_energy_error": energy_error})
        if residual < tolerance and energy_error < energy_tolerance:
            converged = True
            termination_reason = "converged"
            break
        if iteration == max_iter:
            break
        if shooting_active:
            if checkpoint_residual is None:
                checkpoint_residual = residual
            elif iteration % 20 == 0:
                if residual > .5 * checkpoint_residual:
                    precondition = spectral_precondition
                    shooting_active = False
                    shooting_fallback_reason = "spectral residual stalled over 20 iterations"
                    history.clear()
                checkpoint_residual = residual
            if shooting_active:
                shooting_iterations += 1
        q = g.copy()
        alphas = []
        for s, y, rho in reversed(history):
            alpha = rho * np.dot(s, q)
            alphas.append(alpha)
            q -= alpha * y
        direction = precondition(q)
        for (s, y, rho), alpha in zip(history, reversed(alphas), strict=True):
            direction += s * (alpha - rho * np.dot(y, direction))
        direction = -tangent(x, direction)
        derivative = float(g @ direction)
        if derivative >= 0:
            history.clear()
            direction = -preconditioned_gradient
            derivative = float(g @ direction)
        step = 1.0
        accepted = False
        for attempt in range(2):
            if attempt:
                # Secant curvature can become unreliable near the energy
                # resolution limit. Retry the physical gradient before
                # declaring failure, without relaxing either tolerance.
                history.clear()
                if shooting_active:
                    shooting_active = False
                    precondition = spectral_precondition
                    shooting_fallback_reason = "shooting/secant line search exhausted"
                line_search_fallbacks += 1
                direction = -preconditioned_gradient
                derivative = float(g @ direction)
                step = 1.0
            for _ in range(28):
                try:
                    candidate = retract(x + step * direction)
                    trial_energy, trial_ci = evaluate(candidate, gradient=False)
                except (ValueError, FloatingPointError):
                    step *= 0.5
                    continue
                if np.isfinite(trial_energy) and trial_energy <= energy + 1e-4 * step * derivative:
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                break
        if not accepted:
            termination_reason = "line_search_failed"
            break
        old_x, old_g = x, g
        x = candidate
        # The accepted line-search trial already solved this exact CI
        # Hamiltonian. Reuse its amplitudes in the Hellmann–Feynman gradient.
        energy, g, ci = evaluate(x, ci_result=trial_ci)
        energies.append(energy)
        # Vector transport keeps all secant pairs in the current tangent
        # space. Ambient gradient differences otherwise contain changing
        # constraint normals and corrupt the approximate inverse Hessian.
        transported = []
        for old_s, old_y, _ in history:
            st, yt = tangent(x, old_s), tangent(x, old_y)
            sy = float(st @ yt)
            if sy > 1e-12 * np.linalg.norm(st) * np.linalg.norm(yt):
                transported.append((st, yt, 1. / sy))
        history = transported
        s = tangent(x, x - old_x)
        y = g - tangent(x, old_g)
        curvature = float(s @ y)
        if curvature > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            history.append((s, y, 1.0 / curvature))
            history = history[-10:]
    return unpack(x), ci, {
        "converged": converged, "iterations": iteration,
        "termination_reason": termination_reason,
        "line_search_fallbacks": line_search_fallbacks,
        "occupation_floor": occupation_floor,
        "gradient_norm": float(np.linalg.norm(g)),
        "orbital_residual": residual, "orbital_tolerance": tolerance,
        "estimated_energy_error": energy_error, "energy_tolerance": energy_tolerance,
        "energy_history": energies,
        "functional": "correlated Dirac-Coulomb CI",
        "target": "ground-state energy (equal ensemble at exact degeneracy)",
        "radial_labels": "correlation-orbital labels; not guaranteed spectroscopic excited states",
        "projector": (
            "fixed positive-energy nuclear Dirac B-spline subspace"
            if projector_source == "nuclear"
            else "fixed positive-energy Fock-field (screened) B-spline subspace"
        ),
        "no_pair_projector_leakage": no_pair_leakage,
        "excited_shell_constraints": "lower reference roots below each symmetry's first active shell are excluded",
        "preconditioner": "screened Fock spectral gaps" if inverse_blocks else "nuclear Dirac spectral gaps",
        "update_preconditioner": orbital_preconditioner,
        "shooting_iterations": shooting_iterations,
        "shooting_fallback_reason": shooting_fallback_reason,
        "shooting_operator": (
            "positive radial kinetic resolvent; full correlated CI gradient source"
            if orbital_preconditioner == "shooting" else None),
        "breit_qed": "subsequent RCI / perturbative corrections",
    }
