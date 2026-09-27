"""Correlated Dirac–Coulomb orbital energy and analytic radial derivatives.

The angular coefficient table used for the Hamiltonian is contracted with
CI determinant amplitudes to form one-/two-body radial integral weights.
Differentiating those same cell integrals gives the orbital gradient,
including configuration interference absent from occupation-only SCF.
"""
from __future__ import annotations

import numpy as np

from atomx.methods.ci import ci_solve
from atomx.methods.mcdhf import exact_ci, matrix
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
        for kappa in {key[1] for key in self.keys}:
            h, s = build_dirac_matrices(self.ns, kappa, Z, bsp_data, nuclear_potential)
            self.one_body[kappa] = np.tril(h) + np.tril(h, -1).T
            self.metric[kappa] = s
        self.blocks = []
        for (_, j_indices) in matrix._group_csf_indices(csfs):
            signatures = [exact_ci.csf_signature(csfs[i]) for i in j_indices]
            groups = {}
            for sig in signatures:
                groups.setdefault(sig.pattern, []).append(sig)
            structure = exact_ci.block_terms(tuple(groups), signatures[0].total_J2)
            grouped = tuple(tuple(groups[p]) for p in structure.patterns)
            projection = exact_ci.csf_block_matrix(structure.patterns, structure.j2, grouped)
            # Columns of projection are grouped by canonical pattern.
            original = {sig: i for i, sig in enumerate(signatures)}
            columns = [j_indices[original[sig]] for group in grouped for sig in group]
            self.blocks.append((structure, projection, columns))

    def _hellmann_feynman_weights(self, result):
        """Contract CI amplitudes with the angular table: E = sum w_k I_k.

        Averaged over the (near-)degenerate ground multiplet, matching the
        equal-weight ensemble gradient documented on the class.
        """
        energy = float(result.ground_energy)
        ground = abs(result.state_energies - energy) < 1e-10
        weights = {}
        for structure, projection, columns in self.blocks:
            amplitudes = projection @ result.mixing_coefficients[columns][:, ground]
            values = np.mean(amplitudes[structure.pairs[:, 0]] *
                             amplitudes[structure.pairs[:, 1]], axis=1)
            coefficients = np.asarray(structure.table.T @ values).ravel()
            for key, coefficient in zip(structure.keys, coefficients, strict=True):
                if abs(coefficient) > 1e-15:
                    weights[key] = weights.get(key, 0.0) + coefficient
        return weights

    def freeze(self, ci_result, *, screen_tol=1e-12):
        """Snapshot the exact-CI state for variational surrogate evaluations.

        Returns a :class:`FrozenCI` whose energy/gradient evaluations need
        only the radial integral pass — no CI Hamiltonian assembly, no
        diagonalization.  At the freeze point the surrogate equals the
        exact ground energy and its Hellmann–Feynman gradient; it is a
        variational upper bound elsewhere (orbital rotations preserve the
        norm of the fixed determinant amplitudes), which is what makes
        Armijo trials on it sound.  ``screen_tol`` is relative to the
        largest weight; dropped keys shift the surrogate energy by far
        less than any convergence tolerance in use.
        """
        weights = self._hellmann_feynman_weights(ci_result)
        return FrozenCI(self, weights, screen_tol=screen_tol)

    def evaluate(self, orbitals, *, gradient=True, ci_result=None):
        result = ci_result
        if result is None:
            result = ci_solve(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                              nuclear_potential=self.nuclear_potential, convol_cache=self.cache)
        energy = float(result.ground_energy)
        if not gradient:
            return energy, result
        weights = self._hellmann_feynman_weights(result)
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


class FrozenCI:
    """Variational surrogate of the CI ground state with frozen amplitudes.

    ``energy(orbitals)`` and ``evaluate(orbitals)`` reproduce the exact
    ground-state energy and Hellmann–Feynman gradient at the freeze point
    (where the amplitudes solve the exact CI) and stay a variational upper
    bound in a neighbourhood: orbital rotations are unitary, so the fixed
    determinant amplitudes keep their norm.  Used for line-search trials,
    where each evaluation costs one radial integral pass instead of a full
    CI Hamiltonian assembly plus diagonalization.
    """

    def __init__(self, functional, weights, *, screen_tol=1e-12):
        self.functional = functional
        scale = max(abs(w) for w in weights.values())
        self.weights = {k: w for k, w in weights.items()
                        if abs(w) > screen_tol * scale}
        self._one_body = {}
        # Energy counts each unordered integral once; the gradient needs
        # both (a,b|c,d) and (c,d|a,b) orderings because each places the
        # orbital in a different gradient slot.  Both term lists share one
        # pair-potential pass per (rank, source pair) batch.
        self._energy_terms = {}
        self._gradient_terms = {}
        for key, weight in self.weights.items():
            if key[0] == "h":
                self._one_body[key] = weight
                continue
            _, rank, a, b, c, d = key
            self._energy_terms.setdefault(
                (rank, *sorted((c, d))), []).append(((a, b), weight))
            for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                self._gradient_terms.setdefault(
                    (rank, *sorted((source_a, source_b))),
                    []).append(((left, right), weight))

    def _energy_and_gradient(self, orbitals, gradient):
        functional = self.functional
        ns = functional.ns
        energy = 0.0
        vectors = {key: np.concatenate(orbitals[key]) for key in functional.keys}
        grad = {key: np.zeros(2 * ns) for key in functional.keys} if gradient else None
        for key, weight in self._one_body.items():
            _, na, kappa, nb = key
            h = functional.one_body[kappa]
            a, b = (na, kappa), (nb, kappa)
            va, vb = vectors[a], vectors[b]
            energy += weight * float(va @ (h @ vb))
            if gradient:
                grad[a] += weight * (h @ vb)
                grad[b] += weight * (h @ va)
        batches = sorted(set(self._energy_terms) | set(self._gradient_terms))
        for batch in batches:
            pp, qq = functional._pair_potential(
                rank=batch[0], orbital_a=orbitals[batch[1]],
                orbital_b=orbitals[batch[2]])
            for (left, right), weight in self._energy_terms.get(batch, ()):
                pl, ql = orbitals[left]
                pr, qr = orbitals[right]
                energy += weight * (float(pl @ (pp @ pr)) + float(ql @ (qq @ qr)))
            if gradient:
                for (left, right), weight in self._gradient_terms.get(batch, ()):
                    pl, ql = orbitals[left]
                    pr, qr = orbitals[right]
                    grad[left][:ns] += weight * (pp @ pr)
                    grad[left][ns:] += weight * (qq @ qr)
                    grad[right][:ns] += weight * (pp @ pl)
                    grad[right][ns:] += weight * (qq @ ql)
        return energy, grad

    def energy(self, orbitals):
        """Frozen surrogate energy — one radial integral pass, no CI solve."""
        energy, _ = self._energy_and_gradient(orbitals, False)
        return energy

    def evaluate(self, orbitals):
        """Frozen surrogate energy and Hellmann–Feynman gradient."""
        return self._energy_and_gradient(orbitals, True)


def optimize_orbitals(functional, orbitals, *, max_iter=160, tolerance=1e-6,
                      energy_tolerance=1e-10, preconditioner_matrices=None,
                      frozen_line_search=False):
    """Minimize the correlated energy in a fixed positive-energy subspace.

    A nuclear Dirac projector fixes the finite-basis no-pair Hamiltonian.
    Every trial uses a metric-orthonormal set (polar retraction); an Armijo
    line search accepts only energy-lowering steps. The derivative is the
    full CI Hellmann–Feynman derivative, not a configuration average.
    Limited-memory secant updates and orbital-energy preconditioning avoid
    forming the orbital Hessian.

    With ``frozen_line_search=True`` (opt-in), Armijo trials evaluate a
    variational surrogate — the exact CI state frozen at the current
    point — instead of solving the CI Hamiltonian at every trial.  The
    surrogate matches the exact energy and slope at the freeze point
    (Hellmann–Feynman) and remains an upper bound along the trial
    direction.  Measured economics (Fe XVII j-resolved, pinned threads):
    a surrogate evaluation costs ~0.2-0.3 of a CI solve, so the option
    pays only when the line search rejects often (break-even around 1.4
    trials/iteration) — well-preconditioned problems accept the first
    step ~always and are faster on the historical exact-trial path, which
    also reuses the accepted trial's amplitudes for the gradient.  The
    default therefore stays ``False``.

    Accuracy guard: because the surrogate's acceptance is only
    first-order exact, a candidate it accepts is re-verified on the exact
    functional before the step is committed (one exact solve, whose
    amplitudes then feed the Hellmann–Feynman gradient — the same reuse
    the exact-trial path has always had); a mis-accepted candidate falls
    back to exact trials for the rest of that line search.  Monotone
    descent on the exact energy and the convergence gate are preserved
    unchanged: the accepted point is always re-evaluated with the exact
    CI, which alone feeds the gradient, the secant pairs, and the gate.
    """
    from atomx.numerics.bspline.dirac_assembly import solve_dirac

    groups = {}
    for key in functional.keys:
        groups.setdefault(key[1], []).append(key)
    one_electron = sum(sh.occupation for sh in functional.csfs[0].subshells) == 1
    layout = []
    inverse_blocks = []
    initial, inverse_diagonal = [], []
    cursor = 0
    for kappa, keys in sorted(groups.items()):
        metric = functional.metric[kappa]
        result = solve_dirac(functional.one_body[kappa], metric, functional.ns, kappa,
                             nsp=functional.data.get("nsp"), nsq=functional.data.get("nsq"))
        # Preserve an explicitly excited active shell when lower shells of
        # this symmetry are absent (e.g. a hydrogenic 2s/2p calculation).
        # Otherwise unrestricted minimization would silently replace 2s by
        # 1s. Missing lower reference roots are frozen outside the active
        # space, within the same stated nuclear-Dirac projector model.
        l = kappa if kappa > 0 else -kappa - 1
        has_kb = functional.data.get("ksp", functional.data["ks"]) != functional.data.get("ksq", functional.data["ks"])
        first = min(key[0] for key in keys) - l - 1 + int(kappa > 0 and not has_kb)
        projector = result.coefficients[:, first:]
        coefficients = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
        y = projector.T @ metric @ coefficients
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
        layout.append((keys, projector, y.shape, slice(cursor, cursor + extent)))
        cursor += extent
        initial.append(y.ravel())
        eps = result.eigenvalues[first:]
        orbital_eps = np.sum(y * (eps[:, None] * y), axis=0)
        inverse_diagonal.append((1.0 / (2 * np.maximum(abs(eps[:, None] - orbital_eps), 0.5))).ravel())
        if preconditioner_matrices is not None:
            blocks = []
            for column, key in enumerate(keys):
                fock = projector.T @ preconditioner_matrices[key] @ projector
                fock = 0.5 * (fock + fock.T)
                values, eigenvectors = np.linalg.eigh(fock)
                reference = float(y[:, column] @ fock @ y[:, column])
                inverse = 1. / (2 * np.maximum(abs(values - reference), 0.5))
                blocks.append((eigenvectors * inverse) @ eigenvectors.T)
            inverse_blocks.append(blocks)
    x = np.concatenate(initial)
    inverse_diagonal = np.concatenate(inverse_diagonal)

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
    frozen = functional.freeze(ci) if frozen_line_search else None
    # The orbital Hessian scales with its occupation. Omitting this factor
    # makes a nearly empty correlation orbital converge orders of magnitude
    # more slowly than the closed core. Keep a finite floor for null-density
    # directions, whose radial form is not defined by a ground-state target.
    for group, (keys, _, shape, where) in enumerate(layout):
        populations = np.array([max(functional.last_occupations[key], 0.05) for key in keys])
        inverse_diagonal[where] /= np.broadcast_to(populations, shape).ravel()
        if inverse_blocks:
            inverse_blocks[group] = [block / population for block, population
                                     in zip(inverse_blocks[group], populations, strict=True)]

    def precondition(vector):
        if not inverse_blocks:
            return inverse_diagonal * vector
        result = np.zeros_like(vector)
        for (_, _, shape, where), blocks in zip(layout, inverse_blocks, strict=True):
            matrix = vector[where].reshape(shape)
            result[where] = np.column_stack([block @ matrix[:, i]
                                            for i, block in enumerate(blocks)]).ravel()
        return result
    energies = [energy]
    history = []
    converged = False
    for iteration in range(max_iter + 1):
        preconditioned_gradient = tangent(x, precondition(g))
        residual = float(np.linalg.norm(preconditioned_gradient))
        energy_error = 0.5 * abs(float(g @ preconditioned_gradient))
        if residual < tolerance and energy_error < energy_tolerance:
            converged = True
            break
        if iteration == max_iter:
            break
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
            direction = -tangent(x, precondition(g))
            derivative = float(g @ direction)
        step = 1.0
        accepted = False
        use_frozen = frozen is not None
        verified_ci = None
        trial_ci = None
        while True:
            for _ in range(28):
                candidate = retract(x + step * direction)
                if use_frozen:
                    trial_energy = frozen.energy(unpack(candidate))
                else:
                    trial_energy, trial_ci = evaluate(candidate, gradient=False)
                if trial_energy <= energy + 1e-4 * step * derivative:
                    accepted = True
                    break
                step *= 0.5
            if not accepted or not use_frozen:
                break
            # Accuracy guard: the surrogate accepted this candidate, but its
            # acceptance is only first-order exact.  Verify on the exact
            # functional (same Armijo test — the slopes agree to first
            # order) before committing; the solved amplitudes are reused for
            # the gradient below.  A failure disables the surrogate for the
            # rest of this line search, restoring the historical path.
            _exact_trial, verified_ci = evaluate(candidate, gradient=False)
            if _exact_trial <= energy + 1e-4 * step * derivative:
                break
            use_frozen = False
            verified_ci = None
            step *= 0.5
        if not accepted:
            break
        old_x, old_g = x, g
        x = candidate
        energy, g, ci = evaluate(x, ci_result=verified_ci
                                 if verified_ci is not None else trial_ci)
        if frozen is not None:
            frozen = functional.freeze(ci)
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
        "gradient_norm": float(np.linalg.norm(g)),
        "orbital_residual": residual, "orbital_tolerance": tolerance,
        "estimated_energy_error": energy_error, "energy_tolerance": energy_tolerance,
        "energy_history": energies,
        "functional": "correlated Dirac-Coulomb CI",
        "target": "ground-state energy (equal ensemble at exact degeneracy)",
        "radial_labels": "correlation-orbital labels; not guaranteed spectroscopic excited states",
        "projector": "fixed positive-energy nuclear Dirac B-spline subspace",
        "excited_shell_constraints": "lower nuclear-Dirac roots below each symmetry's first active shell are excluded",
        "preconditioner": "screened Fock spectral gaps" if inverse_blocks else "nuclear Dirac spectral gaps",
        "breit_qed": "subsequent RCI / perturbative corrections",
    }
