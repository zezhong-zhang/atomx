"""Correlated Dirac–Coulomb orbital energy and analytic radial derivatives.

The angular coefficient table used for the Hamiltonian is contracted with
CI determinant amplitudes to form one-/two-body radial integral weights.
Differentiating those same cell integrals gives the orbital gradient,
including configuration interference absent from occupation-only SCF.
"""
from __future__ import annotations

import math

import numpy as np

from atomx.methods.ci import CIResult, ci_solve
from atomx.numerics.bspline.convol import (
    SEMISEPARABLE_DIRECT_NS,
    ConvolCache,
    _banded_to_full_add,
    _convol_case1_semiseparable_t,
    _convol_case1_t,
    _density_sym,
    compute_slater_rk_cell,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices


class CSFOrbitalFunctional:
    """Ground or fixed-ensemble CI functional on orthonormal radial spinors.

    The gradient is Euclidean in the P/Q coefficient vectors. Project it
    into an orthogonality-preserving tangent space before an update. At an
    exact ground-state degeneracy we return the equal-weight ensemble
    gradient. An explicit CIStateAverage selects excited roots within each
    symmetry block. Breit/QED are subsequent RCI/perturbative corrections.
    ``level_tracking`` optionally binds that ensemble to a frozen physical
    CI reference, resolved on every energy and gradient trial.
    ``dense_cluster_gap`` explicitly refines near-root numerical subspaces
    of the same full real Hamiltonian; it is not a physical degeneracy gate.
    ``retain_hamiltonian`` explicitly opts into the complete dense DC
    snapshot owned by each CI trial. Supplied retained results validate the
    same physical point before signed density/gradient contraction. Legacy
    results without an owner remain gradient inputs; they do not provide H.
    """

    def __init__(self, csfs, bsp_data, Z, nuclear_potential=None, cache=None,
                 *, state_average=None, level_tracking=None, dense_cluster_gap=None,
                 max_determinants=None, retain_hamiltonian=False):
        if not isinstance(retain_hamiltonian, bool):
            raise TypeError('retain_hamiltonian must be bool')
        self.retain_hamiltonian = retain_hamiltonian
        from atomx.physics.csf.state_average import CIStateAverage
        if state_average is not None and not isinstance(state_average, CIStateAverage):
            raise TypeError('state_average must be CIStateAverage or None')
        if level_tracking is not None:
            from atomx.physics.csf.tracking import CIBasisTracking
            if not isinstance(level_tracking, CIBasisTracking) or level_tracking.state_average != state_average:
                raise ValueError('level_tracking must reference this CIStateAverage target')
        self.level_tracking = level_tracking
        self.dense_cluster_gap = dense_cluster_gap
        self.state_average = state_average
        self.target_description = ('ground-state energy (equal ensemble at exact degeneracy)'
                                   if state_average is None else
                                   'fixed CI level average in shared orthonormal orbitals')
        if level_tracking is not None:
            self.target_description = 'frozen-reference physical CI subspace average in shared orthonormal orbitals'
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
        from atomx.physics.csf.radial_density import CIRadialDensity
        self.radial_density = CIRadialDensity(csfs, max_determinants=max_determinants)
        self.max_determinants = self.radial_density.max_determinants
        self.blocks = self.radial_density.blocks

    def evaluate(self, orbitals, *, gradient=True, ci_result=None, gradient_keys=None):
        result = ci_result
        if result is None:
            result = ci_solve(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                              nuclear_potential=self.nuclear_potential, convol_cache=self.cache,
                              dense_cluster_gap=self.dense_cluster_gap,
                              max_determinants=self.max_determinants,
                              retain_hamiltonian=self.retain_hamiltonian)
        elif isinstance(result, CIResult) and result.hamiltonian_snapshot is not None:
            result.hamiltonian_for(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                                  nuclear_potential=self.nuclear_potential,
                                  max_determinants=self.max_determinants)
        if self.level_tracking is not None:
            state_weights, tracking = self.level_tracking.resolve(self.csfs, result, orbitals, self.metric)
        else:
            state_weights = (None if self.state_average is None else
                             self.state_average.weights(self.csfs, result))
        energy = (float(result.ground_energy) if state_weights is None else
                  float(state_weights @ result.state_energies))
        if not gradient:
            return energy, result
        weights = self.radial_density.contract(result, state_weights)
        self.last_state_weights = state_weights
        if self.level_tracking is not None:
            self.last_tracking_diagnostics = tracking
        gradient = self.gradient_from_weights(orbitals, weights, gradient_keys=gradient_keys)
        return energy, gradient, result

    def gradient_from_weights(self, orbitals, weights, *, gradient_keys=None,
                              record_diagnostics=True):
        """Differentiate a contracted Hamiltonian RDM using shared radial pairs.

        Linear combinations from several electron sectors can be contracted
        before this step, so each distinct Coulomb source is evaluated once.
        ``weights`` retains the Hamiltonian angular kernel's signed terms.
        ``record_diagnostics=False`` evaluates a transition/response gradient
        without replacing the accepted energy state's RDM or populations.
        """
        if not isinstance(record_diagnostics, (bool, np.bool_)):
            raise TypeError('record_diagnostics must be boolean')
        if record_diagnostics:
            self.last_weights = weights
        selected = self.keys if gradient_keys is None else tuple(gradient_keys)
        if any(key not in self.keys for key in selected):
            raise ValueError('gradient requested for an orbital outside the CSF basis')
        gradient = {key: np.zeros(2 * self.ns) for key in selected}
        if record_diagnostics:
            self.last_occupations = {key: max(0., weights.get(("h", key[0], key[1], key[0]), 0.))
                                     for key in self.keys}
        vectors = {key: np.concatenate(orbitals[key]) for key in self.keys}
        one_body_actions = {}
        gradient_terms = {}

        for key, weight in weights.items():
            if key[0] == "h":
                _, na, kappa, nb = key
                a, b = (na, kappa), (nb, kappa)
                h = self.one_body[kappa]
                if a in gradient:
                    if b not in one_body_actions:
                        one_body_actions[b] = h @ vectors[b]
                    gradient[a] += weight * one_body_actions[b]
                if b in gradient:
                    if a not in one_body_actions:
                        one_body_actions[a] = h @ vectors[a]
                    gradient[b] += weight * one_body_actions[a]
            else:
                _, rank, a, b, c, d = key
                for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                    if left not in gradient and right not in gradient:
                        continue
                    pair = (rank, *sorted((source_a, source_b)))
                    gradient_terms.setdefault(pair, []).append((left, right, weight))
        # Contract each source pair once and discard its radial matrices
        # immediately: memory does not grow with the two-particle RDM size.
        for (rank, source_a, source_b), terms in gradient_terms.items():
            pp, qq = self._pair_potential(rank, orbitals[source_a], orbitals[source_b])
            # Each signed RDM term keeps its original accumulation order.
            # Reuse only exact actions on the same target radial spinor;
            # these arrays belong to this source pair and this evaluation.
            actions = {}
            for left, right, weight in terms:
                if left in gradient:
                    if right not in actions:
                        pr, qr = orbitals[right]
                        actions[right] = pp @ pr, qq @ qr
                    gp, gq = actions[right]
                    gradient[left][:self.ns] += weight * gp
                    gradient[left][self.ns:] += weight * gq
                if right in gradient:
                    if left not in actions:
                        pl, ql = orbitals[left]
                        actions[left] = pp @ pl, qq @ ql
                    gp, gq = actions[left]
                    gradient[right][:self.ns] += weight * gp
                    gradient[right][self.ns:] += weight * gq
        if record_diagnostics:
            self.last_gradient = gradient
        return gradient

    def energy_for_weights(self, orbitals, weights, *, integral_cache=None):
        """Evaluate a fixed coherent state without constructing its gradient."""
        energy = 0.
        # Exact content keys: a changed coefficient can never reuse an old
        # integral. The caller scopes this cache to one occupied rotation.
        signatures = ({key: (p.tobytes(), q.tobytes()) for key, (p, q) in orbitals.items()}
                      if integral_cache is not None else None)
        for term, coefficient in weights.items():
            if term[0] == "h":
                _, na, kappa, nb = term
                a = np.concatenate(orbitals[na, kappa])
                b = np.concatenate(orbitals[nb, kappa])
                value = a @ self.one_body[kappa] @ b
            else:
                _, rank, a, b, c, d = term
                if integral_cache is None:
                    value = compute_slater_rk_cell(*orbitals[a], *orbitals[b],
                        *orbitals[c], *orbitals[d], self.data, rank, cache=self.cache)
                else:
                    # Many terms share the same density pair. Contract its
                    # Coulomb potential once; unchanged pairs also survive
                    # the next trial angle of the same occupied rotation.
                    cache_key = (rank, *sorted((signatures[c], signatures[d])))
                    potential = integral_cache.get(cache_key)
                    if potential is None:
                        potential = self._pair_potential(rank, orbitals[c], orbitals[d])
                        integral_cache[cache_key] = potential
                    pa, qa = orbitals[a]
                    pb, qb = orbitals[b]
                    value = pa @ potential[0] @ pb + qa @ potential[1] @ qb
            energy += coefficient * value
        return float(energy)

    def fixed_density_energy_difference(self, reference, trial, reference_ci, trial_ci):
        """Exact finite-cell energy change when the objective RDM is unchanged.

        Expand the complete quadratic/quartic integrals in orbital increments,
        rather than subtract two large core totals. This is explicitly limited
        to one-CSF symmetry blocks and identical objective densities. Mixed CI
        response and charge-sector targets require another implementation;
        ``None`` reports that unsupported case, without freezing its density.
        The ket-normalized one-body convention matches the CI evaluator.
        """
        return self._fixed_density_energy_difference(reference, trial, reference_ci, trial_ci)

    def _fixed_density_energy_difference(self, reference, trial, reference_ci, trial_ci,
                                         *, increments=None):
        if (type(self) is not CSFOrbitalFunctional or self.level_tracking is not None
                or any(len(columns) != 1 for _, _, columns in self.blocks)):
            return None
        def objective_density(ci):
            weights = (None if self.state_average is None else
                       self.state_average.weights(self.csfs, ci))
            return self.radial_density.contract(ci, weights)
        weights = objective_density(reference_ci)
        if weights != objective_density(trial_ci):
            return None
        explicit_increments = increments is not None
        if increments is None:
            increments = {key: tuple(new - old for new, old in
                                      zip(trial[key], reference[key], strict=True))
                          for key in self.keys}
        changes, sources = [], {}
        for term, coefficient in weights.items():
            if term[0] != "h":
                _, rank, a, b, c, d = term
                sources.setdefault((rank, *sorted((c, d))), []).append((a, b, coefficient))
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            va, vb = np.concatenate(reference[a]), np.concatenate(reference[b])
            da, db = np.concatenate(increments[a]), np.concatenate(increments[b])
            h, metric = self.one_body[kappa], self.metric[kappa]
            hb, hdb = h @ vb, h @ db
            dn = math.fsum((float(db @ metric @ vb), float(vb @ metric @ db),
                            float(db @ metric @ db)))
            norm = float(vb @ metric @ vb)
            dh = math.fsum((float(da @ hb), float(va @ hdb), float(da @ hdb)))
            delta = (dh - float(va @ hb) / norm * dn) / (norm + dn)
            changes.append(coefficient * delta)
        # Stream one source pair at a time, as in the gradient: memory does
        # not grow with the number of two-body density coordinates.
        for (rank, c, d), terms in sources.items():
            original = self._pair_potential(rank, reference[c], reference[d])
            delta = self._pair_potential(
                rank, reference[c], reference[d],
                change=(increments[c], increments[d]))
            for a, b, coefficient in terms:
                components = []
                for side in (0, 1):
                    va, vb = reference[a][side], reference[b][side]
                    da, db = increments[a][side], increments[b][side]
                    old_b, old_db = original[side] @ vb, original[side] @ db
                    components.extend((float(da @ old_b), float(va @ old_db),
                                       float(da @ old_db)))
                    if explicit_increments:
                        # Evaluate the complete polynomial in the declared
                        # linear-map increment. Adding it to the reference
                        # first would reintroduce deep-core rounding noise.
                        delta_b, delta_db = delta[side] @ vb, delta[side] @ db
                        components.extend((float(va @ delta_b), float(da @ delta_b),
                                           float(va @ delta_db), float(da @ delta_db)))
                    else:
                        components.append(float(trial[a][side] @ delta[side] @ trial[b][side]))
                changes.append(coefficient * math.fsum(components))
        return math.fsum(changes)

    def constrained_energy_difference(self, reference, trial, reference_ci, trial_ci,
                                      *, increments=None):
        """Center a fixed-density difference on the same orthogonal manifold.

        Subtract only the complete metric constraint defect. This Lagrangian
        has exactly the same objective on the constrained physical manifold.
        Both endpoints must already be orthogonal to floating-point precision;
        larger constraint changes are not numerical noise and return ``None``.
        The complete, uncentered difference API above retains its semantics.
        """
        if (type(self) is not CSFOrbitalFunctional or self.level_tracking is not None
                or any(len(columns) != 1 for _, _, columns in self.blocks)):
            return None
        wide = np.longdouble
        groups = {}
        for key in self.keys:
            groups.setdefault(key[1], []).append(key)
        geometry = []
        for kappa, keys in groups.items():
            c = np.column_stack([np.concatenate(reference[key]) for key in keys]).astype(wide)
            t = np.column_stack([np.concatenate(trial[key]) for key in keys]).astype(wide)
            metric = self.metric[kappa].astype(wide)
            identity = np.eye(len(keys), dtype=wide)
            guard = 64 * np.finfo(float).eps * len(keys)
            for columns in (c, t):
                defect = columns.T @ metric @ columns - identity
                if not np.all(np.isfinite(defect)) or np.max(np.sum(np.abs(defect), axis=1)) > guard:
                    return None
            dc = (t - c if increments is None else np.column_stack(
                [np.concatenate(increments[key]) for key in keys]).astype(wide))
            if increments is not None:
                physical_trial = c + dc
                defect = physical_trial.T @ metric @ physical_trial - identity
                if (not np.all(np.isfinite(defect))
                        or np.max(np.sum(np.abs(defect), axis=1)) > guard
                        or np.max(np.abs(physical_trial - t)) >
                        32 * np.finfo(float).eps * max(1, np.max(np.abs(c)))):
                    return None
            # Retain the quadratic term: the linear normal motion in a valid
            # polar step cancels its curvature, rather than changing energy.
            dg = c.T @ metric @ dc + dc.T @ metric @ c + dc.T @ metric @ dc
            geometry.append((keys, c, dg))
        delta = self._fixed_density_energy_difference(
            reference, trial, reference_ci, trial_ci, increments=increments)
        if delta is None:
            return None
        state_weights = (None if self.state_average is None else
                         self.state_average.weights(self.csfs, reference_ci))
        weights = self.radial_density.contract(reference_ci, state_weights)
        actual = {key: value.astype(wide) for key, value in
                  self.gradient_from_weights(reference, weights, record_diagnostics=False).items()}
        # CI one-body integrals divide by the ket norm. Its ambient normal
        # derivative differs from the raw polynomial gradient; their tangent
        # derivatives agree on the normalized physical manifold.
        for term, coefficient in weights.items():
            if term[0] != "h":
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            va, vb = np.concatenate(reference[a]), np.concatenate(reference[b])
            h, metric = self.one_body[kappa], self.metric[kappa]
            ha, hb, sb = h @ va, h @ vb, metric @ vb
            norm = float(vb @ sb)
            actual[a] += coefficient * (1 / norm - 1) * hb
            actual[b] += (coefficient * (1 / norm - 1) * ha
                          - (2 * coefficient * float(va @ hb) / norm**2) * sb)
        normal_changes = []
        for keys, c, dg in geometry:
            g = np.column_stack([actual[key] for key in keys])
            multiplier = c.T @ g
            lagrange = .25 * (multiplier + multiplier.T)
            normal_changes.append(float(np.sum(lagrange * dg)))
        return math.fsum((delta, -math.fsum(normal_changes)))

    def _pair_potential(self, rank, orbital_a, orbital_b, *, change=None):
        """Differentiate the exact four-component cell Slater integral."""
        data, ns = self.data, self.ns
        orders = (data.get("ksp", data["ks"]), data.get("ksq", data["ks"]))
        output = [np.zeros((ns, ns)), np.zeros((ns, ns))]
        def source_density(right, extent):
            if change is None:
                return _density_sym(orbital_a[right], orbital_b[right], extent, orders[right])
            da, db = change
            # Complete product increment, including the quadratic term.
            return (_density_sym(da[right], orbital_b[right], extent, orders[right])
                    + _density_sym(orbital_a[right], db[right], extent, orders[right])
                    + _density_sym(da[right], db[right], extent, orders[right]))
        for left, label_left in enumerate(("p", "q")):
            for right, label_right in enumerate(("p", "q")):
                components = label_left + label_right + label_left + label_right
                if ns >= SEMISEPARABLE_DIRECT_NS:
                    factors = self.cache.get_coulomb_factors(rank, components)
                    extent = ns
                    density = source_density(right, ns)
                    band = _convol_case1_semiseparable_t(*factors, density,
                        ns, orders[left], orders[right])
                else:
                    bank, extent, _ = self.cache.get_rkb_t(rank, components)
                    density = source_density(right, extent)
                    band = _convol_case1_t(bank, density, extent, orders[left], orders[right])
                _banded_to_full_add(output[left], band, extent, orders[left], 1.0)
        return tuple(output)

    def mixed_ci_energy_difference(self, reference, trial, reference_ci, trial_ci,
                                   *, increments):
        """Diagnostic complete finite DC difference for matched isolated roots.

        This opt-in consumer leaves the fixed-density/constrained APIs and
        optimizer unchanged. Two complete physical H owners are mandatory;
        ownership alone does not certify their mutable eigenpairs. Reported
        roots and retained-vector Rayleigh quotients have separate semantics.
        Concurrent CI eigenpair mutation is unsupported. Original columns
        are kept; only one whole-column reference phase is recorded. No lost eigenvalue precision or decision bound is claimed.
        """
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.numerics.ci_energy_increment import real_rank_one_response
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.radial_increment import bspline_dc_radial_increment

        if (type(self) is not CSFOrbitalFunctional or self.state_average is None
                or self.level_tracking is None
                or self.level_tracking.state_average != self.state_average
                or any(len(group) != 1 for group, _ in self.level_tracking._groups)):
            raise ValueError('mixed finite differences require frozen real rank-one DC targets')
        current_keys = tuple(sorted({(sh.n, sh.kappa) for c in self.csfs for sh in c.subshells}))
        if (tuple(csf_signature(c) for c in self.csfs) != self.radial_density.signatures
                or current_keys != self.keys
                or self.max_determinants != self.radial_density.max_determinants
                or len({c.n_electrons for c in self.csfs}) != 1
                or any(c.parity != (-1) ** sum(
                    (sh.kappa if sh.kappa > 0 else -sh.kappa - 1) * sh.occupation
                    for sh in c.subshells) for c in self.csfs)):
            raise ValueError('mixed finite difference has a stale ordered angular consumer')
        if (set(reference) != set(trial) or set(reference) != set(increments)
                or not set(self.keys).issubset(reference)):
            raise ValueError('mixed finite difference endpoint/increment keys mismatch')
        for waves in (reference, trial, increments):
            for pair in waves.values():
                if (len(pair) != 2 or any(np.asarray(v).shape != (self.ns,)
                        or np.asarray(v).dtype != np.dtype(np.float64)
                        or np.iscomplexobj(v) or not np.isfinite(v).all() for v in pair)):
                    raise ValueError('mixed finite difference requires finite real P/Q arrays')
        for ci in (reference_ci, trial_ci):
            if not isinstance(ci, CIResult) or ci.hamiltonian_snapshot is None:
                raise ValueError('mixed finite difference requires complete retained DC CI owners')
            if (np.asarray(ci.state_energies).dtype != np.dtype(np.float64)
                    or np.asarray(ci.mixing_coefficients).dtype != np.dtype(np.float64)):
                raise ValueError('mixed finite difference requires native float64 CI arrays')
            owner = ci.hamiltonian_snapshot
            if (owner.hamiltonian_terms != ('dirac-coulomb',)
                    or owner.representation != 'dense-real-full-CSF' or owner.units != 'Hartree'):
                raise ValueError('mixed finite difference supports only complete native DC owners')
        h0, h1 = tuple(ci.hamiltonian_for(
            self.csfs, waves, self.keys, self.data, self.Z, self.ns,
            nuclear_potential=self.nuclear_potential, max_determinants=self.max_determinants)
            for ci, waves in ((reference_ci, reference), (trial_ci, trial)))
        full_controls = []
        for ci, h, waves in ((reference_ci, h0, reference), (trial_ci, h1, trial)):
            energies, _ = ci_level_blocks(self.csfs, ci, require_full=True)
            mixing = ci.mixing_coefficients
            ci_gram = float(np.max(np.abs(mixing.T @ mixing - np.eye(len(self.csfs)))))
            eigen_residual = float(np.max(np.linalg.norm(h @ mixing - mixing * energies, axis=0)))
            radial_gram = 0.
            for kappa, metric in self.metric.items():
                keys = [key for key in self.keys if key[1] == kappa]
                columns = np.column_stack([np.concatenate(waves[key]) for key in keys])
                radial_gram = max(radial_gram, float(np.max(np.abs(
                    columns.T @ metric @ columns - np.eye(len(keys))))))
            if (not np.isfinite((eigen_residual, ci_gram, radial_gram)).all()
                    or eigen_residual > 1e-8 or ci_gram > 1e-10 or radial_gram > 1e-10):
                raise ValueError('mixed finite difference full eigenpair/Gram validation failed')
            full_controls.append(dict(eigenpair_residual=eigen_residual,
                                      CI_Gram=ci_gram, radial_Gram=radial_gram))
        reconstruction = max(float(np.max(np.abs(
            (np.asarray(t, dtype=np.longdouble) - np.asarray(r, dtype=np.longdouble))
            - np.asarray(d, dtype=np.longdouble))))
            for key in reference for r, d, t in zip(reference[key], increments[key], trial[key], strict=True))
        wave_scale = max(1., max(float(np.max(np.abs(v)))
                                for pair in reference.values() for v in pair))
        if reconstruction > 32 * np.finfo(float).eps * wave_scale:
            raise ValueError('mixed finite difference explicit increment does not reconstruct the trial')
        w0, binding0 = self.level_tracking.resolve(self.csfs, reference_ci, reference, self.metric)
        w1, binding1 = self.level_tracking.resolve(self.csfs, trial_ci, trial, self.metric)
        if len(binding0) != len(binding1):
            raise ValueError('mixed finite difference target ranks changed')
        _, blocks0 = ci_level_blocks(self.csfs, reference_ci, require_full=True)
        _, blocks1 = ci_level_blocks(self.csfs, trial_ci, require_full=True)
        signed_weights, target_components = {}, []
        reported_responses, rayleigh_responses = [], []
        reported_deltas, quotient_deltas = [], []
        for old, new in zip(binding0, binding1, strict=True):
            reference_level = tuple(tuple(v) for v in old['reference_levels'])
            if (reference_level != tuple(tuple(v) for v in new['reference_levels'])
                    or len(reference_level) != 1 or len(old['selected_levels']) != 1
                    or len(new['selected_levels']) != 1
                    or old['per_level_weight'] != new['per_level_weight']):
                raise ValueError('mixed finite difference physical target or weight changed')
            j0, p0, root0 = old['selected_levels'][0]
            j1, p1, root1 = new['selected_levels'][0]
            if (j0, p0) != (j1, p1):
                raise ValueError('mixed finite difference target symmetry changed')
            column0, column1 = int(blocks0[j0, p0][root0]), int(blocks1[j1, p1][root1])
            c0, c1 = reference_ci.mixing_coefficients[:, column0], trial_ci.mixing_coefficients[:, column1]
            weight = float(w0[column0])
            if weight != float(w1[column1]) or weight <= 0:
                raise ValueError('mixed finite difference normalized target weight changed')
            l0, l1 = reference_ci.state_energies[column0], trial_ci.state_energies[column1]
            phase = 1 if np.dot(c0, c1) >= 0 else -1
            response = real_rank_one_response(h0, h1, c0, c1, l0, l1, phase=phase)
            for key, value in self.radial_density.contract_bilinear(c1, c1).items():
                signed_weights[key] = math.fsum((signed_weights.get(key, 0.),
                                                weight * value / response.norm1))
            reported_responses.append(weight * response.reported_response)
            rayleigh_responses.append(weight * response.rayleigh_response)
            root_delta = float(np.longdouble(l1) - np.longdouble(l0))
            reported_deltas.append(weight * root_delta)
            quotient_deltas.append(weight * math.fsum((root_delta,
                response.quotient_minus_reported[1], -response.quotient_minus_reported[0])))
            target_components.append(dict(
                reference_level=reference_level[0], selected_reference=(j0, p0, root0),
                selected_trial=(j1, p1, root1), weight=weight, phase=response.phase,
                norm0=response.norm0, norm1=response.norm1,
                reported_terms=response.reported_terms, rayleigh_terms=response.rayleigh_terms,
                quotient_minus_reported=response.quotient_minus_reported,
                residual_norms=response.residual_norms,
                coefficient_change_norm=response.coefficient_change_norm,
                working_dtype=response.working_dtype,
                mantissa_fraction_bits=int(response.mantissa_fraction_bits)))
        radial = bspline_dc_radial_increment(reference, increments, self.data, self.Z, self.ns,
            nuclear_potential=self.nuclear_potential, cache=self.cache, trial=trial)
        radial_terms, reconstruction_terms = [], []
        for key, coefficient in signed_weights.items():
            explicit, correction = radial.components(key)
            radial_terms.append(coefficient * explicit)
            reconstruction_terms.append(coefficient * correction)
        explicit = math.fsum(radial_terms)
        reconstruction_energy = math.fsum(reconstruction_terms)
        reported_response, rayleigh_response = math.fsum(reported_responses), math.fsum(rayleigh_responses)
        reported_change = math.fsum((explicit, reconstruction_energy, reported_response))
        rayleigh_change = math.fsum((explicit, reconstruction_energy, rayleigh_response))
        direct_reported, direct_quotient = math.fsum(reported_deltas), math.fsum(quotient_deltas)
        floating_eol = float(w1 @ trial_ci.state_energies) - float(w0 @ reference_ci.state_energies)
        # Reject a mutation during the call, without inferring freshness from
        # a matrix pointer, eigenpair contents or convenient provenance flag.
        for ci, waves in ((reference_ci, reference), (trial_ci, trial)):
            ci.hamiltonian_for(self.csfs, waves, self.keys, self.data, self.Z, self.ns,
                nuclear_potential=self.nuclear_potential, max_determinants=self.max_determinants)
        return dict(
            diagnostic_only=True, certified_error_bound=False, optimizer_adapter=False,
            scope='complete real rank-one physically tracked B-spline Dirac-Coulomb',
            explicit_radial=explicit, reconstruction_radial=reconstruction_energy,
            reported_response=reported_response, rayleigh_response=rayleigh_response,
            reported_change=reported_change, rayleigh_change=rayleigh_change,
            direct_reported_root_difference=direct_reported,
            direct_retained_quotient_difference=direct_quotient,
            reported_closure_error=reported_change - direct_reported,
            quotient_closure_error=rayleigh_change - direct_quotient,
            floating_EO_difference=floating_eol,
            floating_averaging_offset=floating_eol - direct_reported,
            signed_density_terms=len(signed_weights),
            negative_signed_density_terms=int(sum(v < 0 for v in signed_weights.values())),
            endpoint_input_sha256=(reference_ci.hamiltonian_snapshot.input_sha256,
                                   trial_ci.hamiltonian_snapshot.input_sha256),
            endpoint_H_sha256=(reference_ci.hamiltonian_snapshot.matrix_sha256,
                               trial_ci.hamiltonian_snapshot.matrix_sha256),
            targets=target_components, full_CI_controls=full_controls, radial=radial.diagnostics())
