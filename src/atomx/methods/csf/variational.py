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
    SEMISEPARABLE_DIRECT_NS,
    _convol_case1_semiseparable_t,
    compute_slater_rk_cell,
    _banded_to_full_add,
    _convol_case1_t,
    _density_sym,
)
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices


class CSFOrbitalFunctional:
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
        from atomx.physics.csf.radial_density import CIRadialDensity
        self.radial_density = CIRadialDensity(csfs)
        self.blocks = self.radial_density.blocks

    def evaluate(self, orbitals, *, gradient=True, ci_result=None, gradient_keys=None):
        result = ci_result
        if result is None:
            result = ci_solve(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                              nuclear_potential=self.nuclear_potential, convol_cache=self.cache)
        energy = float(result.ground_energy)
        if not gradient:
            return energy, result
        weights = self.radial_density.contract(result)
        self.last_weights = weights
        selected = self.keys if gradient_keys is None else tuple(gradient_keys)
        if any(key not in self.keys for key in selected):
            raise ValueError('gradient requested for an orbital outside the CSF basis')
        gradient = {key: np.zeros(2 * self.ns) for key in selected}
        self.last_occupations = {key: max(0., weights.get(("h", key[0], key[1], key[0]), 0.))
                                 for key in self.keys}
        vectors = {key: np.concatenate(orbitals[key]) for key in self.keys}
        gradient_terms = {}

        for key, weight in weights.items():
            if key[0] == "h":
                _, na, kappa, nb = key
                a, b = (na, kappa), (nb, kappa)
                h = self.one_body[kappa]
                if a in gradient:
                    gradient[a] += weight * (h @ vectors[b])
                if b in gradient:
                    gradient[b] += weight * (h @ vectors[a])
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
            for left, right, weight in terms:
                pl, ql = orbitals[left]
                pr, qr = orbitals[right]
                if left in gradient:
                    gradient[left][:self.ns] += weight * (pp @ pr)
                    gradient[left][self.ns:] += weight * (qq @ qr)
                if right in gradient:
                    gradient[right][:self.ns] += weight * (pp @ pl)
                    gradient[right][self.ns:] += weight * (qq @ ql)
        self.last_gradient = gradient
        return energy, gradient, result

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

    def _pair_potential(self, rank, orbital_a, orbital_b):
        """Differentiate the exact four-component cell Slater integral."""
        data, ns = self.data, self.ns
        orders = (data.get("ksp", data["ks"]), data.get("ksq", data["ks"]))
        output = [np.zeros((ns, ns)), np.zeros((ns, ns))]
        for left, label_left in enumerate(("p", "q")):
            for right, label_right in enumerate(("p", "q")):
                components = label_left + label_right + label_left + label_right
                if ns >= SEMISEPARABLE_DIRECT_NS:
                    factors = self.cache.get_coulomb_factors(rank, components)
                    extent = ns
                    density = _density_sym(orbital_a[right], orbital_b[right], ns, orders[right])
                    band = _convol_case1_semiseparable_t(*factors, density,
                        ns, orders[left], orders[right])
                else:
                    bank, extent, _ = self.cache.get_rkb_t(rank, components)
                    density = _density_sym(orbital_a[right], orbital_b[right], extent, orders[right])
                    band = _convol_case1_t(bank, density, extent, orders[left], orders[right])
                _banded_to_full_add(output[left], band, extent, orders[left], 1.0)
        return tuple(output)
