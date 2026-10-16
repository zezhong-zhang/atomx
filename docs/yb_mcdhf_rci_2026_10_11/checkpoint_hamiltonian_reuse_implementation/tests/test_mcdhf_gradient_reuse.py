"""Exact radial action reuse must preserve signed constrained CI derivatives."""
from collections import Counter

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.state_average import CIStateAverage


@pytest.fixture(scope="module", params=(35, 120))
def coherent_problem(request):
    """Native signed CI ensemble, below and above the cell-kernel threshold."""
    ns, ks = request.param, 9
    data = prepare_bspline_data(make_knots(ns, ks, .005, 30), ns, ks)
    configs = ("1s2", "1s1 2s1", "1s1 3s1", "2s2", "2s1 3s1", "3s2",
               "2p2", "2p1 3p1", "3p2")
    csfs = [c for cfg in configs for c in enumerate_jj_csfs(cfg)
            if c.total_J2 == 0 and c.parity == 1]
    functional = CSFOrbitalFunctional(csfs, data, 2,
        state_average=CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3))))
    orbitals, projectors = {}, {}
    for kappa in functional.metric:
        spectrum = solve_dirac(functional.one_body[kappa], functional.metric[kappa], ns, kappa)
        projectors[kappa] = spectrum.coefficients
        ell = kappa if kappa > 0 else -kappa - 1
        for n, k in functional.keys:
            if k == kappa:
                p, q, _ = spectrum.get_orbital(n - ell - 1 + int(kappa > 0))
                orbitals[n, k] = p, q
    _, _, ci = functional.evaluate(orbitals)
    return functional, orbitals, projectors, ci


def literal_gradient(functional, orbitals, weights, selected=None):
    """Preserved pre-optimization contraction, including every signed term."""
    selected = functional.keys if selected is None else selected
    gradient = {k: np.zeros(2 * functional.ns) for k in selected}
    vectors = {k: np.concatenate(orbitals[k]) for k in functional.keys}
    terms = {}
    for key, weight in weights.items():
        if key[0] == "h":
            _, na, kappa, nb = key
            a, b = (na, kappa), (nb, kappa)
            if a in gradient:
                gradient[a] += weight * (functional.one_body[kappa] @ vectors[b])
            if b in gradient:
                gradient[b] += weight * (functional.one_body[kappa] @ vectors[a])
        else:
            _, rank, a, b, c, d = key
            for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                if left in gradient or right in gradient:
                    terms.setdefault((rank, *sorted((source_a, source_b))), []).append((left, right, weight))
    for (rank, source_a, source_b), items in terms.items():
        pp, qq = functional._pair_potential(rank, orbitals[source_a], orbitals[source_b])
        for left, right, weight in items:
            if left in gradient:
                gradient[left][:functional.ns] += weight * (pp @ orbitals[right][0])
                gradient[left][functional.ns:] += weight * (qq @ orbitals[right][1])
            if right in gradient:
                gradient[right][:functional.ns] += weight * (pp @ orbitals[left][0])
                gradient[right][functional.ns:] += weight * (qq @ orbitals[left][1])
    return gradient


@pytest.mark.parametrize("transition", (False, True))
@pytest.mark.parametrize("partial", (False, True))
def test_all_signed_actions_equal_literal_gradient(coherent_problem, transition, partial):
    functional, orbitals, _, ci = coherent_problem
    weights = (functional.radial_density.contract_bilinear(ci.mixing_coefficients[:, 0],
                                                          ci.mixing_coefficients[:, 1])
               if transition else functional.last_weights)
    selected = functional.keys[::2] if partial else None
    expected = literal_gradient(functional, orbitals, weights, selected)
    actual = functional.gradient_from_weights(orbitals, weights, gradient_keys=selected,
                                              record_diagnostics=False)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_response_preserves_accepted_state_and_invalidates_actions(coherent_problem):
    functional, orbitals, _, ci = coherent_problem
    weights = functional.radial_density.contract_bilinear(ci.mixing_coefficients[:, 0],
                                                         ci.mixing_coefficients[:, 1])
    old_weights, old_gradient = functional.last_weights, functional.last_gradient
    old_occupations = dict(functional.last_occupations)
    actual = functional.gradient_from_weights(orbitals, weights, record_diagnostics=False)
    assert functional.last_weights is old_weights
    assert functional.last_gradient is old_gradient
    assert functional.last_occupations == old_occupations
    changed = {key: (p.copy(), q.copy()) for key, (p, q) in orbitals.items()}
    changed[functional.keys[0]][0][7] += 1e-4
    fresh = functional.gradient_from_weights(changed, weights, record_diagnostics=False)
    expected = literal_gradient(functional, changed, weights)
    for key in fresh:
        np.testing.assert_array_equal(fresh[key], expected[key])
    assert any(not np.array_equal(actual[k], fresh[k]) for k in fresh)


def test_reuse_does_not_repeat_matrix_action_on_same_spinor(coherent_problem, monkeypatch):
    functional, orbitals, _, _ = coherent_problem
    weights = dict(functional.last_weights)
    actions = Counter()
    original = functional._pair_potential

    class CountActions:
        def __init__(self, matrix, source, component):
            self.matrix, self.source, self.component = matrix, source, component

        def __matmul__(self, vector):
            actions[self.source, self.component, id(vector)] += 1
            return self.matrix @ vector

    def counted_pair(rank, a, b):
        pp, qq = original(rank, a, b)
        source = rank, id(a[0]), id(b[0])
        return CountActions(pp, source, "p"), CountActions(qq, source, "q")

    monkeypatch.setattr(functional, "_pair_potential", counted_pair)
    functional.gradient_from_weights(orbitals, weights, record_diagnostics=False)
    assert actions
    assert max(actions.values()) == 1


@pytest.mark.parametrize("transition", (False, True))
def test_independent_derivative_in_metric_positive_energy_tangent(coherent_problem, transition):
    functional, orbitals, projectors, ci = coherent_problem
    weights = (functional.radial_density.contract_bilinear(ci.mixing_coefficients[:, 0],
                                                          ci.mixing_coefficients[:, 1])
               if transition else functional.last_weights)
    gradient = functional.gradient_from_weights(orbitals, weights, record_diagnostics=False)
    rng = np.random.default_rng(81217)
    direction = {}
    for kappa, projector in projectors.items():
        keys = [k for k in functional.keys if k[1] == kappa]
        x = np.column_stack([np.concatenate(orbitals[k]) for k in keys])
        metric = functional.metric[kappa]
        d = projector @ rng.normal(size=(projector.shape[1], len(keys)))
        cross = x.T @ metric @ d
        d -= x @ (0.5 * (cross + cross.T))
        d /= np.sqrt(np.trace(d.T @ metric @ d))
        for i, key in enumerate(keys):
            direction[key] = d[:, i]
    analytical = sum(gradient[k] @ direction[k] for k in functional.keys)
    energies = []
    for step in (1e-6, -1e-6):
        trial = {}
        for kappa in functional.metric:
            keys = [k for k in functional.keys if k[1] == kappa]
            x = np.column_stack([np.concatenate(orbitals[k]) + step * direction[k] for k in keys])
            values, vectors = np.linalg.eigh(x.T @ functional.metric[kappa] @ x)
            x = x @ ((vectors / np.sqrt(values)) @ vectors.T)
            for i, key in enumerate(keys):
                trial[key] = x[:functional.ns, i].copy(), x[functional.ns:, i].copy()
        energies.append(functional.energy_for_weights(trial, weights) if transition
                        else functional.evaluate(trial, gradient=False)[0])
    numerical = (energies[0] - energies[1]) / 2e-6
    assert numerical == pytest.approx(analytical, rel=3e-6, abs=2e-7)
