"""Physical projector identity, degeneracy gauge, and variational integration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_ci_basis_overlap import rotated_case

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.physics.csf.levels import CILevel
from atomx.physics.csf.overlap import ci_subspace_overlaps
from atomx.physics.csf.state_average import CIChargeStateAverage, CIStateAverage
from atomx.physics.csf.tracking import CIBasisTracking, CITrackingAmbiguity


@pytest.mark.parametrize('number', [1, 2, 3])
def test_tracking_uses_physical_exterior_power_not_sorted_roots(number):
    csfs, bra, ket, old, new, selectors, metric = rotated_case(number)
    s = selectors[0]
    target = CIStateAverage(((s.J2, s.parity, s.root, 1),))
    tracker = CIBasisTracking(target, csfs, bra, old, metric)
    # Reverse columns AND energies: the physical state retains its energy
    # but now occupies a different global column. Then reverse energy order
    # within its block, retaining all physical eigenvectors.
    permutation = np.arange(len(csfs))[::-1]
    trial = SimpleNamespace(state_energies=-ket.state_energies[permutation]-1,
                            mixing_coefficients=ket.mixing_coefficients[:, permutation])
    weights, diagnostics = tracker.resolve(csfs, trial, new, metric)
    expected = np.flatnonzero(permutation == 0)[0]
    np.testing.assert_allclose(weights, np.eye(len(csfs))[expected], atol=1e-14)
    assert diagnostics[0]['principal_overlap_squared'][0] == pytest.approx(1)
    # A rejected/alternate trial cannot change a subsequent resolution.
    tracker.resolve(csfs, bra, old, metric)
    again, _ = tracker.resolve(csfs, trial, new, metric)
    np.testing.assert_array_equal(again, weights)


def simple_case():
    keys = ((1, -1), (2, -1), (3, -1))
    csfs = enumerate_jj_orbital_space(keys, 1)
    ci = SimpleNamespace(state_energies=np.array([-3., -3., -1.]),
                         mixing_coefficients=np.eye(3))
    waves = {key: (np.eye(3)[:, i], np.zeros(3)) for i, key in enumerate(keys)}
    return csfs, ci, waves, {-1: np.eye(6)}


def test_complete_degenerate_projector_is_invariant_under_internal_rotation():
    csfs, ci, waves, metric = simple_case()
    target = CIStateAverage(((1, 1, 0, 1), (1, 1, 1, 1)))
    tracker = CIBasisTracking(target, csfs, ci, waves, metric)
    for angle in (0., .1, np.pi/4, 1.5):
        rotation = np.eye(3)
        rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]]
        trial = SimpleNamespace(state_energies=ci.state_energies,
                                mixing_coefficients=rotation)
        weights, info = tracker.resolve(csfs, trial, waves, metric)
        np.testing.assert_array_equal(weights, [.5, .5, 0])
        np.testing.assert_allclose(info[0]['principal_overlap_squared'], [1, 1], atol=2e-14)
    with pytest.raises(ValueError, match='degeneracy'):
        CIBasisTracking(CIStateAverage(((1, 1, 0, 1),)), csfs, ci, waves, metric)
    with pytest.raises(ValueError, match='complete degenerate'):
        ci_subspace_overlaps(csfs, ci, waves, [CILevel(1, 1, 0)],
                             csfs, ci, waves, [CILevel(1, 1, 0)], metric)


@pytest.mark.parametrize('number', [2, 3])
def test_degenerate_many_body_projector_survives_both_radial_and_CI_gauge(number):
    csfs, bra, ket, old, new, selectors, metric = rotated_case(number)
    symmetry = selectors[0].J2, selectors[0].parity
    indices = [i for i, s in enumerate(selectors) if (s.J2, s.parity) == symmetry][:2]
    assert len(indices) == 2
    bra.state_energies[indices] = -100
    ket.state_energies[indices] = -100
    target = CIStateAverage(tuple((*symmetry, root, 1.) for root in (0, 1)))
    tracker = CIBasisTracking(target, csfs, bra, old, metric)
    for angle in (.2, np.pi/4, 1.3):
        rotation = np.eye(len(csfs))
        rotation[np.ix_(indices, indices)] = [[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle)]]
        trial = SimpleNamespace(state_energies=ket.state_energies,
                                mixing_coefficients=ket.mixing_coefficients @ rotation)
        weights, info = tracker.resolve(csfs, trial, new, metric)
        expected = np.zeros(len(csfs))
        expected[indices] = .5
        np.testing.assert_array_equal(weights, expected)
        np.testing.assert_allclose(info[0]['principal_overlap_squared'], [1, 1], atol=3e-14)


def test_unresolved_single_root_refuses_and_snapshot_is_owned():
    csfs, ci, waves, metric = simple_case()
    ci.state_energies = np.array([-3., -2., -1.])
    tracker = CIBasisTracking(CIStateAverage(((1, 1, 0, 1),)), csfs, ci, waves, metric)
    old_csfs, old_waves, old_metric = deepcopy(csfs), deepcopy(waves), deepcopy(metric)
    ci.mixing_coefficients[:] = 0
    ci.state_energies[:] = 900
    waves[1, -1][0][:] = 0
    metric[-1][:] = 0
    csfs[0].total_J2 = 9
    trial = SimpleNamespace(state_energies=np.array([-3., -2., -1.]),
                            mixing_coefficients=np.eye(3))
    weights, _ = tracker.resolve(old_csfs, trial, old_waves, old_metric)
    np.testing.assert_array_equal(weights, [1, 0, 0])
    angle = np.pi/4
    trial.mixing_coefficients[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]]
    with pytest.raises(CITrackingAmbiguity, match='ambiguous'):
        tracker.resolve(old_csfs, trial, old_waves, old_metric)
    trial.mixing_coefficients = np.eye(3)
    trial.state_energies[:2] = -3
    with pytest.raises(CITrackingAmbiguity, match='degeneracy'):
        tracker.resolve(old_csfs, trial, old_waves, old_metric)


def test_missing_radial_support_refuses_even_when_score_boundary_is_unique():
    csfs, ci, waves, metric = simple_case()
    ci.state_energies = np.array([-3., -2., -1.])
    # Orthogonal old/new radial spaces: no minor has physical support.
    old = {k: (np.r_[v[0], np.zeros(3)], np.zeros(6)) for k, v in waves.items()}
    new = {k: (np.r_[np.zeros(3), v[0]], np.zeros(6)) for k, v in waves.items()}
    metric = {-1: np.eye(12)}
    tracker = CIBasisTracking(CIStateAverage(), csfs, ci, old, metric)
    with pytest.raises(CITrackingAmbiguity, match='principal overlap'):
        tracker.resolve(csfs, ci, new, metric)


def test_equal_cluster_energy_derivative_survives_a_linear_level_crossing():
    csfs, ci, waves, metric = simple_case()
    target = CIStateAverage(((1, 1, 0, 1), (1, 1, 1, 1)))
    tracker = CIBasisTracking(target, csfs, ci, waves, metric)
    values = []
    # Exact independent trace anchor: the first 2x2 eigenvalue sum equals
    # -6+4*t even when the internal eigenvectors/energy order change.
    for t in (-1e-6, 0, 1e-6):
        h = np.array([[-3+3*t, t, 0], [t, -3+t, 0], [0, 0, -1.]])
        energies, vectors = np.linalg.eigh(h)
        trial = SimpleNamespace(state_energies=energies, mixing_coefficients=vectors)
        weights, _ = tracker.resolve(csfs, trial, waves, metric)
        values.append(weights @ energies)
    assert (values[2]-values[0])/2e-6 == pytest.approx(2., abs=5e-10)


def test_subspace_matrix_preserves_signed_entries_and_rejects_duplicate_selectors():
    csfs, ci, waves, metric = simple_case()
    levels = [CILevel(1, 1, i) for i in range(3)]
    changed = SimpleNamespace(state_energies=ci.state_energies.copy(),
                              mixing_coefficients=np.diag([-1., 1., -1.]))
    matrix = ci_subspace_overlaps(csfs, ci, waves, levels,
                                 csfs, changed, waves, levels, metric)
    np.testing.assert_allclose(matrix, np.diag([-1., 1., -1.]), atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match='duplicate'):
        ci_subspace_overlaps(csfs, ci, waves, levels+levels,
                             csfs, changed, waves, levels, metric)


@pytest.fixture(scope='module')
def atom_seed():
    from atomx import Atom, Basis, Correlation
    return Atom(10, config=['1s2', '1s1 2p1', '2s2'], basis=Basis(ns=40, ks=9),
                corrections='none', correlation=Correlation(radial_optimization='fac_frozen_local')).solve(tol=1e-6)


def test_tracked_joint_energy_and_gradient_share_exact_trial_and_identity(atom_seed, monkeypatch):
    from atomx.methods.csf.charge_variational import ChargeStateOrbitalFunctional
    r = atom_seed.result
    parent = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3), (2, -1, 0, 2)))
    target = CIChargeStateAverage(((2, 7, parent), (1, 3, CIStateAverage(statistical=True))))
    plain = ChargeStateOrbitalFunctional(r.csfs, r.bsp_data, 10, r._nuclear_potential, state_average=target)
    waves = r.orbitals_dict
    e, g, reference = plain.evaluate(waves)
    trackers = {n: CIBasisTracking(levels, plain.sector_csfs[n], reference.sectors[n], waves, plain.metric)
                for n, _, levels in plain.charge_sectors}
    tracked = ChargeStateOrbitalFunctional(r.csfs, r.bsp_data, 10, r._nuclear_potential,
                                         state_average=target, level_tracking=trackers)
    et, gt, trial = tracked.evaluate(waves)
    with pytest.raises(TypeError):
        tracked.sector_tracking[2] = trackers[2]
    with pytest.raises(TypeError):
        trial.tracking[2] = trial.tracking[2]
    with pytest.raises(ValueError):
        trial.tracking[2][0][0] = .9
    assert et == e
    for key in g:
        np.testing.assert_array_equal(gt[key], g[key])
    assert tracked.energy_for_weights(waves, tracked.last_weights) == pytest.approx(et, abs=1e-9)
    keys = [key for key in tracked.keys if key[1] == -1]
    x = np.column_stack([np.concatenate(waves[key]) for key in keys])
    metric = tracked.metric[-1]
    d = np.random.default_rng(53).normal(size=x.shape)
    m = x.T @ metric @ d
    d -= x @ (.5*(m+m.T))
    d /= np.sqrt(np.trace(d.T @ metric @ d))
    analytic = sum(gt[key] @ d[:, i] for i, key in enumerate(keys))
    values = []
    for h in (-1e-6, 1e-6):
        y = x+h*d
        w, v = np.linalg.eigh(y.T @ metric @ y)
        y = y @ ((v/np.sqrt(w)) @ v.T)
        changed = dict(waves)
        for i, key in enumerate(keys):
            changed[key] = y[:tracked.ns, i], y[tracked.ns:, i]
        values.append(tracked.evaluate(changed, gradient=False)[0])
    assert (values[1]-values[0])/2e-6 == pytest.approx(analytic, rel=3e-6, abs=3e-7)
    def forbidden(*args, **kwargs):
        raise AssertionError('accepted trial rediagonalized or rematched')
    monkeypatch.setattr(CIBasisTracking, 'resolve', forbidden)
    monkeypatch.setattr('atomx.methods.ci.ci_solve', forbidden)
    accepted_energy, accepted_g, accepted = tracked.evaluate(waves, ci_result=trial)
    assert accepted is trial and accepted_energy == et
    for key in gt:
        np.testing.assert_array_equal(accepted_g[key], gt[key])


def test_tracking_target_and_geometry_mismatch_refuse(atom_seed):
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    r = atom_seed.result
    target = CIStateAverage()
    f = CSFOrbitalFunctional(r.csfs, r.bsp_data, 10, r._nuclear_potential, state_average=target)
    _, _, ci = f.evaluate(r.orbitals_dict)
    tracker = CIBasisTracking(target, r.csfs, ci, r.orbitals_dict, f.metric)
    with pytest.raises(ValueError, match='reference this'):
        CSFOrbitalFunctional(r.csfs, r.bsp_data, 10, state_average=None, level_tracking=tracker)
    metric = {k: v.copy() for k, v in f.metric.items()}
    metric[-1][0, 0] += 1e-12
    with pytest.raises(ValueError, match='reference radial geometry'):
        tracker.resolve(r.csfs, ci, r.orbitals_dict, metric)
    tracked = CSFOrbitalFunctional(r.csfs, r.bsp_data, 10, r._nuclear_potential,
                                 state_average=target, level_tracking=tracker)
    energy, gradient, _ = tracked.evaluate(r.orbitals_dict, ci_result=ci)
    assert energy == pytest.approx(np.mean(ci.state_energies), abs=1e-12)
    for key in gradient:
        np.testing.assert_array_equal(gradient[key], f.last_gradient[key])
