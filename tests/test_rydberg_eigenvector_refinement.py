"""Targeted refinement must retain the root and occupied-orbital constraints."""
import numpy as np
import pytest
from scipy import linalg

from atomx.numerics.bspline import dirac_assembly as assembly


@pytest.mark.parametrize('constrained', [False, True])
@pytest.mark.parametrize('root,kappa,tolerance', [(12,-1,None),(9,-7,None),(0,-1,1e-12)])
def test_requested_root_refinement_preserves_metric_and_constraints(monkeypatch, constrained, root, kappa, tolerance):
    ns = 20
    keep, grid = assembly._get_keep_grid(ns, ns, ns, 1, 2)
    size = len(keep)
    rng = np.random.default_rng(19)
    q, _ = linalg.qr(rng.normal(size=(size, size)))
    metric = np.diag(np.linspace(0.7, 1.8, size))
    energies = -0.5 / np.arange(1, size+1)**2
    transform = np.diag(1/np.sqrt(np.diag(metric))) @ q
    h = metric @ transform @ np.diag(energies) @ transform.T @ metric
    H, S = np.zeros((2*ns, 2*ns)), np.eye(2*ns)
    H[grid], S[grid] = h, metric
    constraint = np.zeros((2*ns, 1))
    constraint[keep, 0] = transform[:, 0]
    original = assembly._eigh_finite

    def noisy(H, S):
        w, v = original(H, S)
        # Emulate a full-spectrum solver's small error in a diffuse vector.
        v[:, root] += 1e-4 * v[:, root+1]
        v[:, root] /= np.sqrt(v[:, root] @ S @ v[:, root])
        w[root] += 1e-10
        return w, v

    monkeypatch.setattr(assembly, '_eigh_finite', noisy)
    result = assembly.solve_dirac(H, S, ns, kappa,
                                 orthogonal_to=constraint if constrained else None)
    result.get_orbital(root, tolerance=tolerance)
    vector = result.coefficients[keep, root]
    expected = energies[root+1 if constrained else root]
    assert result.bound_energies[root] == pytest.approx(expected, abs=2e-15)
    assert vector @ metric @ vector == pytest.approx(1, abs=2e-14)
    assert linalg.norm(h @ vector - expected * (metric @ vector)) < 2e-14
    if constrained:
        assert abs(constraint[keep, 0] @ metric @ vector) < 2e-14
    assert result._refined_indices == {root}
    # Repeated reads do not repeat the expensive correction.
    before = vector.copy()
    result.get_orbital(root)
    np.testing.assert_array_equal(result.coefficients[keep, root], before)


def test_loose_request_skips_newton_and_tighter_request_rechecks(monkeypatch):
    ns = 8
    keep, grid = assembly._get_keep_grid(ns, ns, ns, 1, 2)
    H, S = np.zeros((2*ns, 2*ns)), np.eye(2*ns)
    H[grid] = np.diag(-.5/np.arange(1, len(keep)+1)**2)
    original = assembly._eigh_finite
    def noisy(h, s):
        w, v = original(h, s)
        v[:, 0] += 1e-7*v[:, 1]
        v[:, 0] /= np.linalg.norm(v[:, 0])
        return w, v
    monkeypatch.setattr(assembly, '_eigh_finite', noisy)
    refine = assembly._refine_eigenpair
    calls = []
    def count(*args):
        calls.append(1)
        return refine(*args)
    monkeypatch.setattr(assembly, '_refine_eigenpair', count)
    result = assembly.solve_dirac(H, S, ns, -1)
    before = result.coefficients.copy()
    result.get_orbital(0, tolerance=1e-5)
    assert not calls
    np.testing.assert_array_equal(result.coefficients, before)
    result.get_orbital(0, tolerance=1e-12)
    assert len(calls) == 1
    vector = result.coefficients[keep, 0]
    assert np.linalg.norm(H[grid] @ vector + .5*vector) < 1e-14
    for bad in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match='tolerance'):
            result.get_orbital(0, tolerance=bad)


def test_neon_tight_scf_reaches_requested_orbital_accuracy():
    from atomx import Atom, Basis
    a = Atom(10, engine='bspline', nuclear='point', basis=Basis(ns=120),
             corrections='none').solve(tol=1e-11, max_iter=50)
    result = a.result
    assert result.converged
    assert result.residual < 1e-11
    assert result.total_energy == pytest.approx(-128.69196949453365, abs=2e-10)
    assert result.calculation_spec.numerics['eigenpair_tolerance'] == pytest.approx(1e-14)


def test_accuracy_estimate_retains_negative_energy_mode_errors(monkeypatch):
    ns = 8
    keep, grid = assembly._get_keep_grid(ns, ns, ns, 1, 2)
    H, S = np.zeros((2*ns, 2*ns)), np.eye(2*ns)
    H[grid] = np.diag(np.r_[-100000., -.5/np.arange(1, len(keep))**2])
    original = assembly._eigh_finite
    def noisy(h, s):
        w, v = original(h, s)
        v[:, 1] += 1e-7*v[:, 0]  # sea admixture, invisible in electron-only projections
        v[:, 1] /= np.linalg.norm(v[:, 1])
        return w, v
    monkeypatch.setattr(assembly, '_eigh_finite', noisy)
    result = assembly.solve_dirac(H, S, ns, -1)
    result.get_orbital(0, tolerance=1e-8)
    assert result._refined_indices == {0}
    assert abs(result.coefficients[keep[0], 0]) < 1e-14
