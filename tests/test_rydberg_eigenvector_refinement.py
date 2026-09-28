"""Targeted refinement must retain the root and occupied-orbital constraints."""
import numpy as np
import pytest
from scipy import linalg

from atomx.numerics.bspline import dirac_assembly as assembly


@pytest.mark.parametrize('constrained', [False, True])
@pytest.mark.parametrize('root,kappa', [(12,-1),(9,-7)])
def test_requested_root_refinement_preserves_metric_and_constraints(monkeypatch, constrained, root, kappa):
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
    result.get_orbital(root)
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
