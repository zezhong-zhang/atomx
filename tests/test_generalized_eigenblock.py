"""Independent full-pencil anchors for jointly orthogonal radial refinements."""

import numpy as np
import pytest

from atomx.numerics.bspline import dirac_assembly


def pencil():
    rng = np.random.default_rng(140325)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    scale = np.geomspace(.1, 10., 8)
    values = np.array([-40., -20., .3, .8, 1.7, 4., 100., 1000.])
    h = (q*values) @ q.T * scale[:, None]*scale[None, :]
    h = np.tril(h)+np.tril(h, -1).T
    s = np.diag(scale**2)
    mixed = q.copy()
    for i, j, theta in [(0, 2, 1e-5), (1, 3, -2e-5), (0, 4, 1.5e-5)]:
        a, b = mixed[:, i].copy(), mixed[:, j].copy()
        mixed[:, i] = np.cos(theta)*a+np.sin(theta)*b
        mixed[:, j] = -np.sin(theta)*a+np.cos(theta)*b
    initial = mixed[:, 2:]/scale[:, None]
    e = np.sum(initial*(h @ initial), axis=0)
    return h, s, e, initial, values[2:], q[:, 2:], scale


def test_full_pencil_correction_repairs_outside_span_and_retains_all_columns():
    h, s, e0, v0, exact, expected, scale = pencil()
    old = [a.copy() for a in (h, s, e0, v0)]
    e, v, info = dirac_assembly.refine_generalized_eigenblock(
        h, s, e0, v0, leading_count=3)
    assert v.shape == (8, 6)
    np.testing.assert_allclose(v.T @ s @ v, np.eye(6), atol=1e-12, rtol=0)
    np.testing.assert_allclose(e[:3], exact[:3], atol=1e-10, rtol=0)
    for j in range(3):
        actual = scale*v[:, j]
        np.testing.assert_allclose(np.outer(actual, actual),
                                   np.outer(expected[:, j], expected[:, j]), atol=1e-10, rtol=0)
    assert info['maximum_leading_dual_residual_Ha'] < 1e-9
    # A fixed-span solver cannot recover the omitted negative-direction part.
    projector = (scale[:, None]*v0) @ (scale[:, None]*v0).T
    assert np.max(abs((np.eye(8)-projector) @ (scale[:, None]*v[:, :3]))) > 1e-6
    for before, after in zip(old, (h, s, e0, v0), strict=True):
        np.testing.assert_array_equal(before, after)
    assert info['bordered_solves_max'] == 6
    assert info['leading_ritz_solves'] == 1


def test_unused_upper_hamiltonian_triangle_does_not_change_operator():
    h, s, e, v, *_ = pencil()
    expected = dirac_assembly.refine_generalized_eigenblock(h, s, e, v, leading_count=3)
    h[np.triu_indices(8, 1)] = 7e8
    actual = dirac_assembly.refine_generalized_eigenblock(h, s, e, v, leading_count=3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize('limits', [dict(dimension_cap=7), dict(column_cap=5),
                                   dict(memory_cap_bytes=1)])
def test_full_block_admission_precedes_factorization(monkeypatch, limits):
    h, s, e, v, *_ = pencil()

    def forbidden(*args, **kwargs):
        pytest.fail('factorization entered before complete block resource admission')

    monkeypatch.setattr(dirac_assembly.linalg, 'cholesky', forbidden)
    monkeypatch.setattr(dirac_assembly.linalg, 'solve', forbidden)
    monkeypatch.setattr(dirac_assembly.linalg, 'eigh', forbidden)
    with pytest.raises(MemoryError, match='resource admission'):
        dirac_assembly.refine_generalized_eigenblock(h, s, e, v, leading_count=3, **limits)


def test_bad_metric_unresolved_roots_and_partial_state_identity_reject():
    h, s, e, v, *_ = pencil()
    with pytest.raises(ValueError, match='isolated'):
        dirac_assembly.refine_generalized_eigenblock(h, s, np.zeros(6), v, leading_count=3)
    with pytest.raises(ValueError, match='Gram'):
        dirac_assembly.refine_generalized_eigenblock(h, s, e, 1.01*v, leading_count=3)
    bad = s.copy()
    bad[0, 1] = .1
    with pytest.raises(ValueError, match='symmetric overlap'):
        dirac_assembly.refine_generalized_eigenblock(h, bad, e, v, leading_count=3)
