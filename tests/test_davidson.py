"""Smoke tests for the Davidson-Liu CI eigensolver."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from atomx.methods.ci import davidson_eigh


def _make_test_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    # Diagonally dominant — Davidson converges fastest in this regime,
    # which is the typical CI structure (large off-diagonals only inside
    # symmetry blocks).
    H += np.diag(np.linspace(-50.0, 50.0, n))
    return 0.5 * (H + H.T)


def test_davidson_matches_lapack_lowest_states():
    H = _make_test_matrix(150, seed=1)
    e_dav, V_dav, _ = davidson_eigh(H, n_states=4, tol=1e-9)
    e_ref, _ = sla.eigh(H)
    np.testing.assert_allclose(e_dav, e_ref[:4], atol=1e-8)
    # Orthonormal eigenvectors
    np.testing.assert_allclose(V_dav.T @ V_dav, np.eye(4), atol=1e-8)


def test_davidson_residuals_below_tol():
    H = _make_test_matrix(100, seed=2)
    e, V, _ = davidson_eigh(H, n_states=3, tol=1e-10)
    res = H @ V - V * e[None, :]
    rnorms = np.linalg.norm(res, axis=0)
    assert np.all(rnorms < 1e-8)


def test_davidson_rejects_oversized_n_states():
    H = _make_test_matrix(20, seed=3)
    import pytest
    with pytest.raises(ValueError, match="exceeds matrix dim"):
        davidson_eigh(H, n_states=25)


def test_davidson_rejects_zero_max_iter():
    H = _make_test_matrix(10, seed=4)
    import pytest
    with pytest.raises(ValueError, match="max_iter"):
        davidson_eigh(H, n_states=2, max_iter=0)


def test_davidson_rejects_zero_block_size():
    H = _make_test_matrix(10, seed=5)
    import pytest
    with pytest.raises(ValueError, match="block_size"):
        davidson_eigh(H, n_states=2, block_size=0)


def test_davidson_raises_when_not_converged():
    H = _make_test_matrix(80, seed=6)
    import pytest
    with pytest.raises(RuntimeError, match="did not converge"):
        davidson_eigh(H, n_states=4, tol=1e-14, max_iter=1)
