"""Same physical matrices and independent high-precision near-zero oracle."""

from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.linalg import eigh

from atomx.numerics.hermitian_clusters import refine_real_symmetric_clusters


@pytest.mark.parametrize("case", [0, 1])
def test_original_near_zero_ci_matrix_passes_only_after_explicit_refinement(case):
    with np.load(Path(__file__).parent / "fixtures/ci_near_zero_residual.npz") as data:
        h, w, c = (
            data[f"case{case}_{key}"].copy()
            for key in ("h", "input_energies", "input_coefficients")
        )
    original = h.copy(), w.copy(), c.copy()
    with pytest.raises(ValueError, match="original-matrix residual"):
        refine_real_symmetric_clusters(h, w, c, gap=1e-4)
    rw, rc, info = refine_real_symmetric_clusters(
        h, w, c, gap=1e-4, residual_iterations=2
    )
    assert info["max_original_relative_residual"] < 1e-10
    assert info["gram_error"] < 1e-10
    assert info["isolated_residual_corrections"][0]["column"] == 79
    assert (
        max(row["correction_norm"] for row in info["isolated_residual_corrections"])
        < 1e-12
    )
    assert np.all(np.diff(rw) >= 0)
    for value, before in zip((h, w, c), original, strict=True):
        np.testing.assert_array_equal(value, before)
    # Isolated other columns do not change, including the selected ground root.
    np.testing.assert_array_equal(rc[:, 0], c[:, 0])


def test_refined_near_zero_root_matches_actual_double_matrix_mpmath_oracle():
    vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q = np.eye(5) - 2 * np.outer(vector, vector) / (vector @ vector)
    h = q @ np.diag([-83.0, -42.0, -0.0001, 3.0, 25.0]) @ q.T
    h = 0.5 * (h + h.T)
    w, c = eigh(h)
    w[2] += 5e-13
    with pytest.raises(ValueError, match="original-matrix residual"):
        refine_real_symmetric_clusters(h, w, c, gap=1e-5)
    rw, rc, info = refine_real_symmetric_clusters(
        h, w, c, gap=1e-5, residual_iterations=2
    )
    with mp.workdps(60):
        oracle_w, oracle_c = mp.eigsy(mp.matrix(h.tolist()))
        energy = float(oracle_w[2])
        v = np.array([float(oracle_c[i, 2]) for i in range(5)])
    assert rw[2] == pytest.approx(energy, abs=3e-14)
    np.testing.assert_allclose(np.outer(rc[:, 2], rc[:, 2]), np.outer(v, v), atol=3e-14)
    assert info["max_original_relative_residual"] < 1e-10


def test_explicit_repair_refuses_large_changes_and_split_clusters():
    with pytest.raises(ValueError, match="local energy bound"):
        refine_real_symmetric_clusters(
            np.diag([1.0, 2.0]),
            [1.0, 2.0],
            np.fliplr(np.eye(2)),
            gap=0.1,
            residual_iterations=2,
        )
    h = np.array([[1.0, 0.0, 0.001], [0.0, 1.0 + 1e-6, 0.0], [0.001, 0.0, 2.0]])
    with pytest.raises(ValueError, match="cannot split"):
        refine_real_symmetric_clusters(
            h, [1.0, 1.0 + 1e-6, 2.0], np.eye(3), gap=1e-4, residual_iterations=2
        )


@pytest.mark.parametrize("iterations", [True, -1, 3, 1.5])
def test_invalid_repair_budget_rejected(iterations):
    with pytest.raises(ValueError, match="residual_iterations"):
        refine_real_symmetric_clusters(
            np.eye(2), [1.0, 1.0], np.eye(2), gap=0.1, residual_iterations=iterations
        )
