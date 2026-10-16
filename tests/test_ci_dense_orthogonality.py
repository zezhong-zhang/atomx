"""Native clustered CI block: closure must survive a change of energy zero."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from atomx.methods.ci import ci_solve
from atomx.methods.mcdhf import make_csf_list


@pytest.mark.parametrize("energy_zero", [0.0, 100.0, -1000.0])
@pytest.mark.parametrize("n_states", [None, 2])
def test_clustered_dense_ci_preserves_orthogonality_and_residual(
    monkeypatch, energy_zero, n_states,
):
    data = json.loads((Path(__file__).parent / "data/ne_clustered_ci_block.json").read_text())
    matrix = np.array(data["matrix_hartree"])
    matrix += energy_zero * np.eye(len(matrix))
    monkeypatch.setattr("atomx.physics.csf.matrix.build_ci_matrix", lambda *a, **k: matrix)
    # Only the eigensolver is under test: use one symmetry label per row.
    csfs = make_csf_list(Z=2, config_strings=["1s2"]) * len(matrix)
    result = ci_solve(csfs, {}, (), None, 10, 60, n_states=n_states)
    values, vectors = result.state_energies, result.mixing_coefficients
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=2e-14, rtol=0)
    np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-12, rtol=0)
    expected, _ = eigh(matrix, driver="ev")
    np.testing.assert_allclose(values, expected[:n_states], atol=2e-12, rtol=0)
    if n_states is None:
        np.testing.assert_allclose(vectors @ vectors.T, np.eye(len(matrix)), atol=2e-14, rtol=0)
        np.testing.assert_allclose((vectors * values) @ vectors.T, matrix, atol=2e-12, rtol=0)


@pytest.mark.parametrize("coupling_sign", [-1.0, 1.0])
@pytest.mark.parametrize("n_states", [None, 1])
def test_signed_large_zero_dense_ci_matches_analytic_projectors(
    monkeypatch, coupling_sign, n_states,
):
    """Public complete/subset CI has an independent signed spectral oracle."""
    offset, diagonal = -float(2**14), 2.0**-13
    coupling = coupling_sign * 2.0**-17
    matrix = np.array([[offset - diagonal, coupling], [coupling, offset + diagonal]])
    original_bytes = matrix.tobytes()
    monkeypatch.setattr("atomx.physics.csf.matrix.build_ci_matrix", lambda *a, **k: matrix)
    csfs = make_csf_list(Z=2, config_strings=["1s2"]) * 2
    result = ci_solve(csfs, {}, (), None, 2, 2, n_states=n_states)
    count = 2 if n_states is None else n_states
    radius = math.hypot(diagonal, coupling)
    angle = 0.5 * math.atan2(coupling, diagonal)
    expected_vectors = np.array([
        [math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)],
    ])
    expected_values = (offset + np.array([-radius, radius]))[:count]
    values, vectors = result.state_energies, result.mixing_coefficients
    np.testing.assert_allclose(values, expected_values, atol=1e-8, rtol=0)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(count), atol=1e-10, rtol=0)
    np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-8, rtol=0)
    for column in range(count):
        np.testing.assert_allclose(
            np.outer(vectors[:, column], vectors[:, column]),
            np.outer(expected_vectors[:, column], expected_vectors[:, column]),
            atol=1e-12, rtol=0,
        )
    assert result.provenance["dense_centering_shifts_Ha"] == [offset]
    assert matrix.tobytes() == original_bytes
