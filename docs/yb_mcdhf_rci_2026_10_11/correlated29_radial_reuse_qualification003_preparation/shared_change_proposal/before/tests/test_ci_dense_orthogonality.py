"""Native clustered CI block: closure must survive a change of energy zero."""
import json
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
