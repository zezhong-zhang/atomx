"""Independent complete42-matrix stress gate for numerical subspaces >16.

This tests a directed pencil similar to a Hermitian arrowhead; it is a
numerical oracle, not an additional atomic physical benchmark.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monovalent_davidson_control import solve  # noqa: E402
from test_monovalent_davidson_control import DirectedProblem  # noqa: E402


class WideProblem(DirectedProblem):
    def __init__(self):
        super().__init__()
        count = 41
        rng = np.random.default_rng(704)
        q, _ = np.linalg.qr(rng.normal(size=(count, count)))
        h = (q * np.geomspace(0.002, 3.0, count)) @ q.T
        scale = np.linspace(0.7, 1.9, count)
        self.a = h * scale[:, None] / scale[None, :] - np.eye(count)
        b = rng.normal(0, 0.006, count)
        self.b, self.e = scale * b, b / scale
        self.diagonal = -np.ones(count)
        self.plan["amplitudes"][2, -1] = count

    def exact(self, coupling):
        count = len(self.b)
        matrix = np.empty((count + 1, count + 1))
        matrix[0, 0] = coupling * self.c
        matrix[0, 1:] = coupling * self.e
        matrix[1:, 0] = coupling * self.b
        matrix[1:, 1:] = coupling * self.a - np.diag(self.diagonal)
        values, vectors = np.linalg.eig(matrix)
        selected = int(np.argmin(values.real))
        assert abs(values[selected].imag) < 1e-13
        return values[selected].real, (
            vectors[1:, selected] / vectors[0, selected]
        ).real


@pytest.mark.parametrize("subspace", [16, 32, 64])
def test_full_directed_pencil_root_with_wide_actual_subspaces(subspace):
    problem = WideProblem()
    result = solve(
        problem,
        (2, -1),
        subspace=subspace,
        max_actions=1000,
        residual_tolerance_Ha=1e-12,
    )
    for coupling, energy, _, raw, defect in result.continuation_path:
        expected, _ = problem.exact(coupling)
        assert abs(energy - expected) < 1e-12
        assert max(raw, defect) <= 1e-12
    expected, vector = problem.exact(1.0)
    assert abs(result.correlation_energy_Ha - expected) < 1e-12
    np.testing.assert_allclose(result.amplitudes, vector, atol=5e-10, rtol=0)
    assert result.residual_max_Ha <= 1e-12
    dimension = max(d["dimension"] for d in result.provenance["diagnostics"])
    if subspace > 16:
        assert dimension > 16
    else:
        assert result.provenance["restarts"] > 0
