"""Independent directed-pencil and rejection gates for the private solver.

Native physical validation is separate. These small nonsymmetric operators
exercise numerical restart, stale images, branch failure and budget rejection.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

OWNER = Path(__file__).resolve().parent
sys.path.insert(0, str(OWNER))
from monovalent_davidson_control import solve  # noqa: E402


class DirectedProblem:
    def __init__(self, *, cap=10**6, stale=False):
        rng = np.random.default_rng(503)
        self.a = rng.normal(0, 0.05, (7, 7))
        self.b = rng.normal(0, 0.04, 7)
        self.e = rng.normal(0, 0.07, 7)
        self.c = -0.003
        self.diagonal = -np.arange(1.0, 8.0)
        self.plan = dict(
            amplitudes={(2, -1): 7},
            single_keys={(2, -1): ((3, -1),)},
            channels={(2, -1): ()},
            complete_payload_bound_bytes=1024,
        )
        self.core = SimpleNamespace(_lock=threading.RLock(), provenance={})
        self.core_result = SimpleNamespace(total_energy_Ha=-1.0)
        self.cap, self.stale, self._matvecs, self.prepares = cap, stale, 0, 0

    def _check(self, extra=0):
        if self.plan["complete_payload_bound_bytes"] + extra > self.cap:
            raise MemoryError("complete toy bound exceeds cap")

    def _prepare(self, key):
        self.prepares += 1
        return None, None, self.diagonal, -0.1

    def _evaluate(self, vector, prepared, *, linear=False):
        self._matvecs += 1
        energy, action = float(self.e @ vector), self.a @ vector
        if self.stale and linear and 0 < np.linalg.norm(vector) < 0.9:
            action[0] += 2e-7
        return (energy, action) if linear else (energy + self.c, action + self.b)

    def exact(self, coupling):
        # Direct full augmented matrix; no Krylov basis or cached images.
        matrix = np.zeros((8, 8))
        matrix[0, 0] = coupling * self.c
        matrix[0, 1:] = coupling * self.e
        matrix[1:, 0] = coupling * self.b
        matrix[1:, 1:] = coupling * self.a - np.diag(self.diagonal)
        values, vectors = np.linalg.eig(matrix)
        index = int(np.argmin(abs(values)))
        return values[index].real, (vectors[1:, index] / vectors[0, index]).real


@pytest.mark.parametrize("subspace", [3, 7, 16])
def test_complete_nonsymmetric_root_and_every_coupling(subspace):
    problem = DirectedProblem()
    result = solve(problem, (2, -1), subspace=subspace, residual_tolerance_Ha=1e-12)
    for coupling, energy, _, raw, defect in result.continuation_path:
        expected, _ = problem.exact(coupling)
        assert abs(energy - expected) < 1e-12
        assert max(raw, defect) <= 1e-12
    expected, amplitudes = problem.exact(1.0)
    assert abs(result.correlation_energy_Ha - expected) < 1e-12
    np.testing.assert_allclose(result.amplitudes, amplitudes, atol=1e-12, rtol=0)
    assert result.residual_max_Ha <= 1e-12
    if subspace == 3:
        assert result.provenance["restarts"] > 0


def test_complete_cap_rejected_before_prepare_or_native_action():
    problem = DirectedProblem(cap=1024)
    with pytest.raises(MemoryError, match="complete"):
        solve(problem, (2, -1))
    assert problem.prepares == problem._matvecs == 0


def test_explicit_native_budget_never_falls_back():
    problem = DirectedProblem()
    with pytest.raises(RuntimeError, match="action budget exhausted"):
        solve(problem, (2, -1), max_actions=2)
    assert problem._matvecs == 2


def test_cached_image_discrepancy_rejected_by_actual_full_map():
    problem = DirectedProblem(stale=True)
    with pytest.raises(RuntimeError, match="cached-image/public complete"):
        solve(problem, (2, -1))


def test_reference_connected_complex_pair_is_rejected():
    problem = DirectedProblem()
    problem.diagonal[:] = -1.0
    problem.a[:] = 0.0
    problem.b[:] = 0.0
    problem.e[:] = 0.0
    problem.b[0], problem.e[0] = 10.0, -10.0
    with pytest.raises(RuntimeError, match="real reference-connected"):
        solve(problem, (2, -1))
