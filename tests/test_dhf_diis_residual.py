"""DIIS must receive the residual of each actual SCF sweep."""
import numpy as np

from atomx import Atom
from atomx.methods import dirac_fock as df


def test_diis_history_uses_actual_inputs(monkeypatch):
    flatten = df.DFSolver._flatten_orbital_vector
    extrapolate = df.diis_extrapolate
    snapshots = []
    checked = []

    def capture(*args, **kwargs):
        vector = flatten(*args, **kwargs)
        snapshots.append(vector.copy())
        return vector

    def verify(outputs, errors):
        assert len(snapshots) % 2 == 0
        actual = [snapshots[i+1]-snapshots[i]
                  for i in range(0, len(snapshots), 2)]
        for expected, supplied in zip(actual[-len(errors):], errors, strict=True):
            np.testing.assert_array_equal(supplied, expected)
        assert np.linalg.norm(errors[0]) > 0
        checked.append(len(errors))
        return extrapolate(outputs, errors)

    monkeypatch.setattr(df.DFSolver, '_flatten_orbital_vector', capture)
    monkeypatch.setattr(df, 'diis_extrapolate', verify)
    result = Atom(Z=12, config='1s2 2s2 2p6 3s2', engine='bspline').solve(
        df.DiracFock(starting_potential='hydrogenic', scf_damping=0.35),
        max_iter=60)
    assert checked
    assert result.converged
