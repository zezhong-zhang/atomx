"""Warm starts must reuse waves without changing the shooting HF problem."""
from dataclasses import replace

import numpy as np
import pytest

from atomx.config.jj_coupling import generate_jj_average
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
from atomx.numerics.shooting.local import ShootingSolver


@pytest.mark.parametrize('z,config', [(2, '1s2'), (10, '1s2 2s2 2p6')])
def test_warm_start_reuses_radials_and_reaches_same_solution(z, config, monkeypatch):
    shells, _ = generate_jj_average(config, averaging='statistical')
    shells = [s for s in shells if s.occupation > 1e-10]
    solver = ShootingHFSolver(z, shells, nuclear='point')
    cold = solver.solve(max_iter=100, tol=1e-10)
    assert cold.converged
    saved_p, saved_q = cold._pa.copy(), cold._qa.copy()
    def forbid_starter(*args, **kwargs):
        pytest.fail('warm start called the local-exchange starter')
    monkeypatch.setattr(ShootingSolver, 'solve', forbid_starter)
    warm = solver.solve(max_iter=100, tol=1e-10, seed=cold)
    assert warm.converged
    assert warm.iterations <= cold.iterations
    if z == 10:
        assert warm.iterations < cold.iterations
    assert abs(warm.total_energy-cold.total_energy) < 2e-7
    np.testing.assert_array_equal(cold._pa, saved_p)
    np.testing.assert_array_equal(cold._qa, saved_q)
    with pytest.raises(ValueError, match='same Z'):
        solver.solve(seed=replace(cold, Z=z+1))
    with pytest.raises(ValueError, match='spinors'):
        solver.solve(seed=replace(cold, _pa=np.full_like(cold._pa, np.nan)))
    with pytest.raises(ValueError, match='radial grid'):
        ShootingHFSolver(z, shells, nuclear='point', n_grid=500).solve(seed=cold)


def test_warm_start_allows_changed_guerra_weights():
    from atomx.config.jj_coupling import JJConfig
    from atomx.methods.dirac_fock import Subshell
    def shells(qm, qp):
        return [Subshell(1, -1, 2.), Subshell(2, -1, 2.),
                Subshell(2, 1, qm), Subshell(2, -2, qp)]
    def solver(weight):
        return ShootingHFSolver(6, shells(weight, 2-weight),
            jj_configs=[JJConfig(subshells=shells(1., 1.), weight=weight),
                        JJConfig(subshells=shells(0., 2.), weight=1-weight)])
    seed = solver(.5).solve(max_iter=150, tol=1e-10)
    target = solver(.6)
    cold = target.solve(max_iter=150, tol=1e-10)
    warm = target.solve(max_iter=150, tol=1e-10, seed=seed)
    assert cold.converged and warm.converged
    assert abs(cold.total_energy-warm.total_energy) < 1e-7
