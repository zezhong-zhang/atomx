"""Energy-stationary frozen-source plateaus must not exhaust the SCF budget."""
import numpy as np
import pytest
from atomx.config.jj_coupling import generate_jj_average
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver


@pytest.mark.parametrize('z,n,mass',[(7,5,14),(9,6,19),(10,6,20)])
def test_adaptive_plateau_recovery_matches_implicit_exchange(z,n,mass):
    shells,configs=generate_jj_average(f'1s2 {n}g1',averaging='statistical')
    solver=ShootingHFSolver(z,shells,jj_configs=configs,nuclear='fermi',aw=mass,n_grid=3200,r_max=75.)
    adaptive=solver.solve(max_iter=100,tol=1e-12,damping=.5)
    implicit=solver.solve(max_iter=100,tol=1e-12,damping=.5,orbital_solver='nonlocal_resolvent')
    assert adaptive.converged and implicit.converged
    assert adaptive.iterations < 100
    d=adaptive.solver_diagnostics
    assert d['nonlocal_stagnation_promotions'] > 0
    assert d['undamped_orbital_update_norm'] <= d['orbital_update_tolerance']
    assert d['nonlocal_resolvent_max_residual'] <= 1e-7
    assert adaptive.total_energy == pytest.approx(implicit.total_energy,abs=2e-10)
    # Orbital shapes have first-order SCF errors; the variational total energy
    # is stationary to second order. Compare shapes at the requested update
    # tolerance rather than imposing energy-like accuracy on eigenvalues.
    weights=adaptive.radial_state.metric.weights
    for p,q,r,s in zip(adaptive._pa,adaptive._qa,implicit._pa,implicit._qa):
        sign=1. if np.sum((p*r+q*s)*weights)>=0 else -1.
        difference=np.sqrt(np.sum(((p-sign*r)**2+(q-sign*s)**2)*weights))
        assert difference <= 2*d['orbital_update_tolerance']
