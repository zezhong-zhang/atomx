"""Independent variational and cross-engine controls for occupied exchange."""
import numpy as np
import pytest

from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.dirac_fock import DFSolver
from atomx.numerics.radial_metric import RadialMetric
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver


def test_lithium_open_shell_matches_variational_bspline():
    shells, _ = generate_jj_average('1s2 2s1', averaging='statistical')
    solver = ShootingHFSolver(3, shells, nuclear='point')
    result = solver.solve(max_iter=120, tol=1e-10)
    reference = DFSolver(3, shells, nuclear='point', ns=100).solve(max_iter=120, tol=1e-9)
    assert result.converged and reference.converged
    assert abs(result.total_energy-reference.total_energy) < 2e-8
    metric = RadialMetric.shooting(result._grid, result._difr)
    gram = (result._pa*metric.weights) @ result._pa.T + (result._qa*metric.weights) @ result._qa.T
    np.testing.assert_allclose(gram, np.eye(len(shells)), atol=2e-10)
    # A further occupied rotation must leave the stationary energy unchanged.
    energy = solver._rotate_occupied(list(result._pa.copy()), list(result._qa.copy()),
        {-1: [0, 1]}, result._grid, result._difr,
        solver.nuclear_rv_screened(result._grid, result._difr, len(result._grid)))
    assert energy == pytest.approx(result.total_energy, abs=2e-10)


@pytest.mark.parametrize('z, configuration', [
    (19, '1s2 2s2 2p6 3s2 3p6 4s1'),
    (27, '1s2 2s2 2p6 3s2 3p6 3d7 4s2')])
def test_full_exchange_tail_matches_refined_bspline(z, configuration):
    shells, _ = generate_jj_average(configuration, averaging='statistical')
    shells = [s for s in shells if s.occupation > 1e-10]
    result = ShootingHFSolver(z, shells, nuclear='point', n_grid=2400).solve(
        max_iter=100, tol=1e-13)
    reference = DFSolver(z, shells, nuclear='point', ns=100).solve(
        max_iter=100, tol=1e-9)
    assert result.converged and reference.converged
    assert abs(result.total_energy-reference.total_energy) < 2e-9
    assert result.solver_diagnostics['green_boundary'] == 'full_box_scaled'


@pytest.mark.parametrize('averaging', ['statistical', 'equal', 'fractional'])
def test_public_shooting_preserves_open_shell_pair_ensemble(averaging):
    """A p3 configuration needs pair weights, not only mean occupations."""
    from atomx import Atom, Correlation
    from atomx.config.jj_coupling import generate_jj_average
    from atomx.methods.dirac_fock import DFSolver

    config = '1s2 2s2 2p3'
    atom = Atom(10, charge=3, config=config, engine='shooting',
                exchange='hf', nuclear='point', corrections='none',
                correlation=Correlation(averaging=averaging))
    atom.solve(tol=1e-12, max_iter=200)
    shells, ensemble = generate_jj_average(config,
        averaging='equal' if averaging == 'equal' else 'statistical')
    reference = DFSolver(10, shells, nuclear='point', ns=100,
                         jj_configs=None if averaging == 'fractional' else ensemble
                         ).solve(max_iter=150, tol=1e-10)
    assert atom.converged and reference.converged
    assert abs(atom.total_energy-reference.total_energy) < 2e-8
    assert bool(atom.calculation_spec.state.get('jj_configurations')) == (averaging != 'fractional')
    assert atom.provenance['averaging'] == averaging


def test_excited_helium_uses_constrained_pole_count():
    """The 1s pole moves under the occupied-2s orthogonality constraint."""
    from atomx.methods.dirac_fock import DiracFock
    shells, configs = generate_jj_average('1s1 2s1', averaging='statistical')
    shooting = ShootingHFSolver(2, shells, jj_configs=configs, aw=4,
        nuclear='fermi', n_grid=1600).solve(max_iter=150, tol=1e-12)
    reference = DiracFock(breit=False, qed=False, nuclear_mass_number=4).solve(
        2, shells, jj_configs=configs, max_iter=150, tol=1e-9)
    assert shooting.converged and reference.converged
    assert abs(shooting.total_energy-reference.total_energy) < 2e-9
    assert shooting.solver_diagnostics['occupied_rotation_residual_Ha'] < 1e-9
