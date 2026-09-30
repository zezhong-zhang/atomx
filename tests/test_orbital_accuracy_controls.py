"""Independent orbital stopping accuracy must survive public dispatch."""
import numpy as np
import pytest

from atomx import Atom, Basis, Sampling, RadialRefinement
from atomx.config.subshell import Subshell
from atomx.methods.dirac_fock import DFSolver, DiracFock
from atomx.numerics.radial_metric import orbital_distance


def test_distance_is_phase_and_representation_invariant():
    angle = 1e-10
    left = np.array([1., 0.])
    right = np.array([np.cos(angle), np.sin(angle)])
    # An overlap-subtraction formula would round this distance to zero.
    expected = 2*np.sin(angle/2)
    assert orbital_distance(left, -right, np.ones(2)) == pytest.approx(expected, rel=1e-14)
    transform = np.diag([1e-4, 1e4])
    metric = transform.T @ transform
    assert orbital_distance(np.linalg.solve(transform, left),
                            np.linalg.solve(transform, right), metric) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(FloatingPointError, match='nonfinite'):
        orbital_distance(left, [np.nan, 0.], np.ones(2))


def test_bspline_damping_cannot_certify_undamped_orbital_convergence():
    def solve(threshold):
        return DFSolver(2, [Subshell(1, -1, 2)], ns=50,
            include_breit=False, include_qed=False).solve(
                max_iter=3, tol=.5, damping=.9999,
                damping_release_residual=1e-20, orbital_tolerance=threshold)
    # Deliberately loose energy/coefficient controls isolate the L2 gate.
    assert solve(None).converged
    result = solve(.03)
    diagnostics = result.solver_diagnostics
    assert diagnostics['applied_orbital_update_norm'] < .03
    assert diagnostics['undamped_orbital_update_norm'] > .03
    assert not result.converged


@pytest.mark.parametrize('mode,exchange', [('orbital', 'hf'), ('batch', 'slater'), ('batch', 'gu')])
def test_other_bspline_iterations_measure_the_undamped_field(mode, exchange):
    result = DFSolver(2, [Subshell(1, -1, 2)], ns=60, exchange=exchange,
        include_breit=False, include_qed=False).solve(
            max_iter=120, tol=1e-7, mode=mode, orbital_tolerance=1e-7)
    assert result.converged
    diagnostics = result.solver_diagnostics
    assert diagnostics['undamped_orbital_update_norm'] < 1e-7
    assert diagnostics['applied_orbital_update_norm'] < 1e-7
    if exchange != 'hf':
        # Potential mixing reduces the applied update; report the full
        # unmixed-field orbital defect separately.
        assert diagnostics['undamped_orbital_update_norm'] > 4*diagnostics['applied_orbital_update_norm']


@pytest.mark.parametrize('engine', ['shooting', 'bspline'])
@pytest.mark.parametrize('z', [6, 10])
def test_public_df_checks_and_records_requested_orbital_accuracy(engine, z):
    atom = Atom(z, engine=engine, corrections='none',
        basis=Basis(ns=80, sampling=Sampling(n_grid=1000, orbital_tolerance=1e-5)))
    atom.solve(tol=1e-8, max_iter=100, orbital_tolerance=1e-7)
    result = atom.result
    diagnostics = result.orbital_optimization if z == 6 else result.solver_diagnostics
    assert result.converged
    assert result.calculation_spec.method == 'DHF'
    assert result.calculation_spec.numerics['orbital_tolerance'] == 1e-7
    assert diagnostics['orbital_update_tolerance'] <= 1e-7
    assert diagnostics['undamped_orbital_update_norm'] <= 1e-7
    if engine == 'bspline':
        assert diagnostics['applied_orbital_update_norm'] <= 1e-7
    if z == 6:
        assert diagnostics['state_density_residual'] <= 1e-7
        assert result.iterations == diagnostics['iterations']


def test_sampling_override_reaches_every_adaptive_solve(monkeypatch):
    original = Atom._solve_fixed
    requests = []
    def record(self, *args, **kwargs):
        requests.append(kwargs.get('orbital_tolerance'))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Atom, '_solve_fixed', record)
    atom = Atom(1, nuclear='point', corrections='none',
        basis=Basis(sampling=Sampling(orbital_tolerance=1e-4)))
    atom.solve(sampling=Sampling(orbital_tolerance=1e-8,
                                radial_refinement=RadialRefinement()))
    assert atom.converged
    assert len(requests) >= 2
    assert set(requests) == {1e-8}
    assert atom.calculation_spec.numerics['orbital_tolerance'] == 1e-8


def test_explicit_method_and_persistence_keep_orbital_controls(tmp_path):
    from atomx.methods.dirac_fock import SCFResult
    result = Atom(2, corrections='none', basis=Basis(ns=60)).solve(
        DiracFock(breit=False, qed=False), tol=1e-8, orbital_tolerance=1e-7)
    assert result.converged
    assert result.solver_diagnostics['orbital_update_tolerance'] == 1e-7
    assert result.calculation_spec.numerics['orbital_tolerance'] == 1e-7
    path = tmp_path/'helium.h5'
    result.save(path)
    loaded = SCFResult.load(path)
    assert loaded.solver_diagnostics == result.solver_diagnostics
    assert loaded.calculation_spec.to_dict() == result.calculation_spec.to_dict()


@pytest.mark.parametrize('kwargs', [dict(config=['1s2', '2s2']),
    dict(engine='shooting', exchange='gu')])
def test_unsupported_routes_reject_the_control(kwargs):
    with pytest.raises(ValueError, match='orbital_tolerance'):
        Atom(2, corrections='none', **kwargs).solve(orbital_tolerance=1e-7)


def test_explicit_mcdhf_does_not_reinterpret_l2_as_gradient():
    from atomx.methods.mcdhf import MCDHF
    with pytest.raises(ValueError, match='orbital_tolerance'):
        Atom(2, corrections='none').solve(MCDHF(), orbital_tolerance=1e-7)


def test_single_csf_hamiltonian_df_honors_the_threshold():
    from atomx.methods.hamiltonian_df import HamiltonianDFSolver
    result = HamiltonianDFSolver(2, '1s2', engine='bspline', ns=60,
        include_breit=False, include_qed=False, use_physics_grid=False).solve(
            scf_tol=1e-6, orbital_tolerance=1e-9, max_iter=40)
    assert result.converged
    assert result.orbital_optimization['orbital_update_tolerance'] == 1e-9
    assert result.orbital_optimization['undamped_orbital_update_norm'] < 1e-9


@pytest.mark.parametrize('value', [0., -1., np.inf, np.nan])
def test_invalid_orbital_controls(value):
    with pytest.raises(ValueError, match='orbital_tolerance'):
        Sampling(orbital_tolerance=value)
    with pytest.raises(ValueError, match='orbital_tolerance'):
        Atom(2, corrections='none').solve(orbital_tolerance=value)
