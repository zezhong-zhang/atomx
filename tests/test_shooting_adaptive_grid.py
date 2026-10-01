"""Refinement must expose unresolved discretization and preserve the model."""
from dataclasses import replace

import numpy as np
import pytest

from atomx import Atom, Basis, RadialRefinement, Sampling
from atomx.numerics.shooting.adaptive import RadialSample, refine_radial_grid
from atomx.numerics.shooting.dirac_orbital import RadialBoxError


def manufactured(n, radius, *, error=0., decay=2.):
    r = np.linspace(0., radius, n)
    weights = np.full(n, radius/(n-1))
    weights[[0, -1]] *= .5
    return RadialSample(None, -1.+error, {'1s': -.5+error}, r, weights,
        np.array([r*r*np.exp(-decay*r)]), True, np.zeros(1))


def test_independent_grid_and_box_checks():
    calls = []
    def solve(n, radius):
        calls.append((n, radius))
        return manufactured(n, radius, error=100/n**4 + np.exp(-radius))
    final, audit = refine_radial_grid(solve, n_grid=100, r_max=30.,
        policy=RadialRefinement(), vary_box=True)
    assert audit['radial_space_converged']
    assert audit['history'][-1]['check'] == 'box'
    assert abs(final.energy+1) < 1e-6
    assert calls[0][1] == calls[1][1] < calls[-1][1]


def test_tail_expansion_tracks_diffuse_orbital_not_core():
    def solve(n, radius):
        sample = manufactured(n, radius, decay=.4)
        sample.densities = np.vstack((sample.densities,
            sample.radius**2*np.exp(-20*sample.radius)))
        return sample
    _, audit = refine_radial_grid(solve, n_grid=100, r_max=40.,
        policy=RadialRefinement(max_solves=10), vary_box=True)
    assert audit['radial_space_converged']
    assert any(row['check'] == 'tail_expansion' for row in audit['history'])
    assert audit['history'][-1]['tail_norm'] <= 1e-10


def test_only_box_errors_are_recoverable():
    def solve(n, radius):
        if radius < 20:
            raise RadialBoxError('bound-state bracket collapsed')
        return manufactured(n, radius)
    _, audit = refine_radial_grid(solve, n_grid=100, r_max=15.,
        policy=RadialRefinement(), vary_box=True)
    assert audit['radial_space_converged']
    assert audit['history'][0]['status'] == 'box_too_small'
    with pytest.raises(RadialBoxError):
        refine_radial_grid(solve, n_grid=100, r_max=15.,
            policy=RadialRefinement(), vary_box=False)
    def broken(n, r):
        raise FloatingPointError('failed matching')
    with pytest.raises(FloatingPointError, match='matching'):
        refine_radial_grid(broken, n_grid=100, r_max=30.,
            policy=RadialRefinement(), vary_box=True)


@pytest.mark.parametrize('control,reason', [
    (dict(max_points=100), 'point_limit'),
    (dict(max_solves=2), 'solve_limit'),
    (dict(max_radius=30.), 'radius_limit'),
])
def test_limits_are_not_convergence(control, reason):
    _, audit = refine_radial_grid(manufactured, n_grid=100, r_max=30.,
        policy=RadialRefinement(**control), vary_box=True)
    assert not audit['converged']
    assert audit['termination_reason'] == reason


def test_scf_failure_does_not_trigger_grid_refinement():
    def solve(n, radius):
        return replace(manufactured(n, radius), converged=False)
    _, audit = refine_radial_grid(solve, n_grid=100, r_max=30.,
        policy=RadialRefinement(), vary_box=True)
    assert not audit['converged']
    assert audit['termination_reason'] == 'scf_not_converged'
    assert len(audit['history']) == 1


def test_refinement_uses_actual_point_count():
    def solve(n, radius):
        return manufactured(max(n, 160), radius)
    _, audit = refine_radial_grid(solve, n_grid=50, r_max=30.,
        policy=RadialRefinement(), vary_box=False)
    assert audit['history'][1]['actual_points'] > audit['history'][0]['actual_points']
    assert audit['grid_converged']


def test_fixed_box_has_no_box_convergence_claim():
    _, audit = refine_radial_grid(manufactured, n_grid=100, r_max=30.,
        policy=RadialRefinement(), vary_box=False)
    assert audit['converged'] and audit['grid_converged']
    assert not audit['radial_space_converged']
    assert {row['r_max'] for row in audit['history']} == {30.}


@pytest.mark.parametrize('kwargs', [dict(energy_atol=0), dict(tail_norm=1),
    dict(scf_tol=np.nan), dict(max_solves=1), dict(max_points=50.5),
    dict(max_solves=8.0)])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        RadialRefinement(**kwargs)


def test_hydrogenic_radial_refinement_and_persistence(tmp_path):
    from atomx.numerics.shooting.constants import SL
    from atomx.results import ShootingResult
    atom = Atom(1, nuclear='point', corrections='none',
        basis=Basis(sampling=Sampling(radial_refinement=RadialRefinement())))
    atom.solve()
    assert atom.converged
    assert atom.result.radial_convergence['radial_space_converged']
    exact = -1/(1+np.sqrt(1-1/SL**2))
    assert abs(atom.total_energy-exact) < 1e-8
    assert atom.r_max is None and atom._n_grid is None
    path = tmp_path/'hydrogen.h5'
    atom.result.save(path)
    loaded = ShootingResult.load(path)
    assert loaded.radial_convergence == atom.result.radial_convergence


def test_coherent_df_refines_complete_state():
    atom = Atom(6, corrections='none', basis=Basis(r_max=50.,
        sampling=Sampling(radial_refinement=RadialRefinement()))).solve()
    audit = atom.result.radial_convergence
    assert atom.converged and audit['grid_converged']
    assert not audit['box_converged']
    assert audit['history'][-1]['excitation_change_Ha'] < 1e-6
    assert atom.result.method_name == 'DHF'
    assert atom.r_max == 50.
    assert atom.calculation_spec.numerics['radial_refinement']['energy_atol'] == 1e-6


def test_explicit_mcdhf_does_not_silently_use_df_refinement():
    atom = Atom(2, config=['1s2', '1s1 2s1'],
        basis=Basis(sampling=Sampling(radial_refinement=RadialRefinement())))
    with pytest.raises(ValueError, match='single-configuration'):
        atom.solve()


def test_atom_does_not_report_success_after_refinement_limit():
    atom = Atom(2, corrections='none', basis=Basis(sampling=Sampling(
        radial_refinement=RadialRefinement(max_points=1000)))).solve()
    assert not atom.converged
    assert atom.result.radial_convergence['termination_reason'] == 'point_limit'
    assert atom.result.radial_convergence['history'][0]['scf_converged']


def test_fixed_box_failure_restores_requested_controls():
    atom = Atom(1, config='7s1', corrections='none', basis=Basis(r_max=2.,
        sampling=Sampling(radial_refinement=RadialRefinement())))
    with pytest.raises(RadialBoxError):
        atom.solve()
    assert atom.r_max == 2. and atom._n_grid is None
    assert atom.result is None
