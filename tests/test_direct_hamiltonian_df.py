"""Direct DF routing, independent radial discretizations, and coherent fields."""
from copy import deepcopy
import numpy as np
import pytest
from atomx import Atom, Basis, Correlation
from atomx.sampling import Sampling


@pytest.fixture(scope='module')
def carbon():
    return Atom(6, engine='shooting', corrections='none',
                basis=Basis(sampling=Sampling(n_grid=2000))).solve(tol=1e-11)


def test_direct_shooting_avoids_continuum_optimizer(monkeypatch):
    from atomx.methods.mcdhf.solver import MCDHFSolver
    from atomx.methods.csf import grid_solver
    def rejected(*args, **kwargs):
        raise AssertionError('single-configuration DF entered the continuum optimizer')
    monkeypatch.setattr(MCDHFSolver, 'solve', rejected)
    monkeypatch.setattr(grid_solver, 'solve_variational_shooting', rejected)
    atom = Atom(6, engine='shooting', corrections='none').solve()
    assert atom.converged
    diagnostics = atom.result.orbital_optimization
    assert diagnostics['continuum_basis'] is False
    assert diagnostics['exchange'] == 'nonlocal_coherent_csf'
    assert diagnostics['orbital_residual'] >= diagnostics['undamped_orbital_update_norm']
    assert atom.calculation_spec.physics['electron_projector'] == 'self_consistent'
    assert 'electron_space' not in atom.calculation_spec.numerics
    assert atom.calculation_spec.numerics['n_grid'] == 1000


def test_independent_radial_engines_solve_same_stationary_df(carbon):
    bspline = Atom(6, engine='bspline', corrections='none').solve(tol=1e-11)
    assert carbon.converged and bspline.converged
    assert abs(carbon.total_energy-bspline.total_energy) < 2e-9
    assert carbon.calculation_spec.numerics['n_grid'] == 2000
    for key, population in carbon.result.occupations.items():
        assert population == pytest.approx(bspline.result.occupations[key], abs=2e-6)
    for label, energy in carbon.orbital_energies.items():
        assert energy == pytest.approx(bspline.orbital_energies[label], abs=2e-6)


@pytest.mark.parametrize('ns', [80, 120])
def test_explicit_basis_size_keeps_physical_positive_kappa_branch(carbon, ns):
    # Equal P/Q orders produced an oscillatory spurious 2p1/2 root and a
    # falsely converged -49.27 Ha ground state. Direct source equations need
    # the balanced representation for user-specified sizes as well as the
    # default physics grid. Shooting supplies an independent radial oracle.
    result = Atom(6, engine='bspline', corrections='none', basis=Basis(ns=ns)).solve(
        tol=1e-9, max_iter=100, orbital_tolerance=1e-8).result
    assert result.converged
    assert result.bsp_data['ksp'] == result.bsp_data['ksq']-1
    assert result.calculation_spec.numerics['kinetic_balance'] is True
    assert result.calculation_spec.numerics['basis_orders'] == (8, 9)
    assert result.total_energy == pytest.approx(carbon.total_energy, abs=2e-9)
    for label, energy in carbon.orbital_energies.items():
        assert result.orbital_energies[label] == pytest.approx(energy, abs=2e-6)


def test_coherent_coulomb_field_is_energy_derivative(carbon):
    from atomx.config.jj_coupling import generate_jj_average
    from atomx.physics.hartree_fock import RadialHartreeFock
    from atomx.physics.csf.radial_fock import CSFRadialFock
    from atomx.physics.csf.matrix import GridCIWorkspace
    from atomx.potentials.radial_nucleus import screened_nuclear_rv
    result = carbon.result
    r, difr = result._grid, result._difr
    ws = GridCIWorkspace(r, difr, screened_nuclear_rv(6, r, difr, len(r), nuclear='fermi'))
    shells, ensemble = generate_jj_average(carbon._config_strings[0])
    field = CSFRadialFock(deepcopy(shells), result.csfs, ws, RadialHartreeFock(shells, ensemble))
    field.set_state(result)
    pa, qa = np.array([result.orbitals_dict[k] for k in field.keys]).transpose(1,0,2)
    for a, shell in enumerate(field.subshells):
        direction = np.array([pa[a]*np.sin(r), qa[a]*np.cos(r)])
        analytic = 2*shell.occupation*np.sum(field._two_body_action(a, pa, qa)*direction*field.weights)
        energies = []
        for step in (1e-5, -1e-5):
            p, q = pa.copy(), qa.copy()
            p[a] += step*direction[0]
            q[a] += step*direction[1]
            energies.append(field.two_electron_energy(p, q, r, difr, len(r))[0])
        assert (energies[0]-energies[1])/2e-5 == pytest.approx(analytic, rel=2e-7, abs=2e-8)


def test_rotation_energy_preserves_ensemble_and_coherent_functionals(carbon):
    from atomx.config.jj_coupling import generate_jj_average
    from atomx.physics.hartree_fock import RadialHartreeFock
    from atomx.physics.csf.radial_fock import CSFRadialFock
    from atomx.physics.csf.matrix import GridCIWorkspace
    from atomx.potentials.radial_nucleus import screened_nuclear_rv
    result = carbon.result
    r, difr = result._grid, result._difr
    rv = screened_nuclear_rv(6, r, difr, len(r), nuclear='fermi')
    shells, ensemble = generate_jj_average(carbon._config_strings[0])
    average = RadialHartreeFock(deepcopy(shells), ensemble)
    coherent = CSFRadialFock(deepcopy(shells), result.csfs,
                            GridCIWorkspace(r, difr, rv), average)
    coherent.set_state(result)
    p0, q0 = np.array([result.orbitals_dict[k] for k in coherent.keys]).transpose(1,0,2)
    for field in (average, RadialHartreeFock(deepcopy(shells)), coherent):
        p, q = p0.copy(), q0.copy()
        for a, b in ((0, 1), (1, 0)):
            # The second callback starts after the first rotation has changed
            # the orbital set: no fixed contributions may leak between pairs.
            fast = field.rotation_energy(p, q, r, difr, rv, a, b)
            pa, pb, qa, qb = p[a].copy(), p[b].copy(), q[a].copy(), q[b].copy()
            for angle in (-.31, 0., .17, .9):
                c, s = np.cos(angle), np.sin(angle)
                p[a], p[b] = c*pa+s*pb, c*pb-s*pa
                q[a], q[b] = c*qa+s*qb, c*qb-s*qa
                assert fast() == pytest.approx(field.total_energy(p, q, r, difr, rv), abs=3e-12)


def test_direct_df_respects_sweep_limit():
    atom = Atom(6, engine='shooting', corrections='none').solve(max_iter=1)
    assert not atom.converged
    assert atom.result.iterations == 1
    assert atom.result.orbital_optimization['initializer_iterations'] == 1


def test_finite_continuum_space_is_explicit_mcdhf_control():
    from atomx.numerics.shooting.electron_space import ShootingSpace
    with pytest.raises(ValueError, match='explicit MCDHF'):
        Atom(6, basis=Basis(shooting_space=ShootingSpace()))


def test_bspline_selected_gradient_and_scalar_energy_keep_same_hamiltonian():
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    atom = Atom(6, engine='bspline', corrections='none').solve(tol=1e-9)
    result = atom.result
    functional = CSFOrbitalFunctional(result.csfs, result.bsp_data, 6, result._nuclear_potential)
    energy, gradient, ci = functional.evaluate(result.orbitals_dict)
    weights = functional.last_weights
    assert functional.energy_for_weights(result.orbitals_dict, weights) == pytest.approx(energy, abs=1e-10)
    for key in functional.keys:
        _, selected, _ = functional.evaluate(result.orbitals_dict, ci_result=ci, gradient_keys=(key,))
        np.testing.assert_allclose(selected[key], gradient[key], rtol=2e-13, atol=2e-13)
    # A radial variation checks the derivative against independently
    # contracted scalar Slater integrals, with the state amplitudes frozen.
    key = (2, 1)
    p, q = result.orbitals_dict[key]
    direction = np.sin(np.arange(2*len(p))+.3)
    directional = float(gradient[key]@direction)
    energies = []
    integral_cache = {}
    for step in (1e-6, -1e-6):
        orbitals = dict(result.orbitals_dict)
        orbitals[key] = (p+step*direction[:len(p)], q+step*direction[len(p):])
        scalar = functional.energy_for_weights(orbitals, weights)
        assert functional.energy_for_weights(orbitals, weights, integral_cache=integral_cache) == pytest.approx(scalar, abs=1e-10)
        energies.append(scalar)
    assert (energies[0]-energies[1])/2e-6 == pytest.approx(directional, rel=2e-6, abs=2e-8)
