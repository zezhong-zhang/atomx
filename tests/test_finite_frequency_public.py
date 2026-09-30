"""Public transverse-exchange requests must retain their physical meaning."""
import numpy as np
import pytest

from atomx import Atom, Basis, Corrections, Correlation, Sampling
from atomx.atom_inputs import resolve_corrections
from atomx.corrections.stack import CorrectionStack, finalize_with_corrections
from atomx.physics.csf.breit_ci import (
    _ci_radial_inputs, breit_operator_energy, breit_configuration_average)


def finite(**kwargs):
    return Corrections(qed=False, breit_frequency='orbital_differences', **kwargs)


@pytest.mark.parametrize('engine', ['shooting', 'bspline'])
def test_dhf_public_finite_operator_replaces_static_without_mc(engine, monkeypatch):
    # Explicit operator mode must not compute a redundant static matrix.
    import atomx.physics.csf.breit_ci as breit
    monkeypatch.setattr(breit, 'attach_breit_operator_energy',
                        lambda *a, **k: pytest.fail('redundant static Breit'))
    stack = CorrectionStack(se_backend=None, include_vp=False,
                            breit_mode='operator', breit_frequency='orbital_differences')
    atom = Atom(10, engine=engine, nuclear='point', basis=Basis(ns=80),
                corrections=Corrections(stack=stack)).solve(tol=1e-9, max_iter=80)
    result = atom.result
    assert result.converged
    assert result.calculation_spec.method == 'DHF'
    orbitals, workspace, z, ns = _ci_radial_inputs(result)
    energies = {(s.n, s.kappa): result.orbital_energies[s.label] for s in result.subshells}
    expected = breit_operator_energy(orbitals, workspace, z, ns, '1s2 2s2 2p6',
                                    orbital_energies=energies)
    static = breit_operator_energy(orbitals, workspace, z, ns, '1s2 2s2 2p6')
    assert expected != pytest.approx(static, abs=1e-6)
    ledger = result.correction_application
    assert set(dict(ledger.shifts)) == {'breit'}
    assert dict(ledger.shifts)['breit'] == pytest.approx(expected, abs=2e-12)
    assert result.total_energy == pytest.approx(ledger.base_energy+expected, abs=2e-12)
    provenance = result.__dict__['_breit_provenance']
    assert provenance['energy_source'] == 'converged_orbital_energies'
    assert len(provenance['orbital_energies_Ha']) == 4
    assert 'finite-frequency' in provenance['operator']
    with pytest.raises(ValueError, match='already finalized'):
        finalize_with_corrections(result, stack)


def test_explicit_zero_frequencies_recover_static_open_shell_ensemble():
    atom = Atom(6, engine='shooting', nuclear='point',
                basis=Basis(sampling=Sampling(n_grid=2000)),
                correlation=Correlation(averaging='statistical'),
                corrections=finite(breit_orbital_energies={
                    (1, -1): 0., (2, -1): 0., (2, 1): 0., (2, -2): 0.})).solve(
                        tol=1e-9, max_iter=100)
    result = atom.result
    assert result.converged
    orbitals, workspace, z, ns = _ci_radial_inputs(result)
    expected = breit_configuration_average(orbitals, workspace, z, ns, result._jj_configs)
    assert result.energy_components['E_Breit'] == pytest.approx(expected, abs=2e-12)
    assert result._breit_provenance['energy_source'] == 'explicit'


@pytest.mark.parametrize('engine', ['shooting', 'bspline'])
def test_public_multiconfiguration_final_hamiltonian(engine):
    from atomx.methods.ci import ci_solve
    from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
    from atomx.physics.csf.matrix import GridCIWorkspace
    from atomx.numerics.shooting.electron_space import ShootingSpace
    energies = {(1, -1): -1.9, (2, -1): -.3}
    space = ShootingSpace(n_grid=600, outer_intervals=300, continuum_roots=12)
    basis = Basis(r_max=80, shooting_space=space) if engine == 'shooting' else Basis(ns=40, r_max=25)
    atom = Atom(2, config=['1s2', '2s2'], engine=engine, nuclear='point',
                basis=basis, corrections=finite(breit_orbital_energies=energies)).solve(
                    tol=1e-5, max_iter=2)
    result = atom.result
    # A deliberately small orbital space/sweep budget checks finalization,
    # not complete variational convergence or spectroscopic accuracy.
    csfs = enumerate_jj_csfs_many(['1s2', '2s2'])
    if engine == 'shooting':
        workspace = GridCIWorkspace(result._grid, result._difr, np.full_like(result._grid, -2.))
        ns, potential = workspace.n, None
    else:
        workspace = result.bsp_data
        ns, potential = workspace['ns'], result._nuclear_potential
    kwargs = dict(csfs=csfs, orbitals_dict=result.orbitals_dict,
                  orbital_basis=get_shared_orbital_basis(csfs), bsp_data=workspace,
                  Z=2, ns=ns, nuclear_potential=potential)
    expected = ci_solve(**kwargs, include_breit=True, breit_orbital_energies=energies)
    dc = ci_solve(**kwargs, include_breit=False)
    np.testing.assert_allclose(result.state_energies, expected.state_energies, atol=2e-10, rtol=0)
    assert result.energy_components['E_Breit'] == pytest.approx(
        expected.state_energies[0]-dc.state_energies[0], abs=2e-10)
    assert result.breit_provenance['energy_source'] == 'explicit'
    assert ('breit', 'ci_hamiltonian') in result.operator_stages
    assert '_E_Breit_scalar_not_applied' not in vars(result)
    stack = CorrectionStack(se_backend=None, include_vp=False,
                            breit_frequency='orbital_differences',
                            breit_orbital_energies=energies)
    with pytest.raises(ValueError, match='already included'):
        finalize_with_corrections(result, stack)


def test_frequency_inputs_fail_before_correlation_solve():
    with pytest.raises(ValueError, match='explicit orbital energies'):
        Atom(2, config=['1s2', '2s2'], corrections=finite()).solve()
    with pytest.raises(ValueError, match='every orbital'):
        Atom(2, config=['1s2', '2s2'], corrections=finite(
            breit_orbital_energies={(1, -1): -2.})).solve()
    with pytest.raises(ValueError, match='finite'):
        finite(breit_orbital_energies={(1, -1): np.nan})
    with pytest.raises(ValueError, match='requires'):
        Corrections(breit=False, breit_frequency='orbital_differences')
    with pytest.raises(ValueError, match='requires'):
        Corrections(breit_orbital_energies={(1, -1): -2.})
    with pytest.raises(ValueError, match='legacy'):
        CorrectionStack(breit_mode='scalar_legacy', breit_frequency='orbital_differences')
    with pytest.raises(ValueError, match='contradicts'):
        resolve_corrections(Corrections(stack=CorrectionStack(), breit_frequency='orbital_differences'))
    stack = CorrectionStack(breit_frequency='orbital_differences',
                            breit_orbital_energies={(1, -1): -2.})
    with pytest.raises(ValueError, match='contradicts'):
        resolve_corrections(finite(stack=stack, breit_orbital_energies={(1, -1): -1.}))
    assert resolve_corrections(finite()).breit_frequency == 'orbital_differences'


def test_method_objects_share_the_frequency_controls():
    from atomx.methods.dirac_fock import DiracFock
    from atomx.methods.mcdhf import MCDHF
    from atomx.config.subshell import Subshell
    from atomx.types import NuclearModel
    df = DiracFock(breit=True, qed=False, nuclear=NuclearModel.POINT,
                   breit_frequency='orbital_differences')
    result = df.solve(2, [Subshell(1, -1, 2)], tol=1e-8, ns=45)
    assert result.converged
    assert result._breit_provenance['operator'] == 'finite-frequency-transverse'
    mc = MCDHF(breit=True, qed=False, nuclear=NuclearModel.POINT,
                breit_frequency='orbital_differences',
                breit_orbital_energies={(1, -1): -1.9})
    result = mc.solve(2, ['1s2'], tol=1e-8, ns=45)
    assert result.converged
    assert result.breit_provenance['operator'] == 'finite-frequency-transverse-in-blocks'
