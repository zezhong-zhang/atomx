"""Hamiltonian DF: intermediate coupling, coherent energy, and Eq. 10 density."""
import numpy as np
import pytest

from atomx import Atom, Correlation
from atomx.config.jj_coupling import compute_WW, generate_jj_average
from atomx.methods.dirac_fock import SCFResult
from atomx.methods.mcdhf.solver import MCDHFSolver
from atomx.results import ShootingResult


@pytest.mark.parametrize('configuration,ell,electrons', [
    ('2p1', 1, 1), ('2p2', 1, 2), ('3d7', 2, 7), ('5f10', 3, 10),
])
def test_statistical_pair_moments_match_integer_configuration_trace(
        configuration, ell, electrons):
    shells, ensemble = generate_jj_average(configuration, averaging='statistical')
    capacity = 4*ell+2
    degeneracies = [2*ell, 2*ell+2]
    for i in range(2):
        assert shells[i].occupation == pytest.approx(electrons*degeneracies[i]/capacity)
        for j in range(2):
            # Select two distinct electrons from a uniformly filled LS shell.
            expected = (electrons*(electrons-1)*degeneracies[i]
                        *(degeneracies[j]-(i == j))/(capacity*(capacity-1)))
            assert compute_WW(i, j, ensemble) == pytest.approx(expected, abs=1e-14)
    assert sum(compute_WW(i, j, ensemble) for i in range(2) for j in range(2)) == pytest.approx(electrons*(electrons-1))
    # In particular a one-electron shell has no two-electron self interaction.
    if electrons == 1:
        assert compute_WW(0, 1, ensemble) == 0
        assert shells[0].occupation*shells[1].occupation > 0


@pytest.fixture(scope='module', params=['bspline', 'shooting'])
def carbon(request):
    return Atom(6, engine=request.param, corrections='none').solve()


def test_hamiltonian_occupations_come_from_eigenvectors(carbon):
    from atomx.methods.hamiltonian_df import HamiltonianDFResult
    result = carbon.result
    assert carbon.converged
    assert isinstance(result, HamiltonianDFResult)
    assert not isinstance(result, MCDHFSolver.result_type)
    assert carbon._csf_result is result
    assert result.method_name == carbon.calculation_spec.method == 'DHF'
    assert carbon.provenance['averaging'] == 'hamiltonian'
    assert carbon.calculation_spec.state['averaging']['weights'] == 'hamiltonian_eigenvectors'
    assert carbon.calculation_spec.state['configuration_expansion'] is False
    keys = tuple(result.occupations)
    q = np.array([c.occupation_vector(keys) for c in result.csfs])
    ground = abs(result.state_energies-result.ground_energy) < 1e-10
    weights = np.mean(abs(result.mixing_coefficients[:, ground])**2, axis=1)
    np.testing.assert_allclose(q.T@weights, list(result.occupations.values()), atol=1e-13)
    assert sum(result.occupations.values()) == pytest.approx(6)
    # Nonrelativistic 3P0 recoupling predicts 4/3, 2/3, not the statistical
    # 2/3, 4/3. Small relativistic departures are expected for carbon.
    assert result.occupations[2, 1] == pytest.approx(4/3, abs=.01)
    assert result.occupations[2, -2] == pytest.approx(2/3, abs=.01)
    for csf in result.csfs:
        nr = {}
        for sh in csf.subshells:
            nr[sh.n, sh.l] = nr.get((sh.n, sh.l), 0)+sh.occupation
        assert nr == {(1, 0): 2, (2, 0): 2, (2, 1): 2}
    diagnostics = result.orbital_optimization
    assert diagnostics['orbital_residual'] < diagnostics['orbital_tolerance']
    assert np.max(np.diff(diagnostics['energy_history'])) < 1e-10


def test_hamiltonian_state_differs_from_statistical_ensemble(carbon):
    statistical = Atom(6, engine=carbon.engine, corrections='none',
                       correlation=Correlation(averaging='statistical')).solve()
    assert carbon.total_energy < statistical.total_energy-.01
    assert statistical.calculation_spec.state['averaging']['method'] == 'configuration_average'


def test_hamiltonian_keeps_two_body_coherence(carbon):
    from atomx.physics.csf.matrix import build_ci_matrix, GridCIWorkspace
    result = carbon.result
    keys = tuple(result.occupations)
    if carbon.engine == 'shooting':
        from atomx.potentials.radial_nucleus import screened_nuclear_rv
        r, difr = result._grid, result._difr
        rv = screened_nuclear_rv(6, r, difr, len(r), nuclear='fermi')
        data = GridCIWorkspace(r, difr, rv)
        ns = len(r)
        extra = {}
    else:
        data = result.bsp_data
        ns = data['ns']
        extra = dict(nuclear_potential=result._nuclear_potential)
    h = build_ci_matrix(result.csfs, result.orbitals_dict, keys, data, 6, ns, **extra)
    c = result.mixing_coefficients[:, 0]
    coherent = float(c@h@c)
    incoherent = float(abs(c)**2@np.diag(h))
    assert coherent == pytest.approx(result.ground_energy, abs=1e-9)
    assert incoherent-coherent > .01


def test_explicit_hamiltonian_uses_shared_kernel_without_mcdhf_dispatch(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError('monoconfiguration DF entered the MCDHF entry point')
    monkeypatch.setattr(MCDHFSolver, 'solve', reject)
    atom = Atom(6, engine='bspline', correlation=Correlation(averaging='hamiltonian')).solve()
    assert atom.converged
    assert atom.calculation_spec.method == 'DHF'
    assert np.isfinite(atom.result.energy_components['E_Breit'])
    assert np.isfinite(atom.result.energy_components['E_QED_total'])


@pytest.mark.parametrize('engine', ['shooting', 'bspline'])
def test_unique_csf_keeps_fast_dhf_equations(engine):
    options = dict(Z=10, engine=engine, corrections='none')
    default = Atom(**options).solve()
    statistical = Atom(**options, correlation=Correlation(averaging='statistical')).solve()
    assert default.total_energy == statistical.total_energy
    assert isinstance(default.result, ShootingResult if engine == 'shooting' else SCFResult)


def test_fractional_occupation_products_remain_explicitly_different():
    statistical = Atom(6, corrections='none', correlation=Correlation(averaging='statistical')).solve()
    fractional = Atom(6, corrections='none', correlation=Correlation(averaging='fractional')).solve()
    assert abs(statistical.total_energy-fractional.total_energy) > 1e-3
    assert fractional.result._jj_configs is None
    assert fractional.calculation_spec.state['averaging']['method'] == 'fractional_occupation'


def test_fixed_ensemble_helper_rejects_hamiltonian_alias():
    with pytest.raises(ValueError, match='variational state'):
        generate_jj_average('2p2', averaging='hamiltonian')


def test_single_configuration_ci_request_has_explicit_migration_error():
    with pytest.raises(ValueError, match="averaging='hamiltonian'"):
        Atom(98, correlation=Correlation(averaging='ci'))


def test_explicit_configuration_list_keeps_mcdhf(monkeypatch):
    calls = []
    class Result:
        pass
    def record(solver, **kwargs):
        calls.append(solver)
        return Result()
    monkeypatch.setattr(MCDHFSolver, 'solve', record)
    atom = Atom(4, config=['1s2 2s2', '1s2 2p2'], engine='shooting').solve()
    assert len(calls) == 1
    assert calls[0].engine == 'shooting'
    assert atom.provenance['averaging'] == 'ci'


@pytest.mark.parametrize('exchange', ['gu', 'slater'])
def test_local_exchange_has_explicit_separate_default(exchange):
    with pytest.raises(ValueError, match="requires exchange='hf'"):
        Atom(6, exchange=exchange, correlation=Correlation(averaging='hamiltonian'))
    atom = Atom(2, exchange=exchange, corrections='none').solve()
    assert atom.converged
    assert atom.provenance['averaging'] == 'statistical'


def _co_csfs():
    from atomx.config.csf import enumerate_jj_csfs
    return enumerate_jj_csfs('1s2 2s2 2p6 3s2 3p6 3d7 4s2')


def test_mcdhf_engine_validates():
    with pytest.raises(ValueError, match='engine'):
        MCDHFSolver(Z=27, csfs=_co_csfs(), engine='finite-difference')


def test_mcdhf_shooting_engine_rejects_bspline_only_options():
    csfs = _co_csfs()
    with pytest.raises(ValueError, match='radial_optimization'):
        MCDHFSolver(Z=27, csfs=csfs, engine='shooting', radial_optimization='fac_frozen')
    with pytest.raises(ValueError, match='per_block_orbital_sets'):
        MCDHFSolver(Z=27, csfs=csfs, engine='shooting', per_block_orbital_sets=True)
    with pytest.raises(ValueError, match='nuclear'):
        MCDHFSolver(Z=27, csfs=csfs, engine='shooting', nuclear='uniform')


def test_californium_dispatch_has_one_nr_configuration_and_no_mc_entry(monkeypatch):
    from types import SimpleNamespace
    from atomx.methods.hamiltonian_df import HamiltonianDFSolver
    calls = []
    def forbid(*args, **kwargs):
        raise AssertionError('implicit MCDHF expansion')
    def record(solver, **kwargs):
        calls.append(solver)
        return SimpleNamespace()
    monkeypatch.setattr(MCDHFSolver, 'solve', forbid)
    monkeypatch.setattr(HamiltonianDFSolver, 'solve', record)
    atom = Atom(98, engine='shooting').solve()
    assert len(calls) == 1 and len(calls[0].csfs) == 107
    assert calls[0].configuration == atom._config_strings[0]
    assert len(atom._config_strings) == 1
    assert not atom._multi_config
    assert atom.provenance['averaging'] == 'hamiltonian'
