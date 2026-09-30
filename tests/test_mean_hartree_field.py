"""A mean continuum field must retain the nucleus and remove one self-field."""
from types import SimpleNamespace
import numpy as np
import pytest
from atomx import Atom
from atomx.potentials.static_field import static_hartree_potential


@pytest.mark.parametrize('z', [2, 26, 82])
def test_one_electron_mean_field_is_exact_coulomb_everywhere(z, monkeypatch):
    atom = Atom(z, config='1s1', engine='bspline', nuclear='point', corrections='none').solve(tol=1e-10)
    import atomx.numerics.yk as yk
    monkeypatch.setattr(yk, 'compute_yk', lambda *a: pytest.fail('zero screening must not be integrated'))
    field = static_hartree_potential(atom.result, screening='mean')
    r = np.geomspace(1e-12, 1e3, 101)
    np.testing.assert_allclose(r*field(r), -z, rtol=3e-16, atol=0)
    assert field.residual_charge == z


@pytest.mark.parametrize('engine,nuclear', [('bspline', 'fermi'), ('bspline', 'uniform'),
                                          ('shooting', 'fermi')])
def test_mean_field_retains_finite_nucleus_of_native_result(engine, nuclear):
    atom = Atom(26, config='1s1', engine=engine, nuclear=nuclear, corrections='none').solve(tol=1e-10)
    result = atom.result
    field = static_hartree_potential(result, screening='mean')
    if engine == 'bspline':
        r = np.geomspace(1e-12, 100., 100)
        expected = np.array([result._nuclear_potential(x) for x in r])
    else:
        r = result._grid[1:]
        expected = result._rv_nuc[1:]/r
    np.testing.assert_allclose(field(r), expected, rtol=2e-13, atol=1e-10)
    # Finite nuclear potential: rV vanishes at the origin, unlike -Z/r.
    assert abs(1e-12*field(np.array([1e-12]))[0]) < 1e-5


@pytest.mark.parametrize('engine', ['bspline', 'shooting'])
def test_mean_hartree_density_and_tail_use_the_same_electron_number(engine):
    atom = Atom(10, engine=engine, nuclear='point', corrections='none').solve(tol=1e-10)
    full = static_hartree_potential(atom.result)
    mean = static_hartree_potential(atom.result, screening='mean')
    r = np.geomspace(1e-4, 100., 200)
    np.testing.assert_allclose(mean(r)+10/r, .9*(full(r)+10/r), rtol=1e-10, atol=3e-11)
    np.testing.assert_allclose(r[-5:]*mean(r[-5:]), -1., atol=1e-8)
    assert mean.residual_charge == 1.


def test_mean_screening_and_missing_nuclear_data_are_explicit():
    result = SimpleNamespace(Z=10, subshells=[SimpleNamespace(occupation=1.)],
                             energy_components={'_nuclear_profile': {'model':'fermi'}})
    with pytest.raises(ValueError, match='finite-nucleus field is missing'):
        static_hartree_potential(result, screening='mean')
    with pytest.raises(ValueError, match='twice'):
        static_hartree_potential(result, screening='mean', exclude_label='1s1/2')
    with pytest.raises(ValueError, match='screening must'):
        static_hartree_potential(result, screening='fac')


def test_eii_uses_selected_field_and_reports_actual_screening():
    from atomx.constants import HARTREE_EV
    from atomx.scattering import ContinuumModel, PotentialModel
    from atomx.scattering.electron_ionization import eii_cross_section

    atom = Atom(2, config='1s2', engine='bspline', nuclear='point',
                corrections='none').solve(tol=1e-10)
    energy = [-2 * atom.orbital_energies['1s1/2'] * HARTREE_EV]

    def run(potential):
        return eii_cross_section(atom.result, energy, method='DW',
            kappa_max=3, n_outgoing_energies=5,
            continuum_model=ContinuumModel.shooting(potential=potential, n_grid=1000))

    default = run(None)
    mean = run(PotentialModel.static_hartree(screening='mean'))
    bare = run(PotentialModel.bare())
    exchange = run(PotentialModel.static_hartree_exchange(screening='mean'))
    np.testing.assert_array_equal(default.sigma_total_cm2, mean.sigma_total_cm2)
    # A real two-electron target distinguishes these models; a one-electron
    # target would let an ignored bare-field request pass unnoticed.
    assert abs(bare.sigma_total_cm2[0] / mean.sigma_total_cm2[0] - 1) > 1e-2
    assert abs(exchange.sigma_total_cm2[0] / mean.sigma_total_cm2[0] - 1) > 1e-3
    for result, charge, screening, local_exchange in (
        (default, 1., 'mean', False), (mean, 1., 'mean', False),
        (bare, 2., None, False), (exchange, 1., 'mean', True),
    ):
        assert np.isfinite(result.sigma_total_cm2).all()
        assert (result.sigma_total_cm2 > 0).all()
        provenance = result.provenance['per_orbital']['1s1/2']
        assert provenance['residual_charge'] == charge
        assert provenance['screening'] == screening
        assert result.provenance['include_local_exchange'] == local_exchange
    with pytest.raises(ValueError, match='bare Coulomb field cannot'):
        run(PotentialModel(kind='bare', include_exchange=True))
