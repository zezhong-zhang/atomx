"""Energy-sharing integration through the actual collision kernel."""
import numpy as np
import pytest

from atomx import Atom
from atomx.constants import BOHR_RADIUS_M, C_SQ, HARTREE_EV
from atomx.scattering import ContinuumModel
from atomx.scattering.electron_ionization import _eii_dw_total, eii_cross_section


@pytest.fixture(scope='module')
def helium_ion():
    return Atom(2, config='1s1', engine='bspline', nuclear='point',
                corrections='none').solve(tol=1e-10)


def test_full_interval_matches_independent_linear_gauss_integration(helium_ion):
    atom = helium_ion
    binding = -atom.orbital_energies['1s1/2']
    incident = 2*binding
    model = ContinuumModel.shooting(n_grid=1000)
    result = eii_cross_section(atom.result, [incident*HARTREE_EV], method='DW',
        kappa_max=10, n_outgoing_energies=16, continuum_model=model)
    x, w = np.polynomial.legendre.leggauss(32)
    capture = {}
    _eii_dw_total(atom.result, [incident*HARTREE_EV], kappa_max=10,
        n_outgoing_energies=32, continuum_model=model,
        e1_grid_frac=.25*(x+1), _capture=capture)
    _, integrand, available = next(iter(capture.values()))
    p2 = 2*incident + incident**2/C_SQ
    reference = (np.pi/p2*.5*BOHR_RADIUS_M**2*1e4
                 * available*.25*np.dot(w, integrand))
    assert result.sigma_total_cm2[0] == pytest.approx(reference, rel=1e-5)
    provenance = result.provenance['per_orbital']['1s1/2']
    assert provenance['energy_sharing_fraction_interval'] == [0., .5]
    diagnostic = provenance['partial_wave_diagnostics'][0]
    assert diagnostic['explicit_partial_wave_sigma_cm2'] + diagnostic['incident_tail_sigma_cm2'] == pytest.approx(
        result.sigma_total_cm2[0], rel=1e-14)


def test_soft_energies_stay_positive_and_repeated_incident_waves_are_reused(helium_ion, monkeypatch):
    from atomx.numerics.shooting import dirac_continuum
    original = dirac_continuum.solve_dirac_continuum_many
    batches = []

    def record(requests, *args, **kwargs):
        batches.append(list(requests))
        return original(requests, *args, **kwargs)

    monkeypatch.setattr(dirac_continuum, 'solve_dirac_continuum_many', record)
    atom = helium_ion
    energy = -atom.orbital_energies['1s1/2']*HARTREE_EV*(1+1e-10)
    result = eii_cross_section(atom.result, [energy, energy], method='DW',
        kappa_max=1, n_outgoing_energies=3,
        continuum_model=ContinuumModel.shooting(n_grid=1000))
    requests = [req for batch in batches for req in batch]
    # Repeated incident energies reuse the completed collision result;
    # no required wave is solved twice. Each batch contains at most the
    # two outgoing energies, independent of sharing quadrature order.
    assert len(requests) == len(set(requests))
    assert max(len({energy for _, energy in batch}) for batch in batches) <= 2
    assert 0 < min(e for _, e in requests) < 1e-10
    assert np.isfinite(result.sigma_total_cm2).all()
    assert result.sigma_total_cm2[0] > 0
    assert result.sigma_total_cm2[0] == result.sigma_total_cm2[1]


def test_incident_tail_is_explicit_and_kept_separate_from_computed_sum(helium_ion):
    from atomx.scattering.electron_ionization.dw import dw_eii_cross_section
    energy = -helium_ion.orbital_energies['1s1/2']*HARTREE_EV*5
    settings = dict(kappa_max=3, n_outgoing_energies=8,
                    continuum_model=ContinuumModel.shooting(n_grid=1000))
    explicit = eii_cross_section(helium_ion.result, [energy], method='DW', **settings)
    completed = dw_eii_cross_section(helium_ion.result, [energy],
                                    partial_wave_tail='geometric', **settings)
    plain = explicit.provenance['per_orbital']['1s1/2']
    added = completed.provenance['per_orbital']['1s1/2']
    assert plain['incident_tail_model'] == 'none'
    assert added['incident_tail_model'] == 'geometric'
    p, q = plain['partial_wave_diagnostics'][0], added['partial_wave_diagnostics'][0]
    assert p['incident_tail_sigma_cm2'] == 0
    assert p['estimated_incident_tail_sigma_cm2'] > 0
    assert q['explicit_partial_wave_sigma_cm2'] == pytest.approx(explicit.sigma_total_cm2[0], rel=1e-14)
    assert completed.sigma_total_cm2[0] == pytest.approx(
        explicit.sigma_total_cm2[0]+p['estimated_incident_tail_sigma_cm2'], rel=1e-14)
    with pytest.raises(ValueError, match='partial_wave_tail'):
        dw_eii_cross_section(helium_ion.result, [energy], partial_wave_tail='automatic')


def test_missing_sharing_node_cannot_reuse_previous_wave_storage(helium_ion, monkeypatch):
    from atomx.numerics.shooting import dirac_continuum
    energy = -helium_ion.orbital_energies['1s1/2']*HARTREE_EV*2
    options = dict(kappa_max=1, n_outgoing_energies=3,
                   continuum_model=ContinuumModel.shooting(n_grid=700))
    baseline = {}
    _eii_dw_total(helium_ion.result, [energy], _capture=baseline, **options)
    nodes, reference, _ = next(iter(baseline.values()))
    original = dirac_continuum.solve_dirac_continuum_many

    def missing_middle(requests, *args, **kwargs):
        grid, waves = original(requests, *args, **kwargs)
        for req in requests:
            if req[1] == nodes[1]:
                waves[req] = None
        return grid, waves

    monkeypatch.setattr(dirac_continuum, 'solve_dirac_continuum_many', missing_middle)
    capture = {}
    result = _eii_dw_total(helium_ion.result, [energy], _capture=capture,
                          continuum_failure_mode='diagnostic', **options)
    _, actual, _ = next(iter(capture.values()))
    assert actual[1] == 0.
    np.testing.assert_array_equal(actual[[0, 2]], reference[[0, 2]])
    assert result.provenance['continuum_failures']


def test_projected_channel_spaces_retain_complete_antisymmetric_amplitudes(helium_ion):
    energy = -helium_ion.orbital_energies['1s1/2']*HARTREE_EV*2
    options = dict(kappa_max=3, n_outgoing_energies=3,
                   continuum_model=ContinuumModel.shooting(n_grid=700))
    full = _eii_dw_total(helium_ion.result, [energy], **options)
    same = _eii_dw_total(helium_ion.result, [energy], _channel_limits=(3,3,3), **options)
    restricted = _eii_dw_total(helium_ion.result, [energy], _channel_limits=(2,1,3), **options)
    np.testing.assert_array_equal(same.sigma_total_cm2, full.sigma_total_cm2)
    assert 0 < restricted.sigma_total_cm2[0] < full.sigma_total_cm2[0]
    limits = restricted.provenance['per_orbital']['1s1/2']['channel_l_limits']
    assert limits == dict(incident=2, slow=1, fast=3)
    with pytest.raises(ValueError, match='cannot exceed'):
        _eii_dw_total(helium_ion.result, [energy], _channel_limits=(4,1,3), **options)
