"""Physical resonance branching and independent matter/photon energy controls."""

import numpy as np
import pytest
from scipy.optimize import brentq

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.charge_mixture import ChargePartition, SahaMixture
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMAutoionizationChannel,
    CRMBoundFreeChannel,
    CRMDRSatelliteSource,
    CRMLevel,
    CRMTransition,
    radiative_recombination_rate_cm3_s,
)
from atomx.plasma.nonlte_source import NonLTEKineticPlasma, evolve_nonlte_isochoric
from atomx.scattering.autoionization.auger import AutoionizationState
from atomx.scattering.dielectronic_recombination.solver import (
    dielectronic_capture_rate_cm3_s,
    dr_rate_coefficient,
)

C = CODATA[DEFAULT_CODATA_VERSION]


def make_model(*, k=1e-9, sigma=1e-18, auger=0., A=0., processes=('eii', 'three-body', 'rr')):
    levels = [CRMLevel('bound', 0, g=1), CRMLevel('continuum', 200, g=2)]
    transitions, auto = [], []
    charges, ids = [11, 12], ['b', 'c']
    if auger:
        levels.append(CRMLevel('resonance', 240, g=3, autoionization_rate_s=auger))
        charges.append(11)
        ids.append('r')
        transitions.append(CRMTransition(2, 0, A))
        auto.append(CRMAutoionizationChannel(2, 1, auger, 'analytic-partial-width'))
    crm = CollisionalRadiativeModel(levels, transitions,
        bound_free_channels=[CRMBoundFreeChannel(0, 1, 200, eii_rate_cm3_s=k,
            photoionization_sigma_cm2=lambda E: sigma * (200/E)**2)],
        autoionization_channels=auto)
    model = NonLTEKineticPlasma(crm, charges, ids, source_id='manufactured-global-states',
        rate_source_id='manufactured-analytic-rates', excitation_mode='none',
        bound_free_processes=processes, coverage='finite analytic control; not Sn data')
    return model, crm


@pytest.mark.parametrize('T', [5., 20., 100.])
def test_unbranched_capture_matches_independent_electron_phase_volume(T):
    E, gamma, gr, gc = 40., 2e12, 3., 2.
    wavelength_m = C['_hplanck'] / np.sqrt(2*np.pi*C['_me']*T*C['_e'])
    expected = wavelength_m**3 * 1e6 * gr/(2*gc) * gamma * np.exp(-E/T)
    value = dielectronic_capture_rate_cm3_s(T, E, gamma, g_resonance=gr, g_continuum=gc)
    assert value == pytest.approx(expected, rel=2e-10, abs=0)


def test_explicit_resonance_schur_elimination_yields_one_dr_branch_factor():
    T, ne, gamma, A = 30., 1e17, 4e12, 2e12
    model, _ = make_model(auger=gamma, A=A, processes=())
    p = np.array([0., 1., 0.])
    R = model.evaluate(T, ne/12*1e6, p)['generator'].toarray()
    capture = R[2, 1] / ne
    assert R[1, 2] == gamma
    assert -R[2, 2] == gamma + A
    effective = capture * R[0, 2] / -R[2, 2]
    reference = dr_rate_coefficient([AutoionizationState('analytic', 40., 1., gamma, A)],
                                     T, initial_state_g=2).alpha_cm3_s[0]
    assert effective == pytest.approx(reference, rel=1e-12)
    other, _ = make_model(auger=gamma, A=100*A, processes=())
    R2 = other.evaluate(T, ne/12*1e6, p)['generator'].toarray()
    assert R2[2, 1] == R[2, 1]  # Capture is independent of the radiative branching.


@pytest.mark.parametrize('T', [5., 20., 100.])
def test_rr_photon_moment_uses_cross_section_weighted_electron_energy(T):
    I = 200.
    def sigma(E):
        return 1e-18 * (I/E)**2
    kwargs = dict(g_bound=1, g_continuum=2, cross_section_cm2=sigma)
    zero = radiative_recombination_rate_cm3_s(I, T, **kwargs)
    first = radiative_recombination_rate_cm3_s(I, T, photon_energy_moment=1, **kwargs)
    # sigma(E)*E² is constant, leaving an exponential kinetic-energy density.
    assert first/zero == pytest.approx(I + T, rel=2e-9)
    with pytest.raises(ValueError, match='explicit PI'):
        radiative_recombination_rate_cm3_s(I, T, photon_energy_moment=1)


def test_eii_consumes_ionization_and_new_electron_thermal_energy():
    m, _ = make_model(processes=('eii',))
    T = 30.
    r = m.evaluate(T, 1e22, [1, 0])
    rate = 11e16 * 1e-9
    assert r['population_rate_per_s'][1] == rate
    assert r['temperature_rate_eV_per_s'] == pytest.approx(-(200 + 1.5*T)*rate/(1.5*12))
    assert r['line_power_eV_per_ion_s'] == r['rr_power_eV_per_ion_s'] == 0


def test_rr_preferential_capture_changes_temperature_without_double_counting_ip():
    m, _ = make_model(processes=('rr',))
    T = 20.
    r = m.evaluate(T, 1e22, [0, 1])
    rate = r['population_rate_per_s'][0]
    assert rate > 0
    assert r['rr_power_eV_per_ion_s']/rate == pytest.approx(200 + T, rel=2e-9)
    assert r['temperature_rate_eV_per_s'] == pytest.approx(.5*T*rate/(1.5*13), rel=2e-8)


def test_auger_releases_resonance_energy_and_adds_an_electron():
    m, _ = make_model(auger=2e9, processes=())
    r = m.evaluate(20., 1e22, [0, 0, 1])
    assert r['charge_rate_per_s'] == 2e9
    assert r['atomic_energy_rate_eV_per_ion_s'] == -40 * 2e9
    assert r['temperature_rate_eV_per_s'] == pytest.approx((40 - 1.5*20)*2e9/(1.5*12))
    assert r['rr_power_eV_per_ion_s'] == 0


def test_collision_only_closed_system_relaxes_to_independent_saha_energy_root():
    m, _ = make_model(k=1e-9, processes=('eii', 'three-body'))
    ni, T0, p0 = 1e24, 50., np.array([.9, .1])
    U0 = m.evaluate(T0, ni, p0)['internal_energy_eV_per_ion']
    eos = SahaMixture([ChargePartition(q, np.array([0.]), np.array([g]), (str(q),), 'test')
                       for q, g in ((11, 1), (12, 2))], [200], threshold_source_id='analytic-test')
    Teq = brentq(lambda T: eos.equilibrium(T, ni)['internal_energy_eV'] - U0, 1, 100)
    eq = eos.equilibrium(Teq, ni)
    r = evolve_nonlte_isochoric(m, ni, T0, p0, duration_s=1e-7, heating_W_m3=lambda t: 0)
    assert r['temperature_eV'][-1] == pytest.approx(Teq, rel=2e-7)
    np.testing.assert_allclose(r['populations'][:, -1], eq['populations'], rtol=2e-6)
    assert r['maximum_relative_energy_defect'] < 1e-8
    np.testing.assert_array_equal(r['ledger_eV_per_ion'], 0)


def test_radiative_decay_emits_stored_excitation_without_cooling_thermal_gas():
    A, gap, T, ni = 2e8, 90., 30., 1e22
    crm = CollisionalRadiativeModel([CRMLevel('g', 0), CRMLevel('e', gap)], [CRMTransition(1, 0, A)])
    m = NonLTEKineticPlasma(crm, [12, 12], ['g', 'e'], source_id='test', rate_source_id='test',
                           excitation_mode='none', bound_free_processes=(), coverage='analytic two-level')
    r = evolve_nonlte_isochoric(m, ni, T, [.5, .5], duration_s=3/A, heating_W_m3=lambda t: 0)
    np.testing.assert_allclose(r['temperature_eV'], T, atol=1e-11)
    np.testing.assert_allclose(r['populations'][1], .5*np.exp(-A*r['time_s']), rtol=2e-8)
    np.testing.assert_allclose(r['ledger_eV_per_ion'][1], gap*(.5-r['populations'][1]), atol=2e-12)


def test_full_resonant_multicharge_heating_and_rr_photon_ledger():
    m, _ = make_model(auger=2e9, A=1e9)
    r = evolve_nonlte_isochoric(m, 1e22, 30, [.45, .45, .1], duration_s=1e-8,
                                heating_W_m3=lambda t: 3e13)
    assert r['maximum_relative_energy_defect'] < 1e-8
    assert r['ledger_eV_per_ion'][1, -1] > 0
    assert r['ledger_eV_per_ion'][2, -1] > 0
    assert r['ledger_eV_per_ion'][0, -1] == pytest.approx(3e13*1e-8/(1e22*C['_e']))
    assert r['electron_density_per_m3'][-1] != r['electron_density_per_m3'][0]
    assert r['minimum_population'] >= -1e-12


def test_global_energy_zero_does_not_change_kinetics_or_condition_energy_gate():
    first, crm = make_model(processes=('eii',))
    for lev in crm.levels:
        lev.energy_eV += 10000
    shifted = NonLTEKineticPlasma(crm, [11, 12], ['b', 'c'], source_id='shifted', rate_source_id='test',
        excitation_mode='none', bound_free_processes=('eii',), coverage='energy gauge test')
    a = first.evaluate(30, 1e22, [.8, .2])
    b = shifted.evaluate(30, 1e22, [.8, .2])
    np.testing.assert_array_equal(a['population_rate_per_s'], b['population_rate_per_s'])
    assert a['temperature_rate_eV_per_s'] == pytest.approx(b['temperature_rate_eV_per_s'], rel=1e-13)


def test_reject_incomplete_physical_resonances_rates_and_fallbacks():
    m, crm = make_model(auger=2e9)
    args = dict(source_id='test', rate_source_id='test', excitation_mode='none',
                bound_free_processes=(), coverage='admission')
    crm.autoionization_channels.clear()
    with pytest.raises(ValueError, match='widths'):
        NonLTEKineticPlasma(crm, [11, 12, 11], ['b', 'c', 'r'], **args)
    _, crm = make_model()
    crm.dr_satellite_sources.append(CRMDRSatelliteSource(0, 1, lambda T: 1e-10))
    with pytest.raises(ValueError, match='synthetic'):
        NonLTEKineticPlasma(crm, [11, 12], ['b', 'c'], **args)
    _, crm = make_model()
    crm.bound_free_channels[0].ionization_potential_eV = 201
    with pytest.raises(ValueError, match='threshold'):
        NonLTEKineticPlasma(crm, [11, 12], ['b', 'c'], **args)
    with pytest.raises(MemoryError, match='before copying'):
        NonLTEKineticPlasma(crm, [11, 12], ['b', 'c'], max_states=1, **args)
    negative, _ = make_model(k=lambda T: -1, processes=('eii',))
    with pytest.raises(ValueError, match='nonnegative'):
        negative.evaluate(20, 1e22, [1, 0])
    underflow, _ = make_model(processes=('eii', 'three-body'))
    with pytest.raises(ArithmeticError, match='underflow'):
        underflow.evaluate(.1, 1e22, [1, 0])
    with pytest.raises(MemoryError, match='Jacobian'):
        evolve_nonlte_isochoric(m, 1e22, 20, [.5, .5, 0], duration_s=1e-9,
                               heating_W_m3=lambda t: 0, jacobian_memory_cap_bytes=1)


def test_model_owns_numeric_crm_and_hashes_real_scalar_rates():
    first, crm = make_model(k=1e-9, processes=('eii',))
    other, _ = make_model(k=2e-9, processes=('eii',))
    assert first.source_id != other.source_id
    before = first.evaluate(30, 1e22, [1, 0])['population_rate_per_s']
    crm.bound_free_channels[0].eii_rate_cm3_s = 100
    crm.levels[0].energy_eV = 100
    np.testing.assert_array_equal(first.evaluate(30, 1e22, [1, 0])['population_rate_per_s'], before)


def test_multiple_auger_channels_keep_distinct_parent_energies_and_weights():
    T, ne, widths = 30., 1e17, (1e12, 2e12)
    crm = CollisionalRadiativeModel([
        CRMLevel('b', 0, g=1), CRMLevel('c0', 200, g=2),
        CRMLevel('c1', 220, g=4), CRMLevel('r', 250, g=3, autoionization_rate_s=sum(widths)),
    ], [CRMTransition(3, 0, 3e12)], autoionization_channels=[
        CRMAutoionizationChannel(3, 1, widths[0], 'channel0'),
        CRMAutoionizationChannel(3, 2, widths[1], 'channel1'),
    ])
    m = NonLTEKineticPlasma(crm, [11, 12, 12, 11], ['b', 'c0', 'c1', 'r'],
        source_id='manufactured-multiparent', rate_source_id='test', excitation_mode='none',
        bound_free_processes=(), coverage='all supplied analytic channels')
    R = m.evaluate(T, ne/12*1e6, [0, .5, .5, 0])['generator'].toarray()
    volume = (C['_hplanck']**2/(2*np.pi*C['_me']*T*C['_e']))**1.5*1e6
    assert R[3, 1] == pytest.approx(ne*volume*3/4*widths[0]*np.exp(-50/T), rel=2e-10)
    assert R[3, 2] == pytest.approx(ne*volume*3/8*widths[1]*np.exp(-30/T), rel=2e-10)
    assert R[1, 3] == widths[0] and R[2, 3] == widths[1]
    assert -R[3, 3] == 6e12
    np.testing.assert_allclose(R.sum(axis=0), 0, atol=1e-3)


def test_fixed_charge_slab_rejects_new_charge_changing_channels():
    from atomx.plasma.euv_source import LTELinePlasma, SlabSourceConfig, evolve_cr_slab

    _, crm = make_model(auger=2e9, A=1e9, processes=())
    crm.bound_free_channels.clear()
    crm.charge = 12
    plasma = LTELinePlasma([0, 200, 240], [1, 2, 3], [2], [0], [1e9], charge=12,
        ion_mass_amu=120, energy_edges_eV=np.arange(0, 301), fwhm_eV=0,
        source_id='test', emission_scope='fixed-charge guard')
    config = SlabSourceConfig(1e22, 1e-5, 20, 10.6e-6, 5, 1e-9)
    with pytest.raises(ValueError, match='autoionization'):
        evolve_cr_slab(plasma, config, lambda t: 0, crm, [.5, .5, 0])
