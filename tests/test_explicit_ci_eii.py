"""Independent magnetic trace and matched-threshold native coherent CI DW."""

import numpy as np
import pytest
from test_explicit_ci_photoionization import helium as helium_fixture

from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
from atomx.angular.dyson import dyson_annihilation, reduced_dyson_annihilation
from atomx.config.csf import enumerate_jj_csfs
from atomx.constants import HARTREE_EV
from atomx.scattering.electron_ionization import _eii_dw_channels
from atomx.scattering.electron_ionization.channels import EIIDWBoundChannel
from atomx.scattering.electron_ionization.ci import ExplicitCIElectronIonization

helium = helium_fixture


def test_complete_magnetic_trace_preserves_kappa_and_signed_radial_coherence():
    parents = [c for cfg in ('4d2', '4d1 5d1', '5d2') for c in enumerate_jj_csfs(cfg)
               if c.total_J2 == 4]
    daughters = [c for cfg in ('4d1', '5d1') for c in enumerate_jj_csfs(cfg)
                 if c.total_J2 == 3]
    keys = tuple(sorted({(s.n, s.kappa) for c in parents+daughters for s in c.subshells}))
    spinors = tuple((n, k, m) for n, k in keys for m in range(-2*abs(k)+1, 2*abs(k), 2))
    rng = np.random.default_rng(5192)
    left = np.array([.6, -.8])
    right = rng.normal(size=len(parents))
    right /= np.linalg.norm(right)

    def states(csfs, coefficients, j):
        out = {}
        for m in range(-j, j+1, 2):
            vector = {}
            for c, coefficient in zip(csfs, coefficients, strict=True):
                for mask, value in csf_magnetic_vector(csf_signature(c), m, keys):
                    vector[mask] = vector.get(mask, 0.)+coefficient*value
            out[m] = vector
        return out

    initial, final = states(parents, right, 4), states(daughters, left, 3)
    radial_keys, reduced = reduced_dyson_annihilation(final[3], initial[4], spinors, 4, 3)
    rho = np.zeros((len(spinors), len(spinors)))
    for ket in initial.values():
        for bra in final.values():
            d = dyson_annihilation(bra, ket, len(spinors))
            rho += np.outer(d, d)/5
    expected = np.zeros_like(rho)
    for i, (n, k, m) in enumerate(spinors):
        for j, (nb, kb, mb) in enumerate(spinors):
            if (kb, mb) == (k, m):
                expected[i, j] = reduced[radial_keys.index((n, k))]*reduced[radial_keys.index((nb, k))]/(2*abs(k))
    np.testing.assert_allclose(rho, expected, atol=1e-14, rtol=1e-12)
    assert any(abs(expected[i, j]) > .001 for i in range(len(spinors)) for j in range(len(spinors))
               if spinors[i][0] != spinors[j][0])
    # Replacing the coherent density by occupations changes an observable.
    response = rng.normal(size=len(spinors))
    assert abs(response@rho@response-np.dot(response**2, np.diag(rho))) > .001


def test_native_helium_threshold_strength_and_full_dw_normalization(helium):
    _, _, _, snapshot = helium
    consumer = ExplicitCIElectronIonization(snapshot, coverage='retained1s He->He+ only')
    channel, = consumer.bound_channels(1, 0)
    assert channel.occupation == 1.
    assert channel.threshold_Ha == snapshot.levels[0].energy_Ha-snapshot.levels[1].energy_Ha
    reference_p, reference_q = snapshot.orbitals[1, -1]
    # A full closed1s pair has total removal strength2; no second g or q.
    nonzero = np.flatnonzero(reference_p)
    ratio = channel.P[nonzero]/reference_p[nonzero]
    np.testing.assert_allclose(ratio**2, 2., atol=2e-14, rtol=1e-14)
    threshold = snapshot.threshold_eV(1, 0)
    energies = np.array([threshold-1., threshold, threshold+40.])
    actual = consumer.cross_section(1, 0, energies, kappa_max=1, sharing_order=3)
    expected = _eii_dw_channels([EIIDWBoundChannel('independent closed1s', -1,
            channel.threshold_Ha, 2., reference_p, reference_q)], snapshot.data, snapshot.Z, energies,
        V_func=snapshot._field(0), Z_res=1., potential_kind='same independent selected field',
        pauli_projectors={-1: ((reference_p, reference_q),)}, kappa_max=1,
        n_outgoing_energies=3, continuum_model=snapshot.model)
    np.testing.assert_allclose(actual.sigma_total_cm2, expected.sigma_total_cm2, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(actual.sigma_total_cm2[:2], [0., 0.])
    assert actual.sigma_total_cm2[2] > 0
    assert actual.provenance['matched_threshold_eV'] == threshold
    assert actual.provenance['pauli_projection_max_defect'] < 1e-10
    assert actual.provenance['full_ionic_EII_certified'] is False
    assert actual.provenance['experimental_accepted'] is False
    assert threshold == channel.threshold_Ha*HARTREE_EV


def test_explicit_input_ownership_and_full_work_admission(helium, monkeypatch):
    _, _, _, snapshot = helium
    consumer = ExplicitCIElectronIonization(snapshot, coverage='test', max_channel_work=1)
    channel, = consumer.bound_channels(1, 0)
    with pytest.raises(ValueError):
        channel.P.setflags(write=True)
    monkeypatch.setattr('atomx.scattering.electron_ionization.ci._eii_dw_channels',
                        lambda *args, **kw: pytest.fail('cap must reject before continuum'))
    with pytest.raises(MemoryError, match='channel-work'):
        consumer.cross_section(1, 0, [100.])
    with pytest.raises(ValueError, match='adjacent'):
        consumer.bound_channels(0, 1)
    with pytest.raises(ValueError, match='snapshot'):
        ExplicitCIElectronIonization(object(), coverage='test')
    for threshold in (0., -1., float('nan')):
        with pytest.raises(ValueError, match='threshold'):
            EIIDWBoundChannel('bad', -1, threshold, 1., [1.], [0.])


def test_direct_thermal_consumer_uses_actual_threshold_and_retains_every_node(helium, monkeypatch):
    from scipy.constants import e, m_e

    from atomx.scattering.electron_ionization import EIIResult

    _, _, _, snapshot = helium
    consumer = ExplicitCIElectronIonization(snapshot, coverage='analytic thermal entry control')
    calls = []
    def analytic(initial, final, energy, **options):
        calls.append(energy.copy())
        return EIIResult(energy, np.full_like(energy, 1e-18), method='DW')
    monkeypatch.setattr(consumer, 'cross_section', analytic)
    result = consumer.rate_coefficient(1, 0, 20., order=4)
    threshold = snapshot.threshold_eV(1, 0)
    expected = 1e-18*100*np.sqrt(8*e*20/(np.pi*m_e))*(1+threshold/20)*np.exp(-threshold/20)
    assert result['rate_cm3_s'] == pytest.approx(expected, rel=1e-12)
    assert len(calls) == 1 and len(result['energies_eV']) == 4
    np.testing.assert_array_equal(result['energies_eV'], calls[0])
    assert min(result['energies_eV']) > threshold
    assert result['full_ionic_EII_certified'] is False
