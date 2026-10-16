"""Independent limits for population, transport, laser and source energy flow."""

import numpy as np
import pytest
from scipy.integrate import quad

from atomx.constants import BOLTZMANN_EV, CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMBoundFreeChannel,
    CRMDRSatelliteSource,
    CRMLevel,
    CRMTransition,
)
from atomx.plasma.euv_source import (
    EV_J,
    HC_EV_NM,
    LTELinePlasma,
    SlabSourceConfig,
    evolve_cr_slab,
    evolve_lte_slab,
    integrated_escaped_spectrum,
)
from atomx.plasma.laser_absorption import (
    critical_density_per_m3,
    drude_slab_fractions,
    electron_ion_collision_frequency,
)
from atomx.plasma.population_network import (
    advance_populations,
    omega_collision_pair,
    population_generator,
)
from atomx.spectrum.slab_transfer import slab_escape_factor, slab_escaped_flux

CONSTANTS = CODATA[DEFAULT_CODATA_VERSION]


@pytest.mark.parametrize("tau", [0, 1e-14, 1e-6, 0.1, 1, 10, 100, 1e8])
def test_hemispheric_slab_integral(tau):
    beta = float(slab_escape_factor(tau))
    if tau == 0:
        assert beta == 1
    else:
        # Independently integrate mu(1-exp(-tau/mu)); cancellation-free integrand.
        integral = quad(lambda mu: mu * -np.expm1(-tau / mu), 0, 1,
                        epsabs=1e-30, epsrel=2e-10, limit=200)[0]
        assert beta == pytest.approx(integral / tau, rel=2e-9)
    assert 0 < beta <= 1
    if tau >= 100:
        assert beta == pytest.approx(0.5 / tau)


def test_slab_thin_and_blackbody_limits():
    # In LTE epsilon_nu=4pi alpha B_nu: each thick face tends to pi B_nu.
    B, alpha, length = 3.2, 100.0, 100.0
    flux = slab_escaped_flux(np.array([4 * np.pi * alpha * B]), np.array([alpha]), length)
    assert flux[0] / 2 == pytest.approx(np.pi * B)
    assert slab_escaped_flux(np.array([7.0]), np.array([0.0]), 3)[0] == 21
    with pytest.raises(ValueError, match="masers"):
        slab_escape_factor(-1)


def test_nonlte_two_level_and_detailed_balance():
    T, gap, gl, gu = 30.0, 90.0, 1.0, 3.0
    up, down = omega_collision_pair([gap, gap + 1000], [2, 2], gap, T, gl, gu)
    assert down == pytest.approx(8.6291e-6 * 2 / (gu * np.sqrt(T / BOLTZMANN_EV)))
    assert up / down == pytest.approx(gu / gl * np.exp(-gap / T))
    ne, A, duration = 1e17, 2e9, 2e-9
    a, b = ne * up, ne * down + A
    R = population_generator(2, [0, 1, 1], [1, 0, 0], [a, ne * down, A])
    p = advance_populations(R, [1, 0], duration)
    expected_upper = a / (a + b) * (1 - np.exp(-(a + b) * duration))
    assert p[1] == pytest.approx(expected_upper, rel=2e-12)
    assert p.sum() == pytest.approx(1, abs=2e-14)
    # LTE is reached in the collision-only limit, without a fitted power scale.
    C = population_generator(2, [0, 1], [1, 0], [a, ne * down])
    late = advance_populations(C, [1, 0], 20 / a)
    assert late[1] / late[0] == pytest.approx(gu / gl * np.exp(-gap / T), rel=1e-10)


def test_collision_underflow_and_sparse_storage():
    up, down = omega_collision_pair([1000, 1100], [1, 1], 1000, 1, 1, 1)
    assert up == 0 and down > 0
    n = 115715
    R = population_generator(n, np.arange(n - 1), np.arange(1, n), np.ones(n - 1))
    assert R.nnz == 2 * (n - 1)
    assert R.data.nbytes + R.indices.nbytes + R.indptr.nbytes < 4 * 1024**2
    assert np.max(np.abs(R.sum(axis=0))) == 0


def test_invalid_population_generator_is_rejected():
    with pytest.raises(ValueError):
        population_generator(2, [0], [1], [-1])
    with pytest.raises(ValueError):
        advance_populations(population_generator(2, [0], [1], [1]), [0.6, 0.6], 1)
    with pytest.raises(ValueError):
        advance_populations(population_generator(2, [0], [1], [1]).T, [1, 0], 1)


def test_drude_energy_partition_and_classical_limits():
    wavelength, T, ne, q, log = 10.6e-6, 30.0, 1e24, 12, 5.0
    nc = critical_density_per_m3(wavelength)
    assert nc == pytest.approx(1.1148542e27 / 10.6**2, rel=1e-6)
    nu = electron_ion_collision_frequency(ne, T, q, log)
    # CODATA evaluation is 2.9063169e-6. The often rounded 2.91e-6 is
    # only a three-significant-figure reference, not a 2e-4 relative gate.
    assert nu == pytest.approx(2.9063169e-6 * (ne / 1e6) * q * log / T**1.5, rel=2e-7)
    f = drude_slab_fractions(ne, nu, wavelength, 30e-6)
    assert f["absorption"] + f["reflection"] + f["transmission"] == pytest.approx(1, abs=2e-16)
    omega = 2 * np.pi * CONSTANTS["_c"] / wavelength
    weak_nu = omega * 1e-7
    weak = drude_slab_fractions(ne, weak_nu, wavelength, 30e-6)
    expected_alpha = weak_nu / CONSTANTS["_c"] * (ne / nc) / np.sqrt(1 - ne / nc)
    assert weak["attenuation_per_m"] == pytest.approx(expected_alpha, rel=1e-12)
    vacuum = drude_slab_fractions(0, 0, wavelength, 30e-6)
    assert vacuum["transmission"] == 1 and vacuum["absorption"] == vacuum["reflection"] == 0


def plasma_fixture(A=1e10, width=0.0, extra_level=False):
    # Choose a physical photon at a bin center for an exact Einstein/Planck test.
    return LTELinePlasma([0, 92.125] + ([150] if extra_level else []),
                         [1, 3] + ([5] if extra_level else []), [1], [0], [A],
                         charge=12, ion_mass_amu=120, energy_edges_eV=np.arange(0, 200.25, .25),
                         fwhm_eV=width, source_id="analytic-two-level", emission_scope="one-line")


def test_absolute_power_einstein_kirchhoff_and_partition():
    plasma = plasma_fixture(extra_level=True)
    T, ni = 30.0, 1e23
    p, U, cv = plasma.lte(T)
    eps, alpha = plasma.coefficients(p, ni)
    photon = 92.125
    assert eps.sum() == pytest.approx(ni * p[1] * 1e10 * photon * EV_J)
    frequency = photon * EV_J / CONSTANTS["_hplanck"]
    B = 2 * CONSTANTS["_hplanck"] * frequency**3 / CONSTANTS["_c"]**2 / np.expm1(photon / T)
    idx = np.argmax(eps)
    assert eps[idx] / plasma.width_Hz[idx] / (4 * np.pi * alpha[idx]) == pytest.approx(B, rel=1e-14)
    # A supplied level without radiating endpoints still belongs in the partition/EOS.
    assert p[2] > 0 and p[1] < 3 * np.exp(-photon / T) / (1 + 3 * np.exp(-photon / T))
    h = 1e-3
    assert cv == pytest.approx((plasma.lte(T + h)[1] - plasma.lte(T - h)[1]) / (2 * h), rel=2e-9)
    assert U > 1.5 * 13 * T


def test_profile_conserves_power_and_inversion_rejects():
    narrow, wide = plasma_fixture(), plasma_fixture(width=0.8)
    p = narrow.lte(30)[0]
    e0, a0 = narrow.coefficients(p, 1e23)
    e1, a1 = wide.coefficients(p, 1e23)
    assert e1.sum() == pytest.approx(e0.sum(), rel=2e-15)
    assert a1 @ wide.width_Hz == pytest.approx(a0 @ narrow.width_Hz, rel=2e-15)
    with pytest.raises(ValueError, match="inversion"):
        wide.coefficients([0, 1], 1e23)
    with pytest.raises(MemoryError):
        LTELinePlasma([0, 1], [1, 1], [1], [0], [1], charge=1, ion_mass_amu=1,
                      energy_edges_eV=[0, 1, 2], fwhm_eV=0, source_id="test",
                      emission_scope="test", memory_cap_bytes=1)


def source_config(**kwargs):
    return SlabSourceConfig(**(dict(ion_density_per_m3=1e23, thickness_m=30e-6,
                                   temperature_eV=30, laser_wavelength_m=10.6e-6,
                                   coulomb_log=5, duration_s=2e-9) | kwargs))


def test_dark_adiabatic_slab_and_pressure_work():
    # Remove excitation heat capacity for the analytic polytropic reference.
    plasma = LTELinePlasma([0], [1], np.array([], int), np.array([], int), [],
                         charge=12, ion_mass_amu=120, energy_edges_eV=[0, 1, 2],
                         fwhm_eV=0, source_id="dark-ideal-gas", emission_scope="no-lines")
    c = source_config(rtol=2e-9)
    result = evolve_lte_slab(plasma, c, lambda t: 0)
    T, L, v = result["state"][:3]
    assert L[-1] > L[0] and v[-1] > 0 and T[-1] < T[0]
    np.testing.assert_allclose(T * L**(2 / 3), T[0] * L[0]**(2 / 3), rtol=4e-7)
    assert result["maximum_relative_energy_defect"] < 2e-8


def test_laser_radiation_and_preheat_energy_ledger():
    plasma = plasma_fixture(width=0.5)
    c = source_config()
    on = evolve_lte_slab(plasma, c, lambda t: 1e12)
    off = evolve_lte_slab(plasma, c, lambda t: 0)
    y = on["state"]
    assert y[3, -1] == pytest.approx(2000, rel=2e-14)
    assert y[4, -1] + y[5, -1] + y[6, -1] == pytest.approx(y[3, -1], rel=2e-14)
    assert y[8, -1] > off["state"][8, -1] > 0  # preheat radiates even without a laser
    assert on["maximum_relative_energy_defect"] < 2e-7
    assert off["maximum_relative_energy_defect"] < 2e-7
    assert np.all(y[8] <= y[7])
    spectrum, audit = integrated_escaped_spectrum(plasma, on, c.ion_density_per_m3 * c.thickness_m)
    assert spectrum.sum() == pytest.approx(y[7, -1], rel=5e-4)
    assert audit["band_relative_error"] < 5e-4
    assert HC_EV_NM / 13.635 < 92.125 < HC_EV_NM / 13.365
    with pytest.raises(ValueError, match="density ratio"):
        evolve_lte_slab(plasma, source_config(ion_density_per_m3=1e27), lambda t: 1e12)


def two_level_crm(omega=0):
    return CollisionalRadiativeModel(
        [CRMLevel("ground", 0, 1), CRMLevel("upper", 92.125, 3)],
        [CRMTransition(1, 0, 1e10, omega_E_eV=np.array([92.125, 1092.125]),
                       omega=np.array([omega, omega]))], Z=50, charge=12)


def test_crm_sparse_dense_and_escape_use_identical_rates():
    crm = two_level_crm(2)
    dense = crm.rate_matrix(1e18, 30)
    sparse = crm.rate_matrix(1e18, 30, sparse=True)
    np.testing.assert_array_equal(dense, sparse.toarray())
    trapped = crm.rate_matrix(1e18, 30, radiative_factors=[0.1])
    assert dense[1, 0] == trapped[1, 0]  # excitation unchanged
    assert dense[0, 1] - trapped[0, 1] == pytest.approx(0.9e10)
    assert np.max(np.abs(sparse.sum(axis=0))) == 0


def test_sparse_crm_preserves_charge_changing_and_dr_channels():
    I = 100.0
    crm = CollisionalRadiativeModel(
        [CRMLevel("bound", 0, 1), CRMLevel("satellite", 90, 3), CRMLevel("continuum", I, 2)],
        [CRMTransition(1, 0, 1e10)],
        bound_free_channels=[CRMBoundFreeChannel(
            0, 2, I, eii_rate_cm3_s=lambda T: 2e-8,
            photoionization_sigma_cm2=lambda E: 1e-18 * (I / E)**3)],
        dr_satellite_sources=[CRMDRSatelliteSource(1, 2, lambda T: 1e-10)])
    options = dict(radiation_temperature_eV=30)
    dense = crm.rate_matrix(1e19, 50, **options)
    sparse = crm.rate_matrix(1e19, 50, sparse=True, **options)
    np.testing.assert_array_equal(dense, sparse.toarray())
    assert dense[1, 2] == 1e9 and dense[2, 0] > 0 and dense[0, 2] > 0
    assert np.max(np.abs(dense.sum(axis=0))) / np.max(abs(dense)) < 2e-16


def test_nonlte_radiation_cools_excitation_without_double_cooling_electrons():
    plasma = plasma_fixture(width=0.5)
    config = source_config(expansion=False, duration_s=1e-10, rtol=1e-9)
    result = evolve_cr_slab(plasma, config, lambda t: 0, two_level_crm(), [0.8, 0.2])
    y = result["state"]
    assert y[10, -1] < 0.2
    np.testing.assert_allclose(y[0], 30, rtol=0, atol=2e-12)
    assert result["maximum_relative_energy_defect"] < 1e-12
    # Net loss of excitation equals emitted/escaped photon energy exactly.
    expected_loss = config.ion_density_per_m3 * config.thickness_m * (0.2 - y[10, -1]) * 92.125 * EV_J
    assert y[7, -1] == pytest.approx(expected_loss, rel=1e-12)


def test_line_escape_is_adjoint_to_emission_and_requires_collision_data():
    plasma = plasma_fixture(width=0.5)
    p = np.array([0.8, 0.2])
    ni, L = 1e23, 30e-6
    eps, alpha = plasma.coefficients(p, ni)
    factors = plasma.line_escape_factors(alpha * L)
    assert slab_escaped_flux(eps, alpha, L).sum() == pytest.approx(
        ni * L * p[1] * 1e10 * 92.125 * EV_J * factors[0], rel=1e-14)
    missing = two_level_crm()
    missing.transitions[0].omega = None
    with pytest.raises(ValueError, match="collision strengths"):
        evolve_cr_slab(plasma, source_config(), lambda t: 0, missing, p)
    with pytest.raises(MemoryError, match="no levels removed"):
        evolve_cr_slab(plasma, source_config(), lambda t: 0, two_level_crm(), p, max_dynamic_levels=1)
