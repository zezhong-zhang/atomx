"""Independent Saha response, layered transfer and continuum reciprocity gates."""

import numpy as np
import pytest

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.charge_mixture import ChargePartition, SahaMixture
from atomx.plasma.crm import (
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
)
from atomx.plasma.euv_source import LTELinePlasma
from atomx.plasma.lagrangian_source import (
    evolve_lagrangian_slab,
    spitzer_face_heat_flux,
)
from atomx.plasma.laser_absorption import (
    critical_density_per_m3,
    drude_layer_stack,
    drude_slab_fractions,
)
from atomx.plasma.mixture_radiation import MixtureBoundFreeEdge, MixtureRadiation
from atomx.spectrum.continuum import free_free_absorption
from atomx.spectrum.lte_continuum import LTEBoundFreeEdge, lte_continuum_coefficients
from atomx.spectrum.slab_transfer import layered_transfer, slab_escaped_flux

C = CODATA[DEFAULT_CODATA_VERSION]


def make_mixture():
    return SahaMixture([
        ChargePartition(11, np.array([0, 20]), np.array([2, 4]), ("a", "b"), "finite-test-11"),
        ChargePartition(12, np.array([0, 30]), np.array([1, 3]), ("a", "b"), "finite-test-12"),
        ChargePartition(13, np.array([0]), np.array([2]), ("a",), "finite-test-13"),
    ], [200, 260], threshold_source_id="manufactured-diagnostic-not-Sn-data")


@pytest.mark.parametrize("T,ni", [(10, 1e23), (30, 1e24), (100, 1e25)])
def test_charge_neutrality_and_independent_adjacent_saha(T, ni):
    m = make_mixture()
    r = m.equilibrium(T, ni)
    p, f, ne = r["populations"], r["charge_fractions"], r["electron_density_per_m3"]
    assert p.sum() == pytest.approx(1, abs=4e-14)
    assert ne == pytest.approx(ni * f @ m.charges, rel=4e-14)
    for i, I in enumerate(m.ionization_potentials_eV):
        partitions = [float(np.sum(g.statistical_weights * np.exp(-g.energies_eV / T))) for g in m.partitions[i:i+2]]
        expected = saha_ionization_density_cm3(I, T, g_bound=partitions[0], g_continuum=partitions[1]) * 1e6
        assert f[i + 1] * ne / f[i] == pytest.approx(expected, rel=2e-11)


def test_eos_derivatives_include_density_and_electron_response():
    m = make_mixture()
    T, ni, dT, dlog = 30.0, 1e24, 1e-3, 1e-5
    r = m.equilibrium(T, ni)
    numerical_T = (m.equilibrium(T + dT, ni)["internal_energy_eV"] - m.equilibrium(T - dT, ni)["internal_energy_eV"]) / (2 * dT)
    numerical_n = (m.equilibrium(T, ni * np.exp(dlog))["internal_energy_eV"] - m.equilibrium(T, ni * np.exp(-dlog))["internal_energy_eV"]) / (2 * dlog)
    assert r["heat_capacity_eV_per_eV"] == pytest.approx(numerical_T, rel=3e-8)
    assert r["energy_derivative_log_density_eV"] == pytest.approx(numerical_n, rel=3e-8)
    assert abs(r["neutrality_log_residual"]) < 1e-12


def test_mixture_owns_partitions_and_binds_actual_eos_and_radiative_inputs():
    E = np.array([0., 20.])
    first = ChargePartition(11, E, np.array([1., 3.]), ("a", "b"), "same-provenance")
    second = ChargePartition(12, np.array([0.]), np.ones(1), ("a",), "same-upper")
    original = SahaMixture([first, second], [200], threshold_source_id="same-threshold-source")
    E[1] = 30
    assert original.partitions[0].energies_eV[1] == 20
    changed = SahaMixture([first, second], [200], threshold_source_id="same-threshold-source")
    threshold = SahaMixture(original.partitions, [201], threshold_source_id="same-threshold-source")
    assert len({original.source_id, changed.source_id, threshold.source_id}) == 3
    assert radiation_mixture(1e9).source_id != radiation_mixture(1e10).source_id


@pytest.mark.parametrize("tau", [0, .001, .1, 1, 100])
def test_layered_solution_reduces_to_analytic_uniform_slab(tau):
    eps = np.full((5, 2), 10.0)
    alpha = np.full_like(eps, tau / 2)
    dz = np.full(5, .4)
    r = layered_transfer(eps, alpha, dz, angular_order=96)
    expected = slab_escaped_flux(eps[0], alpha[0], 2)
    np.testing.assert_allclose(r["left_flux_W_m2_bin"] + r["right_flux_W_m2_bin"], expected, rtol=2e-6)
    # Independent local emission-minus-absorption balance as well as face telescoping.
    loss = (eps - 4 * np.pi * alpha * r["mean_intensity_W_m2_sr_bin"]) * dz[:, None]
    np.testing.assert_allclose(loss, r["cell_net_loss_W_m2_bin"], rtol=2e-10, atol=1e-12)
    assert np.max(abs(r["energy_balance_defect_W_m2_bin"])) < 1e-12


def test_cold_foreground_absorbs_heats_and_changes_viewing_direction():
    eps, alpha, dz = np.array([[100.], [0.]]), np.array([[0.], [20.]]), np.ones(2)
    r = layered_transfer(eps, alpha, dz, angular_order=48)
    assert r["cell_net_loss_W_m2_bin"][1, 0] < 0
    assert r["right_flux_W_m2_bin"][0] < r["left_flux_W_m2_bin"][0] * 1e-8
    assert abs(r["cell_net_loss_W_m2_bin"].sum() - r["left_flux_W_m2_bin"].sum() - r["right_flux_W_m2_bin"].sum()) < 1e-12
    with pytest.raises(MemoryError, match="no cells/bins removed"):
        layered_transfer(eps, alpha, dz, memory_cap_bytes=1)


def test_planck_bath_is_stationary_in_every_layer():
    source = np.array([2., 3.])
    alpha = np.array([[1., 8.], [3., .2], [0., 9.]])
    r = layered_transfer(4 * np.pi * alpha * source, alpha, [.1, .2, .3],
                         incident_left_W_m2_sr_bin=source, incident_right_W_m2_sr_bin=source)
    np.testing.assert_allclose(r["mean_intensity_W_m2_sr_bin"], np.tile(source, (3, 1)), atol=2e-14)
    assert np.max(abs(r["cell_net_loss_W_m2_bin"])) < 1e-13


def test_bound_free_lte_photon_number_matches_independent_milne_rr():
    I, T, ne, nc = 50., 20., 1e24, 1e22
    saha_m3 = saha_ionization_density_cm3(I, T, g_bound=2, g_continuum=1) * 1e6
    nb = ne * nc / saha_m3

    def sigma(E):
        return 3e-18 * (I / E)**3

    edges = np.linspace(I, I + 40 * T, 32001)
    r = lte_continuum_coefficients(edges, T, ne, nc + nb, 1,
        free_free_gaunt=lambda x: np.ones_like(x), gaunt_source_id="unit-diagnostic",
        bound_free_edges=[LTEBoundFreeEdge(I, nb, sigma, "analytic-PI-diagnostic")])
    E = (edges[:-1] + edges[1:]) / 2
    photons = np.sum(r["free_bound_emissivity_W_m3_bin"] / (E * C["_e"]))
    rr_cm3 = radiative_recombination_rate_cm3_s(I, T, g_bound=2, g_continuum=1, cross_section_cm2=sigma)
    assert photons == pytest.approx(ne * nc * rr_cm3 / 1e6, rel=2e-6)


def test_free_free_kirchhoff_and_absorption_without_emission_underflow():
    edges = np.linspace(10, 200, 101)
    E, T = (edges[:-1] + edges[1:]) / 2, 30.
    r = lte_continuum_coefficients(edges, T, 1e24, 1e23, 144,
         free_free_gaunt=lambda x: np.ones_like(x), gaunt_source_id="unit-diagnostic")
    nu = E * C["_e"] / C["_hplanck"]
    B = 2 * C["_hplanck"] * nu**3 / C["_c"]**2 / np.expm1(E / T)
    ratio = r["emissivity_W_m3_bin"] / (r["absorption_per_m"] * 4 * np.pi * np.diff(edges) * C["_e"] / C["_hplanck"])
    np.testing.assert_allclose(ratio, B, rtol=2e-9)
    a = free_free_absorption([10000.], 1., 12, n_e_cm3=1e18, n_i_cm3=1e17)
    assert np.isfinite(a[0]) and a[0] > 0


def test_charge_partition_requires_missing_stages_instead_of_inventing_them():
    m = make_mixture()
    with pytest.raises(ValueError, match="contiguous"):
        SahaMixture([m.partitions[0], m.partitions[2]], [200], threshold_source_id="test")
    bad = ChargePartition(11, np.array([0, 201]), np.array([1, 1]), ("a", "b"), "test")
    with pytest.raises(ValueError, match="resonance/DR"):
        SahaMixture([bad, m.partitions[1]], [200], threshold_source_id="test")


def radiation_mixture(A=0, continuum="none"):
    tables = {q: LTELinePlasma([0, 90], [1, 3], [1], [0], [A], charge=q, ion_mass_amu=120,
                energy_edges_eV=np.arange(0, 400.5, .5), fwhm_eV=1,
                source_id=f"manufactured-charge-{q}", emission_scope="one synthetic line") for q in (11, 12)}
    eos = SahaMixture([ChargePartition(q, t.energy, t.g, ("a", "b"), t.source_id)
                       for q, t in tables.items()], [200], threshold_source_id="manufactured-threshold")
    edges = [] if continuum != "free-free+bound-free" else [MixtureBoundFreeEdge(11, 0, 0, lambda E: 1e-18 * (200 / E)**3, "analytic-test-edge")]
    return MixtureRadiation(eos, tables, continuum_mode=continuum, nuclear_source_id="diagnostic-120amu",
            free_free_gaunt=lambda x: np.ones_like(x), gaunt_source_id="unit-diagnostic",
            bound_free_edges=edges, bound_free_coverage="one synthetic channel")


def test_mixture_radiation_matches_state_identity_and_adds_continua():
    bare, full = radiation_mixture(1e10), radiation_mixture(1e10, "free-free+bound-free")
    left, right = bare.cell(30, 1e23), full.cell(30, 1e23)
    assert np.sum(right["emissivity_W_m3_bin"]) > np.sum(left["emissivity_W_m3_bin"])
    assert np.sum(right["absorption_per_m"]) > np.sum(left["absorption_per_m"])
    state = bare.eos.equilibrium(30, 1e23)
    expected = 1e23 * sum(state["populations"][s][1] * 1e10 * 90 * C["_e"] for s in bare.eos.slices)
    assert np.sum(left["emissivity_W_m3_bin"]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="one explicit"):
        MixtureRadiation(bare.eos, {11: bare.tables[11]}, continuum_mode="none", nuclear_source_id="test")
    with pytest.raises(ValueError, match="integers"):
        MixtureRadiation(bare.eos, bare.tables, continuum_mode="free-free+bound-free",
            nuclear_source_id="test", free_free_gaunt=lambda x: np.ones_like(x),
            gaunt_source_id="test", bound_free_coverage="invalid fractional charge",
            bound_free_edges=[MixtureBoundFreeEdge(11.5, 0, 0, lambda E: 1e-18, "test")])


def test_heat_conduction_is_internal_and_flows_down_temperature_gradient():
    flux = spitzer_face_heat_flux([10, 30, 20], [1e24]*3, [12]*3, [1e-5]*3, 5, .05)
    assert flux[0] == flux[-1] == 0
    assert flux[1] < 0 and flux[2] > 0
    # The source is a telescoping divergence; compare to its floating-point
    # summation bound, not an absolute 1e-6 W/m² below the ulp of ~1e11 fluxes.
    assert abs(np.diff(flux).sum()) <= 2 * np.finfo(float).eps * np.sum(abs(np.diff(flux)))
    qsat = .05 * 1e24 * 10 * C["_e"] * np.sqrt(10 * C["_e"] / C["_me"])
    assert abs(flux[1]) < qsat


def test_multicharge_lagrangian_pressure_work_conserves_total_energy():
    model = radiation_mixture()
    r = evolve_lagrangian_slab(model, [1e-5]*3, [1e23]*3, [30]*3,
        duration_s=1e-10, laser_intensity_W_m2=lambda t: 0, laser_wavelength_m=10.6e-6,
        coulomb_log=5, conduction_flux_limit=None, angular_order=12, rtol=1e-9)
    assert r["positions_m"][0, -1] < 0 and r["positions_m"][-1, -1] > 3e-5
    assert r["kinetic_J_m2"][-1] > 0
    assert r["maximum_relative_energy_defect"] < 2e-8
    np.testing.assert_array_equal(r["ledger_J_m2"], 0)


def test_spatial_source_laser_radiation_and_conduction_close_together():
    model = radiation_mixture(1e10, "free-free+bound-free")
    r = evolve_lagrangian_slab(model, [1e-5]*3, [1e23]*3, [20, 30, 20],
        duration_s=1e-10, laser_intensity_W_m2=lambda t: 1e12, laser_wavelength_m=10.6e-6,
        coulomb_log=5, conduction_flux_limit=.05, angular_order=12)
    led = r["ledger_J_m2"]
    assert led[0, -1] == pytest.approx(100)
    assert led[1, -1] + led[2, -1] + led[3, -1] == pytest.approx(100)
    assert led[4, -1] > 0 and led[5, -1] > 0
    assert r["maximum_relative_energy_defect"] < 1e-7


def test_wave_deposition_lossless_quarter_wave_and_opaque_limits():
    wavelength = 2e-6
    nc = critical_density_per_m3(wavelength)
    lossless = drude_layer_stack([.75 * nc], [0], wavelength, [wavelength / 2])
    assert lossless["reflection"] == pytest.approx(.36, abs=2e-15)
    assert lossless["transmission"] == pytest.approx(.64, abs=2e-15)
    assert lossless["absorption_by_cell"][0] == pytest.approx(0, abs=2e-15)
    omega = 2 * np.pi * C["_c"] / wavelength
    thick = drude_layer_stack([.6 * nc], [.1 * omega], wavelength, [100 * wavelength])
    single = drude_slab_fractions(.6 * nc, .1 * omega, wavelength, 100 * wavelength)
    assert thick["reflection"] == pytest.approx(single["reflection"], abs=2e-15)
    assert thick["absorption_by_cell"].sum() == pytest.approx(single["absorption"], abs=2e-15)


def test_wave_deposition_crosses_critical_density_without_a_clamp():
    wavelength = 10.6e-6
    nc = critical_density_per_m3(wavelength)
    omega = 2 * np.pi * C["_c"] / wavelength
    ne = nc * np.array([.2, .8, 1, 1.5, 5, 20])
    result = drude_layer_stack(ne, np.full(6, .03 * omega), wavelength, np.full(6, 3e-6))
    assert np.max(result["critical_density_ratio"]) == 20
    assert result["reflection"] > 0 and result["transmission"] < 1e-10
    assert result["absorption_by_cell"].min() >= 0
    assert abs(result["energy_balance_defect"]) < 2e-14
