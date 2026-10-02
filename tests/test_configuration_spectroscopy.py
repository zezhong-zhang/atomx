"""Atomic occupation moments and energy/normalization anchors."""

import numpy as np
import pytest

from atomx.constants import ALPHA, AU_RATE_PER_S, HARTREE_EV
from atomx.methods.dirac_fock import DFSolver
from atomx.properties.transitions.fluorescence import fluorescence_rates
from atomx.scattering.models import ContinuumModel, PotentialModel
from atomx.scattering.photoionization import subshell_photoionization
from atomx.scattering.radiative_recombination import _rr_potential


def test_one_electron_configuration_has_no_electron_pairs():
    solver = DFSolver.from_configuration(1, "2p1", ns=40, ks=7, r_max=25)
    # Neither jj member can hold two electrons, nor can both hold one.
    assert all(solver._get_WW(i, j) == 0 for i in range(2) for j in range(2))
    assert sum(s.occupation for s in solver.subshells) == pytest.approx(1)


def test_p2_pair_moments_from_microstate_combinatorics():
    solver = DFSolver.from_configuration(6, "2p2", ns=40, ks=7, r_max=25)
    # There are C(6,2)=15 microstates: one contains both j=1/2 states,
    # six contain two j=3/2 states, eight contain one of each.
    assert solver._get_WW(0, 0) == pytest.approx(2 / 15)
    assert solver._get_WW(1, 1) == pytest.approx(12 / 15)
    assert solver._get_WW(0, 1) == pytest.approx(8 / 15)


def test_statistical_p2_fock_field_agrees_between_solver_modes():
    solver = DFSolver.from_configuration(6, "2p2", ns=50, ks=7, r_max=25)
    result = solver.solve(tol=1e-9, max_iter=100, mode="orbital")
    assert result.converged
    # Compare the separate direct/exchange builders with the independent
    # angular-coefficient assembly used by batch mode, including both
    # Dirac components. This catches inconsistent occupation moments and
    # missing P-density contributions to the small-component field.
    for i, sh in enumerate(solver.subshells):
        direct = solver._build_direct_potential(result.orbitals_raw, sh.kappa, 0)
        exchange = solver._build_exchange_matrix_hf(sh.kappa, 0, result.orbitals_raw)
        intra = solver._build_intra_shell_potential(sh.kappa, 0, result.orbitals_raw)
        field = (
            direct[0] - exchange[0] + intra[0],
            direct[1] - exchange[1] + intra[1],
            -exchange[2],
            -exchange[3],
        )
        batch_field = solver._build_hf_matrix_dbsr(i, result.orbitals_raw)
        for actual, expected in zip(field, batch_field, strict=True):
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)


@pytest.fixture(scope="module")
def hydrogen_excited():
    solver = DFSolver.from_configuration(
        1, "2p1", ns=160, ks=9, r_max=40, nuclear="point"
    )
    result = solver.solve(tol=1e-9, max_iter=100, mode="orbital")
    assert result.converged
    # Add an independently solved hydrogenic 1s to the same radial basis
    # for a fixed-orbital energy/fluorescence anchor (not a virtual solve).
    from atomx.methods.dirac_fock import Subshell

    ground_solver = DFSolver.from_configuration(
        1, "1s1", ns=160, ks=9, r_max=40, nuclear="point"
    )
    ground = ground_solver.solve(tol=1e-9, max_iter=100, mode="orbital")
    np.testing.assert_array_equal(solver.bsp_data["t"], ground_solver.bsp_data["t"])
    P, Q, _ = ground.orbitals_by_subshell()[1, -1]
    result.subshells.append(Subshell(1, -1, 0))
    # The solver/result share this basis inventory.
    result.orbitals_raw[-1] = [(P, Q, 0)]
    return solver, result


def test_hydrogen_frozen_configuration_and_fluorescence(hydrogen_excited):
    solver, result = hydrogen_excited
    before = [s.occupation for s in result.subshells]
    Ef = solver.frozen_configuration_energy(result, "1s1")
    assert Ef == pytest.approx(-0.5, abs=1e-5)
    assert result.total_energy == pytest.approx(-0.125, abs=4e-6)
    energy = (result.total_energy - Ef) * HARTREE_EV
    rates = fluorescence_rates(result, (1, 0), photon_energies_eV={(2, 1): energy})
    # Two available 1s magnetic states: total emission is twice the
    # per-vacancy rate, with the physical hydrogen Ly-alpha lifetime.
    assert rates["vacancies"] == 2
    radial_1s_2p = 256 / (81 * np.sqrt(6))
    analytic_rate = (4 / 9) * ALPHA**3 * (3 / 8) ** 3 * radial_1s_2p**2 * AU_RATE_PER_S
    assert rates["total_per_s"] == pytest.approx(analytic_rate, rel=3e-5)
    assert rates["per_vacancy_per_s"] == rates["total_per_s"] / 2
    assert before == [s.occupation for s in result.subshells]


def test_continuum_charge_comes_from_actual_field(hydrogen_excited):
    _, result = hydrogen_excited
    sh = next(s for s in result.subshells if s.occupation > 0)
    # Full one-electron Hartree field has neutral tail; a frozen model
    # must not announce charge +1 merely because it is used by PI/RR.
    model = ContinuumModel.shooting(
        potential=PotentialModel.static_hartree(), orbital_removal="none"
    )
    potential, charge = _rr_potential(result, sh, model)
    assert charge == 0
    assert potential.residual_charge == charge
    assert abs(1e4 * potential(np.array([1e4]))[0]) < 1e-10


def test_pi_threshold_and_occupation_are_explicit(hydrogen_excited):
    _, result = hydrogen_excited
    occupied = [s.label for s in result.subshells if s.occupation > 0]
    old = dict(result.orbital_energies)
    thresholds = {label: 5.0 for label in occupied}
    out = subshell_photoionization(
        result,
        [4.0],
        continuum_model=ContinuumModel.shooting(potential=PotentialModel.bare()),
        thresholds_eV=thresholds,
    )
    assert np.array_equal(out.sigma_total_a0_sq, [0.0])
    assert out.provenance["occupation_weighted"]
    assert out.provenance["thresholds_eV"] == thresholds
    assert result.orbital_energies == old
