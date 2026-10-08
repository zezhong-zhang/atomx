"""Independent angular anchors and actual Dirac-Coulomb CI loss dynamics."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan, wigner_3j

from atomx import Atom, Basis, Correlation
from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
from atomx.config.csf import enumerate_jj_csfs
from atomx.constants import ALPHA, AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.ci_continuum import TwoElectronCIChannels
from atomx.scattering.models import ContinuumModel
from atomx.xfel import CoherentGaussianPulse, TwoElectronCoherentModel


def continuum(nodes=16000):
    return ContinuumModel.shooting(n_grid=nodes, r_max=80, orbital_removal="none")


def atom(config, frozen=False):
    value = Atom(
        Z=10,
        config=config,
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(radial_optimization="fac_frozen_local")
        if frozen
        else None,
    )
    value.solve(max_iter=60, tol=1e-6)
    assert value.converged
    return value


@pytest.fixture(scope="module")
def neon():
    return atom(["1s2", "1s1 2p1"])


@pytest.fixture(scope="module")
def joint(neon):
    energy = max(
        neon.transitions(multipoles=["E1"]), key=lambda l: l.A_rate_per_s
    ).energy_eV
    pulse = CoherentGaussianPulse(energy, 1e17, 10)
    return TwoElectronCoherentModel(
        neon,
        pulse,
        photon_numbers={i: int(i != 0) for i in range(5)},
        ionic_photon_numbers={i: 0 for i in range(3)},
        continuum_model=continuum(),
    ), pulse


def test_joint_driver_embeds_selected_parent_population_without_ionic_initial_leak(neon, joint):
    from atomx import CILevel, CILevelPopulation

    baseline, pulse = joint
    selected = TwoElectronCoherentModel(
        neon,
        pulse,
        photon_numbers={i: int(i != 0) for i in range(5)},
        ionic_photon_numbers={i: 0 for i in range(3)},
        continuum_model=continuum(),
        initial_state=CILevelPopulation(((CILevel(0, 1), 1), (CILevel(2, -1, 1), 3))),
    )
    np.testing.assert_array_equal(selected.system.dipole, baseline.system.dipole)
    assert selected.system.provenance["basis_sha256"] == baseline.system.provenance["basis_sha256"]
    populations = np.diag(selected.initial_density).real
    for (number, level), magnetic in selected.by_level.items():
        actual = sum(populations[selected.indices[state]] for state in magnetic.values())
        expected = {0: 0.25, 4: 0.75}.get(level, 0) if number == 2 else 0
        assert actual == pytest.approx(expected, abs=1e-15)
    assert np.trace(selected.initial_density) == pytest.approx(1)


def test_all_integer_J_M_determinants_match_independent_physical_j_CG():
    for csf in enumerate_jj_csfs("1s1 2p1"):
        sig = csf_signature(csf)
        keys = tuple(sorted((n, k) for n, k, _ in sig.subshells))
        spinors = [
            (n, k, m) for n, k in keys for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
        ]
        for m in range(-sig.total_J2, sig.total_J2 + 1, 2):
            actual = dict(csf_magnetic_vector(sig, m, keys))
            expected = {}
            for a, (na, ka, ma) in enumerate(spinors):
                for b, (nb, kb, mb) in enumerate(spinors):
                    if (na, ka) != keys[0] or (nb, kb) != keys[1]:
                        continue
                    value = float(
                        clebsch_gordan(
                            Rational(2 * abs(ka) - 1, 2),
                            Rational(2 * abs(kb) - 1, 2),
                            Rational(sig.total_J2, 2),
                            Rational(ma, 2),
                            Rational(mb, 2),
                            Rational(m, 2),
                        )
                    )
                    if value:
                        expected[(1 << a) | (1 << b)] = value
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(
                list(actual.values()), [expected[k] for k in actual], atol=1e-15
            )


def test_closed_shell_PI_has_occupation_once_and_independent_angular_prefactor():
    channels = TwoElectronCIChannels(atom(["1s2"]), continuum_model=continuum())
    photon = 2000.0
    omega = photon / HARTREE_EV
    kinetic = (
        omega + channels.initial.state_energies[0] - channels.ion.state_energies[0]
    )
    expected = 0.0
    for kappa in (1, -2):
        p, q = channels._wave(0, kappa, kinetic)
        radial = radial_eL(
            *channels.bound[1, -1],
            p,
            q,
            channels.workspace,
            1,
            kappa_i=-1,
            kappa_f=kappa,
            omega=omega,
        )
        j = Rational(2 * abs(kappa) - 1, 2)
        # Spin-angular reduced norm from physical-j SymPy, independent of atomx ckm.
        strength = float(
            (2 * j + 1)
            * 2
            * wigner_3j(j, 1, Rational(1, 2), Rational(-1, 2), 0, Rational(1, 2)) ** 2
        )
        expected += strength * radial**2
    expected *= 4 * np.pi**2 * ALPHA * omega / 3 * (BOHR_RADIUS_M * 100) ** 2
    values = channels.photoionization(photon)
    actual = sum(
        amplitude**2
        for ch in values
        if ch.initial_electrons == 2
        for _, amplitude in ch.components
    )
    assert actual == pytest.approx(expected, rel=1e-14)
    for polarization in (-1, 1):
        circular = channels.photoionization(photon, polarization_q=polarization)
        total = sum(
            a * a
            for ch in circular
            if ch.initial_electrons == 2
            for _, a in ch.components
        )
        assert total == pytest.approx(expected, rel=1e-14)
        for ch in circular:
            if ch.initial_electrons == 2:
                for (mf, mi), _ in ch.components:
                    assert mf + ch.electron_m2 == mi + 2 * polarization
    assert not channels.photoionization(1.0)
    assert not channels.auger()
    model = TwoElectronCoherentModel(
        channels.atom,
        CoherentGaussianPulse(photon, 1e18, 10),
        photon_numbers={0: 0},
        ionic_photon_numbers={0: 0},
        continuum_model=continuum(),
    )
    pulse = CoherentGaussianPulse(photon, 1e18, 10)
    result = model.solve(pulse)
    from scipy.special import erf

    start = result.times_fs[0]
    sigma_t = pulse.fwhm_fs / np.sqrt(8 * np.log(2))
    exposure = (
        pulse.fluence_photons_per_cm2
        / 2
        * (
            erf((result.times_fs - pulse.center_fs) / (np.sqrt(2) * sigma_t))
            - erf((start - pulse.center_fs) / (np.sqrt(2) * sigma_t))
        )
    )
    np.testing.assert_allclose(
        model.level_populations(result)[2, 0], np.exp(-expected * exposure), atol=1e-10
    )


def test_pure_pair_Auger_spin_anchor_counts_distinct_electron_baths(monkeypatch):
    # Synthetic radial unit anchor ONLY; final Slater spin functions derived
    # from Kronecker deltas, independent of atomx two_body_angular.
    source = TwoElectronCIChannels.__new__(TwoElectronCIChannels)
    source.quantum = {2: {0: (0, 1)}, 1: {0: (1, 1)}}
    source.initial = SimpleNamespace(state_energies=np.array([0.0]))
    source.ion = SimpleNamespace(state_energies=np.array([-1.0]))
    source.spinors = ((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1))
    source.vectors = {2: {(0, 0): {12: 1.0}}, 1: {(0, -1): {1: 1.0}, (0, 1): {2: 1.0}}}
    source.bound = {
        (1, -1): (np.ones(2), np.zeros(2)),
        (2, -1): (np.ones(2), np.zeros(2)),
    }
    source.data = {}
    source.weights = np.array([0.5, 0.5])
    source._h_bound = {key: (np.zeros(2), np.zeros(2)) for key in source.bound}
    source._wave = lambda *args: (np.ones(2), np.zeros(2))
    source._prepare_waves = lambda *args: None
    monkeypatch.setattr("atomx.scattering.ci_continuum.slater_rk", lambda *args: 1.0)
    output = source.auger()
    assert len(output) == 2
    for ch in output:
        (((mf, mi), amplitude),) = ch.components
        assert mi == 0
        spin = float(mf == -1 and ch.electron_m2 == 1) - float(
            mf == 1 and ch.electron_m2 == -1
        )
        assert amplitude == pytest.approx(
            np.sqrt(2 * np.pi * AU_RATE_PER_S) * spin, rel=1e-14
        )
    assert sum(a * a for c in output for _, a in c.components) == pytest.approx(
        4 * np.pi * AU_RATE_PER_S, rel=1e-14
    )

    # Now isolate h (Coulomb zero): a normalized 1s2s singlet loses the
    # 2s electron with total Dyson norm one, not the closed-pair norm two.
    source.vectors[2][0, 0] = {6: 1 / np.sqrt(2), 9: -1 / np.sqrt(2)}
    source._h_bound[2, -1] = (np.ones(2), np.zeros(2))
    monkeypatch.setattr("atomx.scattering.ci_continuum.slater_rk", lambda *args: 0.0)
    singles = source.auger()
    assert sum(a * a for c in singles for _, a in c.components) == pytest.approx(
        2 * np.pi * AU_RATE_PER_S, rel=1e-14
    )


def test_real_joint_resonant_PI_trace_positivity_and_CI_phase_covariance(neon, joint):
    original, pulse = joint
    assert len(original.system.state_ids) == 22
    assert original.pi_channels and not original.auger_channels
    result = original.solve(pulse, after_pulse_fs=30)
    assert 8 < result.mean_charge[-1] < 8.01
    np.testing.assert_allclose(
        np.trace(result.density_matrices, axis1=1, axis2=2), 1, atol=1e-12
    )
    assert np.linalg.eigvalsh(result.density_matrices).min() > -1e-10
    changed = copy.copy(neon)
    changed._result = copy.copy(neon.result)
    changed._result.mixing_coefficients = neon.result.mixing_coefficients.copy()
    changed._result.mixing_coefficients[:, 4] *= -1
    other = TwoElectronCoherentModel(
        changed,
        pulse,
        photon_numbers={i: int(i != 0) for i in range(5)},
        ionic_photon_numbers={i: 0 for i in range(3)},
        continuum_model=continuum(),
    )
    signs = np.array(
        [
            -1 if original.level_by_state[s] == (2, 4) else 1
            for s in original.system.state_ids
        ]
    )
    for a, b in zip(
        original.system.pi_operators, other.system.pi_operators, strict=True
    ):
        np.testing.assert_allclose(b, signs[:, None] * a * signs[None, :], atol=1e-24)
    flipped = other.solve(pulse, after_pulse_fs=30)
    np.testing.assert_allclose(
        flipped.density_matrices,
        signs[None, :, None] * result.density_matrices * signs[None, None, :],
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="another joint"):
        original.level_populations(flipped)


def test_actual_open_Auger_frozen_control_matches_exponential_decay():
    value = atom(["1s2", "1s1 2p1", "2s2"], frozen=True)
    pulse = CoherentGaussianPulse(923, 0, 10)
    model = TwoElectronCoherentModel(
        value,
        pulse,
        photon_numbers={i: 0 for i in range(6)},
        ionic_photon_numbers={i: 0 for i in range(4)},
        continuum_model=continuum(),
    )
    assert len(model.auger_channels) == 2
    provider = model.channels
    h = np.array(
        [
            [
                np.sum(
                    provider.weights
                    * (
                        provider.bound[a][0] * provider._h_bound[b][0]
                        + provider.bound[a][1] * provider._h_bound[b][1]
                    )
                )
                if a[1] == b[1]
                else 0
                for b in provider.keys
            ]
            for a in provider.keys
        ]
    )
    # Independent CI assembly of h_D and its differential radial form agree.
    np.testing.assert_allclose(
        np.linalg.eigvalsh(h), provider.ion.state_energies, atol=1e-11, rtol=0
    )
    upper = int(np.argmax(value.state_energies))
    rho = np.zeros_like(model.initial_density)
    index = model.indices[model.by_level[2, upper][0]]
    rho[index, index] = 1
    result = model.solve(pulse, initial_density=rho, after_pulse_fs=40)
    rate = model._decay_rates[index]
    width = sum(a * a for c in model.auger_channels for _, a in c.components)
    assert 1e14 < width < 3e14
    assert model.system.provenance["parent_radial_optimization"] == "fac_frozen_local"
    elapsed_s = (result.times_fs - result.times_fs[0]) * 1e-15
    np.testing.assert_allclose(
        model.level_populations(result)[2, upper], np.exp(-rate * elapsed_s), atol=1e-10
    )
    expected_q = 8 + width / rate * (1 - np.exp(-rate * elapsed_s))
    np.testing.assert_allclose(result.mean_charge, expected_q, atol=1e-10)
    changed = copy.copy(value)
    changed._result = copy.copy(value.result)
    changed._result.mixing_coefficients = value.result.mixing_coefficients.copy()
    changed._result.mixing_coefficients[:, upper] *= -1
    phase_channels = TwoElectronCIChannels(changed, continuum_model=continuum()).auger()
    for a, b in zip(model.auger_channels, phase_channels, strict=True):
        assert a.initial_level == b.initial_level == upper
        for (label, amplitude), (other_label, other_amplitude) in zip(
            a.components, b.components, strict=True
        ):
            assert label == other_label
            assert other_amplitude == pytest.approx(-amplitude, rel=1e-14)


def test_physics_convention_and_stale_tensor_gates(neon, joint):
    model, pulse = joint
    for photon, duration in (
        (pulse.photon_energy_eV + 1, 10),
        (pulse.photon_energy_eV, 11),
    ):
        with pytest.raises(ValueError, match="changed carrier or duration"):
            model.solve(CoherentGaussianPulse(photon, 1e17, duration))
    changed = copy.copy(neon)
    changed.breit = True
    with pytest.raises(ValueError, match="corrections='none'"):
        TwoElectronCIChannels(changed, continuum_model=continuum())
    with pytest.raises(ValueError, match="orbital_removal"):
        TwoElectronCIChannels(neon, continuum_model=ContinuumModel.shooting())
    changed = copy.copy(neon)
    changed._result = copy.copy(neon.result)
    changed._result.mixing_coefficients = neon.result.mixing_coefficients.copy()
    # Same eigenvalues, orthonormal vectors, WRONG Hamiltonian eigenstates.
    changed._result.mixing_coefficients[:, [0, 4]] = (
        changed._result.mixing_coefficients[:, [4, 0]]
    )
    with pytest.raises(ValueError, match="diagonalize"):
        TwoElectronCIChannels(changed, continuum_model=continuum())
    threshold = (
        model.channels.ion.state_energies[0] - neon.state_energies[0]
    ) * HARTREE_EV
    with pytest.raises(ValueError, match="near-threshold"):
        TwoElectronCoherentModel(
            neon,
            CoherentGaussianPulse(threshold, 1e14, 10),
            photon_numbers={i: 0 for i in range(5)},
            ionic_photon_numbers={i: 0 for i in range(3)},
            continuum_model=continuum(),
        )
    with pytest.raises(ValueError, match="1000"):
        TwoElectronCIChannels(neon, continuum_model=continuum(100))


def test_ground_correlation_orbitals_cannot_be_relabelled_as_bound_ionic_levels():
    value = atom(["1s2", "1s1 2p1", "2s2"])
    assert value.state_energies.max() > 50  # measured: +57.7936 Ha
    with pytest.raises(ValueError, match="unbound ionic"):
        TwoElectronCIChannels(value, continuum_model=continuum())


def test_prepared_atomic_snapshot_and_energy_caches_do_not_mix_rate_tables(neon):
    source = copy.copy(neon)
    source._result = copy.copy(neon.result)
    source._result.state_energies = neon.result.state_energies.copy()
    source._result.orbitals_dict = {
        key: tuple(v.copy() for v in waves)
        for key, waves in neon.result.orbitals_dict.items()
    }
    prepared = TwoElectronCIChannels(source, continuum_model=continuum())
    first = prepared.photoionization(923)
    assert first
    # Later mutation of the caller's result cannot change this prepared atom.
    source.result.state_energies += 1
    for waves in source.result.orbitals_dict.values():
        for component in waves:
            component *= 2
    assert prepared.photoionization(923) == first
    shifted = prepared.photoionization(950)
    assert shifted != first
    assert prepared.photoionization(923) == first
    assert prepared.auger() == ()
