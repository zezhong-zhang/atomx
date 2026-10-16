"""Independent Fock-space anchors and live general-N CI continuum checks."""

from itertools import combinations
from math import comb

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j

from atomx import Atom, Basis, CIStateAverage, Correlation
from atomx.angular.coefficients import ckm
from atomx.angular.csf_hamiltonian import (
    continuum_hamiltonian_terms,
    csf_magnetic_vector,
    csf_signature,
)
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.constants import ALPHA, BOHR_RADIUS_M, HARTREE_EV
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.ci_continuum import TwoElectronCIChannels
from atomx.scattering.models import ContinuumModel
from atomx.scattering.multielectron_ci import MultiElectronCIChannels
from atomx.xfel import CoherentGaussianPulse, MultiElectronCoherentModel


def continuum():
    return ContinuumModel.shooting(n_grid=16000, r_max=80, orbital_removal="none")


def build(configs, Z=10, ns=60):
    atom = Atom(Z, config=configs, basis=Basis(ns=ns, ks=9), corrections="none",
                correlation=Correlation(state_average=CIStateAverage()))
    atom.solve(tol=1e-6, max_iter=100)
    assert atom.converged
    return atom


def operate(occupied, operations):
    """Independent sorted-list Fermi algebra, operations in action order."""
    occupied, sign = list(occupied), 1
    for index, create in operations:
        if (index in occupied) == create:
            return None, 0
        position = sum(value < index for value in occupied)
        sign *= (-1) ** position
        if create:
            occupied.insert(position, index)
        else:
            occupied.remove(index)
    return tuple(occupied), sign


def radial(a, c, b, d):
    left, right = sorted((a, c)), sorted((b, d))
    return 0.13 * (sum(left) ** 2 + sum(right) ** 2) + 0.07 * (np.prod(left) + np.prod(right))


@pytest.mark.parametrize("number", [2, 3, 4])
def test_full_continuum_hamiltonian_matches_independent_second_quantization(number):
    bound = ((1, -1, -1), (1, -1, 1), (2, -1, -1), (2, -1, 1))
    # C^0_0 is the identity for physical j=1/2, independently from SymPy.
    for m in (-1, 1):
        j, mj = Rational(1, 2), Rational(m, 2)
        angular = (-1) ** (j - mj) * wigner_3j(j, 0, j, -mj, 0, mj) * np.sqrt(2)
        assert float(angular) == pytest.approx(1, abs=1e-15)
    for m in (-1, 1):
        labels = bound + ((-1, -1, m),)
        def h(a, c, labels=labels):
            return (0.29 + 0.17 * (labels[a][0] + labels[c][0])) if labels[a][1:] == labels[c][1:] else 0
        def v(a, b, c, d, labels=labels):
            if labels[a][2] != labels[c][2] or labels[b][2] != labels[d][2]:
                return 0
            return radial(labels[a][0], labels[c][0], labels[b][0], labels[d][0])
        values = {}
        for initial in combinations(range(4), number):
            for final in combinations(range(4), number - 1):
                bra = final + (4,)
                expected = 0.0
                for a in range(5):
                    for c in range(5):
                        state, sign = operate(initial, ((c, False), (a, True)))
                        if state == bra:
                            expected += sign * h(a, c)
                for a, b in combinations(range(5), 2):
                    for c, d in combinations(range(5), 2):
                        state, sign = operate(initial, ((c, False), (d, False), (b, True), (a, True)))
                        if state == bra:
                            expected += sign * (v(a, b, c, d) - v(a, b, d, c))
                terms = continuum_hamiltonian_terms(
                    {sum(1 << i for i in final): 1.0},
                    {sum(1 << i for i in initial): 1.0}, bound, -1, m,
                )
                actual = 0.0
                for key, coefficient in terms:
                    if key[0] == "h":
                        value = 0.29 + 0.17 * (key[1] + key[3])
                    else:
                        _, rank, a, c, b, d = key
                        assert rank == 0
                        value = radial(a[0], c[0], b[0], d[0])
                    actual += coefficient * value
                assert actual == pytest.approx(expected, abs=1e-14)
                values[initial, final] = expected

        initial_vectors = {sum(1 << i for i in det): (-1)**j * (j + 1) / 7
                           for j, det in enumerate(combinations(range(4), number))}
        final_vectors = {sum(1 << i for i in det): (-1)**j * (j + 2) / 9
                         for j, det in enumerate(combinations(range(4), number - 1))}
        expected = sum(initial_vectors[sum(1 << i for i in a)] * final_vectors[sum(1 << i for i in b)] * v
                       for (a, b), v in values.items())
        actual = 0.0
        for key, coefficient in continuum_hamiltonian_terms(final_vectors, initial_vectors, bound, -1, m):
            value = 0.29 + 0.17 * (key[1] + key[3]) if key[0] == "h" else radial(*(pair[0] for pair in key[2:]))
            actual += coefficient * value
        assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_full_fixed_N_CSF_space_is_orthonormal_and_complete(number):
    keys = ((1, -1), (2, -1), (2, 1), (2, -2))
    csfs = enumerate_jj_orbital_space(keys, number)
    dimension = comb(10, number)
    assert sum(c.total_J2 + 1 for c in csfs) == dimension
    vectors = []
    masks = sorted(sum(1 << i for i in occupied) for occupied in combinations(range(10), number))
    for csf in csfs:
        for m in range(-csf.total_J2, csf.total_J2 + 1, 2):
            vector = dict(csf_magnetic_vector(csf_signature(csf), m, keys))
            vectors.append([vector.get(mask, 0) for mask in masks])
    np.testing.assert_allclose(np.asarray(vectors) @ np.asarray(vectors).T, np.eye(dimension), atol=2e-14)
    with pytest.raises(ValueError, match="cap"):
        enumerate_jj_orbital_space(keys, number, magnetic_state_cap=dimension - 1)


@pytest.fixture(scope="module")
def neon3():
    return build(["1s2 2s1", "1s1 2s1 2p1"])


@pytest.fixture(scope="module")
def joint(neon3):
    provider = MultiElectronCIChannels(neon3, continuum_model=continuum())
    energy = max(neon3.transitions(multipoles=["E1"]), key=lambda l: l.A_rate_per_s).energy_eV
    pulse = CoherentGaussianPulse(energy, 1e17, 10)
    ladders = {n: {i: int(n == 3 and i != 0) for i in range(len(r.state_energies))}
               for n, r in provider.sectors.items()}
    model = MultiElectronCoherentModel(neon3, pulse, photon_numbers=ladders,
                                      continuum_model=continuum(),
                                      radiative_reservoir="geometric_clustered",
                                      electron_reservoir="clustered")
    return model, pulse


def assert_complete_pi_fock_tensor(provider, photon_energy_eV):
    """Compare every signed PI coordinate with dense Fock annihilation matrices.

    Share physical angular/radial operators, but use the independent sorted-list
    Fermi algebra above instead of the channel builder's bit-mask contraction.
    Channel counts include roundoff-sized components and are not invariants.
    """
    omega = photon_energy_eV / HARTREE_EV
    scale = np.sqrt(4 * np.pi**2 * ALPHA * omega) * (BOHR_RADIUS_M * 100)
    actual = {
        (ch.initial_electrons, ch.initial_level, ch.final_level,
         ch.electron_kappa, ch.electron_m2, mf, mi): amplitude / scale
        for ch in provider.photoionization(photon_energy_eV)
        for (mf, mi), amplitude in ch.components
    }
    spinors = provider.spinors
    kmax = max(abs(k) for _, k, _ in spinors) + 1
    kappas = [k for k in range(-kmax, kmax + 1) if k and any(
        ckm(k, me, kb, mb, 1, 0) != 0
        for _, kb, mb in spinors
        for me in range(-2 * abs(k) + 1, 2 * abs(k), 2))]
    for number, result in provider.sectors.items():
        parents = list(combinations(range(len(spinors)), number))
        daughters = list(combinations(range(len(spinors)), number - 1))
        daughter_indices = {det: i for i, det in enumerate(daughters)}
        parent_labels = list(provider.vectors[number])
        final_vectors = provider.vectors[number - 1] if number > 1 else {(-1, 0): {0: 1.0}}
        final_labels = list(final_vectors)
        parent_matrix = np.array([
            [provider.vectors[number][label].get(sum(1 << bit for bit in det), 0)
             for label in parent_labels] for det in parents])
        final_matrix = np.array([
            [final_vectors[label].get(sum(1 << bit for bit in det), 0)
             for label in final_labels] for det in daughters])
        # Complete daughters are essential for closure of every removal path.
        np.testing.assert_allclose(final_matrix @ final_matrix.T, np.eye(len(daughters)),
                                   atol=2e-14, rtol=0)
        dyson = []
        for bit in range(len(spinors)):
            annihilator = np.zeros((len(daughters), len(parents)))
            for col, det in enumerate(parents):
                removed, sign = operate(det, ((bit, False),))
                if sign:
                    annihilator[daughter_indices[removed], col] = sign
            dyson.append(final_matrix.T @ annihilator @ parent_matrix)
        dyson = np.asarray(dyson)
        for initial in range(len(result.state_energies)):
            parent_indices = [i for i, (level, _) in enumerate(parent_labels) if level == initial]
            targets = provider.quantum[number - 1] if number > 1 else {-1: (0, 1)}
            for final in targets:
                ef = provider.sectors[number - 1].state_energies[final] if number > 1 else 0
                energy = float(omega + result.state_energies[initial] - ef)
                if energy <= 0:
                    continue
                final_indices = [i for i, (level, _) in enumerate(final_labels) if level == final]
                block = dyson[:, final_indices][:, :, parent_indices]
                for kappa in kappas:
                    p, q = provider._wave(provider._field_token(number - 1, final), kappa, energy)
                    radial = {key: radial_eL(*provider.bound[key], p, q, provider.workspace,
                                            1, kappa_i=key[1], kappa_f=kappa, omega=omega)
                              for key in provider.keys}
                    for me in range(-2 * abs(kappa) + 1, 2 * abs(kappa), 2):
                        weights = [ckm(kappa, me, k, m, 1, 0) * radial[n, k]
                                   for n, k, m in spinors]
                        expected = np.tensordot(weights, block, axes=(0, 0))
                        observed = np.array([
                            [actual.pop((number, initial, final, kappa, me,
                                         final_labels[f][1], parent_labels[i][1]), 0)
                             for i in parent_indices] for f in final_indices])
                        # Same dimensionless 1e-14 gate as the independent
                        # Fock Hamiltonian anchor, including zeros/interference.
                        np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0)
    assert not actual, "PI tensor contains coordinates outside the complete Fock model"


def test_live_three_electron_residual_spaces_particle_number_and_Hartree_tail(joint):
    model, pulse = joint
    provider = model.channels
    assert len(model.system.state_ids) == 82
    assert {n: len(r.state_energies) for n, r in provider.sectors.items()} == {3: 8, 2: 17, 1: 4}
    for number, result in provider.sectors.items():
        for level in range(len(result.state_energies)):
            _, rho = result.one_body_density_matrix(level)
            assert np.trace(rho) == pytest.approx(number, abs=1e-12)
            token = provider._field_token(number, level)
            assert float(provider._field(token)(1000)) == pytest.approx(-(10 - number) / 1000, abs=1e-14)
    assert_complete_pi_fock_tensor(provider, pulse.photon_energy_eV)
    assert len(model.auger_channels) == 68
    assert len(model.electron_clusters) == 7
    assert max(c["signed_tensor_relative_change"] for c in model.electron_clusters) < 2e-5
    for bound in model.bound_models.values():
        np.testing.assert_allclose(list(bound.system.provenance["prepared_radiative_level_rates_per_s"].values()),
                                   list(bound.system.provenance["radiative_level_rates_per_s"].values()), rtol=2e-14)


def test_clustered_three_electron_pulse_changes_charge_and_rejects_stale_carrier(joint):
    model, pulse = joint
    start, stop = pulse.support_fs
    result = model.solve(pulse, after_pulse_fs=100, times_fs=np.linspace(start, stop + 100, 151))
    assert result.mean_charge[-1] > 7.18
    np.testing.assert_allclose(np.trace(result.density_matrices, axis1=1, axis2=2), 1, atol=2e-14)
    assert min(np.linalg.eigvalsh(result.density_matrices[-1])) > -1e-9
    with pytest.raises(ValueError, match="changed carrier"):
        model.solve(CoherentGaussianPulse(pulse.photon_energy_eV + 1, 1e17, 10))


def test_general_N2_tensor_matches_existing_full_h_plus_V_provider():
    value = build(["1s2", "1s1 2p1", "2s2"], ns=40)
    old = TwoElectronCIChannels(value, continuum_model=continuum())
    new = MultiElectronCIChannels(value, continuum_model=continuum())
    for left, right in ((old.photoionization(1500), new.photoionization(1500)), (old.auger(), new.auger())):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert (a.process, a.initial_electrons, a.initial_level, a.final_level,
                    a.electron_kappa, a.electron_m2) == (b.process, b.initial_electrons,
                    b.initial_level, b.final_level, b.electron_kappa, b.electron_m2)
            assert [label for label, _ in a.components] == [label for label, _ in b.components]
            np.testing.assert_allclose([v for _, v in a.components], [v for _, v in b.components], rtol=2e-12, atol=1e-12)


def test_four_electron_live_cascade_and_preflight_capacity():
    value = build(["1s2 2s2"], Z=8)
    with pytest.raises(ValueError, match="needs 16"):
        MultiElectronCIChannels(value, continuum_model=continuum(), magnetic_state_cap=15)
    with pytest.raises(ValueError, match="integer"):
        MultiElectronCIChannels(value, continuum_model=continuum(), magnetic_state_cap=True)
    provider = MultiElectronCIChannels(value, continuum_model=continuum())
    pulse = CoherentGaussianPulse(1500, 1e18, 10)
    model = MultiElectronCoherentModel(value, pulse, continuum_model=continuum(),
        photon_numbers={n: dict.fromkeys(range(len(r.state_energies)), 0)
                        for n, r in provider.sectors.items()}, electron_reservoir="clustered")
    assert len(model.system.state_ids) == 16
    result = model.solve(pulse)
    assert result.mean_charge[-1] > 4.05



def test_prepared_multielectron_snapshot_reuses_native_channel_caches(joint, monkeypatch):
    original, pulse = joint
    provider = original.channels
    before = (len(provider._waves), len(provider._angular_terms))
    def unexpected_solve(*args, **kwargs):
        pytest.fail("reused immutable snapshot must not solve the same DW again")
    monkeypatch.setattr("atomx.scattering.ci_continuum.solve_dirac_continuum_many", unexpected_solve)
    ladders = {n: {i: int(n == 3 and i != 0) for i in range(len(r.state_energies))}
               for n, r in provider.sectors.items()}
    reused = MultiElectronCoherentModel(provider, pulse, photon_numbers=ladders,
        continuum_model=provider.model, radiative_reservoir="geometric_clustered",
        electron_reservoir="clustered")
    assert reused.channels is provider
    assert (len(provider._waves), len(provider._angular_terms)) == before
    np.testing.assert_array_equal(reused.system.dipole, original.system.dipole)
    for a, b in zip(reused.system.decay_operators, original.system.decay_operators, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(reused.system.pi_operators, original.system.pi_operators, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="same continuum"):
        MultiElectronCoherentModel(provider, pulse, photon_numbers=ladders,
            continuum_model=ContinuumModel.shooting(n_grid=24000, r_max=80, orbital_removal="none"))
