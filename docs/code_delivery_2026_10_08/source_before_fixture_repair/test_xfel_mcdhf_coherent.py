"""Real two-electron CI phases, angular anchors and explicit model gates."""

import copy

import numpy as np
import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j

from atomx import Atom, Basis
from atomx.constants import ALPHA, AU_RATE_PER_S, HARTREE_EV
from atomx.xfel import CoherentGaussianPulse, MCDHFCoherentModel
from atomx.xfel.coherent import magnetic_e1_component


@pytest.fixture(scope="module")
def neon():
    atom = Atom(Z=10, config=["1s2", "1s1 2p1"], basis=Basis(ns=40, ks=9))
    atom.solve(max_iter=60, tol=1e-6)
    assert atom.converged
    return atom


def model(atom, **kwargs):
    return MCDHFCoherentModel(
        atom,
        photon_numbers={
            i: int(i != 0) for i in range(len(atom._csf_result.state_energies))
        },
        bound_only=True,
        **kwargs,
    )


def test_ci_signed_amplitude_and_integer_J_components_match_independent_anchors(neon):
    ci = model(neon)
    assert len(ci.system.state_ids) == 13
    rates = sum(c.conj().T @ c for c in ci.system.decay_operators)
    for line in ci.lines:
        rme = line.reduced_matrix_element_au
        assert rme**2 == pytest.approx(line.line_strength, rel=1e-14)
        ju, jl = line.final_two_J, line.initial_two_J
        expected_A = (
            4
            / 3
            * ALPHA**3
            * line.energy_diff_hartree**3
            * rme**2
            / (ju + 1)
            * AU_RATE_PER_S
        )
        for mu, state in ci.by_level[line.final_state_index].items():
            assert rates[ci.indices[state], ci.indices[state]] == pytest.approx(
                expected_A, rel=1e-14
            )
            for ml in range(-jl, jl + 1, 2):
                for q in (-1, 0, 1):
                    value = magnetic_e1_component(ju, mu, jl, ml, q, rme)
                    reference = (
                        float(
                            (-1) ** ((ju - mu) // 2)
                            * wigner_3j(
                                Rational(ju, 2),
                                1,
                                Rational(jl, 2),
                                -Rational(mu, 2),
                                q,
                                Rational(ml, 2),
                            )
                        )
                        * rme
                    )
                    assert value == pytest.approx(reference, abs=1e-15)
                    reverse = (-1.0) ** ((jl - ju) // 2) * rme
                    hermitian = (-1.0) ** q * magnetic_e1_component(
                        jl, ml, ju, mu, -q, reverse
                    )
                    assert value == pytest.approx(hermitian, abs=1e-15)


def test_ci_column_phase_flip_preserves_dynamics_by_unitary_covariance(neon):
    original = model(neon)
    changed = copy.copy(neon)
    changed._result = copy.copy(neon._csf_result)
    changed._csf_result.mixing_coefficients = (
        neon._csf_result.mixing_coefficients.copy()
    )
    changed._csf_result.mixing_coefficients[:, 4] *= -1
    flipped = model(changed)
    signs = np.array(
        [
            -1 if original.level_by_state[s] == 4 else 1
            for s in original.system.state_ids
        ]
    )
    np.testing.assert_allclose(
        flipped.system.dipole,
        signs[:, None] * original.system.dipole * signs[None, :],
        atol=1e-15,
    )
    for a, b in zip(
        original.system.decay_operators, flipped.system.decay_operators, strict=True
    ):
        np.testing.assert_allclose(b, signs[:, None] * a * signs[None, :], atol=1e-10)
    pulse = CoherentGaussianPulse(original.lines[0].energy_eV, 1e17, 10)
    a, b = original.solve(pulse), flipped.solve(pulse)
    np.testing.assert_allclose(
        b.density_matrices,
        signs[None, :, None] * a.density_matrices * signs[None, None, :],
        atol=1e-12,
    )
    assert (
        np.max(abs(a.density_matrices[:, 0, original.indices[original.by_level[4][0]]]))
        > 0.1
    )
    with pytest.raises(ValueError, match="different CI"):
        original.level_populations(b)


def test_real_ci_zero_flux_matches_exact_m_resolved_spontaneous_decay(neon):
    ci = model(neon, initial_state=4)
    pulse = CoherentGaussianPulse(ci.lines[0].energy_eV, 0, 10)
    result = ci.solve(pulse)
    elapsed_s = (result.times_fs - result.times_fs[0]) * 1e-15
    expected = np.exp(-ci.lines[0].A_rate_per_s * elapsed_s)
    np.testing.assert_allclose(ci.level_populations(result)[4], expected, atol=1e-12)
    for state in ci.by_level[4].values():
        np.testing.assert_allclose(
            result.populations[:, ci.indices[state]], expected / 3, atol=1e-12
        )


def test_symmetry_selected_excited_and_mixed_initial_states_share_the_same_operators(neon):
    from atomx import CILevel, CILevelPopulation, CIStateAverage

    excited = CILevel(2, -1, 1)
    assert excited.resolve(neon.result.csfs, neon.result) == 4
    population = CILevelPopulation(((CILevel(0, 1), 1), (excited, 3)))
    ci = model(neon, initial_state=population)
    legacy = model(neon)
    selected = model(neon, initial_state=excited)
    by_column = model(neon, initial_state=4)
    np.testing.assert_array_equal(selected.initial_density, by_column.initial_density)
    np.testing.assert_array_equal(ci.system.dipole, legacy.system.dipole)
    assert ci.system.provenance["basis_sha256"] == legacy.system.provenance["basis_sha256"]
    assert ci.system.provenance["default_initial_level_populations"] == ((0, 0.25), (4, 0.75))
    result = ci.solve(CoherentGaussianPulse(ci.lines[0].energy_eV, 0, 10))
    elapsed_s = (result.times_fs - result.times_fs[0]) * 1e-15
    upper = 0.75 * np.exp(-ci.lines[0].A_rate_per_s * elapsed_s)
    np.testing.assert_allclose(ci.level_populations(result)[4], upper, atol=1e-12)
    np.testing.assert_allclose(ci.level_populations(result)[0], 1 - upper, atol=1e-12)
    for state in ci.by_level[4].values():
        np.testing.assert_allclose(result.populations[:, ci.indices[state]], upper / 3, atol=1e-12)
    with pytest.raises(TypeError, match="initial_state"):
        model(neon, initial_state=CIStateAverage())


def test_average_rates_nonorthogonal_and_unconverged_states_cannot_enter_ci_model(neon):
    with pytest.raises(ValueError, match="bound_only"):
        MCDHFCoherentModel(neon, photon_numbers={})
    with pytest.raises(ValueError, match="every CI level"):
        MCDHFCoherentModel(neon, photon_numbers={0: 0}, bound_only=True)
    for modification, message in (
        (dict(converged=False), "converged"),
        (
            dict(per_block_orbitals={(0, 1): neon._csf_result.orbitals_dict}),
            "shared orthonormal",
        ),
    ):
        changed = copy.copy(neon)
        changed._result = copy.copy(neon._csf_result)
        changed._csf_result.__dict__.update(modification)
        with pytest.raises(ValueError, match=message):
            model(changed)
    changed = copy.copy(neon)
    changed._result = copy.copy(neon._csf_result)
    changed._csf_result.orbitals_dict = dict(neon._csf_result.orbitals_dict)
    key = next(iter(changed._csf_result.orbitals_dict))
    changed._csf_result.orbitals_dict[key] = tuple(
        2 * v for v in changed._csf_result.orbitals_dict[key]
    )
    with pytest.raises(ValueError, match="orthonormal"):
        model(changed)


def test_missing_signed_amplitude_is_not_reconstructed_from_A(neon):
    changed = copy.copy(neon)
    lines = copy.deepcopy(neon.transitions(multipoles=["E1"]))
    lines[0].reduced_matrix_element_au = None
    changed.transitions = lambda **kwargs: lines
    with pytest.raises(ValueError, match="signed"):
        model(changed)


def test_exactly_degenerate_ci_lines_share_a_bath_and_preserve_a_dark_state(neon):
    changed = copy.copy(neon)
    changed._result = copy.copy(neon._csf_result)
    changed._result.state_energies = neon._csf_result.state_energies.copy()
    changed._result.state_energies[1] = changed._result.state_energies[4]
    ci = model(changed)
    assert len(ci.system.decay_operators) == 3
    rme = {line.final_state_index: line.reduced_matrix_element_au for line in ci.lines}
    dark = np.zeros(len(ci.system.state_ids))
    dark[ci.indices[ci.by_level[1][0]]] = rme[4]
    dark[ci.indices[ci.by_level[4][0]]] = -rme[1]
    dark /= np.linalg.norm(dark)
    rho = np.outer(dark, dark)
    result = ci.solve(
        CoherentGaussianPulse(ci.lines[0].energy_eV, 1e17, 10), initial_density=rho
    )
    np.testing.assert_allclose(
        result.density_matrices,
        np.broadcast_to(rho, result.density_matrices.shape),
        atol=1e-12,
    )


def test_nearly_degenerate_ci_lines_require_a_nonsecular_model(neon):
    changed = copy.copy(neon)
    changed._result = copy.copy(neon._csf_result)
    changed._result.state_energies = neon._csf_result.state_energies.copy()
    changed._result.state_energies[1] = (
        changed._result.state_energies[4] - 0.001 / HARTREE_EV
    )
    with pytest.raises(ValueError, match="nonsecular"):
        model(changed)


def test_degenerate_bath_requires_consistent_rotating_frame_ladders(neon):
    changed = copy.copy(neon)
    changed._result = copy.copy(neon._csf_result)
    changed._result.state_energies = neon._csf_result.state_energies.copy()
    changed._result.state_energies[1] = changed._result.state_energies[4]
    ladders = {i: int(i != 0 and i != 1) for i in range(5)}
    with pytest.raises(ValueError, match="photon-ladder difference"):
        MCDHFCoherentModel(changed, photon_numbers=ladders, bound_only=True)


@pytest.mark.parametrize("reservoir", ["clustered", "geometric_clustered"])
def test_opt_in_near_degenerate_reservoir_retains_signed_dark_combination(neon, reservoir):
    changed = copy.copy(neon)
    changed._result = copy.copy(neon.result)
    changed._result.state_energies = neon.result.state_energies.copy()
    changed._result.state_energies[1] = changed._result.state_energies[4] - 0.001 / HARTREE_EV
    ci = model(changed, radiative_reservoir=reservoir)
    assert len(ci.system.decay_operators) == 3
    indices = [ci.indices[ci.by_level[i][0]] for i in (1, 4)]
    stacked = np.concatenate([op[:, indices] for op in ci.system.decay_operators])
    _, singular, vh = np.linalg.svd(stacked, full_matrices=False)
    assert singular[-1] < 1e-9
    assert abs(vh[-1][0]) > 1e-3 and abs(vh[-1][1]) > 1e-3
    np.testing.assert_allclose(stacked @ vh[-1].conj(), 0, atol=1e-9)
    rates = sum(op.conj().T @ op for op in ci.system.decay_operators)
    assert abs(rates[indices[0], indices[1]]) > 1e6
    if reservoir == "geometric_clustered":
        np.testing.assert_allclose(
            list(ci.system.provenance["prepared_radiative_level_rates_per_s"].values()),
            list(ci.system.provenance["radiative_level_rates_per_s"].values()), rtol=1e-14)
    # The dissipative dark combination does not imply a stationary state:
    # its two Hamiltonian frequencies remain distinct.
    assert ci.system.energies[indices[0]] != ci.system.energies[indices[1]]
    records = ci.system.provenance["radiative_clusters"]
    assert any(len(r["frequencies_eV"]) == 2 for r in records)



def test_cluster_modes_reject_a_vacuum_spectrum_with_large_variation(neon):
    changed = copy.copy(neon)
    changed._result = copy.copy(neon.result)
    changed._result.state_energies = neon.result.state_energies.copy()
    changed._result.state_energies[1] = changed._result.state_energies[4] - 100 / HARTREE_EV
    # Deliberately broaden both lines before operator preparation. The gate
    # must reject their large spectral variation before any rate-closure use.
    lines = copy.deepcopy(changed.transitions(multipoles=["E1"]))
    for line in lines:
        line.A_rate_per_s = 1e18
    changed.transitions = lambda **kwargs: lines
    for reservoir in ("clustered", "geometric_clustered"):
        with pytest.raises(ValueError, match="vacuum spectral approximation"):
            model(changed, radiative_reservoir=reservoir)
