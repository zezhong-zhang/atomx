"""Independent ensemble, symmetry-order and analytic-gradient checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from atomx import MCDHF, Atom, Basis, CIStateAverage, Correlation
from atomx.methods.csf.variational import CSFOrbitalFunctional


def spectrum(energies, sectors, mixing=None):
    csfs = [SimpleNamespace(total_J2=j, parity=p) for j, p in sectors]
    result = SimpleNamespace(
        state_energies=np.array(energies, dtype=float),
        mixing_coefficients=np.eye(len(csfs)) if mixing is None else mixing,
    )
    return csfs, result


def test_local_root_selector_survives_cross_symmetry_energy_reordering():
    csfs, result = spectrum([-3, -2, -1], [(0, 1), (2, -1), (0, 1)])
    target = CIStateAverage(((0, 1, 1, 1), (2, -1, 0, 3)))
    np.testing.assert_array_equal(target.weights(csfs, result), [0, 0.75, 0.25])
    result.state_energies = result.state_energies[[0, 2, 1]]
    result.mixing_coefficients = result.mixing_coefficients[:, [0, 2, 1]]
    np.testing.assert_array_equal(target.weights(csfs, result), [0, 0.25, 0.75])
    statistical = CIStateAverage(statistical=True)
    np.testing.assert_allclose(statistical.weights(csfs, result), [0.2, 0.2, 0.6])


def test_exact_degeneracy_has_basis_independent_weights_or_rejects():
    csfs, result = spectrum([-3, -3], [(0, 1), (0, 1)])
    theta = 0.6
    result.mixing_coefficients = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    np.testing.assert_array_equal(CIStateAverage().weights(csfs, result), [0.5, 0.5])
    with pytest.raises(ValueError, match="degeneracy"):
        CIStateAverage(((0, 1, 0, 1),)).weights(csfs, result)


def test_wrong_root_or_mixed_symmetry_cannot_silently_select_another_level():
    csfs, result = spectrum([-3, -2], [(0, 1), (2, -1)])
    with pytest.raises(ValueError, match="outside the space"):
        CIStateAverage(((0, 1, 1, 1),)).weights(csfs, result)
    result.mixing_coefficients = np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    with pytest.raises(ValueError, match="symmetry-pure"):
        CIStateAverage().weights(csfs, result)


@pytest.mark.parametrize(
    "states",
    [
        (),
        ((0, 1, 0, 0),),
        ((0, 1, 0, np.nan),),
        ((0, 0, 0, 1),),
        ((0, 1, -1, 1),),
        ((0, 1, 0, 1), (0, 1, 0, 2)),
    ],
)
def test_invalid_objectives_raise(states):
    with pytest.raises(ValueError):
        CIStateAverage(states)


@pytest.fixture(scope="module")
def seed():
    return Atom(
        10,
        config=["1s2", "1s1 2p1", "2s2"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(radial_optimization="fac_frozen_local"),
    ).solve(tol=1e-6)


def test_weighted_CI_gradient_matches_rediagonalized_energy_difference(seed):
    result = seed.result
    # Unequal weights on the two even J=0 roots retain coherent CI interference.
    target = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3), (2, -1, 0, 2)))
    functional = CSFOrbitalFunctional(
        result.csfs,
        result.bsp_data,
        10,
        result._nuclear_potential,
        state_average=target,
    )
    energy, gradient, ci = functional.evaluate(result.orbitals_dict)
    weights = target.weights(result.csfs, ci)
    assert energy == pytest.approx(weights @ ci.state_energies, abs=1e-12)
    assert functional.energy_for_weights(
        result.orbitals_dict, functional.last_weights
    ) == pytest.approx(energy, abs=1e-10)
    key = (2, -1)
    keys = [item for item in functional.keys if item[1] == key[1]]
    x = np.column_stack([np.concatenate(result.orbitals_dict[item]) for item in keys])
    metric = functional.metric[key[1]]
    direction = np.random.default_rng(12).normal(size=x.shape)
    cross = x.T @ metric @ direction
    direction -= x @ (0.5 * (cross + cross.T))
    direction /= np.sqrt(np.trace(direction.T @ metric @ direction))
    analytical = sum(
        gradient[item] @ direction[:, column] for column, item in enumerate(keys)
    )
    energies = []
    for step in (1e-6, -1e-6):
        orbitals = dict(result.orbitals_dict)
        y = x + step * direction
        eigenvalues, eigenvectors = np.linalg.eigh(y.T @ metric @ y)
        y = y @ ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T)
        for column, item in enumerate(keys):
            orbitals[item] = (
                y[: functional.ns, column].copy(),
                y[functional.ns :, column].copy(),
            )
        energies.append(functional.evaluate(orbitals, gradient=False)[0])
    derivative = (energies[0] - energies[1]) / 2e-6
    assert derivative == pytest.approx(analytical, rel=3e-6, abs=2e-7)
    # Independently average scalar energies of each resolved CI column.
    from atomx.physics.csf.radial_density import CIRadialDensity

    density = CIRadialDensity(result.csfs)
    scalar = 0.0
    for column, weight in enumerate(weights):
        if weight:
            one = np.eye(len(weights))[column]
            scalar += weight * functional.energy_for_weights(
                result.orbitals_dict, density.contract(ci, one)
            )
    assert scalar == pytest.approx(energy, abs=1e-10)


def test_public_ensemble_solve_retains_ground_density_and_records_objective():
    atom = Atom(
        10,
        config=["1s2", "1s1 2p1", "2s2"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(state_average=CIStateAverage()),
    )
    atom.solve(tol=1e-6, max_iter=60)
    result = atom.result
    assert result.converged
    diag = result.solver_diagnostics["_variational_diagnostics"]
    assert "fixed CI level" in diag["target"]
    np.testing.assert_allclose(diag["state_weights"], np.ones(6) / 6)
    assert sum(result.occupations.values()) == pytest.approx(2)
    assert sum(diag["ensemble_occupations"].values()) == pytest.approx(2)
    assert diag["ensemble_occupations"][2, -1] > 0.3
    assert result.occupations[2, -1] < 0.01
    assert result.calculation_spec.to_dict()["state"]["ci_state_average"] == {
        "states": None,
        "statistical": False,
    }
    history = diag["energy_history"]
    assert np.max(np.diff(history)) < 1e-10
    # 2s must now be a bound residual orbital, not the positive-energy
    # correlation orbital produced by ground-only optimization of this space.
    from atomx.scattering.ci_continuum import TwoElectronCIChannels
    from atomx.scattering.models import ContinuumModel

    provider = TwoElectronCIChannels(
        atom,
        continuum_model=ContinuumModel.shooting(
            n_grid=16000, r_max=80, orbital_removal="none"
        ),
    )
    assert np.max(provider.ion.state_energies) < 0
    assert provider.auger()


def test_unsupported_ensemble_paths_raise_before_solve():
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.mcdhf.solver import MCDHFSolver

    with pytest.raises(ValueError, match="shared B-spline"):
        MCDHFSolver(
            10,
            enumerate_jj_csfs("1s2"),
            engine="shooting",
            state_average=CIStateAverage(),
        )
    with pytest.raises(ValueError, match="shared MCDHF"):
        Correlation(state_average=CIStateAverage(), radial_optimization="fac_frozen")
    with pytest.raises(TypeError, match="CIStateAverage"):
        Correlation(state_average="equal")
    result = MCDHF(state_average=CIStateAverage()).solve(
        10, ["1s2", "1s1 2p1"], ns=40, tol=1e-6, max_iter=60
    )
    assert result.converged
    np.testing.assert_allclose(
        result.solver_diagnostics["_variational_diagnostics"]["state_weights"], np.ones(5) / 5
    )


def test_three_electron_relaxation_and_line_specific_reservoir_widths():
    atom = Atom(
        10,
        config=["1s2 2s1", "1s1 2s1 2p1"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(state_average=CIStateAverage()),
    ).solve(tol=1e-6, max_iter=60)
    result = atom.result
    assert result.converged
    functional = CSFOrbitalFunctional(
        result.csfs,
        result.bsp_data,
        10,
        result._nuclear_potential,
        state_average=CIStateAverage(),
    )
    energy, _, ci = functional.evaluate(result.orbitals_dict)
    from atomx.physics.csf.matrix import build_ci_matrix

    matrix = build_ci_matrix(
        result.csfs,
        result.orbitals_dict,
        functional.keys,
        result.bsp_data,
        10,
        result.bsp_data["ns"],
        nuclear_potential=result._nuclear_potential,
    )
    assert energy == pytest.approx(np.trace(matrix) / len(matrix), abs=1e-10)
    assert sum(functional.last_occupations.values()) == pytest.approx(3)
    assert energy == pytest.approx(np.mean(ci.state_energies), abs=1e-10)
    # Independent per-line widths admit the narrow upper pair; a broad
    # unrelated lower-frequency line must not determine its resolution.
    from atomx.xfel import MCDHFCoherentModel

    model = MCDHFCoherentModel(
        atom,
        photon_numbers={i: int(i != 0) for i in range(len(ci.state_energies))},
        bound_only=True,
    )
    assert len(model.system.state_ids) == 26
    from atomx.constants import AU_RATE_PER_S, HARTREE_EV

    widths = np.zeros(len(ci.state_energies))
    for line in model.lines:
        widths[line.final_state_index] += line.A_rate_per_s * HARTREE_EV / AU_RATE_PER_S
    lines = sorted(model.lines, key=lambda line: line.energy_eV)
    lower, upper = lines[-2:]
    difference = upper.energy_eV - lower.energy_eV
    assert difference < 10 * widths.max()  # historical false rejection
    assert difference > 10 * max(
        widths[lower.final_state_index], widths[upper.final_state_index]
    )
    # Actually unresolved lines continue to require a nonsecular model.
    import copy

    changed = copy.copy(atom)
    changed._result = copy.copy(result)
    changed._result.state_energies = result.state_energies.copy()
    changed._result.state_energies[upper.final_state_index] = (
        result.state_energies[lower.final_state_index] + 0.0005 / HARTREE_EV
    )
    with pytest.raises(ValueError, match="nonsecular reservoir"):
        MCDHFCoherentModel(
            changed,
            photon_numbers={i: int(i != 0) for i in range(len(ci.state_energies))},
            bound_only=True,
        )


def test_optical_line_width_includes_decay_of_both_levels():
    from atomx.constants import AU_RATE_PER_S, HARTREE_EV
    from atomx.xfel import MCDHFCoherentModel

    atom = Atom(
        2,
        config=["1s1", "2p1", "3s1"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(state_average=CIStateAverage()),
    ).solve(tol=1e-6, max_iter=60)
    model = MCDHFCoherentModel(
        atom, photon_numbers={0: 0, 1: 1, 2: 1, 3: 2}, bound_only=True
    )
    # The 3s -> 2p line has a decaying lower level. Read the optical
    # coherence damping from the actual magnetic Lindblad operators,
    # independently of the line-width construction in the driver.
    line = min(model.lines, key=lambda l: l.energy_eV)
    lower, upper = line.initial_state_index, line.final_state_index
    i = model.indices[next(iter(model.by_level[upper].values()))]
    j = model.indices[next(iter(model.by_level[lower].values()))]
    coherence = np.zeros_like(model.initial_density)
    coherence[i, j] = 1
    derivative = np.zeros_like(coherence)
    for jump in model.system.decay_operators:
        loss = jump.conj().T @ jump
        derivative += jump @ coherence @ jump.conj().T - 0.5 * (
            loss @ coherence + coherence @ loss
        )
    energy, fwhm = min(model.system.provenance["emission_group_fwhm_eV"])
    assert energy == pytest.approx(line.energy_eV)
    assert fwhm == pytest.approx(
        -2 * derivative[i, j].real * HARTREE_EV / AU_RATE_PER_S, rel=1e-12
    )
    widths = model.system.provenance["radiative_level_rates_per_s"]
    assert widths[lower] > widths[upper] > 0
