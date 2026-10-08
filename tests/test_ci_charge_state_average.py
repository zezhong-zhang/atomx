"""Charge-ensemble Hellmann-Feynman gradients and public solver contracts."""

import numpy as np
import pytest

from atomx import MCDHF, Atom, Basis, CIChargeStateAverage, CIStateAverage, Correlation
from atomx.methods.csf.charge_variational import ChargeStateOrbitalFunctional
from atomx.methods.csf.variational import CSFOrbitalFunctional


@pytest.mark.parametrize(
    "sectors",
    [
        (),
        ((0, 1, CIStateAverage()),),
        ((2, 0, CIStateAverage()),),
        ((2, np.inf, CIStateAverage()),),
        ((True, 1, CIStateAverage()),),
        ((2, 1, CIStateAverage()), (2, 2, CIStateAverage())),
    ],
)
def test_invalid_charge_objectives_raise(sectors):
    with pytest.raises(ValueError):
        CIChargeStateAverage(sectors)


def test_charge_normalization_and_parent_contract():
    target = CIChargeStateAverage(
        ((1, 1e300, CIStateAverage()), (2, 3e300, CIStateAverage()))
    )
    assert [n for n, _, _ in target.normalized_sectors(2)] == [2, 1]
    np.testing.assert_array_equal(
        [w for _, w, _ in target.normalized_sectors(2)], [0.75, 0.25]
    )
    with pytest.raises(ValueError, match="include the parent"):
        target.normalized_sectors(3)
    with pytest.raises(ValueError, match="exceed the parent"):
        target.normalized_sectors(1)
    with pytest.raises(TypeError, match="requires CIStateAverage"):
        CIChargeStateAverage(((2, 1, "ground"),))
    with pytest.raises(ValueError, match="underflow"):
        CIChargeStateAverage(
            ((2, 1e300, CIStateAverage()), (1, 1e-300, CIStateAverage()))
        ).normalized_sectors(2)


@pytest.fixture(scope="module")
def seed():
    return Atom(
        10,
        config=["1s2", "1s1 2p1", "2s2"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(radial_optimization="fac_frozen_local"),
    ).solve(tol=1e-6)


def functional(seed, target):
    r = seed.result
    return ChargeStateOrbitalFunctional(
        r.csfs, r.bsp_data, 10, r._nuclear_potential, state_average=target
    )


def test_joint_RDM_gradient_equals_independent_sector_sum_and_tangent_difference(seed):
    parent = CIStateAverage(((0, 1, 0, 1), (0, 1, 1, 3), (2, -1, 0, 2)))
    target = CIChargeStateAverage(
        ((2, 7, parent), (1, 3, CIStateAverage(statistical=True)))
    )
    f = functional(seed, target)
    orbitals = seed.result.orbitals_dict
    energy, gradient, trial = f.evaluate(orbitals)
    direct_energy = 0.0
    direct_gradient = {key: np.zeros(2 * f.ns) for key in f.keys}
    for number, weight, levels in target.normalized_sectors(2):
        separate = CSFOrbitalFunctional(
            f.sector_csfs[number], f.data, 10, f.nuclear_potential, state_average=levels
        )
        e, g, ci = separate.evaluate(orbitals)
        direct_energy += weight * e
        for key in g:
            direct_gradient[key] += weight * g[key]
        np.testing.assert_array_equal(
            trial.sectors[number].state_energies, ci.state_energies
        )
    assert energy == pytest.approx(direct_energy, abs=1e-12)
    for key in f.keys:
        np.testing.assert_allclose(
            gradient[key], direct_gradient[key], rtol=0, atol=2e-10
        )
    assert f.energy_for_weights(orbitals, f.last_weights) == pytest.approx(
        energy, abs=1e-9
    )
    assert sum(f.last_occupations.values()) == pytest.approx(1.7, abs=1e-12)
    keys = [key for key in f.keys if key[1] == -1]
    x = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
    metric = f.metric[-1]
    direction = np.random.default_rng(53).normal(size=x.shape)
    multiplier = x.T @ metric @ direction
    direction -= x @ (0.5 * (multiplier + multiplier.T))
    direction /= np.sqrt(np.trace(direction.T @ metric @ direction))
    analytic = sum(gradient[key] @ direction[:, i] for i, key in enumerate(keys))
    values = []
    for step in (1e-6, -1e-6):
        y = x + step * direction
        w, v = np.linalg.eigh(y.T @ metric @ y)
        y = y @ ((v / np.sqrt(w)) @ v.T)
        changed = dict(orbitals)
        for i, key in enumerate(keys):
            changed[key] = (y[: f.ns, i].copy(), y[f.ns :, i].copy())
        values.append(f.evaluate(changed, gradient=False)[0])
    assert (values[0] - values[1]) / 2e-6 == pytest.approx(analytic, rel=3e-6, abs=3e-7)


def test_single_charge_reduces_exactly_and_accepted_trial_reuses_CI(seed, monkeypatch):
    levels = CIStateAverage()
    joint = functional(seed, CIChargeStateAverage(((2, 1, levels),)))
    ordinary = CSFOrbitalFunctional(
        seed.result.csfs, joint.data, 10, joint.nuclear_potential, state_average=levels
    )
    e, g, _ = ordinary.evaluate(seed.result.orbitals_dict)
    energy, gradient, trial = joint.evaluate(seed.result.orbitals_dict)
    assert energy == e
    for key in g:
        np.testing.assert_array_equal(g[key], gradient[key])

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted charge trial re-diagonalized CI")

    monkeypatch.setattr("atomx.methods.ci.ci_solve", forbidden)
    energy, gradient, _ = joint.evaluate(seed.result.orbitals_dict, ci_result=trial)
    assert energy == e
    changed = {
        key: tuple(v.copy() for v in waves)
        for key, waves in seed.result.orbitals_dict.items()
    }
    key = next(iter(changed))
    changed[key][0][0] += 1e-9
    with pytest.raises(ValueError, match="orbital snapshot"):
        joint.evaluate(changed, ci_result=trial)
    other = functional(seed, CIChargeStateAverage(((2, 1, levels),)))
    with pytest.raises(ValueError, match="orbital snapshot"):
        other.evaluate(seed.result.orbitals_dict, ci_result=trial)


def test_cap_and_unsupported_paths_reject(seed, monkeypatch):
    target = CIChargeStateAverage(
        ((2, 1, CIStateAverage()), (1, 1, CIStateAverage())), magnetic_state_cap=1
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("enumerated a charge space beyond its cap")

    monkeypatch.setattr(
        "atomx.methods.csf.charge_variational.enumerate_jj_orbital_space", forbidden
    )
    with pytest.raises(ValueError, match="magnetic states"):
        functional(seed, target)
    from atomx.methods.mcdhf.solver import MCDHFSolver

    target = CIChargeStateAverage(((2, 1, CIStateAverage()), (1, 1, CIStateAverage())))
    with pytest.raises(ValueError, match="corrections=none"):
        MCDHFSolver(10, seed.result.csfs, state_average=target, include_breit=True)
    with pytest.raises(ValueError, match="shared B-spline"):
        MCDHFSolver(10, seed.result.csfs, state_average=target, engine="shooting")


def test_daughter_occupied_virtual_has_joint_lagrange_multiplier():
    target = CIChargeStateAverage(
        ((3, 1, CIStateAverage(((1, 1, 0, 1),))), (1, 1, CIStateAverage()))
    )
    atom = Atom(
        10,
        config=["1s2 2s1", "1s1 2s1 2p1"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(state_average=target),
    ).solve(tol=1e-6, max_iter=100)
    assert atom.converged
    from atomx.config.subshell import Subshell

    f = functional(atom, target)
    _, _, _ = f.evaluate(atom.result.orbitals_dict)
    for key in ((2, -2), (2, 1)):
        assert atom.result.occupations[key] == 0
        assert f.last_occupations[key] > 0
        vector = np.concatenate(atom.result.orbitals_dict[key])
        expected = vector @ f.last_gradient[key] / (2 * f.last_occupations[key])
        label = Subshell(n=key[0], kappa=key[1], occupation=0).label
        assert atom.result.orbital_energies[label] == pytest.approx(expected, abs=1e-10)


def test_public_joint_solve_keeps_parent_density_and_rebuilds_ionic_Hamiltonian():
    target = CIChargeStateAverage(tuple((n, 1, CIStateAverage()) for n in (3, 2, 1)))
    atom = Atom(
        10,
        config=["1s2 2s1", "1s1 2s1 2p1"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(state_average=target),
    ).solve(tol=1e-6, max_iter=80)
    assert atom.converged
    diag = atom.result.orbital_optimization
    assert diag["objective_mean_electrons"] == pytest.approx(2)
    assert sum(diag["ensemble_occupations"].values()) == pytest.approx(2, abs=1e-10)
    assert sum(atom.result.occupations.values()) == pytest.approx(3, abs=1e-10)
    assert np.max(np.diff(diag["energy_history"])) < 1e-10
    spec = atom.result.calculation_spec.to_dict()["state"]["ci_state_average"]
    assert "sectors" in spec
    from atomx.scattering.models import ContinuumModel
    from atomx.scattering.multielectron_ci import MultiElectronCIChannels

    provider = MultiElectronCIChannels(
        atom,
        continuum_model=ContinuumModel.shooting(
            n_grid=1000, r_max=80, orbital_removal="none"
        ),
    )
    for number, energies in diag["charge_sector_energies_hartree"].items():
        np.testing.assert_allclose(
            provider.sectors[number].state_energies, energies, rtol=0, atol=1e-10
        )
    assert "jointly optimized" in provider.ion.solver_diagnostics["_radial_optimization"]
    # Direct public Method route exercises the same protocol independently.
    method = MCDHF(
        state_average=CIChargeStateAverage(
            ((2, 1, CIStateAverage()), (1, 1, CIStateAverage()))
        )
    )
    result = method.solve(10, ["1s2", "1s1 2p1"], ns=40, tol=1e-6, max_iter=80)
    assert result.converged
    assert result.orbital_optimization["objective_mean_electrons"] == pytest.approx(1.5)
