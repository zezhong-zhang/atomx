"""A frozen local continuum must use the bound Hamiltonian's field."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, SCFResult
from atomx.potentials.frozen_field import FrozenLocalField
from atomx.scattering.models import ContinuumModel, PotentialModel
from atomx.scattering.photoionization import subshell_photoionization
from atomx.scattering.radiative_recombination import _rr_potential_matrices


@pytest.fixture(scope="module")
def neutral_gu():
    solver = DFSolver.from_configuration(
        10,
        "1s2 2s2 2p6",
        exchange="gu",
        ns=180,
        ks=9,
        r_max=80,
        starting_potential="tfda",
    )
    result = solver.solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    return solver, result


def test_continuum_matches_bound_gu_hamiltonian(neutral_gu):
    solver, result = neutral_gu
    # Compare the continuum's assembled electronic field to the bound
    # solver's independent direct/exchange matrix builders, not a second
    # call to the snapshot constructor.
    direct = solver._build_direct_potential(result.orbitals_raw)
    exchange = solver._build_exchange_matrix_slater(result.orbitals_raw)
    model = ContinuumModel.fac_frozen_core()
    continuum = _rr_potential_matrices(result, result.subshells[0], model)
    for actual, vh, vx in zip(continuum, direct, exchange, strict=True):
        np.testing.assert_allclose(actual, vh + vx, rtol=3e-11, atol=3e-12)


def test_frozen_field_never_refits_sic_or_retables_density(neutral_gu, monkeypatch):
    _, result = neutral_gu

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen field must not rebuild density or refit SIC")

    monkeypatch.setattr(
        "atomx.potentials.static_field.static_hartree_potential", forbidden
    )
    monkeypatch.setattr(
        "atomx.numerics.bspline.local_potential._fac_csi_screening_correction",
        forbidden,
    )
    model = ContinuumModel.fac_frozen_core().potential
    radius = np.geomspace(1e-9, 160, 16000)
    a = replace(model, n_radial=16).build_callable(result)
    b = replace(model, n_radial=128000).build_callable(result)
    np.testing.assert_array_equal(a(radius), b(radius))
    assert a.origin == "bound_scf_snapshot"
    assert a.residual_charge == 1
    np.testing.assert_allclose(
        a(np.array([80.0, 160.0])), [-1 / 80, -1 / 160], rtol=1e-13
    )


@pytest.mark.parametrize("configuration, charge", [("1s2 2s2 2p5", 2), ("1s1", 10)])
def test_ionic_and_one_electron_tail(configuration, charge):
    result = DFSolver.from_configuration(
        10,
        configuration,
        exchange="gu",
        ns=100,
        ks=9,
        r_max=40,
        starting_potential="tfda",
    ).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    potential = PotentialModel.frozen_scf().build_callable(result)
    assert potential.residual_charge == charge
    radius = np.array([80.0, 160.0])
    np.testing.assert_allclose(radius * potential(radius), -charge, atol=1e-13)


def test_frozen_field_retains_finite_nucleus():
    result = DFSolver.from_configuration(
        10,
        "1s2",
        exchange="gu",
        ns=100,
        ks=9,
        r_max=40,
        nuclear="fermi",
        starting_potential="tfda",
    ).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    potential = PotentialModel.frozen_scf().build_callable(result)
    radius = np.array([1e-12, 2e-12])
    # A finite nucleus and electronic field stay finite at the origin.
    np.testing.assert_allclose(potential(radius)[0], potential(radius)[1], rtol=1e-7)
    assert abs(potential(radius)[0]) < 1e9
    snapshot = result._local_central_field
    radii = snapshot.radius
    expected = (
        np.vectorize(result._nuclear_potential)(radii)
        + snapshot.screening_charge / radii
    )
    np.testing.assert_allclose(potential(radii), expected, rtol=1e-14)


@pytest.mark.parametrize("mutation", ["occupation", "Z", "converged", "missing"])
def test_stale_or_missing_snapshot_is_rejected(neutral_gu, mutation):
    _, original = neutral_gu
    result = copy.copy(original)
    if mutation == "occupation":
        result.subshells = [copy.copy(sh) for sh in result.subshells]
        result.subshells[0].occupation -= 1
    elif mutation == "Z":
        result.Z += 1
    elif mutation == "converged":
        result.converged = False
    else:
        result._local_central_field = None
    with pytest.raises(ValueError):
        ContinuumModel.fac_frozen_core().potential.build_callable(result)


def test_frozen_field_does_not_remove_another_electron(neutral_gu):
    _, result = neutral_gu
    with pytest.raises(ValueError, match="cannot remove"):
        PotentialModel.frozen_scf().build_callable(result, exclude_label="1s1/2")


def test_hf_gu_hybrid_stays_explicit():
    result = DFSolver.from_configuration(2, "1s2", ns=80, ks=9).solve(tol=1e-9)
    assert result.converged and result._local_central_field is None
    with pytest.raises(ValueError, match="no saved local SCF field"):
        PotentialModel.frozen_scf().build_callable(result)
    hybrid = ContinuumModel.fac_frozen_core().potential.build_callable(result)
    assert getattr(hybrid, "origin", None) != "bound_scf_snapshot"
    assert hybrid.residual_charge == 1


def test_snapshot_storage_copies_radial_tables():
    r = np.array([0.1, 1.0])
    q = np.array([0.01, 1.0])
    snapshot = FrozenLocalField(r, q, 2, 2, 1)
    r[:] = 3
    q[:] = 0
    np.testing.assert_array_equal(snapshot.radius, [0.1, 1.0])
    assert not snapshot.radius.flags.writeable
    assert not snapshot.screening_charge.flags.writeable


@pytest.mark.parametrize("nuclear", ["point", "fermi"])
def test_frozen_field_checkpoint_roundtrip(nuclear, neutral_gu, tmp_path):
    _, result = neutral_gu
    if nuclear == "fermi":
        result = DFSolver.from_configuration(
            10,
            "1s2",
            exchange="gu",
            ns=100,
            ks=9,
            r_max=40,
            nuclear="fermi",
            starting_potential="tfda",
        ).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    checkpoint = tmp_path / "gu.h5"
    result.save(checkpoint)
    loaded = SCFResult.load(checkpoint)
    model = PotentialModel.frozen_scf()
    radius = np.geomspace(1e-12, 160, 4000)
    np.testing.assert_array_equal(
        model.build_callable(result)(radius), model.build_callable(loaded)(radius)
    )
    np.testing.assert_array_equal(
        loaded._local_central_field.screening_charge,
        result._local_central_field.screening_charge,
    )
    pi_model = ContinuumModel.fac_frozen_core(n_grid=4000, r_max=40)
    before = subshell_photoionization(
        result, [2000], orbitals=[result.subshells[0].label], continuum_model=pi_model
    )
    after = subshell_photoionization(
        loaded, [2000], orbitals=[loaded.subshells[0].label], continuum_model=pi_model
    )
    np.testing.assert_array_equal(before.sigma_total_a0_sq, after.sigma_total_a0_sq)
    assert set(after.provenance["continuum_potential_origins"].values()) == {
        "bound_scf_snapshot"
    }


def test_custom_gu_parameters_require_explicit_snapshot_model():
    result = DFSolver.from_configuration(
        10,
        "1s2",
        exchange="gu",
        ns=100,
        ks=9,
        r_max=40,
        fac_hxs=0.8,
        starting_potential="tfda",
    ).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    with pytest.raises(ValueError, match="parameters differ"):
        ContinuumModel.fac_frozen_core().potential.build_callable(result)
    potential = PotentialModel.frozen_scf().build_callable(result)
    assert potential.origin == "bound_scf_snapshot"
