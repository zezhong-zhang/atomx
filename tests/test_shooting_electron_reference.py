"""Matched electron projectors must recover the same occupied HF problem."""

import hashlib

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from atomx.config.csf import enumerate_jj_csfs_many
from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.csf.grid_functional import (
    GridCSFFunctional,
    occupied_1s2_reference_potential,
)
from atomx.numerics.shooting.electron_space import (
    bound_electron_space,
    continuum_enriched_electron_space,
)
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
from atomx.numerics.shooting.orbital_optimization import optimize_grid_shooting
from atomx.physics.csf.matrix import GridCIWorkspace


def compare_reference(z, *, grid=4000, continuum=160, nuclear="fermi"):
    shells, _ = generate_jj_average("1s2", averaging="statistical")
    hf = ShootingHFSolver(z, shells, n_grid=4000, nuclear=nuclear)
    ref = hf.solve(max_iter=200, tol=1e-13)
    if not ref.converged:
        raise RuntimeError("HF reference did not converge")
    csfs = enumerate_jj_csfs_many(["1s2"])
    original = GridCSFFunctional(
        csfs,
        GridCIWorkspace(
            ref._grid, ref._difr, hf._nuclear_rv(ref._grid, ref._difr, len(ref._grid))
        ),
        z,
    )
    original_energy, _ = original.evaluate(
        {(1, -1): np.array([ref._pa[0], ref._qa[0]])}, gradient=False
    )
    radius = 120.0 / z
    r, dr, n = sgrid(radius, 1e-7, radius / 2400, grid)
    rv = hf._nuclear_rv(r, dr, n)
    f = GridCSFFunctional(csfs, GridCIWorkspace(r, dr, rv), z)
    phi = np.array([CubicSpline(ref._grid, x)(r) for x in (ref._pa[0], ref._qa[0])])
    phi /= np.sqrt(np.sum(phi * phi * f.weights))
    interpolated_energy, _ = f.evaluate({(1, -1): phi}, gradient=False)
    screened = occupied_1s2_reference_potential(f, phi)
    report = {
        "z": z,
        "nuclear": nuclear,
        "grid": grid,
        "radius_bohr": radius,
        "continuum_roots": continuum,
        "hf_energy_ha": ref.total_energy,
        "hf_energy_evaluated_by_ci_ha": original_energy,
        "hf_ci_energy_difference_ha": original_energy - ref.total_energy,
        "hf_interpolated_ci_energy_ha": interpolated_energy,
        "hamiltonian_rv_sha256": hashlib.sha256(rv.tobytes()).hexdigest(),
        "models": [],
    }
    for label, potential in [("nuclear", rv), ("occupied_1s2_hf", screened)]:
        space, initial = continuum_enriched_electron_space(
            f,
            n_bound=2,
            n_continuum=continuum,
            reference_rv=potential,
            reference_label=label,
        )
        shifted = {(1, -1): 1.1 * min(space.root_energies[-1])}
        outside = phi - space.project(phi, -1)
        waves, ci, diagnostics = optimize_grid_shooting(
            f, initial, space, shifted, max_iter=300, tolerance=1e-6
        )
        report["models"].append(
            {
                "reference": label,
                "energy_ha": ci.ground_energy,
                "difference_from_hf_ha": ci.ground_energy - ref.total_energy,
                "hf_outside_reference_space_norm": float(
                    np.sqrt(np.sum(outside * outside * f.weights))
                ),
                "diagnostics": diagnostics,
                "space_diagnostics": space.continuum_diagnostics,
            }
        )
    if not np.array_equal(rv, f.data.rv_nuc):
        raise AssertionError("electron reference changed the physical Hamiltonian")
    return report


@pytest.mark.parametrize("z", [2, 26, 92])
def test_matched_projector_recovers_hf_without_changing_hamiltonian(z):
    result = compare_reference(z, continuum=160)
    nuclear, matched = result["models"]
    assert abs(result["hf_ci_energy_difference_ha"]) < 2e-8
    assert matched["diagnostics"]["converged"]
    assert matched["diagnostics"]["orbital_residual"] < 1e-6
    assert abs(matched["difference_from_hf_ha"]) < 2e-8
    assert matched["diagnostics"]["electron_reference"] == "occupied_1s2_hf"
    assert (
        matched["diagnostics"]["electron_reference_sha256"]
        != nuclear["diagnostics"]["electron_reference_sha256"]
    )
    assert abs(nuclear["difference_from_hf_ha"]) > 1e-6
    assert (
        matched["hf_outside_reference_space_norm"]
        < nuclear["hf_outside_reference_space_norm"] * 0.01
    )
    # The remaining energy difference is shared by the two energy evaluators,
    # rather than being an arbitrary fit to the HF scalar total energy.
    assert abs(matched["energy_ha"] - result["hf_energy_evaluated_by_ci_ha"]) < 2e-9


def test_reference_changes_projector_only_and_preserves_provenance():
    r, dr, n = sgrid(60.0, 1e-7, 0.1, 1000)
    rv = np.full(n, -2.0)
    f = GridCSFFunctional(
        enumerate_jj_csfs_many(["1s2"]), GridCIWorkspace(r, dr, rv), 2
    )
    phi = np.array([r * np.exp(-2 * r), -0.001 * r * np.exp(-2 * r)])
    phi /= np.sqrt(np.sum(phi * phi * f.weights))
    before, _ = f.evaluate({(1, -1): phi}, gradient=False)
    reference = occupied_1s2_reference_potential(f, phi)
    # One other electron: direct 2J minus self-exchange J, not 2J or J/2.
    assert reference[-1] - rv[-1] == pytest.approx(1.0, abs=1e-12)
    space, _ = bound_electron_space(
        f, 1, reference_rv=reference, reference_label="test occupied 1s2 potential"
    )
    assert np.array_equal(f.data.rv_nuc, rv)
    assert f.evaluate({(1, -1): phi}, gradient=False)[0] == before
    stored = space.reference_rv.copy()
    reference[:] = 0.0
    np.testing.assert_array_equal(space.reference_rv, stored)
    assert not space.reference_rv.flags.writeable
    assert len(space.reference_sha256) == 64
    with pytest.raises(ValueError, match="reference_label"):
        bound_electron_space(f, 1, reference_rv=stored)
    with pytest.raises(ValueError, match="reference_label"):
        bound_electron_space(f, 1, reference_label="mislabelled nuclear")
    with pytest.raises(ValueError, match="finite"):
        bound_electron_space(
            f, 1, reference_rv=np.full(n, np.nan), reference_label="invalid"
        )
    with pytest.raises(ValueError, match="normalized"):
        occupied_1s2_reference_potential(f, phi * 2)
    other = GridCSFFunctional(enumerate_jj_csfs_many(["1s1"]), f.data, 2)
    with pytest.raises(ValueError, match="two-electron"):
        occupied_1s2_reference_potential(other, phi)
