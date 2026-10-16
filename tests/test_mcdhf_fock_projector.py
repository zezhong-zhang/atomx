"""Complete screened Fock inputs define one fixed native no-pair projector."""

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import optimize_orbitals
from atomx.numerics.bspline.orbital_subspace import radial_projector_start
from atomx.physics.csf.matrix import build_ci_matrix


@pytest.fixture
def screened_problem():
    data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9)
    csfs = [csf for config in ("1s2", "2s2")
            for csf in enumerate_jj_csfs(config)]
    functional = CSFOrbitalFunctional(csfs, data, 4)
    metric = functional.metric[-1]
    nuclear = solve_dirac(functional.one_body[-1], metric, 35, -1)
    waves = {}
    complete = {}
    for key, charge in (((1, -1), 2.), ((2, -1), 1.)):
        column = nuclear.coefficients[:, key[0] - 1]
        waves[key] = (column[:35].copy(), column[35:].copy())
        matrix, _ = build_dirac_matrices(35, -1, charge, data)
        complete[key] = np.tril(matrix) + np.tril(matrix, -1).T
    # The first reference has q(1s)=2; the absent 2s gets fallback weight 1.
    average = (2 * complete[1, -1] + complete[2, -1]) / 3
    return functional, waves, complete, average


@pytest.mark.parametrize("projector", ["nuclear", "fock"])
def test_complete_fock_input_is_averaged_without_extra_nuclear_operator(
    screened_problem, monkeypatch, projector,
):
    functional, waves, complete, average = screened_problem
    seen = []

    def capture(matrix, metric, ns, kappa, **kwargs):
        seen.append(matrix.copy())
        return solve_dirac(matrix, metric, ns, kappa, **kwargs)

    monkeypatch.setattr("atomx.numerics.bspline.dirac_assembly.solve_dirac", capture)
    optimize_orbitals(functional, waves, max_iter=0,
                      preconditioner_matrices=complete, projector_source=projector)
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], functional.one_body[-1])
    # The kernel receives full operators, including the same mass/kinetic
    # terms. Adding nuclear.one_body a second time changes the model.
    np.testing.assert_allclose(seen[1], average, rtol=0, atol=2e-12)


def test_fock_projected_spinors_and_ci_match_independent_screened_spectrum(
    screened_problem,
):
    functional, waves, complete, average = screened_problem
    ns, metric = functional.ns, functional.metric[-1]
    spectrum = solve_dirac(average, metric, ns, -1)
    first = radial_projector_start(-1, 1, functional.data)
    projector = spectrum.coefficients[:, first:]
    seed = np.column_stack([np.concatenate(waves[key]) for key in functional.keys])
    projected = projector @ (projector.T @ metric @ seed)
    eigenvalues, eigenvectors = np.linalg.eigh(projected.T @ metric @ projected)
    expected = projected @ ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T)
    optimized, ci, report = optimize_orbitals(
        functional, waves, max_iter=0, preconditioner_matrices=complete,
        projector_source="fock",
    )
    actual = np.column_stack([np.concatenate(optimized[key]) for key in functional.keys])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    np.testing.assert_allclose(actual.T @ metric @ actual, np.eye(2), rtol=0, atol=2e-12)
    np.testing.assert_allclose(actual, projector @ (projector.T @ metric @ actual),
                               rtol=0, atol=2e-10)
    expected_waves = {key: (expected[:ns, i], expected[ns:, i])
                      for i, key in enumerate(functional.keys)}
    matrix = build_ci_matrix(functional.csfs, expected_waves, functional.keys,
                             functional.data, 4, ns)
    np.testing.assert_allclose(ci.state_energies, np.linalg.eigvalsh(matrix),
                               rtol=0, atol=2e-10)
    assert report["projector"] == "fixed positive-energy Fock-field (screened) B-spline subspace"
    assert report["iterations"] == 0
