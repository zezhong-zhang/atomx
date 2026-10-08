"""Analytic hydrogenic virtuals, basis covariance and strict subspace gates."""
import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.numerics.bspline.orbital_subspace import extend_nuclear_virtuals
from atomx.physics.dirac import hydrogenic_dirac_energy


def seed(kappa, balanced):
    ns, ks = 100, 9
    data = prepare_bspline_data(make_exponential_knots(ns, ks, 1e-6, 75.),
                               ns, ks, kinetic_balance=balanced)
    h, s = build_dirac_matrices(ns, kappa, 10, data)
    h = np.tril(h)+np.tril(h, -1).T
    spectrum = solve_dirac(h, s, ns, kappa, nsp=data['nsp'], nsq=data['nsq'])
    # Independent analytic energies choose reference shells; do not use the
    # helper's indexing rule as the test oracle (especially equal-order p+).
    numbers = (1, 2) if kappa == -1 else (2,)
    orbitals = {}
    for n in numbers:
        expected = hydrogenic_dirac_energy(n, kappa, 10)
        index = np.argmin(abs(spectrum.eigenvalues-expected))
        assert abs(spectrum.eigenvalues[index]-expected) < 2e-5
        v = spectrum.coefficients[:, index]
        orbitals[n, kappa] = (v[:ns], v[ns:])
    return orbitals, {kappa: h}, {kappa: s}, data


@pytest.mark.parametrize('balanced', [False, True])
@pytest.mark.parametrize('kappa', [-1, 1, -2])
def test_appended_virtuals_match_analytic_3_and_4_shells_without_changing_core(kappa, balanced):
    frozen, h, metric, data = seed(kappa, balanced)
    saved = {key: tuple(a.copy() for a in pair) for key, pair in frozen.items()}
    extended, diagnostics = extend_nuclear_virtuals(frozen, h, metric, data, layers=2)
    for key, pair in saved.items():
        for original, actual, source in zip(pair, extended[key], frozen[key], strict=True):
            np.testing.assert_array_equal(original, actual)
            np.testing.assert_array_equal(original, source)
            assert not np.shares_memory(actual, source)
    all_vectors = np.column_stack([np.concatenate(pair) for pair in extended.values()])
    np.testing.assert_allclose(all_vectors.T @ metric[kappa] @ all_vectors,
                               np.eye(len(extended)), atol=2e-11, rtol=0)
    for n in (3, 4):
        assert diagnostics[n, kappa]['one_body_energy_hartree'] == pytest.approx(
            hydrogenic_dirac_energy(n, kappa, 10), abs=2e-5)


def test_virtual_space_is_invariant_to_frozen_same_kappa_rotations():
    frozen, h, metric, data = seed(-1, False)
    x = np.column_stack([np.concatenate(pair) for pair in frozen.values()])
    angle = .4
    rotated = x @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    changed = {key: (rotated[:data['ns'], i], rotated[data['ns']:, i])
               for i, key in enumerate(frozen)}
    a, da = extend_nuclear_virtuals(frozen, h, metric, data, layers=2)
    b, db = extend_nuclear_virtuals(changed, h, metric, data, layers=2)
    va = np.column_stack([np.concatenate(a[n, -1]) for n in (3, 4)])
    vb = np.column_stack([np.concatenate(b[n, -1]) for n in (3, 4)])
    np.testing.assert_allclose(abs(va.T @ metric[-1] @ vb), np.eye(2), atol=1e-10)
    np.testing.assert_allclose([da[key]['one_body_energy_hartree'] for key in da],
                               [db[key]['one_body_energy_hartree'] for key in db], atol=1e-10)


def test_missing_lower_shell_cannot_enter_through_virtual_construction():
    frozen, h, metric, data = seed(-1, False)
    # Label an analytic 1s as an explicitly excited-only 2s active space.
    # Orthonormality alone passes; the requested no-pair boundary must reject.
    wrong = {(2, -1): frozen[1, -1]}
    with pytest.raises(ValueError, match='outside.*subspace'):
        extend_nuclear_virtuals(wrong, h, metric, data)
    invalid = {key: tuple(2*a for a in pair) for key, pair in frozen.items()}
    with pytest.raises(ValueError, match='S-orthonormal'):
        extend_nuclear_virtuals(invalid, h, metric, data)


@pytest.mark.parametrize('balanced', [False, True])
@pytest.mark.parametrize('kappa', [2, -3])
def test_new_d_symmetry_starts_at_physical_3d_and_preserves_existing_orbitals(kappa, balanced):
    frozen, _, _, data = seed(-1, balanced)
    h, s = build_dirac_matrices(data['ns'], kappa, 10, data)
    h = np.tril(h)+np.tril(h, -1).T
    extended, diagnostics = extend_nuclear_virtuals(frozen, {kappa: h}, {kappa: s}, data,
                                                    layers=2, kappas=(kappa,))
    for n in (3, 4):
        assert diagnostics[n, kappa]['one_body_energy_hartree'] == pytest.approx(
            hydrogenic_dirac_energy(n, kappa, 10), abs=2e-5)
    for key, pair in frozen.items():
        for original, actual in zip(pair, extended[key], strict=True):
            np.testing.assert_array_equal(original, actual)
