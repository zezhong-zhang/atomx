"""Independent exterior-power and degenerate matrix response oracles."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.physics.csf.levels import CILevel, ci_level_blocks
from atomx.physics.csf.overlap import ci_subspace_overlap_gradient, ci_subspace_overlaps
from atomx.physics.csf.overlap_response import principal_overlap_response
from atomx.physics.csf.radial_density import CIRadialDensity


def matrix_case(angles=(0.3, 0.6)):
    keys = tuple((n, -1) for n in range(1, 5))
    csfs = sorted(enumerate_jj_orbital_space(keys, 1), key=lambda c: c.subshells[0].n)
    waves = {key: (np.eye(4)[:, i], np.zeros(4)) for i, key in enumerate(keys)}
    reference = np.eye(4)
    for i, angle in enumerate(angles):
        j = i + 2
        reference[np.ix_([i, j], [i, j])] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    bra = SimpleNamespace(
        state_energies=np.array([-4.0, -4.0, 0.0, 1.0]), mixing_coefficients=reference
    )
    ket = SimpleNamespace(
        state_energies=np.array([-3.0, -3.0, -1.0, 1.0]), mixing_coefficients=np.eye(4)
    )
    levels = (CILevel(1, 1, 0), CILevel(1, 1, 1))
    return csfs, bra, ket, waves, {-1: np.eye(8)}, levels


def matrix_directional_weight(weights, direction):
    return sum(
        value * direction[key[1] - 1, key[3] - 1]
        for key, value in weights.items()
        if key[0] == "h"
    )


@pytest.mark.parametrize("angles", [(0.3, 0.6), (0.4, 0.4)])
def test_complete_degenerate_response_matches_independent_spectral_projector(angles):
    csfs, bra, ket, waves, metric, levels = matrix_case(angles)
    w = np.array([[0.3, -0.2], [-0.2, 0.7]])
    density = CIRadialDensity(csfs)
    response = principal_overlap_response(
        csfs,
        bra,
        waves,
        levels,
        csfs,
        ket,
        waves,
        levels,
        metric,
        w,
        radial_density=density,
    )
    rng = np.random.default_rng(513)
    dh = rng.normal(size=(4, 4))
    dh = 0.5 * (dh + dh.T)
    analytic = matrix_directional_weight(response.hamiltonian_weights, dh)
    b = bra.mixing_coefficients[:, :2]

    # Independent full 4x4 eigensolve, using a spectral projector, not roots.
    def value(t):
        _, vectors = np.linalg.eigh(np.diag(ket.state_energies) + t * dh)
        projector = vectors[:, :2] @ vectors[:, :2].T
        return np.trace(w @ b.T @ projector @ b)

    numerical = (value(1e-5) - value(-1e-5)) / 2e-5
    assert analytic == pytest.approx(numerical, abs=3e-10, rel=3e-9)
    assert response.nearest_external_gap == 2.0
    # Rotate both internal exact clusters, including a repeated principal value.
    rb = expm(np.array([[0.0, 0.7], [-0.7, 0.0]]))
    rk = expm(np.array([[0.0, -0.4], [0.4, 0.0]]))
    bra2 = SimpleNamespace(
        state_energies=bra.state_energies,
        mixing_coefficients=bra.mixing_coefficients.copy(),
    )
    ket2 = SimpleNamespace(
        state_energies=ket.state_energies,
        mixing_coefficients=ket.mixing_coefficients.copy(),
    )
    bra2.mixing_coefficients[:, :2] = b @ rb
    ket2.mixing_coefficients[:, :2] = ket.mixing_coefficients[:, :2] @ rk
    w2 = rb.T @ w @ rb
    w2 = 0.5 * (w2 + w2.T)
    changed = principal_overlap_response(
        csfs,
        bra2,
        waves,
        levels,
        csfs,
        ket2,
        waves,
        levels,
        metric,
        w2,
        radial_density=density,
    )
    assert matrix_directional_weight(changed.hamiltonian_weights, dh) == pytest.approx(
        analytic, abs=3e-14
    )
    for key in waves:
        np.testing.assert_allclose(
            changed.basis_gradient[key], response.basis_gradient[key], atol=2e-14
        )
    np.testing.assert_allclose(
        changed.principal_matrix, rb.T @ response.principal_matrix @ rb, atol=2e-14
    )


@pytest.mark.parametrize("number", [1, 2, 3])
def test_zero_overlap_singular_minors_have_correct_nonzero_radial_derivative(number):
    # One active orbital changes to an orthogonal spatial wave. N-1 spectator
    # spin orbitals retain unit overlap: the occupied minor has rank N-1.
    keys = ((1, -1),) if number == 1 else ((1, -1), (2, -1))
    csfs = enumerate_jj_orbital_space(keys, number)
    energies = np.arange(len(csfs), dtype=float)
    ci = SimpleNamespace(state_energies=energies, mixing_coefficients=np.eye(len(csfs)))
    old = {keys[0]: (np.array([1.0, 0.0, 0.0]), np.zeros(3))}
    new = (
        {keys[0]: (np.array([0.0, 0.0, 1.0]), np.zeros(3))}
        if number == 1
        else dict(old)
    )
    if number > 1:
        old[keys[1]] = np.array([0.0, 1.0, 0.0]), np.zeros(3)
        new[keys[1]] = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    wanted = {(1, -1): 1} if number == 1 else {(1, -1): number - 1, (2, -1): 1}
    # For N=2 choose the triplet M=J determinant, eliminating recoupling choices.
    selected = next(
        i
        for i, c in enumerate(csfs)
        if {(s.n, s.kappa): s.occupation for s in c.subshells} == wanted
        and (number != 2 or c.total_J2 == 2)
    )
    _, blocks = ci_level_blocks(csfs, ci)
    c = csfs[selected]
    block = blocks[c.total_J2, c.parity]
    level = CILevel(c.total_J2, c.parity, int(np.flatnonzero(block == selected)[0]))
    overlaps, gradient = ci_subspace_overlap_gradient(
        csfs,
        ci,
        old,
        (level,),
        csfs,
        ci,
        new,
        (level,),
        {-1: np.eye(6)},
        np.ones((1, 1)),
    )
    active = keys[-1]
    np.testing.assert_allclose(overlaps, [[0.0]], atol=1e-14)
    direction = np.r_[old[active][0], np.zeros(3)]
    assert gradient[active] @ direction == pytest.approx(1.0, abs=2e-14)
    for t in (-1e-5, 1e-5):
        changed = dict(new)
        changed[active] = (
            np.cos(t) * new[active][0] + np.sin(t) * old[active][0],
            np.zeros(3),
        )
        actual = ci_subspace_overlaps(
            csfs, ci, old, (level,), csfs, ci, changed, (level,), {-1: np.eye(6)}
        )[0, 0]
        assert actual == pytest.approx(np.sin(t), abs=2e-14)


def test_rectangular_reference_and_complete_ket_cluster_response():
    csfs, bra, ket, waves, metric, levels = matrix_case()
    bra.state_energies = np.array([-4.0, -2.0, 0.0, 1.0])
    response = principal_overlap_response(
        csfs,
        bra,
        waves,
        levels[:1],
        csfs,
        ket,
        waves,
        levels,
        metric,
        np.ones((1, 1)),
        radial_density=CIRadialDensity(csfs),
    )
    assert response.overlaps.shape == (1, 2)
    assert response.principal_matrix.shape == (1, 1)
    dh = np.array(
        [
            [0.2, 0.3, -0.4, 0.1],
            [0.3, -0.1, 0.2, -0.3],
            [-0.4, 0.2, 0.1, 0.4],
            [0.1, -0.3, 0.4, -0.2],
        ]
    )
    reference = bra.mixing_coefficients[:, 0]

    def value(t):
        _, vectors = np.linalg.eigh(np.diag(ket.state_energies) + t * dh)
        return np.sum((reference @ vectors[:, :2]) ** 2)

    numerical = (value(1e-5) - value(-1e-5)) / 2e-5
    assert matrix_directional_weight(response.hamiltonian_weights, dh) == pytest.approx(
        numerical,
        abs=3e-10,
        rel=3e-9,
    )


def test_response_refuses_split_cluster_complex_and_wrong_basis():
    csfs, bra, ket, waves, metric, levels = matrix_case()
    density = CIRadialDensity(csfs)

    def call(bra_levels=levels, ket_levels=levels, w=None, d=density):
        w = np.eye(2) if w is None else w
        return principal_overlap_response(
            csfs,
            bra,
            waves,
            bra_levels,
            csfs,
            ket,
            waves,
            ket_levels,
            metric,
            w,
            radial_density=d,
        )

    with pytest.raises(ValueError, match="complete degenerate"):
        call(ket_levels=levels[:1])
    with pytest.raises(ValueError, match="complete degenerate"):
        call(bra_levels=levels[:1], w=np.ones((1, 1)))
    with pytest.raises(ValueError, match="exact ket CSF"):
        call(d=CIRadialDensity(list(reversed(csfs))))
    with pytest.raises(ValueError, match="real symmetric"):
        call(w=np.eye(2, dtype=complex))
    with pytest.raises(ValueError, match="real symmetric"):
        call(w=np.array([[1.0, 0.1], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="determinant pairs"):
        principal_overlap_response(
            csfs,
            bra,
            waves,
            levels,
            csfs,
            ket,
            waves,
            levels,
            metric,
            np.eye(2),
            radial_density=density,
            max_determinant_pairs=1,
        )


def test_bilinear_density_matches_independent_matrix_polarization_and_signed_amplitudes():
    csfs, bra, ket, waves, metric, levels = matrix_case()
    density = CIRadialDensity(csfs)
    rng = np.random.default_rng(6)
    left, right = rng.normal(size=(2, 4))
    h = rng.normal(size=(4, 4))
    h = 0.5 * (h + h.T)
    terms = density.contract_bilinear(left, right)
    assert matrix_directional_weight(terms, h) == pytest.approx(
        left @ h @ right, abs=2e-14
    )
    for defect in (left.astype(complex), left[:-1], left * np.nan):
        with pytest.raises(ValueError, match="finite real CSF"):
            density.contract_bilinear(defect, right)


def test_transition_gradient_preserves_energy_diagnostics_and_matches_physical_matrix_difference():
    from atomx import Atom, Basis, Correlation
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.matrix import build_ci_matrix

    atom = Atom(
        10,
        config=["1s2", "1s1 2p1", "2s2"],
        basis=Basis(ns=40, ks=9),
        corrections="none",
        correlation=Correlation(radial_optimization="fac_frozen_local"),
    ).solve(tol=1e-6)
    r = atom.result
    f = CSFOrbitalFunctional(r.csfs, r.bsp_data, 10, r._nuclear_potential)
    waves = r.orbitals_dict
    _, _, ci = f.evaluate(waves)
    u, v = np.random.default_rng(27).normal(size=(2, len(r.csfs)))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    terms = f.radial_density.contract_bilinear(u, v)
    h = build_ci_matrix(
        r.csfs,
        waves,
        f.keys,
        r.bsp_data,
        10,
        f.ns,
        nuclear_potential=r._nuclear_potential,
    )
    assert f.energy_for_weights(waves, terms) == pytest.approx(u @ h @ v, abs=2e-12)
    saved = {
        name: getattr(f, name)
        for name in ("last_weights", "last_occupations", "last_gradient")
    }
    gradient = f.gradient_from_weights(waves, terms, record_diagnostics=False)
    for name, value in saved.items():
        assert getattr(f, name) is value
    keys = [key for key in f.keys if key[1] == -1]
    c = np.column_stack([np.concatenate(waves[key]) for key in keys])
    d = np.random.default_rng(31).normal(size=c.shape)
    cross = c.T @ f.metric[-1] @ d
    d -= c @ (0.5 * (cross + cross.T))
    d /= np.sqrt(np.trace(d.T @ f.metric[-1] @ d))
    analytic = sum(gradient[key] @ d[:, i] for i, key in enumerate(keys))
    values = []
    for t in (-1e-6, 1e-6):
        x = c + t * d
        eigenvalues, eigenvectors = np.linalg.eigh(x.T @ f.metric[-1] @ x)
        x = x @ ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T)
        changed = dict(waves)
        for i, key in enumerate(keys):
            changed[key] = x[: f.ns, i], x[f.ns :, i]
        matrix = build_ci_matrix(
            r.csfs,
            changed,
            f.keys,
            r.bsp_data,
            10,
            f.ns,
            nuclear_potential=r._nuclear_potential,
        )
        values.append(u @ matrix @ v)
    assert (values[1] - values[0]) / 2e-6 == pytest.approx(analytic, abs=2e-7, rel=3e-6)
