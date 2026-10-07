"""Regression gates for the September 23 physics/numerics review."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from atomx import Atom
from atomx.angular import csf_hamiltonian as exact_ci
from atomx.config.csf import enumerate_jj_csfs


def test_rank_deficient_diis_trial_keeps_valid_orbitals(monkeypatch):
    from atomx import Basis
    from atomx.methods import dirac_fock
    calls = []

    def collapsed_trial(history, errors):
        calls.append(True)
        return np.zeros_like(history[-1])

    monkeypatch.setattr(dirac_fock, "diis_extrapolate", collapsed_trial)
    atom = Atom(18, engine="bspline", corrections="none", basis=Basis(ns=40)).solve()
    assert calls, "the regression must exercise a rejected DIIS extrapolation"
    assert atom.converged
    assert atom._result.energy_components["max_same_kappa_overlap"] < 1e-12


@pytest.mark.parametrize("kinetic_balance", [False, True])
def test_virtual_p_basis_has_physical_root_and_same_nuclear_field(kinetic_balance):
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.numerics.bspline.basis import prepare_bspline_data
    from atomx.numerics.bspline.knots import make_knots
    data = prepare_bspline_data(make_knots(60, 9, .0001, 50.), 60, 9,
                                kinetic_balance=kinetic_balance)
    result = SimpleNamespace(_bsp_data=data, Z=1, subshells=[], orbitals_raw={})
    _, virtual = build_virtual_basis(result, 1, e_max=0.)
    assert virtual[0][2] == pytest.approx(-.125, abs=3e-6)
    result._nuclear_potential = lambda r: -1. / r + .01
    _, shifted = build_virtual_basis(result, 1, e_max=0.)
    assert shifted[0][2] - virtual[0][2] == pytest.approx(.01, abs=1e-9)


def test_cap_preflight_does_not_construct_layout(monkeypatch):
    pattern = ((4, -4, 4), (4, 3, 3), (5, -5, 5))
    exact_ci._block_structure.cache_clear()
    def forbidden(*args):
        pytest.fail("over-budget space enumerated before dimension check")
    monkeypatch.setattr(exact_ci, "pattern_layout", forbidden)
    with pytest.raises(exact_ci.ExactCISpaceError, match="cap of 1"):
        exact_ci.block_terms((pattern,), 0, max_determinants=1)


def test_lazy_m_sector_matches_exhaustive_enumeration():
    pattern = ((2, -2, 2), (3, 2, 2), (3, -3, 1))
    layout = exact_ci.PatternLayout(pattern)
    assert not layout.by_m2
    choices = []
    for (_n, k, q), offset in zip(layout.pattern, layout.offsets, strict=True):
        choices.append([sum(1 << (i + offset) for i in combo)
                        for combo in itertools.combinations(range(2 * abs(k)), q)])
    all_masks = [sum(combo) for combo in itertools.product(*choices)]
    magnetic = tuple(s[2] for s in layout.spinors)
    expected = sorted(m for m in all_masks if exact_ci._mask_m2(m, magnetic) == 1)
    assert layout.by_m2[1] == tuple(expected)
    assert set(layout.by_m2) == {1}
    assert dict(exact_ci.pattern_m_counts(pattern))[1] == len(expected)


def test_projected_operator_matches_dense_without_dense_hamiltonian():
    csfs = enumerate_jj_csfs("1s1 2s1 2p1")
    signatures = [exact_ci.csf_signature(c) for c in csfs if c.total_J2 == 1]
    # Deterministic synthetic integrals isolate assembly from SCF errors.
    def radial(key):
        return sum(ord(c) for c in repr(key)) * 0.001
    dense = exact_ci.exact_block_matrix(signatures, radial)
    operator = exact_ci.exact_block_matrix(signatures, radial, as_operator=True)
    rng = np.random.default_rng(199)
    vectors = rng.normal(size=(len(signatures), 3))
    np.testing.assert_allclose(operator @ vectors, dense @ vectors, atol=1e-12)
    np.testing.assert_allclose(operator.diagonal(), np.diag(dense), atol=1e-12)
    structure = exact_ci.block_terms(tuple({s.pattern for s in signatures}), 1)
    assert structure.table.shape[0] == len(structure.pairs)


def test_ci_low_states_honor_solver_in_every_symmetry_block(monkeypatch):
    from atomx.methods import ci
    from atomx.physics.csf import matrix
    rng = np.random.default_rng(38)
    h = np.zeros((60, 60))
    for start in (0, 30):
        a = rng.normal(scale=0.05, size=(30, 30))
        h[start:start+30, start:start+30] = a + a.T + np.diag(np.arange(30))
    monkeypatch.setattr(matrix, "build_ci_matrix", lambda *a, **kw: h)
    csfs = [SimpleNamespace(total_J2=0 if i < 30 else 2, parity=1) for i in range(60)]
    calls = []
    real = ci.davidson_eigh
    def wrapped(a, n_states, **kwargs):
        calls.append((a.shape, n_states))
        return real(a, n_states, **kwargs)
    monkeypatch.setattr(ci, "davidson_eigh", wrapped)
    result = ci.ci_solve(csfs, {}, [], {}, 1, 0, n_states=2, solver="davidson", tol=1e-10)
    assert calls == [((30, 30), 2), ((30, 30), 2)]
    np.testing.assert_allclose(result.state_energies, np.linalg.eigvalsh(h)[:2], atol=1e-10)
    assert np.linalg.norm(h @ result.mixing_coefficients - result.mixing_coefficients * result.state_energies) < 1e-9


def test_continuum_cache_reuses_and_invalidates_in_place(monkeypatch):
    from atomx.numerics.bspline import continuum as cont
    from atomx.numerics.bspline.basis import prepare_bspline_data
    from atomx.numerics.bspline.knots import make_knots
    data = prepare_bspline_data(make_knots(40, 9, 0.01, 50), 40, 9)
    potential = np.zeros((40, 40))
    calls = []
    real = cont.linalg.eigh
    def eigh(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(cont.linalg, "eigh", eigh)
    for energy in (0.1, 0.2, 0.3):
        cont.solve_continuum(-1, energy, 1, data, potential)
    assert len(calls) == 1
    potential += np.eye(40) * 0.02
    changed = cont.solve_continuum(-1, 0.2, 1, data, potential)
    assert len(calls) == 2
    data.pop("_continuum_spectra")
    fresh = cont.solve_continuum(-1, 0.2, 1, data, potential)
    for a, b in zip(changed, fresh, strict=True):
        np.testing.assert_array_equal(a, b)
    data["grw"] *= 1.001
    cont.solve_continuum(-1, 0.2, 1, data, potential)
    assert len(calls) == 4
    assert len(data["_continuum_spectra"]) == 1


@pytest.fixture(scope="module")
def helium_s_basis():
    atom = Atom(2, config=["1s2", "1s1 2s1"], corrections="none", nuclear="point")
    atom.solve()
    return atom


def test_constrained_orbitals_and_physical_density(helium_s_basis):
    from atomx.methods.dirac_fock import _orbital_overlap
    atom = helium_s_basis
    result = atom.result
    p, q = result.orbitals_dict[1, -1]
    u, v = result.orbitals_dict[2, -1]
    assert abs(_orbital_overlap(p, q, u, v, result.bsp_data)) < 1e-10
    assert result.converged and result.solver_diagnostics["_inner_converged"]
    assert sum(result.occupations.values()) == pytest.approx(2, abs=1e-13)
    keys, density = result.one_body_density_matrix()
    assert np.trace(density) == pytest.approx(2, abs=1e-12)
    assert np.linalg.eigvalsh(density).min() > -1e-12
    assert abs(density[keys.index((1, -1)), keys.index((2, -1))]) > 1e-6
    assert np.sum(atom.density() * result.bsp_data["grw"].ravel()) == pytest.approx(2, abs=1e-10)


def test_one_electron_tensor_reduces_to_single_particle():
    for config_f, config_i in [("2p1", "1s1"), ("2p1", "2p1"), ("3d1", "1s1")]:
        for bra in enumerate_jj_csfs(config_f):
            for ket in enumerate_jj_csfs(config_i):
                for rank in (0, 1, 2):
                    jf, ji = bra.total_J2, ket.total_J2
                    terms = exact_ci.one_body_tensor_terms(exact_ci.csf_signature(bra), exact_ci.csf_signature(ket), rank)
                    if abs(jf - ji) <= 2 * rank <= jf + ji:
                        assert len(terms) == 1
                        assert terms[0][2] == pytest.approx(1, abs=1e-12)
                    else:
                        assert not terms


def test_correlated_radial_gradient_matches_directional_derivative():
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.numerics.bspline.basis import prepare_bspline_data
    from atomx.numerics.bspline.dirac_assembly import solve_dirac
    from atomx.numerics.bspline.knots import make_knots
    data = prepare_bspline_data(make_knots(35, 9, 0.005, 30), 35, 9)
    csfs = sum([enumerate_jj_csfs(config) for config in ("1s2", "2s2", "2p2")], [])
    functional = CSFOrbitalFunctional(csfs, data, 4)
    orbitals = {}
    for kappa in functional.metric:
        result = solve_dirac(functional.one_body[kappa], functional.metric[kappa], 35, kappa)
        for n, k in functional.keys:
            if k == kappa:
                # Same-order basis has an extra lowest positive-kappa root.
                ell = kappa if kappa > 0 else -kappa - 1
                index = n - ell - 1 + int(kappa > 0)
                p, q, _ = result.get_orbital(index)
                orbitals[n, k] = (p, q)
    _, gradient, _ = functional.evaluate(orbitals)
    rng = np.random.default_rng(237)
    blocks = []
    for kappa in functional.metric:
        keys = [key for key in functional.keys if key[1] == kappa]
        x = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
        metric = functional.metric[kappa]
        direction = rng.normal(size=x.shape)
        cross = x.T @ metric @ direction
        direction -= x @ (0.5 * (cross + cross.T))
        direction /= np.sqrt(np.trace(direction.T @ metric @ direction))
        blocks.append((keys, x, direction, metric))
    analytical = sum(gradient[key] @ direction[:, i]
                     for keys, _, direction, _ in blocks for i, key in enumerate(keys))

    def energy(step):
        trial = {}
        for keys, x, direction, metric in blocks:
            y = x + step * direction
            values, vectors = np.linalg.eigh(y.T @ metric @ y)
            y = y @ ((vectors / np.sqrt(values)) @ vectors.T)
            for i, key in enumerate(keys):
                trial[key] = (y[:35, i].copy(), y[35:, i].copy())
        return functional.evaluate(trial, gradient=False)[0]
    step = 1e-6
    numerical = (energy(step) - energy(-step)) / (2 * step)
    assert abs(analytical) > 1e-4
    assert numerical == pytest.approx(analytical, abs=2e-7)


def test_nonorthogonal_cofactors_reduce_to_slater_condon():
    csfs = enumerate_jj_csfs("1s1 2p1") + enumerate_jj_csfs("1s1 2s1")
    for bra in csfs:
        for ket in csfs:
            for rank in (0, 1, 2):
                a, b = exact_ci.csf_signature(bra), exact_ci.csf_signature(ket)
                expected = {(x, y): c for x, y, c in exact_ci.one_body_tensor_terms(a, b, rank)}
                got = {(x, y): c for x, y, c in exact_ci.nonorthogonal_tensor_terms(
                    a, b, rank, lambda x, y: float(x == y))}
                assert got == pytest.approx(expected, abs=1e-12)


def test_residual_core_continuum_has_correct_charge():
    from atomx.scattering.models import ContinuumModel, PotentialModel
    from atomx.scattering.radiative_recombination import (
        _rr_potential,
        _rr_potential_matrices,
    )
    atom = Atom(2, "1s2", engine="bspline", corrections="none", nuclear="point").solve()
    result = atom.result
    sh = result.subshells[0]
    potential, charge = _rr_potential(result, sh, None)
    radius = 10 * result._bsp_data["gr"].max()
    assert charge == pytest.approx(1)
    assert potential(radius) * radius == pytest.approx(-1, abs=1e-12)
    pp, qq = _rr_potential_matrices(result, sh, None)
    assert np.linalg.norm(pp) > 0
    assert np.linalg.norm(qq) > 0
    bare = ContinuumModel(method="bspline", potential=PotentialModel.bare())
    pp, qq = _rr_potential_matrices(result, sh, bare)
    np.testing.assert_allclose(pp, 0, atol=1e-12)
    np.testing.assert_allclose(qq, 0, atol=1e-12)


def test_mixed_basis_fock_is_derivative_of_hf_energy():
    """Unequal P/Q spline orders need both complete exchange band widths."""
    from atomx.methods.dirac_fock import DFSolver, Subshell
    solver = DFSolver(Z=20, subshells=[Subshell(1, -1, 2), Subshell(2, -1, 1)],
                      ns=30, ks=7, exchange="hf", include_breit=False, include_qed=False)
    result = solver.solve(max_iter=2)
    orbitals = result.orbitals_raw
    p, q, ni = orbitals[-1][1]
    u, v, nj = orbitals[-1][0]
    c, t = np.cos(0.17), np.sin(0.17)
    p, q, u, v = c*p+t*u, c*q+t*v, c*u-t*p, c*v-t*q
    orbitals[-1] = [(u, v, nj), (p, q, ni)]
    vi, vj = np.r_[p, q], np.r_[u, v]
    fields = []
    for index in (1, 0):
        pp, qq, pq, qp = solver._build_hf_matrix_dbsr(index, orbitals)
        np.testing.assert_allclose(pq, qp.T, atol=1e-13)
        fields.append(solver._energy_h0_matrices[-1][0] + np.block([[pp, pq], [qp, qq]]))
    derivative = 2 * (ni * vj @ fields[0] @ vi - nj * vi @ fields[1] @ vj)
    def energy(angle):
        c, t = np.cos(angle), np.sin(angle)
        orbitals[-1] = [(c*u-t*p, c*v-t*q, nj), (c*p+t*u, c*q+t*v, ni)]
        return solver._compute_total_energy(orbitals)[0]
    step = 2e-5
    numerical = (energy(step) - energy(-step)) / (2*step)
    assert abs(derivative) > 1
    assert derivative == pytest.approx(numerical, abs=2e-7)


def test_rydberg_capture_uses_physical_roots_and_pauli_vacancies():
    from atomx import Basis
    from atomx.scattering.radiative_recombination import rr_cross_section_rydberg
    hydrogen = Atom(1, "1s1", engine="bspline", corrections="none", basis=Basis(ns=40)).solve()
    h = rr_cross_section_rydberg(hydrogen.result, [10., 100.], n_max_per_kappa=2, kappa_max=1)
    states = h.provenance["per_state"]
    assert states["2p1/2"]["I_n_eV"] == pytest.approx(states["1s1/2"]["I_n_eV"]/4, rel=1e-4)
    lithium = Atom(3, "1s2 2s1", engine="bspline", corrections="none", basis=Basis(ns=40)).solve()
    li = rr_cross_section_rydberg(lithium.result, [10., 100.], n_max_per_kappa=2, kappa_max=1)
    assert "1s1/2" not in li.sigma_by_orbital
    assert "2s1/2" in li.sigma_by_orbital
    assert np.all(np.isfinite(li.sigma_total_a0_sq)) and np.all(li.sigma_total_a0_sq > 0)


def test_excited_one_electron_space_preserves_roots_and_density_ensemble():
    from atomx import Basis
    atom = Atom(1, config=["2s1", "2p1"], engine="bspline", nuclear="point",
                corrections="none", basis=Basis(ns=40)).solve()
    result = atom.result
    assert atom.converged
    assert result.ground_energy == pytest.approx(-0.1250020802, abs=2e-8)
    keys, density = result.one_body_density_matrix()
    np.testing.assert_allclose(np.diag(density), [result.occupations[k] for k in keys], atol=1e-12)
    assert np.trace(density) == pytest.approx(1, abs=1e-12)
