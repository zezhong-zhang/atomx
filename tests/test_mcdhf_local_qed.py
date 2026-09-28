"""Local QED potentials must retain radial coherence and symmetry."""

import numpy as np
import pytest

from atomx.constants import C
from atomx.corrections.qed import short_range_radial_matrix, uehling_potential
from atomx.corrections.stack import CorrectionStack
from atomx.corrections.workspace import make_grid_workspace
from atomx.methods.mcdhf.config import enumerate_jj_csfs
from atomx.methods.mcdhf.solver import MCDHFSolver


def radial_pair():
    r = np.geomspace(1e-9, 40, 12000)
    w = np.empty_like(r)
    w[1:-1] = (r[2:] - r[:-2]) / 2
    w[0], w[-1] = (r[1] - r[0]) / 2, (r[-1] - r[-2]) / 2
    p = np.array([2*r*np.exp(-r), r*(2-r)*np.exp(-r/2)/np.sqrt(8)])
    return p, -p/(2*C), make_grid_workspace(r, w)


@pytest.mark.parametrize("angle", [0., .37, 1.1])
def test_vp_ci_expectation_is_invariant_under_radial_rotation(angle):
    p, q, workspace = radial_pair()
    u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    csfs = enumerate_jj_csfs("1s1") + enumerate_jj_csfs("2s1")
    stack = CorrectionStack(se_backend=None, include_vp=True,
                            include_wk_vp=True, include_breit=False)
    solver = MCDHFSolver(1, csfs, correction_stack=stack)
    original = {(i+1, -1): (p[i], q[i]) for i in range(2)}
    rotated = {(i+1, -1): ((u@p)[i], (u@q)[i]) for i in range(2)}
    reference = solver._qed_state_components(
        mixing=np.eye(2), orbitals_dict=original, corr_data=workspace, i0=0)
    actual = solver._qed_state_components(
        mixing=u, orbitals_dict=rotated, corr_data=workspace, i0=0)
    assert reference["E_VP"][0] < 0 < reference["E_WK"][0]
    assert set(actual) == {"E_VP", "E_WK"}
    for key in reference:
        np.testing.assert_allclose(actual[key], reference[key], rtol=2e-13)
    # Occupation-only contraction fails this physical invariance.
    radial = short_range_radial_matrix(
        [(*rotated[key], key[1]) for key in sorted(rotated)], workspace,
        lambda r: uehling_potential(r, 1))
    diagonal_estimate = (u*u).T @ np.diag(radial)
    if angle:
        assert np.max(abs(diagonal_estimate-reference["E_VP"])) > .01*np.max(abs(reference["E_VP"]))


def test_local_potential_preserves_kappa_and_diagonal_fast_path():
    p, q, workspace = radial_pair()
    orbitals = [(p[0], q[0], -1), (p[1], q[1], -2)]
    potential = lambda r: uehling_potential(r, 1)  # noqa: E731
    matrix = short_range_radial_matrix(orbitals, workspace, potential)
    assert matrix[0, 1] == matrix[1, 0] == 0
    diagonal = short_range_radial_matrix(orbitals, workspace, potential, diagonal_only=True)
    np.testing.assert_allclose(diagonal, np.diag(matrix), rtol=1e-14)


def test_block_specific_qed_uses_each_blocks_radial_orbitals():
    p, q, workspace = radial_pair()
    csfs = enumerate_jj_csfs("1s1") + enumerate_jj_csfs("2p1")
    stack = CorrectionStack(se_backend=None, include_vp=True, include_breit=False)
    solver = MCDHFSolver(1, csfs, correction_stack=stack)
    orbitals = {(1, -1): (p[0], q[0]), (2, 1): (p[1], q[1]), (2, -2): (p[1], q[1])}
    mixing = np.eye(len(csfs))
    base = solver._qed_state_components(
        mixing=mixing, orbitals_dict=orbitals, corr_data=workspace, i0=0)
    local = {(1, -1): {key: (2*a, 2*b) for key, (a, b) in orbitals.items()}}
    updated = solver._qed_state_components(
        mixing=mixing, orbitals_dict=orbitals, corr_data=workspace, i0=0,
        per_block_orbitals=local)
    for i, csf in enumerate(csfs):
        factor = 4 if (csf.total_J2, csf.parity) == (1, -1) else 1
        assert updated["E_VP"][i] == pytest.approx(factor*base["E_VP"][i], rel=1e-14)


@pytest.mark.parametrize("n,kappa,letter", [(2, 1, "p"), (10, 1, "p"), (3, 2, "d")])
def test_positive_kappa_does_not_append_digits_to_self_energy_n(n, kappa, letter):
    from atomx.corrections.qed import compute_se_energy
    p, q, workspace = radial_pair()
    csfs = [c for c in enumerate_jj_csfs(f"{n}{letter}1")
            if c.subshells[0].kappa == kappa]
    assert len(csfs) == 1
    stack = CorrectionStack(se_backend="mohr", include_vp=False, include_breit=False)
    solver = MCDHFSolver(20, csfs, correction_stack=stack)
    actual = solver._qed_state_components(
        mixing=np.ones((1, 1)), orbitals_dict={(n, kappa): (p[1], q[1])},
        corr_data=workspace, i0=0)
    expected = compute_se_energy(
        [(p[1], q[1], 1., kappa, f"{n}{letter}{2*kappa-1}/2")],
        20, bsp_data=workspace, method="mohr")
    assert abs(expected) > 0
    assert actual["E_SE"][0] == pytest.approx(expected, rel=1e-14)


def test_result_adapter_vp_uses_correlated_density():
    from atomx.corrections.adapters import (
        SelfEnergyCorrection,
        UehlingVPCorrection,
        WichmannKrollVPCorrection,
    )
    from atomx.corrections.qedmod.flambaum_ginges import V_wichmann_kroll
    from atomx.corrections.qedmod.shabaev import _amplitude_A_kappa
    from atomx.methods.mcdhf.solver import MCDHFResult
    p, q, workspace = radial_pair()
    csfs = enumerate_jj_csfs("1s1") + enumerate_jj_csfs("2s1")
    mixing = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    result = MCDHFResult(
        converged=True, iterations=1, state_energies=np.array([-1., 0.]),
        mixing_coefficients=mixing, ground_energy=-1., orbital_energies={},
        csfs=csfs, orbitals_dict={(i+1, -1): (p[i], q[i]) for i in range(2)},
        Z=1, occupations={(1, -1): .5, (2, -1): .5},
        _grid=workspace["gr"], _difr=workspace["grw"])
    psi_p, psi_q = (p[0]+p[1])/np.sqrt(2), (q[0]+q[1])/np.sqrt(2)
    for correction, potential in [(UehlingVPCorrection(), uehling_potential),
                                   (WichmannKrollVPCorrection(), V_wichmann_kroll),
                                   (SelfEnergyCorrection("shabaev_local"),
                                    lambda r, z: _amplitude_A_kappa(z, -1)*r*np.exp(-C*r))]:
        assert correction.supports(result)
        expected = np.sum(workspace["grw"]*(psi_p**2+psi_q**2)*potential(workspace["gr"], 1))
        assert correction.evaluate(result).energy_shift == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("z", [20, 80])
def test_applied_p_self_energy_matches_canonical_orbital_backend(z):
    from atomx import Atom, Corrections
    from atomx.corrections.qed import compute_se_energy, resolve_se_method
    kwargs = dict(Z=z, charge=z-1, config=["2p1"], engine="bspline")
    bare = Atom(**kwargs, corrections="none").solve().result
    method = resolve_se_method("auto", z)
    stack = CorrectionStack(se_backend=method, include_vp=False, include_breit=False)
    corrected = Atom(**kwargs, corrections=Corrections(stack=stack)).solve().result
    for k, j2 in enumerate(bare.state_J2):
        kappa = 1 if j2 == 1 else -2
        p, q = bare.orbitals_dict[2, kappa]
        expected = compute_se_energy([(p, q, 1., kappa, f"2p{j2}/2")], z,
                                     bsp_data=bare.bsp_data, method=method)
        assert corrected.state_energies[k]-bare.state_energies[k] == pytest.approx(expected, abs=1e-11)
    assert corrected.energy_components["E_SE"] == pytest.approx(
        corrected.ground_energy-bare.ground_energy, abs=1e-11)


@pytest.mark.parametrize('angle', [0., .37, 1.1])
def test_local_self_energy_retains_radial_coherence(angle):
    from atomx.corrections.qed import compute_se_energy
    from atomx.corrections.qedmod import get_backend
    p, q, workspace = radial_pair()
    u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    backend = get_backend('shabaev_local')
    matrix = backend.radial_matrix([(p[i], q[i], -1) for i in range(2)], workspace, 1)
    for i in range(2):
        expected = compute_se_energy([(p[i], q[i], 1., -1, f'{i+1}s1/2')], 1,
                                     workspace, method='shabaev_local')
        assert matrix[i, i] == pytest.approx(expected, rel=1e-13)
    csfs = enumerate_jj_csfs('1s1') + enumerate_jj_csfs('2s1')
    solver = MCDHFSolver(1, csfs, correction_stack=CorrectionStack(
        se_backend='shabaev_local', include_vp=False, include_breit=False))
    actual = solver._qed_state_components(
        mixing=u, orbitals_dict={(i+1, -1): ((u@p)[i], (u@q)[i]) for i in range(2)},
        corr_data=workspace, i0=0)
    np.testing.assert_allclose(actual['E_SE'], np.diag(matrix), rtol=1e-13)
