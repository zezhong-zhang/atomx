"""Physical limits and radial-basis covariance of the GRASP NMS operator."""

import numpy as np
import pytest
from scipy.special import gamma

from atomx.constants import C
from atomx.corrections.recoil import recoil_radial_matrix
from atomx.numerics.radial_workspace import make_grid_workspace


@pytest.mark.parametrize("z", [1, 20, 80])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_point_dirac_circular_recoil_has_exact_breit_limit(z, n):
    # Exact first-order point-nucleus recoil: K=(c^4-E_D^2)/(2c^2).
    # For a circular state κ=-n, E_D=c² sqrt(1-(Z/(n*c))²).
    # This checks the 1s, 2p3/2 and 3d5/2 angular barriers independently.
    g = np.sqrt(n**2 - (z / C)**2)
    decay = z / n
    r = np.geomspace(1e-12 / z, 100 / decay, 16000)
    w = np.empty_like(r)
    w[1:-1] = (r[2:] - r[:-2]) / 2
    w[0], w[-1] = (r[1] - r[0]) / 2, (r[-1] - r[-2]) / 2
    ratio = -z / (C * (n + g))
    norm = np.sqrt((2 * decay)**(2 * g + 1) / ((1 + ratio**2) * gamma(2 * g + 1)))
    p = norm * r**g * np.exp(-decay * r)
    q = ratio * p
    k = recoil_radial_matrix([(p, q, -n)], make_grid_workspace(r, w), z)[0, 0]
    assert k == pytest.approx(z**2 / (2 * n**2), rel=2e-6)


def test_recoil_retains_coherence_under_radial_orbital_rotation():
    r = np.linspace(1e-6, 30, 6000)
    w = np.full_like(r, r[1] - r[0])
    w[[0, -1]] *= 0.5
    p = np.array([2 * r * np.exp(-r), r * (2 - r) * np.exp(-r / 2) / np.sqrt(8)])
    q = -p / (2 * C)
    angle = 0.37
    u = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ws = make_grid_workspace(r, w)
    k = recoil_radial_matrix([(p[i], q[i], -1) for i in range(2)], ws, 1)
    kp = recoil_radial_matrix([(a, b, -1) for a, b in zip(u @ p, u @ q, strict=True)], ws, 1)
    assert abs(k[0, 1]) > 1e-3
    np.testing.assert_allclose(kp, u @ k @ u.T, atol=1e-12)
    rho = np.array([[1.7, .2], [.2, .3]])
    assert np.sum(rho * k) == pytest.approx(np.sum((u @ rho @ u.T) * kp), abs=1e-12)


def test_mcdhf_mass_stack_shifts_each_hydrogenic_level():
    from atomx import Atom, Corrections
    from atomx.corrections.stack import CorrectionStack

    kwargs = dict(Z=2, charge=1, config=["1s1", "2s1"], nuclear="point")
    bare = Atom(**kwargs, corrections="none").solve().result
    mass = 7294.3
    stack = CorrectionStack(se_backend=None, include_vp=False,
                            include_breit=False, nuclear_mass_me=mass)
    corrected = Atom(**kwargs, corrections=Corrections(stack=stack)).solve().result
    assert bare.converged and corrected.converged
    shifts = corrected.state_energies - bare.state_energies
    expected = (-bare.state_energies - bare.state_energies**2 / (2 * C**2)) / mass
    np.testing.assert_allclose(shifts, expected, rtol=2e-5, atol=1e-10)
    assert shifts[0] > shifts[1] > 0
    assert corrected.energy_components["nms"] == pytest.approx(shifts[0], abs=1e-12)
    assert "coherence" in corrected.solver_diagnostics["_nms_provenance"]["evaluation"]


def test_mcdhf_vp_only_stack_does_not_enable_self_energy(monkeypatch):
    from atomx import Atom, Corrections
    from atomx.corrections import qed
    from atomx.corrections.stack import CorrectionStack

    def forbidden(*args, **kwargs):
        raise AssertionError("self-energy must stay disabled in a VP-only stack")

    monkeypatch.setattr(qed, "compute_se_energy", forbidden)
    stack = CorrectionStack(se_backend=None, include_vp=True, include_breit=False)
    result = Atom(Z=2, charge=1, config=["1s1", "2s1"],
                  corrections=Corrections(stack=stack)).solve().result
    assert result.converged
    components = result.energy_components
    assert "E_SE" not in components
    assert components["E_QED_total"] == components["E_VP"] < 0


def test_resolved_p_recoil_checks_both_kappa_signs():
    from atomx import Atom, Basis, Corrections
    from atomx.corrections.stack import CorrectionStack

    kwargs = dict(Z=20, charge=19, config=["2p1"], engine="bspline",
                  nuclear="point", basis=Basis(grid_hi=1e-8, grid_he=.2))
    bare = Atom(**kwargs, corrections="none").solve().result
    mass = 72000.
    stack = CorrectionStack(se_backend=None, include_vp=False,
                            include_breit=False, nuclear_mass_me=mass)
    corrected = Atom(**kwargs, corrections=Corrections(stack=stack)).solve().result
    assert set(bare.state_J2) == {1, 3}
    expected = (-bare.state_energies - bare.state_energies**2 / (2*C**2)) / mass
    np.testing.assert_allclose(corrected.state_energies - bare.state_energies,
                               expected, rtol=2e-5, atol=1e-10)


def test_csf_recoil_expectation_includes_off_diagonal_radial_terms():
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.corrections.recoil import recoil_state_parameters

    r = np.linspace(1e-6, 40, 8000)
    w = np.full_like(r, r[1] - r[0])
    w[[0, -1]] *= 0.5
    p1 = 2 * r * np.exp(-r)
    p2 = r * (2 - r) * np.exp(-r / 2) / np.sqrt(8)
    orbitals = {(1, -1): (p1, -p1 / (2 * C)), (2, -1): (p2, -p2 / (2 * C))}
    csfs = enumerate_jj_csfs("1s1") + enumerate_jj_csfs("2s1")
    mixing = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    ws = make_grid_workspace(r, w)
    k = recoil_radial_matrix([(*v, key[1]) for key, v in orbitals.items()], ws, 1)
    actual = recoil_state_parameters(csfs, mixing, orbitals, ws, 1)
    expected = np.diag(mixing.T @ k @ mixing)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert abs(actual[0] - actual[1]) > 0.1


@pytest.mark.parametrize("z", [1, 20, 80])
def test_bspline_dhf_recoil_matches_hydrogenic_identity(z):
    from atomx import Atom, Basis, Corrections
    from atomx.corrections.stack import CorrectionStack

    mass = 2000 * z
    stack = CorrectionStack(se_backend=None, include_vp=False,
                            include_breit=False, nuclear_mass_me=mass)
    # The point-Coulomb cusp makes derivative observables much more
    # sensitive to the innermost cell than energies, especially at high Z.
    basis = Basis(grid_hi=1e-8, grid_he=.2)
    result = Atom(Z=z, charge=z-1, config="1s1", nuclear="point", engine="bspline", basis=basis,
                  corrections=Corrections(stack=stack)).solve().result
    assert result.converged
    assert result.energy_components["nms"] == pytest.approx(z**2 / (2 * mass), rel=2e-5)
    assert "GRASP" in result.solver_diagnostics["_nms_provenance"]["operator"]
