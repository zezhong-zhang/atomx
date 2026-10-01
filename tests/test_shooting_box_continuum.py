"""Independent spectral checks for finite-box continuum shooting."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import spherical_jn

from atomx.numerics.shooting.box_continuum import BoxDiracContinuum
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.grid import sgrid


@pytest.mark.parametrize('kappa', [-1, 1, -2, 2])
def test_free_box_relativistic_bessel_spectrum(kappa):
    r, dr, n = sgrid(20., 1e-7, .1, 1000)
    energies, waves, report = BoxDiracContinuum(r, dr, np.zeros(n)).solve(kappa, 5)
    ell = kappa if kappa > 0 else -kappa-1
    # Independent analytic spherical-Bessel zeros, including both j branches.
    roots = []
    samples = np.arange(.01, 30., .1)
    for a, b in zip(samples[:-1], samples[1:], strict=False):
        if spherical_jn(ell, a)*spherical_jn(ell, b) < 0:
            roots.append(brentq(lambda x: spherical_jn(ell, x), a, b))
    momentum = np.array(roots[:5])/r[-1]
    expected = momentum**2/(np.sqrt(1+(momentum/SL)**2)+1)
    np.testing.assert_allclose(energies, expected, atol=1e-11, rtol=2e-9)
    assert report['orthogonality_defect'] < 1e-8
    assert report['boundary_residual'] < 1e-9
    assert np.max(abs(waves[:, 0, -1])) < 1e-9


@pytest.mark.parametrize('kappa', [-1, 1])
def test_coulomb_box_against_independent_adaptive_ode(kappa):
    z, radius = 2., 30.
    r, dr, n = sgrid(radius, 1e-7, .1, 1000)
    energies, _, report = BoxDiracContinuum(r, dr, np.full(n, -z)).solve(kappa, 3)
    gamma = np.sqrt(kappa*kappa-(z/SL)**2)
    start = 1e-9
    # Frobenius leading term; arbitrary scale. DOP853 uses a separate
    # differential-equation integrator and no production spline/kernel.
    seed = [1., (gamma+kappa)*SL/z]
    def endpoint(energy):
        def rhs(x, y):
            p, q = y
            return [-kappa*p/x+(energy+z/x+2*SL**2)/SL*q,
                    kappa*q/x-(energy+z/x)/SL*p]
        result = solve_ivp(rhs, (start, radius), seed, method='DOP853',
                           rtol=2e-11, atol=1e-10)
        assert result.success
        return result.y[0, -1]
    independent = [brentq(endpoint, e-.002, e+.002, xtol=1e-12) for e in energies]
    np.testing.assert_allclose(energies, independent, atol=2e-9, rtol=2e-8)
    assert report['initial_zero_energy_nodes'] > 0


def test_grid_refinement_and_underresolved_rejection():
    spectra = []
    for size in (600, 1200):
        r, dr, n = sgrid(30., 1e-7, .1, size)
        e, _, _ = BoxDiracContinuum(r, dr, np.full(n, -2.)).solve(-1, 12)
        spectra.append(e)
    np.testing.assert_allclose(*spectra, atol=2e-9, rtol=1e-8)
    r, dr, n = sgrid(30., 1e-7, 3., 40)
    with pytest.raises(ValueError, match='resolve|underresolved'):
        BoxDiracContinuum(r, dr, np.full(n, -2.)).solve(-1, 12)


def test_enrichment_contains_original_space_and_is_nested():
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.csf.grid_functional import GridCSFFunctional
    from atomx.numerics.shooting.electron_space import continuum_enriched_electron_space
    from atomx.physics.csf.matrix import GridCIWorkspace
    r, dr, n = sgrid(100., 1e-7, .2, 1000)
    csfs = [c for c in enumerate_jj_csfs_many(['1s2', '2s2', '2p2'])
            if c.total_J2 == 0 and c.parity == 1]
    f = GridCSFFunctional(csfs, GridCIWorkspace(r, dr, np.full(n, -2.)), 2)
    small, orbitals = continuum_enriched_electron_space(f, n_bound=3, n_continuum=3)
    larger, _ = continuum_enriched_electron_space(f, n_bound=3, n_continuum=6)
    for kappa, basis in small.basis.items():
        for wave in basis:
            error = wave-larger.project(wave, kappa)
            assert np.sum(error*error*f.weights) < 1e-18
        assert small.continuum_diagnostics[kappa]['omitted_negative_box_roots'] == 0
    for key, wave in orbitals.items():
        np.testing.assert_allclose(larger.project(wave, key[1]), wave, atol=1e-10)


def test_diffuse_negative_box_roots_match_bound_limit_and_boundary():
    from atomx.physics.dirac import hydrogenic_dirac_energy
    r, dr, n = sgrid(60., 1e-7, .08, 1800)
    solver = BoxDiracContinuum(r, dr, np.full(n, -2.))
    energies, waves, report = solver.negative_roots_above(-1, -.22, expected_below=3)
    assert report['negative_box_roots'] == 9
    assert report['matched_negative_roots'] == 6
    assert report['maximum_matching_defect'] < 1e-10
    assert np.all(np.diff(energies) > 0) and energies[-1] < 0
    # 4s is localized sufficiently far inside R=60; higher roots should
    # shift upward as the physical finite wall reaches their diffuse tails.
    assert energies[0] == pytest.approx(hydrogenic_dirac_energy(4, -1, 2), abs=1e-10)
    assert energies[-1] > hydrogenic_dirac_energy(9, -1, 2)
    np.testing.assert_allclose(waves[:, 0, -1], 0., atol=1e-15)
    gram = np.einsum('iar,r,jar->ij', waves, solver.weights, waves)
    np.testing.assert_allclose(gram, np.eye(len(waves)), atol=1e-8)
    with pytest.raises(ValueError, match='cutoff'):
        solver.negative_roots_above(-1, -.22, expected_below=2)


def test_continuum_carbon_converges_without_noise_selected_step_stall():
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.methods.csf.grid_functional import GridCSFFunctional
    from atomx.numerics.shooting.electron_space import continuum_enriched_electron_space
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    from atomx.numerics.shooting.orbital_optimization import optimize_grid_shooting
    from atomx.physics.csf.matrix import GridCIWorkspace
    r, dr, n = sgrid(20., 1e-7, 20/2400, 4000)
    nucleus = ShootingHFSolver(6, [], nuclear='fermi')
    f = GridCSFFunctional(enumerate_jj_csfs_many(['1s2']),
        GridCIWorkspace(r, dr, nucleus._nuclear_rv(r, dr, n)), 6)
    space, waves = continuum_enriched_electron_space(f, n_continuum=48)
    _, ci, report = optimize_grid_shooting(f, waves, space,
        {(1, -1): 1.1*min(space.root_energies[-1])}, max_iter=100, tolerance=1e-6)
    assert report['converged']
    assert report['orbital_residual'] < 1e-6
    assert report['electron_space_leakage'] < 1e-12
    assert ci.ground_energy == pytest.approx(-32.37305034815837, abs=1e-9)


def test_heavy_negative_box_matching_preserves_core_amplitude():
    """Z=98, R=75 spans >1000 tail log units; global rescaling lost the core."""
    from atomx.numerics.shooting.dirac_bound import dbound
    from atomx.physics.dirac import hydrogenic_dirac_energy
    from atomx.potentials.radial_nucleus import nuclear_rv
    r, dr, n = sgrid(75., 1e-7, 75./2400, 6000)
    rv = nuclear_rv(98, r, dr, n, nuclear='fermi')
    solver = BoxDiracContinuum(r, dr, rv)
    cutoff = .99*hydrogenic_dirac_energy(7, -1, 98)
    energies, waves, report = solver.negative_roots_above(-1, cutoff, expected_below=7)
    # Independent boundary condition: dbound uses the decaying infinite-domain
    # tail, while negative_roots_above matches P(R)=0 in the full finite box.
    bound, error, *_ = dbound(-80., 1e-11, 8, -1, r, rv,
                              *solver.potential, n, r, n)
    assert error == 0
    assert energies[0] == pytest.approx(bound, abs=1e-8)
    assert report['matched_negative_roots'] == 70
    assert report['maximum_matching_defect'] < 1e-9
    assert np.all(np.isfinite(waves))
    np.testing.assert_allclose(waves[:, 0, -1], 0., atol=1e-15)
    gram = np.einsum('iar,r,jar->ij', waves, solver.weights, waves)
    np.testing.assert_allclose(gram, np.eye(len(waves)), atol=1e-8)
