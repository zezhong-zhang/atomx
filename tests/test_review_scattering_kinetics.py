"""Independent scattering/kinetics gates from the September 23 review."""

import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import quad

from atomx import Atom
from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    radiative_recombination_rate_cm3_s,
)
from atomx.scattering.r_matrix.cc_outer import (
    match_K_matrix,
    scattering_matrices_and_cross_sections,
)
from atomx.scattering.r_matrix.channel_table import Channel
from atomx.scattering.r_matrix.target_states import TargetState


def _channel(index, l_target, l_projectile, threshold, charge=0):
    return Channel(index, TargetState(str(index), l_target + 1, l_target,
                                     charge + 1., threshold, lambda r: r),
                   l_projectile, threshold)


@pytest.mark.parametrize("delta", [0., .1, -.2, .8])
def test_matching_recovers_known_free_wave_phase(delta):
    # This sets the boundary data analytically, not from the R-matrix code.
    a, k = 30., 1.2
    r_boundary = np.array([[np.tan(k*a + delta)/(a*k)]])
    result = match_K_matrix(k*k/2, [_channel(0, 0, 0, 0)], r_boundary, a)
    assert result.K_matrix[0, 0] == pytest.approx(np.tan(delta), abs=2e-13)
    assert result.eigenphases[0] == pytest.approx(delta, abs=2e-13)


def test_matching_unequal_momenta_preserves_physical_reaction_matrix():
    # Unit-flux free references for a prescribed symmetric coupled K.
    a = 17.
    k = np.array([2., 1.])
    expected = np.array([[.1, .2], [.2, -.3]])
    F, G = np.diag(np.sin(k*a)/np.sqrt(k)), np.diag(np.cos(k*a)/np.sqrt(k))
    dF, dG = np.diag(np.sqrt(k)*np.cos(k*a)), np.diag(-np.sqrt(k)*np.sin(k*a))
    U, dU = F + G@expected, dF + dG@expected
    R = np.linalg.solve(dU.T, U.T).T / a
    np.testing.assert_allclose(R, R.T, atol=1e-14)
    channels = [_channel(0, 0, 0, -2.), _channel(1, 0, 0, -.5)]
    got = match_K_matrix(0., channels, R, a)
    np.testing.assert_allclose(got.K_matrix, expected, atol=1e-13)


def test_collision_strength_reciprocity_and_absolute_spin_average():
    channels = [_channel(0, 0, 1, -2.), _channel(1, 1, 0, -.5)]
    k = np.array([2., 1.])
    K = np.array([[.1, .2], [.2, .3]])
    S, T, sigma, omega = scattering_matrices_and_cross_sections(
        channels, np.array([0, 1]), k, K, L_total=1, S_total=0.)
    np.testing.assert_allclose(S.conj().T @ S, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(omega, omega.T, atol=1e-14)
    np.testing.assert_allclose(omega, 1.5 * abs(T)**2, atol=1e-14)
    g_target = np.array([2., 6.])
    np.testing.assert_allclose(omega, g_target[:, None]*k[:, None]**2*sigma/np.pi)


@pytest.mark.parametrize("l,eta,rho", [(0,-1.,30.),(1,-1.,25.),(3,-2.,25.),
                                         (0,-1.,1.),(3,-1.,1.),(2,.5,5.)])
def test_coulomb_reference_waves_against_arbitrary_precision(l, eta, rho):
    import mpmath as mp

    from atomx.scattering.r_matrix.coulomb_fn import coulomb_FG
    with mp.workdps(35):
        expected = [mp.coulombf(l,eta,rho), mp.diff(lambda r:mp.coulombf(l,eta,r),rho),
                    mp.coulombg(l,eta,rho), mp.diff(lambda r:mp.coulombg(l,eta,r),rho)]
    F, dF, G, dG = coulomb_FG(l, eta, rho)
    np.testing.assert_allclose([F,dF,G,dG], np.array(expected, dtype=float),
                               rtol=2e-10, atol=2e-11)
    assert dF*G-F*dG == pytest.approx(1., abs=2e-10)


@pytest.mark.parametrize("temperature", [.01, 1., 13.6, 136., 1360.])
def test_spontaneous_rr_against_direct_milne_maxwell_integral(temperature):
    # Independent SI calculation; no Saha or Planck rate enters the reference.
    I, sigma0 = 13.6, 6.3e-22  # eV, m²
    me, ev, c = sc.m_e, sc.e, sc.c
    integral = quad(lambda t: sigma0*I**3/(I+temperature*t)*ev/(2*me*c*c)
                    *np.exp(-t)*temperature, 0., 100., epsabs=1e-45)[0]
    expected = np.sqrt(8/(np.pi*me))*(temperature*ev)**-1.5*ev**2*integral*1e6
    got = radiative_recombination_rate_cm3_s(I, temperature, sigma0_cm2=sigma0*1e4)
    assert got == pytest.approx(expected, rel=3e-9)


def test_stimulated_rr_requires_an_explicit_radiation_field():
    spontaneous = radiative_recombination_rate_cm3_s(13.6, 136.)
    dark = radiative_recombination_rate_cm3_s(13.6, 136., radiation_field=lambda e:0.)
    # A photon occupation of one doubles the emission probability, at all E.
    def flux(e):
        return 8*np.pi*(e*sc.e)**2/(sc.h**3*sc.c**2)*sc.e/1e4
    illuminated = radiative_recombination_rate_cm3_s(13.6, 136., radiation_field=flux)
    assert dark == spontaneous
    assert illuminated == pytest.approx(2*spontaneous, rel=1e-9)


def test_atom_crm_spontaneous_ground_stability_and_degeneracies():
    atom = Atom(2, config=["1s2", "1s1 2p1"], corrections="none")
    atom.solve()
    crm = CollisionalRadiativeModel.from_atom(atom)
    R = crm._build_rate_matrix(0., 1.)
    np.testing.assert_array_equal(R[:, 0], np.zeros(len(crm.levels)))
    assert len(crm.transitions) == 2
    assert sorted(level.g for level in crm.levels) == [1, 1, 3, 3, 5]
    for tr in crm.transitions:
        assert crm.levels[tr.upper].energy_eV > crm.levels[tr.lower].energy_eV
