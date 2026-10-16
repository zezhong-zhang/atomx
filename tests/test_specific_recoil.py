"""Independent hydrogenic momentum/exchange checks for leading SMS."""
import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.corrections.specific_recoil import momentum_recoil_matrix, vinti_matrix
from atomx.numerics.radial_workspace import make_grid_workspace


def hydrogenic(z):
    r = np.geomspace(1e-8/z, 70/z, 16000)
    w = np.empty_like(r)
    w[1:-1] = (r[2:]-r[:-2])/2
    w[0], w[-1] = (r[1]-r[0])/2, (r[-1]-r[-2])/2
    s = 2*z**1.5*r*np.exp(-z*r)
    p = z**2.5/np.sqrt(24)*r*r*np.exp(-z*r/2)
    zero = np.zeros_like(r)
    return {(1, -1): (s, zero), (2, 1): (p, zero), (2, -2): (p, zero)}, make_grid_workspace(r, w)


@pytest.mark.parametrize('z', [1, 2, 20])
def test_analytic_vinti_and_singlet_triplet_exchange(z):
    orbitals, workspace = hydrogenic(z)
    keys = [(1, -1), (2, 1), (2, -2)]
    v = vinti_matrix([(*orbitals[k], k[1]) for k in keys], workspace)
    expected = 32*z/(27*np.sqrt(6))
    assert v[0, 1] == pytest.approx(expected, rel=5e-7)
    assert v[1, 0] == pytest.approx(-expected, rel=5e-7)
    csfs = [c for c in enumerate_jj_csfs('1s1 2p1') if c.total_J2 == 2]
    matrix = momentum_recoil_matrix(csfs, orbitals, workspace)
    # Independently: the spatial singlet/triplet exchange is +/- |<s|p|p_m>|².
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix),
                               [-expected**2/3, expected**2/3], rtol=1e-6)
    closed = momentum_recoil_matrix(enumerate_jj_csfs('1s2'), orbitals, workspace)
    np.testing.assert_array_equal(closed, 0.)
    one = momentum_recoil_matrix(enumerate_jj_csfs('1s1'), orbitals, workspace)
    np.testing.assert_array_equal(one, 0.)


@pytest.mark.parametrize('ka', [-3, -2, -1, 1, 2, 3])
@pytest.mark.parametrize('kb', [-3, -2, -1, 1, 2, 3])
def test_reduced_spin_against_explicit_spinor_coupling(ka, kb):
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan, wigner_3j

    from atomx.corrections.specific_recoil import reduced_sigma
    la, lb = (k if k > 0 else -k-1 for k in (ka, kb))
    ja, jb = Rational(2*abs(ka)-1, 2), Rational(2*abs(kb)-1, 2)
    m = Rational(1, 2)
    element = 0.
    if la == lb:
        for spin in (m, -m):
            ml = m-spin
            element += float(2*spin*clebsch_gordan(la, m, ja, ml, spin, m)
                             * clebsch_gordan(lb, m, jb, ml, spin, m))
    angular = float((-1)**(ja-m)*wigner_3j(ja, 1, jb, -m, 0, m))
    assert reduced_sigma(ka, kb)*angular == pytest.approx(element, abs=1e-13)


def test_relativistic_terms_vanish_without_small_components():
    from atomx.corrections.specific_recoil import relativistic_vinti_matrix
    orbitals, workspace = hydrogenic(2)
    waves = [(*pq, k[1]) for k, pq in orbitals.items()]
    np.testing.assert_allclose(relativistic_vinti_matrix(waves, workspace, 2),
                               vinti_matrix(waves, workspace), atol=0, rtol=0)


def test_relativistic_sp_half_radial_cancellation():
    from atomx.constants import C
    from atomx.corrections.specific_recoil import relativistic_vinti_matrix
    z, epsilon = 2, .01
    orbitals, workspace = hydrogenic(z)
    waves = [(orbitals[k][0], -epsilon*orbitals[k][0], k[1])
             for k in [(1, -1), (2, 1)]]
    leading = vinti_matrix(waves, workspace)
    full = relativistic_vinti_matrix(waves, workspace, z)
    # For s1/2 -> p1/2, sigma1/CRE=1 and sigma2/CRE=-3.
    # SMS2+SMS3 cancels Q_s P_p and leaves -4Z/c int(P_s Q_p/r).
    # Analytically int(P_1s P_2p/r) = 16 Z/(27 sqrt(6)).
    expected = 4*z*epsilon/C * 16*z/(27*np.sqrt(6))
    assert full[0, 1]-leading[0, 1] == pytest.approx(expected, rel=5e-7)
    np.testing.assert_allclose(full, -full.T, atol=1e-11)


def test_grasp_sigma_source_phases():
    from sympy import Rational
    from sympy.physics.wigner import racah

    from atomx.corrections.specific_recoil import reduced_sigma
    # Direct transcription of RIS4 SIGMA_1/SIGMA_2 IPAR=2 phase factors.
    for ka in (-4, -3, -2, -1, 1, 2, 3, 4):
        for kb in (-4, -3, -2, -1, 1, 2, 3, 4):
            ja, jb = 2*abs(ka)-1, 2*abs(kb)-1
            la, lb = (k if k > 0 else -k-1 for k in (ka, kb))
            for l, allowed, pair in ((ja-la, ja-la == lb, (-ka, kb)),
                                      (la, jb-lb == la, (ka, -kb))):
                expected = 0.
                if allowed:
                    half = Rational(1, 2)
                    expected = float(racah(half, ja*half, half, jb*half, l, 1))
                    if (1+ja+1+jb) % 4 == 0:
                        expected = -expected
                    expected *= np.sqrt(6*(jb+1))
                    if (2*l+1+ja+2) % 4 == 0:
                        expected = -expected
                    expected *= np.sqrt(ja+1)
                assert reduced_sigma(*pair) == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize('relativistic', [False, True])
def test_configuration_average_matches_independent_csf_trace(relativistic):
    from atomx.corrections.specific_recoil import configuration_average_recoil
    orbitals, workspace = hydrogenic(2)
    orbitals = {k: (p, -.01*p) for k, (p, _) in orbitals.items()}
    z = 2 if relativistic else None
    csfs = enumerate_jj_csfs('1s1 2p1')
    matrix = momentum_recoil_matrix(csfs, orbitals, workspace, Z=z)
    weights = np.array([c.total_J2+1 for c in csfs])
    trace_average = np.sum(weights*np.diag(matrix))/np.sum(weights)
    # Independently averaged p shell: 1/3 in p1/2 and 2/3 in p3/2.
    occupancy = {(1, -1): 1., (2, 1): 1/3, (2, -2): 2/3}
    average = configuration_average_recoil(
        [(*pq, occupancy[k], k[1]) for k, pq in orbitals.items()], workspace, Z=z)
    assert average == pytest.approx(trace_average, rel=1e-13)


def test_configuration_mixture_uses_pair_occupations():
    from atomx.config.jj_coupling import JJConfig
    from atomx.corrections.specific_recoil import configuration_average_recoil
    from atomx.methods.dirac_fock import Subshell
    orbitals, workspace = hydrogenic(2)
    keys = [(1, -1), (2, 1)]
    payload = [(*orbitals[k], 1., k[1], Subshell(*k, 1.).label) for k in keys]
    mixture = [JJConfig(subshells=[Subshell(1, -1, 2.), Subshell(2, 1, 0.)], weight=.5),
               JJConfig(subshells=[Subshell(1, -1, 0.), Subshell(2, 1, 2.)], weight=.5)]
    assert configuration_average_recoil(payload, workspace, jj_configs=mixture) == 0.
    assert configuration_average_recoil(payload, workspace) < 0.


def test_sms_stack_requires_mass():
    from atomx.corrections.stack import CorrectionStack
    with pytest.raises(ValueError, match='requires nuclear_mass'):
        CorrectionStack(include_sms=True)


def test_dhf_sms_stack_matches_adapter():
    from atomx import Atom, Corrections
    from atomx.corrections.adapters import SpecificMassShiftCorrection
    from atomx.corrections.stack import CorrectionStack
    common = dict(se_backend=None, include_vp=False, include_breit=False,
                  nuclear_mass_me=20000.)
    bare = Atom(Z=10, config='1s2 2s2 2p6', engine='bspline',
                corrections=Corrections(stack=CorrectionStack(**common))).solve().result
    shifted = Atom(Z=10, config='1s2 2s2 2p6', engine='bspline', corrections=Corrections(
        stack=CorrectionStack(**common, include_sms=True))).solve().result
    expected = SpecificMassShiftCorrection(20000.).evaluate(bare).energy_shift
    assert expected < 0
    assert shifted.total_energy-bare.total_energy == pytest.approx(expected, abs=1e-11)
    assert shifted.energy_components['sms'] == pytest.approx(expected, rel=1e-12)


def test_mcdhf_sms_stack_tracks_all_levels_and_adapter():
    from atomx import Atom, Corrections
    from atomx.corrections.adapters import SpecificMassShiftCorrection
    from atomx.corrections.specific_recoil import momentum_recoil_parameters
    from atomx.corrections.stack import CorrectionStack
    common = dict(se_backend=None, include_vp=False, include_breit=False,
                  nuclear_mass_me=20000.)
    kwargs = dict(Z=10, charge=8, config=['1s1 2p1'], engine='bspline')
    bare = Atom(**kwargs, corrections=Corrections(stack=CorrectionStack(**common))).solve().result
    shifted = Atom(**kwargs, corrections=Corrections(
        stack=CorrectionStack(**common, include_sms=True))).solve().result
    corrections = momentum_recoil_parameters(
        bare.csfs, bare.mixing_coefficients, bare.orbitals_dict, bare.bsp_data, Z=10)/20000.
    predicted = bare.state_energies+corrections
    order = np.argsort(predicted, kind='stable')
    np.testing.assert_allclose(shifted.state_energies, predicted[order], atol=1e-10, rtol=0)
    np.testing.assert_allclose(shifted.solver_diagnostics["_sms_state_shifts"], corrections[order], atol=1e-12)
    assert shifted.energy_components['sms'] == pytest.approx(corrections[order[0]])
    assert SpecificMassShiftCorrection(20000.).evaluate(bare).energy_shift == pytest.approx(corrections[0])


def test_explicit_dirac_fock_method_retains_sms_option():
    from atomx import Atom
    from atomx.methods.dirac_fock import DiracFock
    result = Atom(Z=10, config='1s2 2s2 2p6', engine='bspline').solve(
        DiracFock(nuclear_mass_me=20000., include_sms=True)).result
    assert result.energy_components['sms'] < 0
    assert result.solver_diagnostics['_sms_provenance']['nuclear_mass_me'] == 20000.


@pytest.mark.parametrize('z', [1, 2, 20])
def test_closed_shell_sms_property_matches_analytic_magnetic_substate_sum(z):
    from atomx.properties.isotope_shift import compute_sms_abinitio
    orbitals, workspace = hydrogenic(z)
    keys = [(1,-1),(2,1),(2,-2)]
    payload = [(*orbitals[k], 2*abs(k[1]), k[1], str(k)) for k in keys]
    sms = compute_sms_abinitio(payload, workspace)
    # Explicit spin/m sum for occupied 1s^2 2p^6: sum angular squares = 2.
    # Each closed shell has occupation fraction one, regardless of its size.
    momentum = 32*z/(27*np.sqrt(6))
    assert sms.K_SMS_au == pytest.approx(-2*momentum**2, rel=1e-6)
    assert sms.n_pairs == 2
    assert sms.pair_contributions[str(keys[0]),str(keys[2])] == pytest.approx(
        2*sms.pair_contributions[str(keys[0]),str(keys[1])], rel=1e-13)
    with pytest.raises(ValueError, match='fully occupied'):
        compute_sms_abinitio([(*orbitals[keys[0]], 1, -1, '1s')], workspace)


@pytest.mark.parametrize('ka,kb', [(-1,1),(-1,-2),(1,2),(-2,2),(-2,-3)])
def test_recoil_spherical_harmonic_phase_matches_explicit_spinor_integral(ka,kb):
    """The C denominator must have the same tensor convention as sigma."""
    from sympy import Rational
    from sympy.physics.wigner import clebsch_gordan, wigner_3j

    from atomx.angular.coefficients import reduced_cl_j2
    half = Rational(1,2)
    ja, jb = abs(ka)-half, abs(kb)-half
    la, lb = (k if k>0 else -k-1 for k in (ka,kb))
    value = 0.
    for spin in (half,-half):
        ml = half-spin
        c = (-1)**ml*np.sqrt((2*la+1)*(2*lb+1))*float(
            wigner_3j(la,1,lb,0,0,0)*wigner_3j(la,1,lb,-ml,0,ml))
        value += float(c*clebsch_gordan(la,half,ja,ml,spin,half)
                       *clebsch_gordan(lb,half,jb,ml,spin,half))
    factor = float((-1)**(ja-half)*wigner_3j(ja,1,jb,-half,0,half))
    assert reduced_cl_j2(2*abs(ka)-1,2,2*abs(kb)-1)*factor == pytest.approx(value, abs=1e-14)


@pytest.mark.parametrize('z', [10,82])
def test_leading_and_breit_vinti_against_live_fac(z):
    import json
    from pathlib import Path

    from atomx.config.subshell import Subshell
    from atomx.corrections.specific_recoil import recoil_radial_matrices
    from atomx.numerics.shooting.local import ShootingSolver
    reference = json.loads((Path(__file__).parent/'data/fac_vinti_reference.json').read_text())
    rows = next(c['rows'] for c in reference['cases'] if c['z']==z)
    keys = sorted({tuple(r[side]) for r in rows for side in ('bra','ket')})
    states = [ShootingSolver(z,[Subshell(n,k,1)],nuclear='point',n_grid=2000,
                              r_max=125/z).solve(tol=1e-11) for n,k in keys]
    assert all(s.converged for s in states)
    waves = [(s._pa[0,1:],s._qa[0,1:],k[1]) for k,s in zip(keys,states, strict=False)]
    leading, breit = recoil_radial_matrices(waves,states[0].radial_state.correction_workspace(),Z=z)
    for row in rows:
        a,b = keys.index(tuple(row['bra'])),keys.index(tuple(row['ket']))
        v0,v1,v2 = row['fac_terms']
        # Arbitrary orbital phases cancel in absolute magnitudes and in
        # the relative Breit correction. The latter detects the unequal-j
        # phase defect even when recoil is a small fraction of momentum.
        assert abs(leading[a,b]) == pytest.approx(abs(v0),rel=5e-5)
        assert abs(breit[a,b]) == pytest.approx(abs(v0+v1+v2),rel=5e-5)
        fraction = (breit[a,b]-leading[a,b])/leading[a,b]
        assert fraction == pytest.approx((v1+v2)/v0,rel=1e-4,abs=1e-10)
