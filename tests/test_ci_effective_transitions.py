"""Two-endpoint metric consumption, signed determinants and native photons."""
from unittest.mock import patch

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
from atomx.angular.wigner import wigner_3j
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci_effective import (
    FiniteCIModel,
    FiniteCITransitionOperator,
    build_finite_ci_effective_hamiltonian,
)
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.ci_effective import (
    build_finite_ci_electric_operator,
    finite_ci_electric_line,
)
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity


def _model(h, p, identity):
    return FiniteCIModel(h, p, reference_identity=identity,
                         csf_identities=tuple(range(len(h))), radial_binding=(identity,))


def test_rectangular_complex_transition_and_two_metrics_against_dense_states():
    hb = np.array([[-.6,.02,.04,-.06], [.02,-.2,-.03,.02],
                   [.04,-.03,1.,.17], [-.06,.02,.17,1.5]])
    hk = np.array([[.1,.2,-.15],[.2,1.,.08],[-.15,.08,2.]])
    bra, ket = _model(hb, (0,1), "bra"), _model(hk, (0,), "ket")
    b, k = bra.bloch_reduction((0,1)), ket.bloch_reduction((0,))
    o = np.array([[.1,.2j,-.3],[.2j,-.4,.5],[-.6,.1j,.2],[.7,.8,-.9j]])
    op = FiniteCITransitionOperator(o, bra, ket, operator_identity="signed:T+", unit="au")
    psi_b, psi_k = np.linalg.eigh(hb)[1][:,:2], np.linalg.eigh(hk)[1][:,:1]
    expected = psi_b.T @ o @ psi_k
    for representation in ("bloch", "orthonormal"):
        np.testing.assert_allclose(b.transition_amplitudes(op, k, representation=representation),
                                   expected, atol=1e-14, rtol=1e-14)
        weak = FiniteCITransitionOperator(1e-16*o,bra,ket,operator_identity="weak:T+",unit="au")
        np.testing.assert_allclose(b.transition_amplitudes(weak,k,representation=representation),
                                   1e-16*expected,atol=0,rtol=1e-12)
    reverse = FiniteCITransitionOperator(o.conj().T, ket, bra,
                                         operator_identity="signed:T-", unit="au")
    ob = b.effective_transition(op, k, representation="bloch")
    ok = k.effective_transition(reverse, b, representation="bloch")
    # Reverse adjoint is metric weighted, with the two distinct endpoint metrics.
    np.testing.assert_allclose(k.metric @ ok, ob.conj().T @ b.metric, atol=1e-14)
    assert np.max(abs(ok-ob.conj().T)) > 1e-5
    np.testing.assert_allclose(k.transition_amplitudes(reverse, b, representation="bloch"),
                               expected.conj().T, atol=1e-14)
    assert np.max(abs(expected-b.left_states @ o[:2,:1] @ k.right_states)) > .01
    pb, pk = np.diag([-1.,1.,-1.,1.]), np.diag([1.,-1.,1.])
    fb, fk = _model(pb @ hb @ pb, (0,1), "bra phase"), _model(pk @ hk @ pk, (0,), "ket phase")
    fop = FiniteCITransitionOperator(pb @ o @ pk, fb, fk, operator_identity="phase", unit="au")
    actual = fb.bloch_reduction((0,1)).effective_transition(
        fop, fk.bloch_reduction((0,)), representation="bloch")
    np.testing.assert_allclose(actual, pb[:2,:2] @ ob @ pk[:1,:1], atol=1e-14)
    o[:] = 100
    assert op.matrix[0,0] == .1
    with pytest.raises(ValueError):
        op.matrix.flags.writeable = True
    clone = _model(hb, (0,1), "bra")
    with pytest.raises(ValueError, match="snapshot"):
        clone.bloch_reduction((0,1)).effective_transition(op, k, representation="bloch")
    with pytest.raises(MemoryError, match="cap"):
        b.transition_amplitudes(op, k, representation="bloch", memory_bytes=1)
    with pytest.raises(ValueError, match="explicit"):
        b.effective_transition(op, k, representation="guess")
    with pytest.raises(ValueError, match="rectangular"):
        FiniteCITransitionOperator(np.ones((3,4)), bra, ket, operator_identity="wrong", unit="au")


@pytest.fixture(scope="module")
def native_panel():
    core = DFSolver(Z=4, subshells=[Subshell(1,-1,2)], ns=30, ks=7).solve()
    od = {(1,-1):core.orbitals_by_subshell()[1,-1][:2]}
    for k in (-1,1,-2):
        _, states = build_virtual_basis(core, k, e_max=50, potential_mode="VN",
                                        orthogonal_to_occupied=True)
        for i, state in enumerate(states[:2 if k == -1 else 1]):
            od[i+2,k] = state[:2]
    all_csfs = enumerate_jj_orbital_space(od, 3, magnetic_state_cap=256)
    panel = {}
    for j2, parity in ((1,1),(1,-1),(3,-1)):
        csfs = tuple(c for c in all_csfs if (c.total_J2,c.parity) == (j2,parity))
        p = tuple(i for i,c in enumerate(csfs)
                  if any((s.n,s.kappa,s.occupation) == (1,-1,2) for s in c.subshells))
        model = build_finite_ci_effective_hamiltonian(
            csfs, p, od, core._bsp_data, 4, 30, reference_identity=f"Be+:E1:{j2}:{parity}",
            nuclear_potential=core._nuclear_potential)
        panel[j2,parity] = (csfs, model.bloch_reduction(tuple(range(len(p)))))
    return core, od, panel


def determinant_e1_matrix(bra_csfs, ket_csfs, od, workspace, omega, gauge, approximation):
    """Independent magnetic Slater action, using lowest-M CSFs and full signs.

    Does not call one_body_tensor_terms or any effective-property transform.
    Radial/angular single-electron kernels define the common tested operator.
    """
    keys = tuple(sorted(od))
    spinors = [(n,k,m) for n,k in keys for m in range(-2*abs(k)+1,2*abs(k),2)]
    jf, ji = bra_csfs[0].total_J2, ket_csfs[0].total_J2
    mf, mi = -jf, -ji
    q2 = mf-mi
    denominator = (-1)**((jf-mf)//2)*wigner_3j(jf,2,ji,-mf,q2,mi)
    assert denominator != 0
    radial = {}
    one_body = np.zeros((len(spinors),len(spinors)))
    for ia,(na,ka,ma) in enumerate(spinors):
        for ib,(nb,kb,mb) in enumerate(spinors):
            if ma != mb+q2:
                continue
            ang = angular_eL(kb,ka,1)
            if ang == 0:
                continue
            pair = ((na,ka),(nb,kb))
            if pair not in radial:
                args = (*od[nb,kb],*od[na,ka],workspace,1)
                if gauge == "velocity":
                    r = radial_eL_velocity(*args, kappa_i=kb,kappa_f=ka,omega=omega,
                                           long_wavelength=approximation == "long_wavelength")
                else:
                    r = radial_eL(*args,kappa_i=kb,kappa_f=ka,
                                   omega=omega if approximation == "finite_wavelength" else None)
                radial[pair] = ang*r
            one_body[ia,ib] = ((-1)**((2*abs(ka)-1-ma)//2)
                               * wigner_3j(2*abs(ka)-1,2,2*abs(kb)-1,-ma,q2,mb)*radial[pair])
    vb = [dict(csf_magnetic_vector(csf_signature(c),mf,keys)) for c in bra_csfs]
    vk = [dict(csf_magnetic_vector(csf_signature(c),mi,keys)) for c in ket_csfs]
    matrix = np.zeros((len(vb),len(vk)))
    for j, vector in enumerate(vk):
        action = {}
        for mask, value in vector.items():
            for ib in range(len(spinors)):
                if not mask & (1 << ib):
                    continue
                removed = mask ^ (1 << ib)
                sign_b = (-1)**(mask & ((1 << ib)-1)).bit_count()
                for ia in range(len(spinors)):
                    if removed & (1 << ia) or one_body[ia,ib] == 0:
                        continue
                    final = removed | (1 << ia)
                    sign_a = (-1)**(removed & ((1 << ia)-1)).bit_count()
                    action[final] = action.get(final,0.) + value*sign_b*sign_a*one_body[ia,ib]
        for i, vector in enumerate(vb):
            matrix[i,j] = sum(v*action.get(mask,0.) for mask,v in vector.items())/denominator
    return matrix


@pytest.mark.parametrize("j2", [1,3])
@pytest.mark.parametrize("gauge", ["length","velocity"])
@pytest.mark.parametrize("approximation", ["finite_wavelength","long_wavelength"])
def test_native_signed_lines_against_independent_magnetic_determinants(native_panel,j2,gauge,approximation):
    core, od, panel = native_panel
    lc, lower = panel[1,1]
    uc, upper = panel[j2,-1]
    opts = dict(representation="bloch", gauge=gauge, photon_approximation=approximation,
                nuclear_potential=core._nuclear_potential)
    line = finite_ci_electric_line(upper,lower,uc,lc,od,core._bsp_data,4,30,**opts)
    oracle = determinant_e1_matrix(uc,lc,od,core._bsp_data,line.omega_Ha,gauge,approximation)
    np.testing.assert_allclose(line.full_operator.matrix,oracle,atol=1e-13,rtol=1e-13)
    psi_u = np.linalg.eigh(upper.parent.hamiltonian_Ha)[1][:,:1]
    psi_l = np.linalg.eigh(lower.parent.hamiltonian_Ha)[1][:,:2]
    expected = psi_u.T @ oracle @ psi_l
    actual = upper.transition_amplitudes(line.full_operator,lower,representation="bloch")
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(upper.transition_amplitudes(line.full_operator,lower,
                                                         representation="orthonormal"),
                               expected,atol=1e-13,rtol=1e-13)
    assert line.reduced_matrix_element_au == pytest.approx(expected[0,0],abs=1e-13)
    assert line.line_strength == pytest.approx(expected[0,0]**2,rel=1e-13)
    # Independent E1 radiation prefactor and absorption statistical weight.
    from atomx.constants import ALPHA, AU_RATE_PER_S
    assert line.A_rate_per_s == pytest.approx(4/3*ALPHA**3*line.omega_Ha**3
                                             *line.line_strength/(j2+1)*AU_RATE_PER_S,rel=1e-13)
    assert line.gf_value == pytest.approx(2/3*line.omega_Ha*line.line_strength,rel=1e-13)
    assert line.f_value == pytest.approx(line.gf_value/2,rel=1e-13)
    reverse = build_finite_ci_electric_operator(
        lower.parent,upper.parent,lc,uc,od,core._bsp_data,4,30,omega_Ha=-line.omega_Ha,
        gauge=gauge,photon_approximation=approximation,nuclear_potential=core._nuclear_potential)
    phase = (-1)**((1-j2)//2)
    np.testing.assert_allclose(reverse.matrix,phase*line.full_operator.matrix.T,atol=1e-13,rtol=1e-13)
    bare = oracle[np.ix_(upper.parent.p_indices,lower.parent.p_indices)]
    assert np.max(abs(expected-upper.left_states @ bare @ lower.right_states)) > 1e-5


def test_native_endpoint_binding_and_resource_guards_before_tensor_work(native_panel):
    core,od,panel = native_panel
    lc,lower = panel[1,1]
    uc,upper = panel[1,-1]
    args = (upper.parent,lower.parent,uc,lc,od,core._bsp_data,4,30)
    opts = dict(omega_Ha=.1,nuclear_potential=core._nuclear_potential)
    with patch("atomx.properties.transitions.ci_effective.one_body_tensor_terms",
               side_effect=AssertionError("angular work must not start")):
        for name,value,error in (("dimension_cap",1,ValueError), ("max_determinants",1,ValueError),
                                 ("angular_work_cap",1,ValueError),("memory_bytes",1,MemoryError)):
            with pytest.raises(error,match="cap"):
                build_finite_ci_electric_operator(*args,**opts,**{name:value})
        with pytest.raises(ValueError,match="CSF order"):
            build_finite_ci_electric_operator(upper.parent,lower.parent,uc[::-1],lc,
                                               od,core._bsp_data,4,30,**opts)
        changed = dict(od)
        changed[2,-1] = tuple(-v for v in od[2,-1])
        with pytest.raises(ValueError,match="reference mismatch"):
            build_finite_ci_electric_operator(upper.parent,lower.parent,uc,lc,
                                               changed,core._bsp_data,4,30,**opts)
    with pytest.raises(ValueError,match="positive emitted"):
        finite_ci_electric_line(lower,upper,lc,uc,od,core._bsp_data,4,30,representation="bloch")
    with pytest.raises(ValueError,match="state index"):
        finite_ci_electric_line(upper,lower,uc,lc,od,core._bsp_data,4,30,
                                representation="bloch",upper_state=True)
    with pytest.raises(ValueError,match="nonzero"):
        build_finite_ci_electric_operator(*args,**{**opts,"omega_Ha":0})


def test_selected_second_root_has_its_own_photon_and_upper_statistical_weight(native_panel):
    core,od,panel = native_panel
    uc,upper = panel[1,1]
    lc,lower = panel[3,-1]
    line = finite_ci_electric_line(upper,lower,uc,lc,od,core._bsp_data,4,30,
                                   representation="orthonormal",upper_state=1,
                                   nuclear_potential=core._nuclear_potential)
    assert line.upper_root_index == 1 and line.lower_root_index == 0
    assert line.omega_Ha == upper.energies_Ha[1]-lower.energies_Ha[0]
    assert line.upper_two_J == 1 and line.lower_two_J == 3
    oracle = determinant_e1_matrix(uc,lc,od,core._bsp_data,line.omega_Ha,"length","finite_wavelength")
    psi_u = np.linalg.eigh(upper.parent.hamiltonian_Ha)[1][:,1]
    psi_l = np.linalg.eigh(lower.parent.hamiltonian_Ha)[1][:,0]
    assert line.reduced_matrix_element_au == pytest.approx(psi_u @ oracle @ psi_l,abs=1e-13)
    assert line.f_value == pytest.approx(line.gf_value/4,rel=1e-13)
    with pytest.raises(ValueError,match="finite"):
        FiniteCITransitionOperator(np.full_like(oracle,np.nan),upper.parent,lower.parent,
                                     operator_identity="bad",unit="a0")


def test_small_component_of_normalized_degenerate_angular_state_is_retained():
    """A valid rotation inside a multiplicity-two g^3 J=9/2 subspace.

    The rotation nearly cancels one determinant coefficient, while keeping
    the entire normalized SU(2) multiplet. This is a physical angular basis
    change, not an unnormalized amplitude injected into the constructor.
    """
    import math

    from atomx.angular.csf_hamiltonian import _csf_local_vector, _subshell_series

    first = {m:dict(v) for m,v in _subshell_series(9,3,9,0)}
    second = {m:dict(v) for m,v in _subshell_series(9,3,9,1)}
    mask = 224
    angle = math.atan2(-first[9][mask],second[9][mask])
    angle = np.nextafter(angle,np.inf)
    c,s = math.cos(angle),math.sin(angle)
    mixed = tuple((m,tuple((d,c*first[m].get(d,0)+s*second[m].get(d,0))
                           for d in sorted(set(first[m])|set(second[m])))) for m in sorted(first))
    expected = dict(dict(mixed)[9])[mask]
    assert 0 < abs(expected) < 1e-14
    for _,vector in mixed:
        assert sum(a*a for _,a in vector) == pytest.approx(1,abs=1e-13)
    original = _subshell_series

    def rotated(tj,q,j2,deg):
        return mixed if (tj,q,j2,deg) == (9,3,9,0) else original(tj,q,j2,deg)

    _csf_local_vector.cache_clear()
    try:
        with patch("atomx.angular.csf_hamiltonian._subshell_series",side_effect=rotated):
            actual = dict(_csf_local_vector(((5,-5,3),),9,((9,9,0),)))
        assert actual[mask,] == pytest.approx(expected,rel=1e-14,abs=0)
        assert sum(a*a for a in actual.values()) == pytest.approx(1,abs=1e-13)
    finally:
        _csf_local_vector.cache_clear()
