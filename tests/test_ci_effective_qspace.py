"""Finite P/Q propagation: full-Hamiltonian states, signs and disjoint spaces."""
from unittest.mock import patch

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci_effective import (
    FiniteCIModel,
    build_finite_ci_effective_hamiltonian,
    static_breit_second_order_control,
)
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis


def test_signed_q_mixing_and_state_residue_against_full_hamiltonian():
    h = np.array([[-.6,.02,.04,-.06], [.02,-.2,-.03,.02],
                  [.04,-.03,1.,.17], [-.06,.02,.17,1.5]])
    model = FiniteCIModel(h,(0,1),reference_identity='analytic:signed:2P2Q',
                          csf_identities=('a','b','c','d'),radial_binding=('analytic',))
    en, vec = np.linalg.eigh(h)
    for index in (0,1):
        root = model.solve((-1.,.4),reference_index=index)
        assert root.energy_Ha == pytest.approx(en[index],abs=1e-13)
        np.testing.assert_allclose(np.outer(root.coefficients,root.coefficients),
                                   np.outer(vec[:,index],vec[:,index]),atol=1e-13)
        assert root.p_weight == pytest.approx(np.sum(vec[:2,index]**2),abs=1e-13)
        assert root.residual_Ha < 1e-13
        assert 0 < root.p_weight < 1
        with pytest.raises(ValueError):
            root.coefficients.flags.writeable=True
    energy=-.5
    hpq=h[:2,2:]
    hqq=h[2:,2:]
    expected=h[:2,:2]+hpq@np.linalg.solve(energy*np.eye(2)-hqq,hpq.T)
    np.testing.assert_allclose(model.effective_hamiltonian(energy),expected,atol=1e-15)
    assert np.max(abs(model.effective_hamiltonian(energy)-model.diagonal_q_control(energy)))>1e-5
    step=1e-5
    np.testing.assert_allclose(model.derivative(energy),
        (model.effective_hamiltonian(energy+step)-model.effective_hamiltonian(energy-step))/(2*step),rtol=1e-7,atol=1e-11)
    assert np.linalg.eigvalsh(model.derivative(energy))[-1]<0
    phase=np.diag([-1.,1.,-1.,1.])
    flipped=FiniteCIModel(phase@h@phase,(0,1),reference_identity='phase',
                          csf_identities=('a','b','c','d'),radial_binding=('phase',))
    original=model.solve((-1.,.4))
    other=flipped.solve((-1.,.4))
    assert other.energy_Ha == pytest.approx(original.energy_Ha,abs=1e-13)
    np.testing.assert_allclose(np.outer(other.coefficients,other.coefficients),
        phase@np.outer(original.coefficients,original.coefficients)@phase,atol=1e-13)
    # An owned snapshot cannot be changed through its original input array.
    h[0,0]=10
    assert model.hamiltonian_Ha[0,0]==-.6


def test_empty_q_and_exactly_uncoupled_coincident_q_pole():
    h=np.diag([-.5,-.5,2.])
    model=FiniteCIModel(h,(0,),reference_identity='uncoupled',
                        csf_identities=(0,1,2),radial_binding=('analytic',))
    root=model.solve((-.7,-.3))
    assert root.energy_Ha==-.5 and root.p_weight==1
    np.testing.assert_array_equal(root.coefficients,[1,0,0])
    no_q=FiniteCIModel([[-.5]],(0,),reference_identity='P-only',
                       csf_identities=(0,),radial_binding=('analytic',))
    assert no_q.solve((-.7,-.3)).energy_Ha==-.5
    np.testing.assert_array_equal(no_q.diagonal_q_control(-.5),[[-.5]])


def test_exact_bloch_metric_signed_properties_and_phase_covariance():
    h = np.array([[-.6,.02,.04,-.06], [.02,-.2,-.03,.02],
                  [.04,-.03,1.,.17], [-.06,.02,.17,1.5]])
    opts = dict(reference_identity='Bloch:full:signed', csf_identities=tuple(range(4)),
                radial_binding=('analytic-signed',))
    parent = FiniteCIModel(h, (0,1), **opts)
    reduction = parent.bloch_reduction((0,1))
    energy, full = np.linalg.eigh(h)
    psi = full[:,:2]
    assert np.max(abs(reduction.directed_Ha-reduction.directed_Ha.T)) > 1e-5
    np.testing.assert_allclose(reduction.wave_operator @ reduction.right_states, psi, atol=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(reduction.orthonormal_Ha), energy[:2], atol=1e-14)
    np.testing.assert_allclose(reduction.directed_Ha.T @ reduction.metric,
                               reduction.metric @ reduction.directed_Ha, atol=1e-14)
    # Physical signed property includes Q contributions and complex phases.
    operator = np.array([[0,.1,.2j,-.3],[-.1,1,.4,.2j],
                         [.3,.2j,-.4,.6],[.2j,-.1,.5,0]])
    effective = reduction.effective_operator(operator, representation='bloch')
    np.testing.assert_allclose(reduction.left_states @ effective @ reduction.right_states,
                               psi.T @ operator @ psi, atol=1e-14)
    w = reduction.orthonormal_wave_operator
    orthogonal_states = w.T @ psi
    effective_orthogonal = reduction.effective_operator(operator, representation='orthonormal')
    np.testing.assert_allclose(orthogonal_states.T @ effective_orthogonal @ orthogonal_states,
                               psi.T @ operator @ psi, atol=1e-14)
    assert np.max(abs(effective-operator[:2,:2])) > .001
    phase = np.diag([-1.,1.,-1.,1.])
    other = FiniteCIModel(phase @ h @ phase, (0,1), **opts).bloch_reduction((0,1))
    np.testing.assert_allclose(other.metric, phase[:2,:2] @ reduction.metric @ phase[:2,:2], atol=1e-14)
    np.testing.assert_allclose(other.directed_Ha, phase[:2,:2] @ reduction.directed_Ha @ phase[:2,:2], atol=1e-14)
    np.testing.assert_allclose(other.effective_operator(phase @ operator @ phase, representation='bloch'),
                               phase[:2,:2] @ effective @ phase[:2,:2], atol=1e-14)
    for a in (reduction.wave_operator, reduction.metric, effective):
        with pytest.raises(ValueError):
            a.flags.writeable = True


def test_bloch_rejects_split_degeneracy_singular_projection_and_caps():
    opts = dict(reference_identity='Bloch:guards', csf_identities=(0,1,2), radial_binding=('analytic',))
    degenerate = FiniteCIModel(np.diag([0.,0.,1.]), (0,), **opts)
    with pytest.raises(ValueError, match='degenerate'):
        degenerate.bloch_reduction((0,))
    singular = FiniteCIModel(np.diag([2.,0.,1.]), (0,), **opts)
    with pytest.raises(ValueError, match='projection'):
        singular.bloch_reduction((0,))
    complete = FiniteCIModel(np.diag([0.,0.,1.]), (0,1), **opts)
    np.testing.assert_array_equal(complete.bloch_reduction((0,1)).metric, np.eye(2))
    for indices in ((0,), (0,0), (0,3), (False,1)):
        with pytest.raises(ValueError, match='root'):
            complete.bloch_reduction(indices)
    with pytest.raises(MemoryError, match='cap'):
        complete.bloch_reduction((0,1), memory_bytes=1)
    with pytest.raises(ValueError, match='caps'):
        complete.bloch_reduction((0,1), dimension_cap=2)
    with pytest.raises(ValueError, match='controls'):
        complete.bloch_reduction((0,1), residual_tolerance_Ha=True)
    reduction = complete.bloch_reduction((0,1))
    with pytest.raises(ValueError, match='full finite'):
        reduction.effective_operator(np.eye(2), representation='bloch')
    with pytest.raises(ValueError, match='explicit'):
        reduction.effective_operator(np.eye(3), representation='unknown')


def test_no_pole_crossing_or_ambiguous_branch():
    model=FiniteCIModel([[-.5,.1],[.1,0]],(0,),reference_identity='pole',
                        csf_identities=(0,1),radial_binding=('analytic',))
    with pytest.raises(ValueError,match='pole'):
        model.solve((-1.,1.))
    with pytest.raises(ValueError,match='symmetric'):
        FiniteCIModel([[0,1],[0,0]],(0,),reference_identity='bad',
                       csf_identities=(0,1),radial_binding=('analytic',))


@pytest.fixture(scope='module')
def be_reference():
    core=DFSolver(Z=4,subshells=[Subshell(1,-1,2)],ns=30,ks=7).solve()
    od={(1,-1):core.orbitals_by_subshell()[1,-1][:2]}
    for k in (-1,1,-2):
        _,virtual=build_virtual_basis(core,k,e_max=50,potential_mode='VN',orthogonal_to_occupied=True)
        od[2,k]=virtual[0][:2]
    all_csfs=enumerate_jj_orbital_space(od,3,magnetic_state_cap=256)
    return core,od,all_csfs


def test_native_core_holes_and_complete_small_atom_anchor(be_reference):
    core,od,all_csfs=be_reference
    for j2,parity in ((1,1),(1,-1),(3,-1)):
        csfs=[c for c in all_csfs if (c.total_J2,c.parity)==(j2,parity)]
        p=tuple(i for i,c in enumerate(csfs) if any(sh.n==1 and sh.kappa==-1 and sh.occupation==2
                                                   for sh in c.subshells))
        assert len(p)==1
        model=build_finite_ci_effective_hamiltonian(csfs,p,od,core._bsp_data,4,30,
            reference_identity=f'Be+:complete-small:{j2}:{parity}',nuclear_potential=core._nuclear_potential)
        eps=model.self_energy.reference_energies_Ha[0]
        # A predeclared below-Q interval around the bare P reference.
        bracket=(eps-.2,eps+.01)
        root=model.solve(bracket)
        en,vec=np.linalg.eigh(model.hamiltonian_Ha)
        column=int(np.argmin(abs(en-root.energy_Ha)))
        reduction = model.bloch_reduction((column,))
        assert reduction.energies_Ha[0] == pytest.approx(root.energy_Ha, abs=1e-11)
        # Same physical normalization, independently eliminated fixed-E Q.
        np.testing.assert_allclose(reduction.metric[0,0], 1/root.p_weight, atol=1e-10)
        lift = reduction.wave_operator @ reduction.right_states[:,0]
        np.testing.assert_allclose(np.outer(lift,lift), np.outer(root.coefficients,root.coefficients), atol=1e-10)
        assert root.energy_Ha==pytest.approx(en[column],abs=1e-11)
        np.testing.assert_allclose(np.outer(root.coefficients,root.coefficients),
            np.outer(vec[:,column],vec[:,column]),atol=1e-10)
        assert root.shift_Ha<0 and root.p_weight<1
        assert any(not any(sh.n==1 and sh.occupation==2 for sh in csfs[i].subshells)
                   for i in model.q_indices)


def test_partition_identity_caps_and_sector_guards_before_matrix(be_reference):
    core,od,all_csfs=be_reference
    csfs=[c for c in all_csfs if (c.total_J2,c.parity)==(1,1)]
    args=(od,core._bsp_data,4,30)
    options=dict(reference_identity='Be:guard')
    with patch('atomx.physics.csf.matrix.build_ci_matrix') as native:
        with pytest.raises(ValueError,match='disjoint'):
            build_finite_ci_effective_hamiltonian([csfs[0],csfs[0]],(0,),*args,**options)
        with pytest.raises(ValueError,match='cap'):
            build_finite_ci_effective_hamiltonian(csfs,(0,),*args,dimension_cap=1,**options)
        with pytest.raises(ValueError,match='P indices'):
            build_finite_ci_effective_hamiltonian(csfs,(0,0),*args,**options)
        with pytest.raises(ValueError,match='one N, J and parity'):
            build_finite_ci_effective_hamiltonian(all_csfs,(0,),*args,**options)
        with pytest.raises(ValueError,match='identity'):
            build_finite_ci_effective_hamiltonian(csfs,(0,),*args,reference_identity='')
        native.assert_not_called()


def test_static_breit_full_q_response_and_operator_coverage(be_reference):
    from atomx.config.csf import get_shared_orbital_basis
    from atomx.physics.csf.matrix import build_ci_matrix

    core, od, all_csfs = be_reference
    csfs = [c for c in all_csfs if (c.total_J2, c.parity) == (1, -1)]
    p = tuple(i for i, c in enumerate(csfs) if any(sh.n == 1 and sh.occupation == 2 for sh in c.subshells))
    args = (csfs, p, od, core._bsp_data, 4, 30)
    opts = dict(reference_identity='Be+:DCB-complete', nuclear_potential=core._nuclear_potential)
    dc = build_finite_ci_effective_hamiltonian(*args, **opts)
    dcb = build_finite_ci_effective_hamiltonian(*args, **opts, include_breit=True)
    assert dc.hamiltonian_terms == ('dirac-coulomb',)
    assert dcb.hamiltonian_terms == ('dirac-coulomb', 'static-breit')
    native = build_ci_matrix(csfs, od, get_shared_orbital_basis(csfs), core._bsp_data, 4, 30,
                            nuclear_potential=core._nuclear_potential, include_breit=True, as_operator=True)
    np.testing.assert_allclose(dcb.hamiltonian_Ha, native @ np.eye(len(csfs)), atol=1e-13)
    b = dcb.hamiltonian_Ha-dc.hamiltonian_Ha
    assert np.max(abs(b[np.ix_(dc.p_indices, dc.q_indices)])) > 1e-8
    assert np.max(abs(b[np.ix_(dc.q_indices, dc.q_indices)])) > 1e-8
    e0 = dcb.self_energy.reference_energies_Ha[0]
    root = dcb.solve((e0-.2, e0+.01))
    assert root.energy_Ha == pytest.approx(np.linalg.eigvalsh(dcb.hamiltonian_Ha)[0], abs=1e-11)
    assert root.residual_Ha < 1e-10
    with pytest.raises(ValueError, match='bool'):
        build_finite_ci_effective_hamiltonian(*args, **opts, include_breit=1)


def test_strict_coulomb_breit_interference_against_two_parameter_eigenenergy():
    h0 = np.array([-.7, .6, 1.2])
    c = np.array([[0., .13, -.08], [.13, .07, .04], [-.08, .04, .04]])
    b = np.array([[.009, -.014, .017], [-.014, -.007, .003], [.017, .003, .004]])

    def pair(h, operator):
        opts = dict(reference_identity='analytic:matched', csf_identities=(0, 1, 2), radial_binding=('same',))
        dc = FiniteCIModel(h, (0,), **opts, hamiltonian_terms=('dirac-coulomb',))
        dcb = FiniteCIModel(h+operator, (0,), **opts, hamiltonian_terms=('dirac-coulomb', 'static-breit'))
        return dc, dcb

    dc, dcb = pair(np.diag(h0)+c, b)
    control = static_breit_second_order_control(dc, dcb, h0)

    def energy(g, beta):
        return np.linalg.eigvalsh(np.diag(h0)+g*c+beta*b)[0]

    # Independent mixed derivative of the exact small Hamiltonian. Two
    # step sizes remove the leading finite-difference truncation error.
    def mixed(step):
        return (energy(step, step)-energy(step, -step)-energy(-step, step)+energy(-step, -step))/(4*step**2)

    estimate = (4*mixed(.0005)-mixed(.001))/3
    assert control.coulomb_breit_Ha == pytest.approx(estimate, abs=1e-9)
    assert abs(control.coulomb_breit_Ha) > 1e-3
    assert control.coulomb_second_order_Ha < 0 and control.breit_second_order_Ha < 0
    assert control.first_order_breit_Ha == pytest.approx(.009, abs=1e-15)
    phase = np.diag([-1., 1., -1.])
    other = static_breit_second_order_control(*pair(phase@(np.diag(h0)+c)@phase, phase@b@phase), h0)
    assert other.coulomb_breit_Ha == pytest.approx(control.coulomb_breit_Ha, abs=1e-15)
    with pytest.raises(ValueError):
        control.h0_diagonal_Ha.flags.writeable = True
    with pytest.raises(ValueError, match='zero'):
        static_breit_second_order_control(dc, dcb, h0+.01)
    with pytest.raises(ValueError, match='resonant'):
        static_breit_second_order_control(dc, dcb, np.array([h0[0], h0[0], h0[2]]))
    mismatched = FiniteCIModel(dcb.hamiltonian_Ha, (0,), reference_identity='wrong basis',
                              csf_identities=(0, 1, 2), radial_binding=('different',),
                              hamiltonian_terms=('dirac-coulomb', 'static-breit'))
    with pytest.raises(ValueError, match='matched'):
        static_breit_second_order_control(dc, mismatched, h0)
