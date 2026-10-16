"""Independent occupation/Slater-Condon anchors for a signed matrix consumer."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.methods.ci_mbpt import ci_mbpt_solve
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.mbpt_matrix import MatrixSelfEnergy, build_mbpt2_matrix_self_energy
from atomx.physics.csf.effective_one_body import CIEffectiveOneBody, CIReferenceBinding
from atomx.physics.csf.matrix import build_ci_matrix


@pytest.fixture(scope='module')
def reference():
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2.)], ns=30, ks=7).solve()
    _, states = build_virtual_basis(core, -1, e_max=50., potential_mode='VN', orthogonal_to_occupied=True)
    targets = [(P, Q, -1, e) for P, Q, e in states[:3]]
    od = {(1, -1): core.orbitals_by_subshell()[(1, -1)][:2]}
    od.update({(n, -1): t[:2] for n, t in zip((2, 3, 4), targets, strict=True)})
    return core, od, targets


def operator(reference, matrix):
    core, od, _ = reference
    return CIEffectiveOneBody(((2, -1), (3, -1), (4, -1)), matrix, od, core._bsp_data, 4, 30,
                             correlated_core={(1, -1): 2}, reference_identity='Be-core:test',
                             theory='analytic scalar operator', energy_Ha=-.5,
                             nuclear_potential=core._nuclear_potential)


def matrix(reference, configs, op=None, **kwargs):
    core, od, _ = reference
    csfs = enumerate_jj_csfs_many(configs)
    return build_ci_matrix(csfs, od, get_shared_orbital_basis(csfs), core._bsp_data, 4, 30,
                           nuclear_potential=core._nuclear_potential, one_body_operator=op, **kwargs)


def test_signed_one_electron_matrix_and_native_iterative_consumer(reference):
    # A closed J=0 core plus one electron has precisely δh_ac (no occupation
    # multiplier). This anchors the signs as well as diagonal magnitudes.
    correction = np.array([[.02, -.003, .004], [-.003, -.01, -.005], [.004, -.005, .03]])
    op = operator(reference, correction)
    configs = ['1s2 2s1', '1s2 3s1', '1s2 4s1']
    bare = matrix(reference, configs)
    changed = matrix(reference, configs, op)
    np.testing.assert_allclose(changed-bare, correction, rtol=0, atol=2e-14)
    action = matrix(reference, configs, op, as_operator=True)
    np.testing.assert_allclose(action @ np.eye(3), changed, rtol=0, atol=2e-14)
    core, od, _ = reference
    csfs = enumerate_jj_csfs_many(configs)
    args = (csfs, od, get_shared_orbital_basis(csfs), core._bsp_data, 4, 30)
    dense = ci_mbpt_solve(*args, sigma1_operator=op, nuclear_potential=core._nuclear_potential)
    iterative = ci_mbpt_solve(*args, sigma1_operator=op, nuclear_potential=core._nuclear_potential,
                              n_states=2, solver='davidson', tol=1e-11)
    np.testing.assert_allclose(iterative.state_energies, dense.state_energies[:2], atol=1e-12)
    assert dense.mbpt_provenance['sigma2'] == 'not_implemented'
    assert dense.provenance['effective_one_body']['frozen_energy_argument']
    np.testing.assert_array_equal(matrix(reference, configs), bare)


def test_two_electron_singlet_occupation_and_sqrt2_offdiagonal(reference):
    # s1/2 pair: |aa>, singlet |ab>, |bb>; h couples adjacent states
    # with sqrt(2)*h_ab and never |aa> directly to |bb>. Overall CSF
    # phase may change a signed entry; its square is phase independent.
    delta = .003
    op = operator(reference, [[.02, delta, 0], [delta, -.01, 0], [0, 0, 0]])
    configs = ['1s2 2s2', '1s2 2s1 3s1', '1s2 3s2']
    core, od, _ = reference
    csfs = [c for c in enumerate_jj_csfs_many(configs) if c.total_J2 == 0]
    args = (csfs, od, get_shared_orbital_basis(csfs), core._bsp_data, 4, 30)
    bare = build_ci_matrix(*args, nuclear_potential=core._nuclear_potential)
    diff = build_ci_matrix(*args, nuclear_potential=core._nuclear_potential, one_body_operator=op)-bare
    np.testing.assert_allclose(np.diag(diff), [.04, .01, -.02], atol=1e-13)
    assert diff[0, 1]**2 == pytest.approx(2*delta**2, abs=1e-16)
    assert diff[1, 2]**2 == pytest.approx(2*delta**2, abs=1e-16)
    assert diff[0, 2] == 0


def test_context_phase_and_double_count_guards(reference):
    core, od, targets = reference
    op = operator(reference, np.eye(3)*.01)
    configs = ['1s2 2s1']
    with pytest.raises(ValueError, match='combine'):
        matrix(reference, configs, op, one_body_shifts={})
    with pytest.raises(ValueError, match='correlated-core hole'):
        matrix(reference, ['1s1 2s2'], op)
    with pytest.raises(ValueError, match='mismatch'):
        altered = dict(od)
        altered[2, -1] = tuple(-p for p in od[2, -1])
        matrix((core, altered, targets), configs, op)
    with pytest.raises(ValueError, match='mismatch'):
        op.evaluator_for([], od, core._bsp_data, 5, 30, nuclear_potential=core._nuclear_potential)
    with pytest.raises(ValueError):
        op.matrix_Ha.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        op.energy_Ha = 0
    with pytest.raises(ValueError, match='symmetric'):
        operator(reference, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='mix kappa'):
        CIEffectiveOneBody(((2, -1), (2, 1)), [[0, 1], [1, 0]], od, core._bsp_data, 4, 30,
                           correlated_core={}, theory='test', reference_identity='test', energy_Ha=0)


def test_bound_matrix_export_and_covariant_phase(reference):
    core, od, targets = reference
    binding = CIReferenceBinding.capture(targets, {(1, -1): od[1, -1]},
                                         core._bsp_data, 4, 30, core._nuclear_potential)
    coupling = np.array([.04, -.03, .02])
    model = MatrixSelfEnergy([t[3] for t in targets], [2.], [np.outer(coupling, coupling)],
                            'bound:analytic', ci_binding=binding)
    kwargs = dict(correlated_core={(1, -1): 2}, nuclear_potential=core._nuclear_potential)
    args = (((2, -1), (3, -1), (4, -1)), od, core._bsp_data, 4, 30)
    op = model.as_ci_operator(-.5, *args, **kwargs)
    np.testing.assert_array_equal(op.matrix_Ha, model.evaluate(-.5))
    model.subspace((0, 1)).as_ci_operator(-.5, args[0][:2], *args[1:], **kwargs)
    with pytest.raises(ValueError, match='mismatch'):
        model.as_ci_operator(-.5, tuple(reversed(args[0])), *args[1:], **kwargs)
    unbound = MatrixSelfEnergy([-.5], [2.], [[[.01]]], 'legacy')
    with pytest.raises(ValueError, match='unbound'):
        unbound.as_ci_operator(-.5, *args, **kwargs)
    changed = dict(od)
    changed[3, -1] = tuple(-p for p in od[3, -1])
    with pytest.raises(ValueError, match='mismatch'):
        model.as_ci_operator(-.5, args[0], changed, *args[2:], **kwargs)
    phase = np.diag([1., -1., 1.])
    flipped_binding = CIReferenceBinding.capture([(*changed[n, -1], -1, t[3]) for n, t in zip((2,3,4), targets, strict=True)],
                                                {(1, -1): od[1, -1]}, core._bsp_data, 4, 30, core._nuclear_potential)
    flipped = MatrixSelfEnergy(model.reference_energies_Ha, model.poles_Ha,
                               [phase@model.residues_Ha2[0]@phase], 'phase:analytic', ci_binding=flipped_binding)
    flipped_op = flipped.as_ci_operator(-.5, args[0], changed, *args[2:], **kwargs)
    configs = ['1s2 2s1', '1s2 3s1', '1s2 4s1']
    h = matrix(reference, configs, op)
    hf = matrix((core, changed, targets), configs, flipped_op)
    np.testing.assert_allclose(hf, phase@h@phase, atol=2e-14)


def test_native_mbpt_producer_binding_is_consumable(reference):
    core, od, targets = reference
    model = build_mbpt2_matrix_self_energy(core, targets, reference_identity='native:core',
                                           virtual_kappas=(-1, 1, -2), n_virt_max=2)
    model.as_ci_operator(targets[0][3]-.01, ((2, -1), (3, -1), (4, -1)), od,
                         core._bsp_data, 4, 30, correlated_core={(1, -1): 2},
                         nuclear_potential=core._nuclear_potential)


def test_sampled_finite_nuclear_operator_identity(reference):
    core, od, targets = reference
    parameters = {'strength': 4.}
    def potential(r):
        return -parameters['strength']/np.sqrt(r*r+1e-8)
    op = CIEffectiveOneBody(((2, -1),), [[.01]], od, core._bsp_data, 4, 30,
                            correlated_core={(1, -1): 2}, reference_identity='finite-nucleus:test',
                            theory='analytic', energy_Ha=-.5, nuclear_potential=potential)
    op.evaluator_for([], od, core._bsp_data, 4, 30, nuclear_potential=potential)
    parameters['strength'] += .01
    with pytest.raises(ValueError, match='mismatch'):
        op.evaluator_for([], od, core._bsp_data, 4, 30, nuclear_potential=potential)
