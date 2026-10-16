"""Nonlinear finite CCSD: FCI, disconnected products, phases and failure gates."""
import itertools

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import _ops_sign
from atomx.methods.coupled_cluster import compute


def masks(n, electrons):
    return tuple(sum(1 << i for i in c) for c in itertools.combinations(range(n), electrons))


def test_two_electron_exact_fci_and_owned_phase_covariant_state():
    dets = masks(4, 2)
    h = np.diag([-1.4, -.3, .1, .2, .8, 1.2])
    h[0, 1] = h[1, 0] = .06
    h[0, 5] = h[5, 0] = -.12
    h[1, 5] = h[5, 1] = .04
    opts = dict(spinor_count=4, reference_identity='two-electron:finite-anchor')
    result = compute(h, dets, dets[0], **opts)
    en, vec = np.linalg.eigh(h)
    assert result.total_energy_Ha == pytest.approx(en[0], abs=1e-12)
    assert result.amplitude_residual_Ha < 1e-10
    assert result.full_eigenstate_residual_Ha < 1e-10
    np.testing.assert_allclose(np.outer(result.coefficients, result.coefficients),
                               np.outer(vec[:, 0], vec[:, 0]), atol=1e-11)
    phases = np.array([(-1)**int(d.bit_count()-(d & 0b0101).bit_count()) for d in dets])
    flipped = compute(phases[:, None]*h*phases[None, :], dets, dets[0], **opts)
    assert flipped.total_energy_Ha == pytest.approx(result.total_energy_Ha, abs=1e-12)
    np.testing.assert_allclose(np.outer(flipped.coefficients, flipped.coefficients),
        phases[:, None]*np.outer(result.coefficients, result.coefficients)*phases[None, :], atol=1e-11)
    with pytest.raises(ValueError):
        result.amplitudes.flags.writeable = True
    with pytest.raises(TypeError):
        result.provenance['spinor_count'] = 5
    original = result.provenance['hamiltonian_sha256']
    h[0, 0] = 0
    assert result.provenance['hamiltonian_sha256'] == original


def test_disconnected_doubles_and_size_consistent_four_electron_product():
    dets = masks(8, 4)
    lookup = {d: i for i, d in enumerate(dets)}
    eps = np.array([-.5, -.45, .2, .3, -.6, -.55, .4, .45])
    h = np.diag([sum(eps[i] for i in range(8) if d & (1 << i)) for d in dets])
    pairs = [((0, 1), (2, 3), .12), ((4, 5), (6, 7), -.09)]
    for holes, particles, coupling in pairs:
        ops = tuple((a, True) for a in particles)+tuple((i, False) for i in reversed(holes))
        for col, d in enumerate(dets):
            phase = _ops_sign(d, ops)
            if phase:
                target = (d ^ sum(1 << i for i in holes)) | sum(1 << a for a in particles)
                row = lookup[target]
                h[row, col] += coupling*phase
                h[col, row] += coupling*phase
    ref = sum(1 << i for i in (0, 1, 4, 5))
    result = compute(h, dets, ref, spinor_count=8, reference_identity='two-disconnected-pairs')
    independent = sum(np.linalg.eigvalsh([[sum(eps[list(a)]), coupling],
                                         [coupling, sum(eps[list(b)])]])[0]
                      for a, b, coupling in pairs)
    assert result.total_energy_Ha == pytest.approx(independent, abs=1e-11)
    assert result.full_eigenstate_residual_Ha < 1e-10
    # Quadruple coefficient comes from T2(A)*T2(B), and cannot occur in CISD.
    quadruple = sum(1 << i for i in (2, 3, 6, 7))
    assert abs(result.coefficients[lookup[quadruple]]) > 1e-3
    sd = [i for i, d in enumerate(dets) if (d ^ ref).bit_count() <= 4]
    cisd = np.linalg.eigvalsh(h[np.ix_(sd, sd)])[0]
    assert cisd-result.total_energy_Ha > 1e-6


def test_native_two_electron_dirac_coulomb_against_complete_csf_anchor():
    from atomx.config.csf import enumerate_jj_orbital_space, get_shared_orbital_basis
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.sparse_ci import (
        assemble_sparse_hamiltonian,
        enumerate_determinants,
        patterns_from_csfs,
    )

    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, virtual = build_virtual_basis(core, k, e_max=50, potential_mode='VN', orthogonal_to_occupied=True)
        od[2, k] = virtual[0][:2]
    csfs = enumerate_jj_orbital_space(od, 2, magnetic_state_cap=256)
    dets, spinors = enumerate_determinants(patterns_from_csfs(csfs), od, m2_total=None)
    h = assemble_sparse_hamiltonian(dets, spinors, od, core._bsp_data, 4, 30).toarray()
    # The independently assembled two triangles differ at summation
    # roundoff. Bound and explicitly project that producer roundoff;
    # the CCSD consumer retains its strict symmetric-input contract.
    assert np.max(abs(h-h.T)) < 1e-12
    h = .5*(h+h.T)
    ref = sum(1 << i for i, (n, k, m) in enumerate(spinors) if (n, k) == (1, -1))
    cc = compute(h, dets, ref, spinor_count=len(spinors), reference_identity='Be2+:native-complete-N2')
    ground = [c for c in csfs if (c.total_J2, c.parity) == (0, 1)]
    jj = build_ci_matrix(ground, od, get_shared_orbital_basis(ground), core._bsp_data, 4, 30,
                         nuclear_potential=core._nuclear_potential)
    assert cc.total_energy_Ha == pytest.approx(np.linalg.eigvalsh(jj)[0], abs=1e-10)
    assert cc.full_eigenstate_residual_Ha < 1e-10


def test_ccsd_caps_complete_basis_and_real_matrix_guards():
    dets = masks(4, 2)
    h = np.diag(np.arange(6, dtype=float))
    opts = dict(spinor_count=4, reference_identity='guard')
    with pytest.raises(ValueError, match='complete fixed-N'):
        compute(h, dets, dets[0], **opts, determinant_cap=1)
    with pytest.raises(ValueError, match='complete SD'):
        compute(h, dets, dets[0], **opts, amplitude_cap=1)
    with pytest.raises(ValueError, match='distinct'):
        compute(h, dets[:-1]+(dets[0],), dets[0], **opts)
    with pytest.raises(ValueError, match='symmetric'):
        compute(h.astype(complex), dets, dets[0], **opts)
    with pytest.raises(ValueError, match='continuation'):
        compute(h, dets, dets[0], **opts, continuation=(.5, .25, 1.))
    with pytest.raises(NotImplementedError, match='Fock-space'):
        compute(h, dets, dets[0], **opts, method='FS-CCSDT')
    nontrivial = h.copy()
    nontrivial[0, 5] = nontrivial[5, 0] = .1
    with pytest.raises(RuntimeError, match='stage'):
        compute(nontrivial, dets, dets[0], **opts, max_evaluations=1)
    resonant = nontrivial.copy()
    resonant[5, 5] = resonant[0, 0]
    with pytest.raises(ValueError, match='degenerate'):
        compute(resonant, dets, dets[0], **opts)
    full = compute([[-.5]], (3,), 3, spinor_count=2, reference_identity='no-virtuals')
    assert full.total_energy_Ha == -.5 and len(full.amplitudes) == 0
