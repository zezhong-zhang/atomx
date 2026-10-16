"""Exact blocked storage versus the existing bound complete CI eigensystem."""
from unittest.mock import patch

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
from atomx.methods.ci import ci_solve
from atomx.methods.ci_blocks import plan_ci_blocks, solve_ci_blocks
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis


@pytest.fixture(scope='module')
def native_space():
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {k: pair[:2] for k, pair in core.orbitals_by_subshell().items()}
    for k in (-1, 1, -2):
        _, band = build_virtual_basis(core, k, e_max=50., potential_mode='VN',
                                      orthogonal_to_occupied=True)
        od[2, k] = tuple(band[0][:2])
    csfs = tuple(c for cfg in ('1s2 2s2', '1s2 2s1 2p1', '1s2 2p2')
                 for c in enumerate_jj_csfs(cfg))
    return core, od, csfs


@pytest.mark.parametrize('engine', ['streamed', 'spinor'])
def test_complete_blocks_preserve_every_matrix_entry_and_root(native_space, engine):
    core, od, csfs = native_space
    keys = get_shared_orbital_basis(csfs)
    old = ci_solve(csfs, od, keys, core._bsp_data, 4, 30,
                   nuclear_potential=core._nuclear_potential, retain_hamiltonian=True)
    h = old.hamiltonian_for(csfs, od, keys, core._bsp_data, 4, 30,
                            nuclear_potential=core._nuclear_potential)
    blocked = solve_ci_blocks(csfs, od, core._bsp_data, 4, 30,
                              nuclear_potential=core._nuclear_potential, engine=engine,
                              spinor_reference_keys=((2, -1),) if engine == 'spinor' else None)
    energies = []
    reconstructed = np.zeros_like(h)
    for block in blocked.blocks:
        reconstructed[np.ix_(block.indices, block.indices)] = block.hamiltonian_Ha
        energies.extend(block.energies_Ha)
        assert block.residual_Ha < 1e-10
        assert block.gram_defect < 1e-12
        with pytest.raises(ValueError):
            block.hamiltonian_Ha.flags.writeable = True
    np.testing.assert_allclose(reconstructed, h, atol=2e-13, rtol=2e-14)
    np.testing.assert_allclose(sorted(energies), old.state_energies, atol=2e-13, rtol=2e-14)
    assert sum(len(b.energies_Ha) for b in blocked.blocks) == len(csfs)
    assert blocked.resource_plan['retained_H_C_bytes'] < 16*len(csfs)**2
    if engine == 'streamed':
        assert blocked.radial_diagnostics['matrix_builds'] == len(blocked.blocks)


def test_caps_fail_before_radial_work(native_space):
    core, od, csfs = native_space
    with patch('atomx.methods.ci_blocks.CIRadialSnapshot', side_effect=AssertionError):
        with pytest.raises(MemoryError):
            solve_ci_blocks(csfs, od, core._bsp_data, 4, 30, memory_bytes=1)
        with pytest.raises(ValueError, match='determinant'):
            solve_ci_blocks(csfs, od, core._bsp_data, 4, 30, max_determinants=1)


def test_invalid_identity_rejects(native_space):
    _, _, csfs = native_space
    with pytest.raises(ValueError, match='distinct'):
        plan_ci_blocks(csfs+csfs)
    with pytest.raises(ValueError, match='nonempty'):
        plan_ci_blocks([])


@pytest.mark.parametrize('phase', [1., -1.])
def test_spinor_integrals_preserve_open_core_holes_and_orbital_phase(native_space, phase):
    core, original, _ = native_space
    od = dict(original)
    od[2,-1] = tuple(phase*p for p in od[2,-1])
    csfs = tuple(c for cfg in ('1s2 2s2','1s2 2p2','1s1 2s1 2p2')
                 for c in enumerate_jj_csfs(cfg))
    keys = get_shared_orbital_basis(csfs)
    ordinary = ci_solve(csfs, od, keys, core._bsp_data, 4, 30,
                        nuclear_potential=core._nuclear_potential, retain_hamiltonian=True)
    h = ordinary.hamiltonian_for(csfs, od, keys, core._bsp_data, 4, 30,
                                 nuclear_potential=core._nuclear_potential)
    native = solve_ci_blocks(csfs, od, core._bsp_data, 4, 30, engine='spinor',
                              spinor_reference_keys=((1,-1),),
                              nuclear_potential=core._nuclear_potential)
    actual = np.zeros_like(h)
    for block in native.blocks:
        actual[np.ix_(block.indices,block.indices)] = block.hamiltonian_Ha
    np.testing.assert_allclose(actual,h,atol=5e-13,rtol=5e-14)


@pytest.mark.parametrize('frequency', ['static', 'zero', 'finite'])
@pytest.mark.parametrize('phase', [1., -1.])
def test_spinor_transverse_signed_blocks_match_shared_csf_operator(native_space, frequency, phase):
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian

    core, original, _ = native_space
    od = dict(original)
    od[2, -1] = tuple(phase*p for p in od[2, -1])
    csfs = tuple(c for cfg in ('1s2 2s2', '1s2 2p2', '1s1 2s1 2p2')
                 for c in enumerate_jj_csfs(cfg))
    keys = get_shared_orbital_basis(csfs)
    energies = (None if frequency == 'static' else
                {key: 0. if frequency == 'zero' else float(i-3)
                 for i, key in enumerate(od)})
    controls = dict(include_breit=True, breit_orbital_energies=energies,
                    nuclear_potential=core._nuclear_potential)
    reference = build_ci_matrix(csfs, od, keys, core._bsp_data, 4, 30, **controls)
    native = SpinorCIHamiltonian(csfs, od, core._bsp_data, 4, 30,
                                reference_keys=((1, -1),), **controls)
    actual = np.zeros_like(reference)
    for sector in sorted({(c.total_J2, c.parity) for c in csfs}):
        indices = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
        actual[np.ix_(indices, indices)] = native.block_matrix(
            [csfs[i] for i in indices], max_determinants=2000)
    np.testing.assert_allclose(actual, reference, atol=5e-13, rtol=5e-14)
    dc = build_ci_matrix(csfs, od, keys, core._bsp_data, 4, 30,
                         nuclear_potential=core._nuclear_potential)
    assert np.max(abs(actual-dc)) > 1e-7
    assert np.max(abs((actual-dc)-np.diag(np.diag(actual-dc)))) > 1e-8
