"""Complete signed CI and frozen-core controls for bound one-body composition."""

from unittest.mock import patch

import numpy as np
import pytest
from test_ci_blocks import native_space as native_fixture

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.effective_one_body import CIEffectiveOneBody
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.one_body_composition import add_bound_one_body
from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, build_spinor_integrals

native_space = native_fixture


@pytest.mark.parametrize('phase', [1., -1.])
@pytest.mark.parametrize('representation', ['packed', 'on_demand'])
def test_frozen_core_and_signed_cross_radial_ci_match_independent_native_operator(native_space, phase, representation):
    core, original, _ = native_space
    orbitals = dict(original)
    _, band = build_virtual_basis(core, -1, e_max=50., potential_mode='VN',
                                  orthogonal_to_occupied=True)
    orbitals[3, -1] = tuple(phase*v for v in band[1][:2])
    csfs = tuple(c for cfg in ('1s2 2s2', '1s2 2s1 3s1', '1s2 3s2', '1s2 2s1 2p1')
                 for c in enumerate_jj_csfs(cfg))
    keys = tuple(sorted({(s.n, s.kappa) for c in csfs for s in c.subshells}))
    rng = np.random.default_rng(5518)
    w = rng.normal(size=(len(keys), len(keys)))*.02
    w = .5*(w+w.T)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if a[1] != b[1]:
                w[i, j] = 0.
            w[i, j] *= (phase if a == (3, -1) else 1.)*(phase if b == (3, -1) else 1.)
    op = CIEffectiveOneBody(keys, w, orbitals, core._bsp_data, 4, core._bsp_data['ns'],
        correlated_core={}, reference_identity='finite independent scalar probe',
        theory='explicit scalar test operator', energy_Ha=0., nuclear_potential=core._nuclear_potential)
    builder = SpinorCIHamiltonian(csfs, orbitals, core._bsp_data, 4, core._bsp_data['ns'],
                 reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    old = builder.integrals
    packed_original = old.packed_eri_Ha
    if representation == 'on_demand':
        old = build_spinor_integrals(orbitals, core._bsp_data, 4, core._bsp_data['ns'],
            occupied_keys=old.identity['occupied_keys'], virtual_keys=old.identity['virtual_keys'],
            frozen_keys=old.identity['frozen_keys'], nuclear_potential=core._nuclear_potential,
            representation='on_demand')
    new = add_bound_one_body(old, op, orbitals, core._bsp_data, 4, core._bsp_data['ns'],
                             nuclear_potential=core._nuclear_potential)
    complete_eri = np.concatenate(list(new.iter_packed_eri(max_chunk_bytes=128)))
    np.testing.assert_array_equal(complete_eri, packed_original)
    # The finite CI oracle deliberately consumes a fully materialized copy;
    # production streaming remains a separate representation.
    dense = SpinorIntegrals(new.labels, new.noccupied, new.one_body_Ha, new.fock_Ha,
        complete_eri, new.frozen_energy_Ha, new.reference_energy_Ha, new.identity)
    with patch('atomx.physics.csf.spinor_ci.build_spinor_integrals', return_value=dense):
        modified = SpinorCIHamiltonian(csfs, orbitals, core._bsp_data, 4, core._bsp_data['ns'],
                      reference_keys=((2, -1),), nuclear_potential=core._nuclear_potential)
    expected = build_ci_matrix(csfs, orbitals, keys, core._bsp_data, 4, core._bsp_data['ns'],
                              nuclear_potential=core._nuclear_potential, one_body_operator=op)
    actual = np.zeros_like(expected)
    groups = {}
    for i, c in enumerate(csfs):
        groups.setdefault((c.total_J2, c.parity), []).append(i)
    for ix in groups.values():
        h = modified.block_matrix([csfs[i] for i in ix], max_determinants=80000)
        actual[np.ix_(ix, ix)] = h
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-13)
    assert new.frozen_energy_Ha-old.frozen_energy_Ha == pytest.approx(2*w[keys.index((1, -1)), keys.index((1, -1))], abs=1e-14)
    with pytest.raises(ValueError, match='already included'):
        add_bound_one_body(new, op, orbitals, core._bsp_data, 4, core._bsp_data['ns'],
                           nuclear_potential=core._nuclear_potential)
    changed = dict(orbitals)
    changed[3, -1] = tuple(-v for v in changed[3, -1])
    with pytest.raises(ValueError, match='point mismatch'):
        add_bound_one_body(old, op, changed, core._bsp_data, 4, core._bsp_data['ns'],
                           nuclear_potential=core._nuclear_potential)
    with pytest.raises(MemoryError, match='payload'):
        add_bound_one_body(old, op, orbitals, core._bsp_data, 4, core._bsp_data['ns'], max_added_bytes=1)
