"""Physical equivalence of bounded native ERIs and signed tensor export."""
import itertools
import os

import numpy as np
import pytest
from test_spinor_integrals_fscc import integrals
from test_spinor_integrals_fscc import native as native_fixture

from atomx.methods.fscc import (
    atomic_integrals,
    compute,
    export_integrals,
    load_integrals,
    resource_estimate,
)
from atomx.physics.csf.spinor_integrals import (
    OnDemandSpinorIntegrals,
    build_spinor_integrals,
)

native = native_fixture


@pytest.mark.parametrize('chunk_bytes', [16, 80, 4*1024**2])
def test_streamed_signed_eri_bytes_and_phases(native, tmp_path, chunk_bytes):
    a = integrals(native)
    b = integrals(native, representation='on_demand')
    assert isinstance(b, OnDemandSpinorIntegrals)
    assert np.max(abs(a.one_body_Ha-b.one_body_Ha)) < 1e-13
    assert np.max(abs(a.fock_Ha-b.fock_Ha)) < 1e-13
    for indices in itertools.product(range(10), repeat=4):
        assert b.eri(*indices) == pytest.approx(a.eri(*indices), abs=1e-13)
    pieces = list(b.iter_packed_eri(max_chunk_bytes=chunk_bytes))
    assert max(c.nbytes for c in pieces) <= chunk_bytes
    assert np.array_equal(np.concatenate(pieces), a.packed_eri_Ha)
    old = export_integrals(a, tmp_path/'packed', max_chunk_bytes=chunk_bytes)
    new = export_integrals(b, tmp_path/'stream', max_chunk_bytes=chunk_bytes)
    assert old['files']['ATOMX_TWO'] == new['files']['ATOMX_TWO']
    replay = load_integrals(tmp_path/'stream')
    assert np.array_equal(replay.packed_eri_Ha, a.packed_eri_Ha)
    assert replay.reference_energy_Ha == a.reference_energy_Ha
    # Physical snapshot and pair caches must survive mutation of caller arrays.
    core, od = native
    changed = {key: tuple(x.copy() for x in value) for key, value in od.items()}
    c = build_spinor_integrals(changed, core._bsp_data, 4, 30,
        occupied_keys=[(1, -1)], virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential, representation='on_demand')
    changed[2, -1][0][:] *= -1
    assert c.eri(0, 1, 2, 3) == b.eri(0, 1, 2, 3)
    for array in (c.one_body_Ha, c.fock_Ha):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_frozen_core_scalar_contractions_and_native_cap(native, tmp_path):
    core, od = native
    kwargs = dict(occupied_keys=[(2, -1)], virtual_keys=[(2, 1), (2, -2)],
                  frozen_keys=[(1, -1)], nuclear_potential=core._nuclear_potential)
    a = build_spinor_integrals(od, core._bsp_data, 4, 30, **kwargs)
    b = build_spinor_integrals(od, core._bsp_data, 4, 30, **kwargs, representation='on_demand')
    assert b.reference_energy_Ha == a.reference_energy_Ha
    assert b.frozen_energy_Ha == a.frozen_energy_Ha
    assert np.max(abs(b.fock_Ha-a.fock_Ha)) < 1e-13
    assert np.max(abs(b.one_body_Ha-a.one_body_Ha)) < 1e-13
    assert np.array_equal(np.concatenate(list(b.iter_packed_eri())), a.packed_eri_Ha)
    memory = 2*16*10*10
    with pytest.raises(MemoryError):
        integrals(native, max_integral_bytes=memory)
    c = integrals(native, representation='on_demand', max_integral_bytes=memory)
    assert c.diagnostics()['resident_matrix_bytes'] == memory
    assert resource_estimate(c) == resource_estimate(integrals(native))
    with pytest.raises(MemoryError, match='complete integral files'):
        export_integrals(c, tmp_path/'rejected', max_integral_file_bytes=memory)
    assert not (tmp_path/'rejected').exists()
    with pytest.raises(ValueError):
        integrals(native, representation='unknown')
    with pytest.raises(IndexError):
        c.eri(0, 1, 2, 10)


def test_complete_band_preserved_without_quartic_native_array(native):
    core, _ = native
    a = atomic_integrals(core, virtual_lmax=0, virtual_energy_max_Ha=1)
    b = atomic_integrals(core, virtual_lmax=0, virtual_energy_max_Ha=1, representation='on_demand')
    assert a.labels == b.labels
    assert a.identity['virtual_orbital_energies_Ha'] == b.identity['virtual_orbital_energies_Ha']
    assert np.array_equal(np.concatenate(list(b.iter_packed_eri())), a.packed_eri_Ha)
    assert b.diagnostics()['dense_eri_bytes'] == a.packed_eri_Ha.nbytes


def test_pair_potential_cache_capacity_preserves_signed_operator(native):
    a = integrals(native, representation='on_demand', radial_pair_cache_size=1)
    b = integrals(native, representation='on_demand', radial_pair_cache_size=16384)
    assert np.array_equal(a.fock_Ha, b.fock_Ha)
    assert np.array_equal(np.concatenate(list(a.iter_packed_eri())), np.concatenate(list(b.iter_packed_eri())))
    assert a.diagnostics()['pair_potential_cache_entries'] <= 1
    assert b.diagnostics()['pair_potential_cache_capacity'] == 16384
    assert a.diagnostics()['pair_potential_cache_misses'] > b.diagnostics()['pair_potential_cache_misses']
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='cache capacity'):
            integrals(native, representation='on_demand', radial_pair_cache_size=value)


def test_streamed_fsccsdt_electronic_replay(native, tmp_path):
    executable = os.environ.get('ATOMX_EXPT_EXECUTABLE')
    if not executable:
        pytest.skip('set ATOMX_EXPT_EXECUTABLE for the actual streamed electronic gate')
    a = integrals(native)
    b = integrals(native, representation='on_demand')
    results = [compute(x, executable=executable, directory=tmp_path/name,
        model='CCSDT', sector='0h1p', active_indices=range(2, 10), tile_size=4,
        tolerance=1e-12, residual_bound_tolerance_Ha=1e-9)
        for name, x in [('packed', a), ('stream', b)]]
    assert np.max(abs(np.array(results[0].sector_energies_Ha)-results[1].sector_energies_Ha)) < 1e-11
    assert np.max(abs(results[0].effective_matrices['0h1p', 1]-results[1].effective_matrices['0h1p', 1])) < 1e-11
    assert results[1].equation_residual_bound_Ha < 1e-9
