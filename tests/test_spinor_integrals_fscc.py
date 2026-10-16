"""Independent signed integral, frozen-core and external amplitude anchors."""
import itertools
import os

import numpy as np
import pytest

from atomx.methods.fscc import (
    atomic_integrals,
    compute,
    export_integrals,
    load_integrals,
    read_heff,
    resource_estimate,
)
from atomx.physics.csf.spinor_integrals import (
    SpinorIntegrals,
    build_spinor_integrals,
    packed_size,
)


@pytest.fixture(scope='module')
def native():
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mbpt import build_virtual_basis
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, v = build_virtual_basis(core, k, e_max=50, potential_mode='VN', orthogonal_to_occupied=True)
        od[2, k] = v[0][:2]
    return core, od


def integrals(native, **options):
    core, od = native
    return build_spinor_integrals(od, core._bsp_data, 4, 30,
        occupied_keys=[(1, -1)], virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential, **options)


def test_signed_antisymmetry_and_native_frozen_core_projection(native):
    from atomx.angular.csf_hamiltonian import _ops_sign
    from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian
    core, od = native
    a = integrals(native)
    for p, q, r, s in itertools.product(range(10), repeat=4):
        g = a.eri(p, q, r, s)
        assert g == pytest.approx(-a.eri(q, p, r, s), abs=1e-14)
        assert g == pytest.approx(a.eri(r, s, p, q).conjugate(), abs=1e-14)
    # Freeze the complete 1s multiplet, correlate the complete 2s multiplet.
    b = build_spinor_integrals(od, core._bsp_data, 4, 30,
        occupied_keys=[(2, -1)], virtual_keys=[(2, 1), (2, -2)], frozen_keys=[(1, -1)],
        nuclear_potential=core._nuclear_potential)
    labels = tuple((1, -1, m) for m in (-1, 1))+b.labels
    masks = [3 | sum(1 << (i+2) for i in c) for c in itertools.combinations(range(8), 2)]
    h = assemble_sparse_hamiltonian(masks, labels, od, core._bsp_data, 4, 30).toarray()
    oracle = np.eye(len(masks), dtype=complex)*b.frozen_energy_Ha
    idx = {m >> 2: i for i, m in enumerate(masks)}
    for mask, j in idx.items():
        for p, q in itertools.product(range(8), repeat=2):
            sign = _ops_sign(mask, ((p, True), (q, False)))
            final = (mask ^ (1 << q)) | (1 << p)
            if sign and final in idx:
                oracle[idx[final], j] += sign*b.one_body_Ha[p, q]
        for p, q in itertools.combinations(range(8), 2):
            for r, s in itertools.combinations(range(8), 2):
                sign = _ops_sign(mask, ((p, True), (q, True), (s, False), (r, False)))
                final = (mask ^ (1 << r) ^ (1 << s)) | (1 << p) | (1 << q)
                if sign and final in idx:
                    oracle[idx[final], j] += sign*b.eri(p, q, r, s)
    assert np.max(abs(oracle-h)) < 1e-10
    assert b.reference_energy_Ha == pytest.approx(h[0, 0], abs=1e-10)


def test_phase_covariance_and_owned_arrays(native):
    core, od = native
    a = integrals(native)
    changed = dict(od)
    changed[2, -1] = tuple(-x for x in od[2, -1])
    b = build_spinor_integrals(changed, core._bsp_data, 4, 30,
        occupied_keys=[(1, -1)], virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential)
    phases = np.array([-1 if label[:2] == (2, -1) else 1 for label in a.labels])
    assert np.max(abs(b.fock_Ha-phases[:, None]*a.fock_Ha*phases)) < 1e-12
    assert np.max(abs(a.one_body_Ha-np.diag(np.diag(a.one_body_Ha)))) > 1e-3
    for p, q, r, s in itertools.product(range(10), repeat=4):
        assert b.eri(p, q, r, s) == pytest.approx(a.eri(p, q, r, s)*phases[p]*phases[q]*phases[r]*phases[s], abs=1e-12)
    for array in (a.fock_Ha, a.one_body_Ha, a.packed_eri_Ha):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert a.identity['binding'] != b.identity['binding']


def test_metric_resource_and_model_rejection(native, tmp_path):
    with pytest.raises(MemoryError):
        integrals(native, max_integral_bytes=100)
    core, od = native
    bad = dict(od)
    bad[2, -1] = od[1, -1]
    with pytest.raises(ValueError, match='metric'):
        build_spinor_integrals(bad, core._bsp_data, 4, 30, occupied_keys=[(1, -1)],
            virtual_keys=[(2, -1)], nuclear_potential=core._nuclear_potential)
    a = integrals(native)
    with pytest.raises(ValueError):
        resource_estimate(a, model='CCSD(T)')
    with pytest.raises(MemoryError):
        compute(a, executable='/missing', directory=tmp_path/'blocked', memory_bytes=1)
    assert not (tmp_path/'blocked').exists()
    with pytest.raises(ValueError):
        compute(a, executable='/missing', directory=tmp_path/'bad', sector='0h1p', active_indices=[0])
    with pytest.raises(ValueError, match='complete magnetic'):
        compute(a, executable='/missing', directory=tmp_path/'partial', sector='0h1p', active_indices=[2])
    exported = export_integrals(a, tmp_path/'export')
    assert (tmp_path/'export/ATOMX_ONE').read_bytes().startswith(b'ATXCC001')
    assert (tmp_path/'export/ATOMX_TWO').stat().st_size == 16*packed_size(10)
    assert exported['identity']['binding'] == a.identity['binding']


def test_signed_nonhermitian_heff_parser(tmp_path):
    matrix = np.array([[1., -.2+1j*.3], [.4+1j*.1, 2.]])
    text = 'complex\n0h0p\n1 1\n-1.23456789012345678E+01 0.00000000000000000E+00\n0h1p\n1 2\n'
    text += '\n'.join(f'{z.real:26.17E}{z.imag:26.17E}' for z in matrix.ravel())+'\n'
    (tmp_path/'HEFF').write_text(text)
    blocks = read_heff(tmp_path/'HEFF')
    assert np.array_equal(blocks['0h1p', 1], matrix)
    (tmp_path/'HEFF').write_text(text[:-60])
    with pytest.raises(ValueError):
        read_heff(tmp_path/'HEFF')


def test_complete_virtual_band_and_integral_replay(native, tmp_path):
    from atomx.methods.mbpt import build_virtual_basis
    core, _ = native
    _, expected = build_virtual_basis(core, -1, e_max=1, potential_mode='VN', orthogonal_to_occupied=True)
    a = atomic_integrals(core, virtual_lmax=0, virtual_energy_max_Ha=1)
    assert len(a.labels)-a.noccupied == 2*len(expected)
    assert len(a.identity['virtual_keys']) == len(expected)
    assert max(a.identity['virtual_orbital_energies_Ha'].values()) == pytest.approx(expected[-1][2])
    export_integrals(a, tmp_path/'inputs')
    b = load_integrals(tmp_path/'inputs')
    assert np.array_equal(a.packed_eri_Ha, b.packed_eri_Ha)
    assert np.array_equal(a.fock_Ha, b.fock_Ha)
    assert np.max(abs(a.one_body_Ha-b.one_body_Ha)) < 1e-12
    with pytest.raises(MemoryError):
        load_integrals(tmp_path/'inputs', max_integral_bytes=1)
    with (tmp_path/'inputs/ATOMX_TWO').open('r+b') as f:
        byte = f.read(1)
        f.seek(0)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match='hash'):
        load_integrals(tmp_path/'inputs')


@pytest.mark.parametrize('chunk_bytes', (16, 48, 128, 4*1024**2))
def test_axial_export_complete_signed_stream_and_replay(native, tmp_path, chunk_bytes):
    a = integrals(native)
    first = export_integrals(a, tmp_path/'axial', tensor_symmetry='axial', max_chunk_bytes=chunk_bytes)
    ordinary = export_integrals(a, tmp_path/'C1', max_chunk_bytes=chunk_bytes)
    assert first['files'] == ordinary['files']
    assert first['tensor_symmetry']['forbidden_signed_integral_gate'].startswith('exact zero')
    restored = load_integrals(tmp_path/'axial')
    assert np.array_equal(restored.packed_eri_Ha, a.packed_eri_Ha)
    (tmp_path/'axial/ATOMX_AXIAL').write_text('ATOMX_AXIAL_001 0\n')
    with pytest.raises(ValueError, match='sidecar identity'):
        load_integrals(tmp_path/'axial')


def test_axial_rejects_actual_forbidden_signed_integrals(native, tmp_path):
    from dataclasses import replace
    a = integrals(native)
    h, f = a.one_body_Ha.copy(), a.fock_Ha.copy()
    h[2, 4] = h[4, 2] = f[2, 4] = f[4, 2] = .001
    bad_one = replace(a, one_body_Ha=h, fock_Ha=f)
    with pytest.raises(ValueError, match='signed Fock'):
        export_integrals(bad_one, tmp_path/'fock', tensor_symmetry='axial')
    assert not (tmp_path/'fock').exists()
    packed = a.packed_eri_Ha.copy()
    # Distinct virtual pairs: no occupied contraction or equal-p duplicate.
    indices = [(p, q, r, s) for p in range(9) for q in range(p+1, 10)
               for r in range(p, 9) for s in range(r+1, 10)]
    packed[indices.index((2, 4, 5, 6))] = .02+.01j
    bad_two = replace(a, packed_eri_Ha=packed)
    with pytest.raises(ValueError, match='signed ERI'):
        export_integrals(bad_two, tmp_path/'eri', tensor_symmetry='axial', max_chunk_bytes=48)
    assert not (tmp_path/'eri/ATOMX_AXIAL').exists()
    labels = list(a.labels)
    labels[2] = (6, -6, -11)
    with pytest.raises(ValueError, match='rank-six irrep range'):
        export_integrals(replace(a, labels=tuple(labels)), tmp_path/'range', tensor_symmetry='axial')


def test_live_axial_full_ccsdt_signed_phase_and_all_supported_sectors(native, tmp_path):
    executable = os.environ.get('ATOMX_EXPT_AXIAL_EXECUTABLE')
    if not executable:
        pytest.skip('set ATOMX_EXPT_AXIAL_EXECUTABLE for actual exact tensor blocking')
    from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian
    core, od = native
    a = integrals(native)
    def run(a, name, symmetry, sector='0h1p'):
        active = range(2, 10) if sector == '0h1p' else (range(2) if sector == '1h0p' else ())
        return compute(a, executable=executable, directory=tmp_path/name,
            model='CCSDT', sector=sector, active_indices=active, tensor_symmetry=symmetry,
            tile_size=4, tolerance=1e-12, residual_bound_tolerance_Ha=1e-9)
    c1 = run(a, 'C1', 'C1')
    axial = run(a, 'axial', 'axial')
    assert np.max(abs(c1.model_space_matrix_Ha-axial.model_space_matrix_Ha)) < 1e-9
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(10), 3)]
    expected = np.linalg.eigvalsh(assemble_sparse_hamiltonian(masks, a.labels, od, core._bsp_data, 4, 30).toarray())[:8]
    assert np.max(abs(np.array(axial.sector_energies_Ha)-expected)) < 1e-9
    rng = np.random.default_rng(314519)
    u = np.eye(10, dtype=complex)
    sym = [(m2, (-1)**(k if k > 0 else -k-1)) for _, k, m2 in a.labels]
    for key in sorted(set(sym[2:])):
        indices = [i for i in range(2, 10) if sym[i] == key]
        z = rng.normal(size=(len(indices), len(indices)))+1j*rng.normal(size=(len(indices), len(indices)))
        u[np.ix_(indices, indices)] = np.linalg.qr(z)[0]
    dense = np.array([a.eri(*idx) for idx in itertools.product(range(10), repeat=4)]).reshape((10,)*4)
    rotated = np.einsum('pa,qb,pqrs,rc,sd->abcd', u.conj(), u.conj(), dense, u, u, optimize=True)
    packed = np.array([rotated[p, q, r, s] for p in range(9) for q in range(p+1, 10)
                       for r in range(p, 9) for s in range(r+1, 10)])
    def hermitian(x):
        y = u.conj().T @ x @ u
        return .5*(y+y.conj().T)
    b = SpinorIntegrals(a.labels, 2, hermitian(a.one_body_Ha), hermitian(a.fock_Ha), packed,
        a.frozen_energy_Ha, a.reference_energy_Ha, dict(a.identity, rotation_seed=314519))
    rotated_c1 = run(b, 'rotated_C1', 'C1')
    rotated_axial = run(b, 'rotated_axial', 'axial')
    assert np.max(abs(rotated_c1.model_space_matrix_Ha-rotated_axial.model_space_matrix_Ha)) < 1e-9
    assert np.max(abs(rotated_axial.model_space_matrix_Ha-u[2:, 2:].conj().T@axial.model_space_matrix_Ha@u[2:, 2:])) < 1e-9
    assert np.max(abs(np.array(rotated_axial.sector_energies_Ha)-expected)) < 1e-9
    with pytest.raises(ValueError):
        rotated_axial.model_space_matrix_Ha.setflags(write=True)
    closed = run(a, 'closed', 'axial', '0h0p')
    assert closed.model_space_matrix_Ha is None
    assert closed.core_energy_Ha == pytest.approx(axial.core_energy_Ha, abs=1e-11)
    hole_c1 = run(a, 'hole_C1', 'C1', '1h0p')
    hole = run(a, 'hole_axial', 'axial', '1h0p')
    assert np.max(abs(hole.model_space_matrix_Ha-hole_c1.model_space_matrix_Ha)) < 1e-9
    assert np.max(abs(np.array(hole.sector_energies_Ha)-np.linalg.eigvalsh(a.one_body_Ha)[:2])) < 1e-9


def test_live_tensor_fsccsdt_against_complete_three_electron_ci(native, tmp_path):
    executable = os.environ.get('ATOMX_EXPT_EXECUTABLE')
    if not executable:
        pytest.skip('set ATOMX_EXPT_EXECUTABLE to run external electronic amplitude gate')
    from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian
    core, od = native
    a = integrals(native)
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(10), 3)]
    h = assemble_sparse_hamiltonian(masks, a.labels, od, core._bsp_data, 4, 30).toarray()
    expected = np.linalg.eigvalsh(.5*(h+h.T))[:8]
    energies = []
    for tile in (4, 32):
        result = compute(a, executable=executable, directory=tmp_path/f'tile{tile}',
            model='CCSDT', sector='0h1p', active_indices=range(2, 10), tile_size=tile,
            tolerance=1e-12, residual_bound_tolerance_Ha=1e-9)
        assert np.max(abs(np.array(result.sector_energies_Ha)-expected)) < 1e-9
        energies.append(result.sector_energies_Ha)
        assert 'Norm |T{0h1p}_3| = 0.00262345' in (tmp_path/f'tile{tile}/cc.log').read_text()
    assert np.max(abs(np.array(energies[0])-energies[1])) < 1e-11
    # Complex occupied/virtual block rotations preserve the reference, while
    # generating genuinely non-diagonal complex Fock and two-body matrices.
    rng = np.random.default_rng(83)
    u = np.eye(10, dtype=complex)
    for begin, end in ((0, 2), (2, 10)):
        z = rng.normal(size=(end-begin, end-begin))+1j*rng.normal(size=(end-begin, end-begin))
        u[begin:end, begin:end] = np.linalg.qr(z)[0]
    dense = np.array([a.eri(*idx) for idx in itertools.product(range(10), repeat=4)]).reshape((10,)*4)
    rotated = np.einsum('pa,qb,pqrs,rc,sd->abcd', u.conj(), u.conj(), dense, u, u, optimize=True)
    packed = np.array([rotated[p, q, r, s] for p in range(9) for q in range(p+1, 10)
                       for r in range(p, 9) for s in range(r+1, 10)])
    def hermitian(x):
        y = u.conj().T @ x @ u
        return .5*(y+y.conj().T)
    b = SpinorIntegrals(a.labels, 2, hermitian(a.one_body_Ha), hermitian(a.fock_Ha),
        packed, a.frozen_energy_Ha, a.reference_energy_Ha,
        dict(a.identity, representation='explicit complex occupied/virtual rotation, seed83'))
    result = compute(b, executable=executable, directory=tmp_path/'complex',
        model='CCSDT', sector='0h1p', active_indices=range(2, 10), tile_size=4,
        tolerance=1e-12, residual_bound_tolerance_Ha=1e-9)
    assert np.max(abs(np.array(result.sector_energies_Ha)-expected)) < 1e-9
    unrotated = read_heff(tmp_path/'tile4/scratch/HEFF')['0h1p', 1]
    expected_matrix = u[2:, 2:].conj().T @ unrotated @ u[2:, 2:]
    assert np.max(abs(result.effective_matrices['0h1p', 1]-expected_matrix)) < 1e-9
    hole = compute(a, executable=executable, directory=tmp_path/'hole',
        model='CCSDT', sector='1h0p', active_indices=(0, 1), tile_size=4,
        tolerance=1e-12, residual_bound_tolerance_Ha=1e-9)
    assert np.max(abs(np.array(hole.sector_energies_Ha)-np.linalg.eigvalsh(a.one_body_Ha)[:2])) < 1e-9
