"""Independent full-coordinate, complex-phase and native amplitude anchors."""
import hashlib
import itertools
import json
import os

import numpy as np
import pytest

from atomx.methods.fscc import (
    compute,
    export_integrals,
    load_integrals,
    resource_estimate,
)
from atomx.numerics.axial_eri import AxialPairLayout
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, build_spinor_integrals


def random_integrals():
    rng = np.random.default_rng(627391)
    labels = tuple((n, k, m) for n, k in [(1, -1), (2, -1), (2, 1), (2, -2)]
                   for m in range(-2*abs(k)+1, 2*abs(k), 2))
    n = len(labels)
    pairs = list(itertools.combinations(range(n), 2))
    def quantum(pair):
        return (sum(labels[i][2] for i in pair),
                (-1)**sum(labels[i][1] if labels[i][1] > 0 else -labels[i][1]-1
                          for i in pair))
    a = rng.normal(size=(len(pairs), len(pairs)))+1j*rng.normal(size=(len(pairs), len(pairs)))
    a = a+a.conj().T
    for i, left in enumerate(pairs):
        for j, right in enumerate(pairs):
            if quantum(left) != quantum(right):
                a[i, j] = 0
    index = {p: i for i, p in enumerate(pairs)}
    def eri(p, q, r, s):
        if p == q or r == s:
            return 0j
        return ((1 if p < q else -1)*(1 if r < s else -1)
                *a[index[tuple(sorted((p, q)))], index[tuple(sorted((r, s)))]])
    # Independent complete original ABI enumeration, not the new rank maps.
    packed = np.array([eri(p, q, r, s) for p, q in pairs
                       for r in range(p, n-1) for s in range(r+1, n)])
    h = np.diag(np.arange(n, dtype=float)-10).astype(complex)
    h[0, 2], h[2, 0] = .13+.24j, .13-.24j
    fock = h.copy()
    for p, q in itertools.product(range(n), repeat=2):
        fock[p, q] += sum(eri(p, i, q, i) for i in range(2))
    reference = float((np.trace(h[:2, :2])+.5*sum(eri(i, j, i, j)
                       for i in range(2) for j in range(2))).real)
    obj = SpinorIntegrals(labels, 2, h, fock, packed, 0., reference, {})
    symmetry = np.array([(m, (-1)**(k if k > 0 else -k-1)) for _, k, m in labels])
    expected = np.array([eri(p, q, r, s) for p, q in pairs
                         for r in range(p, n-1) for s in range(r+1, n)
                         if quantum((p, q)) == quantum((r, s))])
    return obj, symmetry, expected, eri


@pytest.mark.parametrize('chunk', [16, 80, 1024, 4*1024**2])
def test_independent_complete_complex_coordinates_and_roundtrip(tmp_path, chunk):
    obj, symmetry, bank, oracle = random_integrals()
    layout = AxialPairLayout(symmetry)
    assert layout.count == len(bank)
    for indices in itertools.product(range(len(obj.labels)), repeat=4):
        assert layout.value(bank, *indices) == oracle(*indices)
    exported = export_integrals(obj, tmp_path/'compact', tensor_symmetry='axial',
                               integral_storage='axial_compact', max_chunk_bytes=chunk)
    np.testing.assert_array_equal(np.fromfile(tmp_path/'compact/ATOMX_TWO', dtype=complex), bank)
    assert exported['eri_storage']['count'] == len(bank)
    replay = load_integrals(tmp_path/'compact')
    np.testing.assert_array_equal(replay.packed_eri_Ha, obj.packed_eri_Ha)
    np.testing.assert_array_equal(replay.fock_Ha, obj.fock_Ha)
    assert np.max(abs(replay.one_body_Ha-obj.one_body_Ha)) < 1e-12
    with pytest.raises(MemoryError, match='decompression'):
        load_integrals(tmp_path/'compact', max_integral_bytes=exported['export']['logical_bytes'])
    for array in (layout.symmetry, layout.ranks, layout.prefix, layout.offsets):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_exact_count_budget_and_missing_source_reject_before_export(tmp_path):
    from atomx.methods.fscc import _LEGACY_DIVEPS_SOURCE_SHA256
    obj, symmetry, bank, _ = random_integrals()
    full = resource_estimate(obj, tensor_symmetry='axial')
    compact = resource_estimate(obj, tensor_symmetry='axial', integral_storage='axial_compact')
    assert compact['compact_eri_bytes'] == len(bank)*16
    assert compact['estimated_working_bytes'] == (full['estimated_working_bytes']-obj.dense_eri_bytes
        +16*len(bank)+2*AxialPairLayout(symmetry).index_bytes)
    with pytest.raises(ValueError, match='requires axial'):
        export_integrals(obj, tmp_path/'invalid', integral_storage='axial_compact')
    # Deliberately mislabeled capability metadata cannot authorize execution.
    executable = tmp_path/'fake.x'
    executable.write_bytes(b'ATOMX_INTEGRALS_V1 ATOMX_AXIAL_V1 ATOMX_AXIAL_ERI_V1')
    (tmp_path/'atomx_backend.json').write_text(json.dumps(dict(abi='ATXCC001',
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        eri_storage_layouts=['packed', 'axial_compact'],
        source_sha256={'src/rcc/engine/diveps.c': _LEGACY_DIVEPS_SOURCE_SHA256})))
    with pytest.raises(ValueError, match='source-bound storage/index'):
        compute(obj, executable=executable, directory=tmp_path/'blocked',
                tensor_symmetry='axial', integral_storage='axial_compact')
    assert not (tmp_path/'blocked').exists()


@pytest.mark.parametrize('bad', ['count', 'truncated', 'axial', 'unexpected_sidecar'])
def test_malformed_saved_layout_rejects(tmp_path, bad):
    obj, _, _, _ = random_integrals()
    d = tmp_path/'inputs'
    export_integrals(obj, d, tensor_symmetry='axial',
                     integral_storage='packed' if bad == 'unexpected_sidecar' else 'axial_compact')
    manifest = json.loads((d/'integrals.json').read_text())
    if bad == 'count':
        manifest['eri_storage']['count'] += 1
    elif bad == 'truncated':
        path = d/'ATOMX_TWO'
        path.write_bytes(path.read_bytes()[:-16])
        manifest['files']['ATOMX_TWO'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif bad == 'axial':
        (d/'ATOMX_AXIAL').write_text('wrong\n')
    else:
        (d/'ATOMX_ERI_COMPACT').write_text('unexpected\n')
    (d/'integrals.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_integrals(d)


@pytest.fixture(scope='module')
def native():
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mbpt import build_virtual_basis
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    od = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    for k in (-1, 1, -2):
        _, v = build_virtual_basis(core, k, e_max=50, potential_mode='VN', orthogonal_to_occupied=True)
        od[2, k] = v[0][:2]
    return build_spinor_integrals(od, core._bsp_data, 4, 30,
        occupied_keys=[(1, -1)], virtual_keys=[(2, -1), (2, 1), (2, -2)],
        nuclear_potential=core._nuclear_potential, include_breit=True)


@pytest.mark.parametrize('model', ['CCSD', 'CCSDT'])
def test_live_native_compact_complete_heff_and_three_electron_fci(native, tmp_path, model):
    executable = os.environ.get('ATOMX_EXPT_COMPACT_ERI_EXECUTABLE')
    if not executable:
        pytest.skip('set ATOMX_EXPT_COMPACT_ERI_EXECUTABLE for complete live storage qualification')
    from atomx.angular.csf_hamiltonian import _ops_sign
    obj = native
    # A complex unitary phase of a complete radial multiplet, including all
    # bra/ket/core/virtual entries, tests the C reader's conjugation direction.
    phase = np.array([np.exp(.37j) if k[:2] == (2, -1) else 1. for k in obj.labels])
    h = phase.conj()[:, None]*obj.one_body_Ha*phase[None, :]
    f = phase.conj()[:, None]*obj.fock_Ha*phase[None, :]
    packed = np.array([obj.eri(p, q, r, s)*phase[p].conjugate()*phase[q].conjugate()*phase[r]*phase[s]
                       for p in range(9) for q in range(p+1, 10)
                       for r in range(p, 9) for s in range(r+1, 10)])
    obj = SpinorIntegrals(obj.labels, 2, .5*(h+h.conj().T), .5*(f+f.conj().T),
                         packed, obj.frozen_energy_Ha, obj.reference_energy_Ha, obj.identity)
    results = []
    for storage in ['packed', 'axial_compact']:
        results.append(compute(obj, executable=executable, directory=tmp_path/storage,
            model=model, sector='0h1p', active_indices=(2, 3), tensor_symmetry='axial',
            integral_storage=storage, tolerance=1e-12, threads=1, timeout_seconds=180))
    assert abs(results[0].core_energy_Ha-results[1].core_energy_Ha) < 1e-9
    assert np.max(abs(results[0].model_space_matrix_Ha-results[1].model_space_matrix_Ha)) < 1e-9
    if model == 'CCSDT':
        # Independent direct fermion assembly, including complete core and
        # virtual determinants, rather than a second CC implementation.
        states = list(itertools.combinations(range(10), 3))
        masks = [sum(1 << i for i in s) for s in states]
        index = {m: i for i, m in enumerate(masks)}
        matrix = np.eye(len(masks), dtype=complex)*obj.frozen_energy_Ha
        for j, mask in enumerate(masks):
            for p, q in itertools.product(range(10), repeat=2):
                sign = _ops_sign(mask, ((p, True), (q, False)))
                target = (mask & ~(1 << q)) | (1 << p)
                if sign:
                    matrix[index[target], j] += sign*obj.one_body_Ha[p, q]
            for p, q in itertools.combinations(range(10), 2):
                for r, s in itertools.combinations(range(10), 2):
                    sign = _ops_sign(mask, ((p, True), (q, True), (s, False), (r, False)))
                    target = (mask & ~(1 << r) & ~(1 << s)) | (1 << p) | (1 << q)
                    if sign:
                        matrix[index[target], j] += sign*obj.eri(p, q, r, s)
        energies = np.linalg.eigvalsh(matrix)
        totals = results[1].core_energy_Ha+np.linalg.eigvals(results[1].model_space_matrix_Ha).real
        for value in totals:
            assert np.min(abs(energies-value)) < 1e-8
