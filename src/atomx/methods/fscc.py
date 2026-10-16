"""Integral-driven all-order CCSD / FS-CCSDT through the EXP-T backend.

Uses the Atomx ATXCC001 interface patch, not upstream's unfinished PySCF
reader. This is a genuine external tensor amplitude solver, distinct from
the native finite determinant CCSD/CCSDT and EOM-EA proof engines. No properties, CI+all-order
effective operator or resonance certificate are implied by an energy result.
"""
import hashlib
import json
import re
import struct
import subprocess
import sys
from collections import Counter
from copy import copy
from dataclasses import dataclass
from math import comb
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np

from atomx.numerics.axial_eri import AxialPairLayout

# Complete sector source from the independently checked pinned Fov*T2 repair.
# A different diagram/source requires its own retained-equation validation.
_NONCANONICAL_01_SOURCE_SHA256 = (
    'f73ee35aead1a0f856012db090d50892e0503691aa0551edd36ffa6b6f32204c'
)
_COMPACT_ERI_INTERFACE_SHA256 = '0af392706d19d36ac6a21a35b21d09b93e6605fc3fd32e6235f4cdee13269c34'
_LEGACY_DIVEPS_SOURCE_SHA256 = 'd8a57389490352b214e188a336735288a87776e12b78ed2277e23c467671384c'
_EXACT_NONZERO_DIVEPS_SOURCE_SHA256 = 'dd17fb3cb207bbe7d6430761c0bc4698abbd49fc994d6bc74f00622dded3b547'


def _numerator_division_threshold(backend, binary):
    """Bind the residual contract to the actual denominator-engine source."""
    source = backend.get('source_sha256', {}).get('src/rcc/engine/diveps.c')
    capability = backend.get('numerator_division')
    if (source == _LEGACY_DIVEPS_SOURCE_SHA256
            and capability in (None, 'ATOMX_LEGACY_NUMERATOR_SKIP_1E14')):
        return 1e-14
    if (source == _EXACT_NONZERO_DIVEPS_SOURCE_SHA256
            and capability == 'ATOMX_EXACT_NONZERO_DIVEPS_001'
            and b'ATOMX_EXACT_NONZERO_DIVEPS_001' in binary):
        return 0.
    raise ValueError('raw residual bound requires a source-bound numerator-division engine')

# The qualified 0h1p producer stores a bra-first matrix. Raw parsing remains
# separate from its physical ket conversion; unknown producer conventions reject.
_ATTACHMENT_HEFF_SOURCE_SHA256 = MappingProxyType({
    'src/rcc/heff/heff.c': '90709f8d34f188bad3bafb0b811379ef622fcc383db87326c1ecdae4c9d261a7',
    'src/rcc/heff/formatted_heff_file.c': 'c345eb795e8734d98b03ca27ece234c4ba50e3885fa733ba7cbc019e650a041a',
    'src/rcc/heff/slater_rules.c': '281081b89884a562f9dc9ad44f7771b5800b3d52b78f2a3983900be3451a6127',
    'src/rcc/linalg/eigensolver.c': '324781a8bd752229c19d72fd069984f6b7fc5b9ac7102ce44b987d398d26856c',
    'src/rcc/heff/model_vectors.c': '8c6cd4f20a1ec8e65418e858086eb1b46ffd15716e08c8a97014c944aa180198',
})


def _attachment_ket_matrix(raw_matrix, backend):
    """Convert the qualified raw bra matrix without Hermitian averaging.

    This is a representation change, not a new wave, physical metric or
    omitted-rank certificate. The matrix is relative to the correlated core.
    """
    _validate_attachment_heff_source(backend)
    matrix = np.asarray(raw_matrix).conj().T
    return np.frombuffer(matrix.tobytes(), dtype=matrix.dtype).reshape(matrix.shape)


def _validate_attachment_heff_source(backend):
    sources = backend.get('source_sha256', {})
    if not isinstance(sources, dict) or any(
            sources.get(path) != sha for path, sha in _ATTACHMENT_HEFF_SOURCE_SHA256.items()):
        raise ValueError('0h1p requires the source-bound qualified HEFF bra-to-ket convention')


def atomic_integrals(core, *, virtual_lmax, virtual_energy_max_Ha, frozen_keys=(),
                     max_integral_bytes=2*1024**3, representation='packed', radial_pair_cache_size=1024,
                     include_breit=False, breit_frequency='static', transverse_cache_size=128):
    """Construct every positive-energy virtual in a declared band and l range.

    Includes bound and pseudo-continuum roots; no per-kappa count truncation.
    All stored core subshells must be closed and converged. Only explicitly
    named frozen subshells are excluded from the amplitude equations.
    Returns native integrals with the energy window and radial identities.
    Explicit ``include_breit=True`` includes complete static Breit.
    ``breit_frequency='orbital_differences'`` selects real transverse exchange
    using the actual core/VN virtual eigenenergies, fixed throughout FSCC.
    It does not feed dressed Fock diagonals back into the photon frequencies.
    """
    from atomx.methods.atomic_space import build_atomic_space
    from atomx.physics.csf.spinor_integrals import build_spinor_integrals
    if breit_frequency not in ('static', 'orbital_differences'):
        raise ValueError('unknown Breit frequency prescription')
    if breit_frequency != 'static' and not include_breit:
        raise ValueError('finite-frequency Breit requires include_breit=True')
    space = build_atomic_space(core, virtual_lmax=virtual_lmax,
                               virtual_energy_max_Ha=virtual_energy_max_Ha)
    od = space.orbitals
    frozen = tuple(tuple(k) for k in frozen_keys)
    if len(set(frozen)) != len(frozen) or any(k not in space.occupied_keys for k in frozen):
        raise ValueError('invalid frozen core partition')
    occupied = tuple(k for k in space.occupied_keys if k not in frozen)
    virtual = space.virtual_keys
    frequencies = None
    if breit_frequency == 'orbital_differences':
        frequencies = {(s.n, s.kappa): core.orbital_energies[s.label]
                       for s in core.subshells}
        frequencies.update({k: space.identity['virtual_orbital_energies_Ha'][str(k)]
                            for k in virtual})
    a = build_spinor_integrals(od, core._bsp_data, core.Z, core._bsp_data['ns'],
        occupied_keys=occupied, virtual_keys=virtual, frozen_keys=frozen,
        nuclear_potential=core._nuclear_potential, max_integral_bytes=max_integral_bytes,
        representation=representation, radial_pair_cache_size=radial_pair_cache_size,
        include_breit=include_breit, breit_orbital_energies=frequencies,
        transverse_cache_size=transverse_cache_size)
    identity = dict(a.identity, **space.identity)
    # Physical arrays/snapshot are already owned and validated. Adding only
    # virtual-band identity must not repeat every spherical core contraction.
    updated = copy(a)
    object.__setattr__(updated, 'identity', MappingProxyType(identity))
    return updated


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def _axial_labels(integrals, *, max_omega_x2=62):
    """Complete physical spinor M/parity, using the pinned Dinfh range."""
    if (isinstance(max_omega_x2, bool) or not isinstance(max_omega_x2, (int, np.integer))
            or max_omega_x2 not in (62, 126)):
        raise ValueError('supported explicit axial ranges are62/126')
    groups = {}
    symmetry = []
    for i, (n, kappa, m2) in enumerate(integrals.labels):
        if (not all(isinstance(x, (int, np.integer)) and not isinstance(x, bool)
                    for x in (n, kappa, m2)) or kappa == 0 or
                m2 == 0 or abs(m2) % 2 != 1 or abs(m2) > 2*abs(kappa)-1 or
                6*abs(m2) > max_omega_x2):
            raise ValueError('invalid axial spinor or insufficient rank-six irrep range')
        groups.setdefault((n, kappa, i < integrals.noccupied), set()).add(m2)
        l = kappa if kappa > 0 else -kappa-1
        symmetry.append((int(m2), (-1)**int(l)))
    for (_, kappa, _), magnetic in groups.items():
        if magnetic != set(range(-2*abs(kappa)+1, 2*abs(kappa), 2)):
            raise ValueError('axial integral space requires complete magnetic multiplets')
    a = np.asarray(symmetry, dtype=np.int32)
    forbidden = np.any(a[:, None, :] != a[None, :, :], axis=2)
    if np.any(integrals.fock_Ha[forbidden] != 0):
        raise ValueError('nonzero signed Fock entries violate declared axial symmetry')
    return a


def _axial_chunks(integrals, symmetry, max_chunk_bytes, *, compact=False):
    """Validate every exported packed coordinate with bounded row masks."""
    n = len(symmetry)
    p, q, offset = 0, 1, 0
    def right_pairs(p):
        r, s = np.triu_indices(n-p, 1)
        r, s = r+p, s+p
        return symmetry[r, 0]+symmetry[s, 0], symmetry[r, 1]*symmetry[s, 1]
    m2, parity = right_pairs(p)
    for chunk in integrals.iter_packed_eri(max_chunk_bytes=max_chunk_bytes):
        begin = 0
        kept = []
        while begin < len(chunk):
            if p >= n-1:
                raise ValueError('oversized packed axial integral stream')
            count = min(len(chunk)-begin, len(m2)-offset)
            allowed = ((m2[offset:offset+count] == symmetry[p, 0]+symmetry[q, 0]) &
                       (parity[offset:offset+count] == symmetry[p, 1]*symmetry[q, 1]))
            if np.any(chunk[begin:begin+count][~allowed] != 0):
                raise ValueError('nonzero signed ERI entries violate declared axial symmetry')
            if compact:
                kept.append(chunk[begin:begin+count][allowed])
            begin += count
            offset += count
            if offset == len(m2):
                q += 1
                offset = 0
                if q == n:
                    p += 1
                    q = p+1
                    if p < n-1:
                        m2, parity = right_pairs(p)
        yield np.concatenate(kept) if compact else chunk
    if p != n-1 or offset != 0:
        raise ValueError('incomplete packed axial integral stream')


def _axial_irrep(m2, parity, *, max_omega_x2=62):
    # Pinned generate_irreps_Dinfh: half-integer g then u, +M then -M.
    return (0 if parity == 1 else max_omega_x2)+abs(m2)-1+int(m2 < 0)


def _model_matrix(blocks, sector, active, symmetry, *, axial_max_omega_x2=62):
    """Restore complete raw model order; reject missing or extra blocks."""
    if sector == '0h0p':
        if set(blocks) != {('0h0p', 1)} or blocks['0h0p', 1].shape != (1, 1):
            raise RuntimeError('incomplete closed-sector effective matrix')
        return None
    if symmetry is None:
        expected = {('0h0p', 1), (sector, 1)}
        if set(blocks) != expected or blocks[sector, 1].shape != (len(active), len(active)):
            raise RuntimeError('incomplete C1 model-space effective matrix')
        return blocks[sector, 1]
    numbers = [_axial_irrep(*symmetry[i], max_omega_x2=axial_max_omega_x2) for i in active]
    # Original formatted HEFF numbers relative to first nonempty irrep.
    expected = {('0h0p', 1)} | {(sector, r-min(numbers)+1) for r in set(numbers)}
    if set(blocks) != expected:
        raise RuntimeError('incomplete axial model-space symmetry blocks')
    matrix = np.zeros((len(active), len(active)), dtype=complex)
    for number in sorted(set(numbers)):
        indices = [i for i, r in enumerate(numbers) if r == number]
        block = blocks[sector, number-min(numbers)+1]
        if block.shape != (len(indices), len(indices)):
            raise RuntimeError('axial model-space block dimensions differ')
        matrix[np.ix_(indices, indices)] = block
    return np.frombuffer(matrix.tobytes(), dtype=complex).reshape(matrix.shape)


def resource_estimate(integrals, *, model='CCSD', sector='0h0p', active_count=0,
                       tensor_symmetry='C1', active_indices=None, axial_max_omega_x2=62,
                       integral_storage='packed'):
    """Conservative tensor working estimate; no determinant enumeration.

    Accounts for uncompressed rank-six arrays and 32 simultaneous diagrams.
    Actual backend memory limit still applies; this is not a peak-memory proof.
    """
    if model not in ('CCSD', 'CCSDT') or sector not in ('0h0p', '0h1p', '1h0p'):
        raise ValueError('supported models CCSD/CCSDT; sectors 0h0p/0h1p/1h0p')
    o = integrals.noccupied
    v = len(integrals.labels)-o
    if not 0 <= active_count <= (v if sector == '0h1p' else o):
        raise ValueError('invalid model-space size')
    if tensor_symmetry not in ('C1', 'axial'):
        raise ValueError('tensor symmetry must be C1 or axial')
    _validate_integral_storage(integral_storage, tensor_symmetry)
    core = o*v+o*o*v*v
    independent = o*v+comb(o, 2)*comb(v, 2)
    valence = 0
    if sector == '0h1p':
        valence = active_count*(v+o*v*v)
    elif sector == '1h0p':
        valence = active_count*(o+o*o*v)
    if model == 'CCSDT':
        core += o**3*v**3
        independent += comb(o, 3)*comb(v, 3)
        if sector == '0h1p':
            valence += active_count*o*o*v**3
        elif sector == '1h0p':
            valence += active_count*o**3*v*v
    if tensor_symmetry == 'axial':
        symmetry = _axial_labels(integrals, max_omega_x2=axial_max_omega_x2)
        active = tuple(() if active_indices is None else active_indices)
        allowed = range(o, o+v) if sector == '0h1p' else range(o)
        if (len(active) != active_count or len(set(active)) != len(active)
                or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                       or i not in allowed for i in active)
                or (sector == '0h0p' and active)):
            raise ValueError('exact axial resource counting requires matching active spinor indices')
        selected = set(active)
        for i in allowed:
            group = {j for j in allowed if integrals.labels[j][:2] == integrals.labels[i][:2]}
            if selected & group and not group.issubset(selected):
                raise ValueError('axial model space requires complete magnetic multiplets')
        occupied, virtual = symmetry[:o], symmetry[o:]
        model_space = symmetry[list(active)]

        def populations(groups):
            counts = Counter({(0, 1): 1})
            for group in groups:
                entries = Counter((int(m), int(p)) for m, p in group)
                combined = Counter()
                for (m, p), number in counts.items():
                    for (mm, pp), other in entries.items():
                        combined[m+mm, p*pp] += number*other
                counts = combined
            return counts

        def match(left, right):
            a, b = populations(left), populations(right)
            return sum(number*b[key] for key, number in a.items())

        core = match([occupied], [virtual])+match([occupied, occupied], [virtual, virtual])
        valence = 0
        if sector == '0h1p':
            valence = match([model_space], [virtual])+match([model_space, occupied], [virtual, virtual])
        elif sector == '1h0p':
            valence = match([model_space], [occupied])+match([model_space, virtual], [occupied, occupied])
        if model == 'CCSDT':
            core += match([occupied]*3, [virtual]*3)
            if sector == '0h1p':
                valence += match([model_space]+[occupied]*2, [virtual]*3)
            elif sector == '1h0p':
                valence += match([model_space]+[virtual]*2, [occupied]*3)
        # Count all ordered quartic blocks, including hppp and every other
        # partition. Pauli/permutation compression is deliberately not used.
        quartic = match([symmetry]*2, [symmetry]*2)
        bank_bytes = integrals.dense_eri_bytes
        index_bytes = 0
        extra = {}
        if integral_storage == 'axial_compact':
            layout = AxialPairLayout(symmetry)
            bank_bytes = 16*layout.count
            # Python planning and C indexing arrays fit together in this
            # explicit envelope; no tensor/intermediate envelope is reduced.
            index_bytes = 2*layout.index_bytes
            extra = dict(eri_storage=integral_storage, compact_eri_coordinates=layout.count,
                         compact_eri_bytes=bank_bytes, compact_index_bound_bytes=index_bytes,
                         original_packed_eri_bytes=integrals.dense_eri_bytes)
        return dict(occupied=o, virtual=v, core_independent_amplitudes=independent,
            raw_tensor_coordinates=core+valence, axial_core_coordinates=core,
            axial_valence_coordinates=valence, axial_all_ordered_quartic_coordinates=quartic,
            axial_max_omega_x2=axial_max_omega_x2,
            estimated_working_bytes=32*16*(core+valence)+32*quartic+bank_bytes+index_bytes,
            estimate_scope=('exact ordered M/parity support,32 amplitude-sized arrays,two ALL ordered quartic tensors,complete signed ERI bank and declared index envelope; no Pauli compression; backend cap still enforced, not peak-RSS proof'
                            if extra else 'exact ordered M/parity support,32 amplitude-sized arrays,two ALL ordered quartic tensors,complete packed ERI; no Pauli compression; backend cap still enforced, not peak-RSS proof'),
            **extra)
    return dict(occupied=o, virtual=v, core_independent_amplitudes=independent,
                raw_tensor_coordinates=core+valence,
                estimated_working_bytes=32*16*(core+valence)+32*(o**4+v**4)+integrals.dense_eri_bytes,
                estimate_scope='32 uncompressed amplitude-sized arrays, two hhhh/pppp tensors, packed ERI; backend cap also enforced')


def _validate_integral_storage(storage, symmetry):
    if storage not in ('packed', 'axial_compact') or (storage == 'axial_compact' and symmetry != 'axial'):
        raise ValueError('explicit axial_compact integral storage requires axial symmetry')


def export_integrals(integrals, directory, *, max_integral_file_bytes=2*1024**3,
                     max_chunk_bytes=4*1024**2, tensor_symmetry='C1', axial_max_omega_x2=62,
                     integral_storage='packed'):
    """Write versioned, signed Fock/ERI files and source/model identity."""
    if sys.byteorder != 'little':
        raise RuntimeError('ATXCC001 currently supports little-endian hosts')
    n = len(integrals.labels)
    _validate_integral_storage(integral_storage, tensor_symmetry)
    symmetry = (_axial_labels(integrals, max_omega_x2=axial_max_omega_x2)
                if tensor_symmetry == 'axial' else None)
    layout = AxialPairLayout(symmetry) if integral_storage == 'axial_compact' else None
    bank_bytes = integrals.dense_eri_bytes if layout is None else 16*layout.count
    required = 28+12*n+16*n*n+bank_bytes
    if required > max_integral_file_bytes:
        raise MemoryError(f'complete integral files need {required} bytes; cap {max_integral_file_bytes}')
    if max_chunk_bytes < 16:
        raise ValueError('integral chunk must hold at least one complex value')
    if tensor_symmetry not in ('C1', 'axial'):
        raise ValueError('tensor symmetry must be C1 or axial')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/'ATOMX_ONE').open('wb') as f:
        f.write(b'ATXCC001')
        f.write(struct.pack('<idd', n, 0., integrals.reference_energy_Ha))
        for i, eps in enumerate(np.diag(integrals.fock_Ha).real):
            f.write(struct.pack('<id', int(i < integrals.noccupied), float(eps)))
        integrals.fock_Ha.astype('<c16', copy=False).tofile(f)
    with (directory/'ATOMX_TWO').open('wb') as stream:
        chunks = (integrals.iter_packed_eri(max_chunk_bytes=max_chunk_bytes)
                  if symmetry is None else _axial_chunks(integrals, symmetry, max_chunk_bytes,
                                                        compact=layout is not None))
        for chunk in chunks:
            chunk.astype('<c16', copy=False).tofile(stream)
    if (directory/'ATOMX_TWO').stat().st_size != bank_bytes:
        raise ValueError('exported complete integral length failed')
    manifest = dict(abi='ATXCC001', labels=integrals.labels, noccupied=integrals.noccupied,
                    reference_energy_Ha=integrals.reference_energy_Ha,
                    frozen_energy_Ha=integrals.frozen_energy_Ha,
                    identity=dict(integrals.identity),
                    export=dict(logical_bytes=required, max_integral_file_bytes=max_integral_file_bytes,
                                max_chunk_bytes=max_chunk_bytes, producer=integrals.diagnostics()),
                    files={name: _sha(directory/name) for name in ('ATOMX_ONE', 'ATOMX_TWO')})
    if symmetry is not None:
        (directory/'ATOMX_AXIAL').write_text(f'ATOMX_AXIAL_001 {n}\n' +
            ''.join(f'{m} {p}\n' for m, p in symmetry))
        manifest['tensor_symmetry'] = dict(kind='axial', max_omega_x2=axial_max_omega_x2,
            spinors=symmetry.tolist(), forbidden_signed_integral_gate='exact zero, every exported coordinate',
            sidecar_sha256=_sha(directory/'ATOMX_AXIAL'))
    if layout is not None:
        (directory/'ATOMX_ERI_COMPACT').write_text(f'ATOMX_AXIAL_ERI_001 {n} {layout.count}\n')
        manifest['eri_storage'] = dict(kind='axial_compact', count=layout.count,
            sidecar_sha256=_sha(directory/'ATOMX_ERI_COMPACT'),
            order='original outer-p,q then allowed r>=p,s>r; all signed duplicates retained',
            original_packed_eri_bytes=integrals.dense_eri_bytes)
    (directory/'integrals.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_integrals(directory, *, max_integral_bytes=2*1024**3):
    """Replay signed exported inputs after byte hashes, sizes and ABI checks."""
    from atomx.physics.csf.spinor_integrals import (
        SpinorIntegrals,
        _packed_offsets,
        normal_ordered_one_body,
        packed_size,
    )
    directory = Path(directory)
    manifest = json.loads((directory/'integrals.json').read_text())
    if manifest['abi'] != 'ATXCC001' or sys.byteorder != 'little':
        raise ValueError('unsupported signed-integral ABI')
    if set(manifest['files']) != {'ATOMX_ONE', 'ATOMX_TWO'}:
        raise ValueError('unexpected integral files')
    total = sum((directory/name).stat().st_size for name in manifest['files'])
    if total > max_integral_bytes:
        raise MemoryError('saved complete integral space exceeds resource cap')
    if any(_sha(directory/name) != digest for name, digest in manifest['files'].items()):
        raise ValueError('saved integral hash mismatch')
    n = len(manifest['labels'])
    storage = manifest.get('eri_storage', {}).get('kind', 'packed')
    symmetry = manifest.get('tensor_symmetry')
    _validate_integral_storage(storage, None if symmetry is None else symmetry.get('kind'))
    if (directory/'ATOMX_ONE').stat().st_size != 28+12*n+16*n*n:
        raise ValueError('saved integral file length mismatch')
    with (directory/'ATOMX_ONE').open('rb') as f:
        if f.read(8) != b'ATXCC001':
            raise ValueError('saved integral magic mismatch')
        count, enuc, reference = struct.unpack('<idd', f.read(20))
        if count != n or enuc != 0 or reference != manifest['reference_energy_Ha']:
            raise ValueError('integral header/manifest mismatch')
        rows = [struct.unpack('<id', f.read(12)) for _ in range(n)]
        fock = np.fromfile(f, dtype='<c16').reshape(n, n)
    o = manifest['noccupied']
    if any(occ != int(i < o) or eps != fock[i, i].real for i, (occ, eps) in enumerate(rows)):
        raise ValueError('saved Fock/occupation mismatch')
    expected = None
    if symmetry is not None:
        # This view carries only archived coordinate arrays for validation;
        # it is not a solved state or a physical operator substitute.
        view = SimpleNamespace(labels=tuple(map(tuple, manifest['labels'])),
                               noccupied=o, fock_Ha=fock)
        expected = _axial_labels(view, max_omega_x2=symmetry.get('max_omega_x2'))
        text = f'ATOMX_AXIAL_001 {n}\n'+''.join(f'{m} {p}\n' for m, p in expected)
        if (symmetry.get('kind') != 'axial' or
                symmetry.get('spinors') != expected.tolist() or
                _sha(directory/'ATOMX_AXIAL') != symmetry.get('sidecar_sha256') or
                (directory/'ATOMX_AXIAL').read_text() != text):
            raise ValueError('saved axial sidecar identity mismatch')
    if storage == 'axial_compact':
        layout = AxialPairLayout(expected)
        info = manifest['eri_storage']
        text = f'ATOMX_AXIAL_ERI_001 {n} {layout.count}\n'
        if (info.get('count') != layout.count or
                info.get('original_packed_eri_bytes') != 16*packed_size(n) or
                _sha(directory/'ATOMX_ERI_COMPACT') != info.get('sidecar_sha256') or
                (directory/'ATOMX_ERI_COMPACT').read_text() != text or
                (directory/'ATOMX_TWO').stat().st_size != 16*layout.count):
            raise ValueError('saved compact integral layout/count mismatch')
        # This loader returns the original owned packed physical object.
        # Decompression is explicit in the resource contract, never a hidden
        # large allocation admitted by the smaller on-disk size alone.
        required = total+32*packed_size(n)+2*layout.index_bytes+32*n*n
        if required > max_integral_bytes:
            raise MemoryError('compact integral replay requires admitted full packed decompression')
        bank = np.fromfile(directory/'ATOMX_TWO', dtype='<c16')
        packed = np.zeros(packed_size(n), dtype=complex)
        offsets = _packed_offsets(n)
        position = 0
        for pq, (p, _q, pairs) in enumerate(layout.rows()):
            r, s = pairs.T
            rx, sx = r-p, s-p
            indices = offsets[pq]+(n-p)*rx+sx-(rx+1)*(rx+2)//2
            packed[indices] = bank[position:position+len(pairs)]
            position += len(pairs)
        assert position == layout.count
        del bank
    else:
        if (directory/'ATOMX_TWO').stat().st_size != 16*packed_size(n):
            raise ValueError('saved integral file length mismatch')
        if (directory/'ATOMX_ERI_COMPACT').exists():
            raise ValueError('unexpected compact integral sidecar for packed storage')
        packed = np.fromfile(directory/'ATOMX_TWO', dtype='<c16')
    h = normal_ordered_one_body(fock, packed, o)
    if np.max(abs(h-h.conj().T)) > 1e-10:
        raise ValueError('saved normal-ordered one-body Hermitian gate failed')
    h = .5*(h+h.conj().T)
    loaded = SpinorIntegrals(tuple(tuple(x) for x in manifest['labels']), o, h, fock, packed,
                            manifest['frozen_energy_Ha'], reference, manifest['identity'])
    if symmetry is not None:
        for _ in _axial_chunks(loaded, expected, 4*1024**2):
            pass
    return loaded


def read_heff(path):
    """Read raw signed formatted blocks, preserving their stored convention.

    Raw EXP-T bra-first blocks are not physical ket effective Hamiltonians.
    The qualified 0h1p conversion is performed separately by compute.
    """
    lines = [line.split('#')[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if not lines or lines[0] != 'complex':
        raise ValueError('expected complex formatted HEFF')
    blocks = {}
    i = 1
    while i < len(lines):
        sector = lines[i]
        if not re.fullmatch(r'[01]h[01]p', sector):
            raise ValueError('unexpected HEFF sector')
        i += 1
        while i < len(lines) and not re.fullmatch(r'[01]h[01]p', lines[i]):
            rep, dim = map(int, lines[i].split())
            i += 1
            values = []
            while len(values) < 2*dim*dim and i < len(lines):
                # Fixed-width values can touch when the sign is negative.
                values.extend(float(x) for x in re.findall(r'[+-]?\d+\.\d+[Ee][+-]\d+', lines[i]))
                i += 1
            if len(values) != 2*dim*dim or dim < 1:
                raise ValueError('truncated or oversized HEFF matrix')
            a = np.asarray(values).reshape(-1, 2)
            block = np.frombuffer((a[:, 0]+1j*a[:, 1]).tobytes(), dtype=complex).reshape(dim, dim)
            if (sector, rep) in blocks:
                raise ValueError('duplicate HEFF sector/representation block')
            blocks[sector, rep] = block
    return blocks


@dataclass(frozen=True)
class FSCCResult:
    """Raw effective blocks and a separately qualified model-space matrix.

    ``effective_matrices`` preserves formatted raw values. In 0h1p,
    ``model_space_matrix_Ha`` uses the physical ket convention, relative to
    the same correlated core. The hole-sector matrix retains its raw convention;
    its physical ket orientation has not been qualified. No wave metric is
    inferred from either matrix.
    """
    model: str
    sector: str
    core_energy_Ha: float
    sector_energies_Ha: tuple
    effective_matrices: object
    amplitude_update_max: float
    equation_residual_bound_Ha: float
    provenance: object
    model_space_matrix_Ha: object = None


def compute(integrals, *, executable, directory, model='CCSD', sector='0h0p',
            active_indices=(), tolerance=1e-10, residual_bound_tolerance_Ha=1e-8,
            maxiter=200, memory_bytes=8*1024**3, tile_size=32, threads=2,
            timeout_seconds=600, environment=None, tensor_symmetry='C1',
            max_integral_file_bytes=2*1024**3, integral_storage='packed'):
    """Solve all retained amplitude equations, with explicit resource limits.

    ``active_indices`` supplies exact global model spinor indices. Complementary
    virtuals remain in the amplitudes. No intruder shifts or selections are
    added. Closed and requested valence sectors must both converge before a
    result is returned. On failure, the complete run log remains in directory.
    Nonzero occupied--virtual Fock coupling in 0h1p requires the source-bound
    ATOMX_FOV_T2_001 repair; original equations are rejected before export.
    ``max_integral_file_bytes`` explicitly admits the complete export payload;
    its2GiB default is independent of the backend tensor/memory budget.
    All 0h1p requests require the qualified HEFF producer source convention;
    their model matrix is the physical ket matrix, with raw blocks preserved.
    All requests source-bind the numerator-division engine. Legacy numerator
    skips add a conservative residual floor; exact nonzero division requires
    its qualified source/capability/marker. Neither changes the Hamiltonian.
    ``integral_storage='axial_compact'`` requires exact axial symmetry and a
    separately source-qualified backend; default packed storage is unchanged.
    """
    if (isinstance(max_integral_file_bytes, (bool, np.bool_))
            or not isinstance(max_integral_file_bytes, (int, np.integer)) or max_integral_file_bytes < 1):
        raise ValueError('positive integer integral export byte cap required')
    active = tuple(sorted(active_indices))
    if tensor_symmetry not in ('C1', 'axial'):
        raise ValueError('tensor symmetry must be C1 or axial')
    _validate_integral_storage(integral_storage, tensor_symmetry)
    n, o = len(integrals.labels), integrals.noccupied
    allowed = range(o, n) if sector == '0h1p' else range(o)
    if sector == '0h0p' and active or sector != '0h0p' and not active:
        raise ValueError('valence sector requires explicit model-space indices')
    if len(set(active)) != len(active) or any(i not in allowed for i in active):
        raise ValueError('model spinors must belong to the requested particle/hole space')
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in active):
        raise ValueError('model spinor indices must be integers')
    model_groups = {}
    for i in allowed:
        model_groups.setdefault(tuple(integrals.labels[i][:2]), set()).add(i)
    chosen = set(active)
    if any(chosen & group and not group.issubset(chosen) for group in model_groups.values()):
        raise ValueError('atomic model space must retain complete magnetic multiplets')
    estimate = resource_estimate(integrals, model=model, sector=sector, active_count=len(active))
    if tensor_symmetry == 'C1' and estimate['estimated_working_bytes'] > memory_bytes:
        raise MemoryError(f'full requested tensors exceed memory cap: {estimate}')
    if not (0 < tolerance < 1e-3 and residual_bound_tolerance_Ha > 0
            and maxiter > 0 and 0 < tile_size <= 2048 and threads > 0 and timeout_seconds > 0):
        raise ValueError('invalid solver controls')
    executable = Path(executable).resolve(strict=True)
    if b'ATOMX_INTEGRALS_V1' not in executable.read_bytes():
        raise ValueError('executable does not implement the Atomx integral ABI')
    backend = json.loads((executable.parent/'atomx_backend.json').read_text())
    if backend['abi'] != 'ATXCC001' or backend['executable_sha256'] != _sha(executable):
        raise ValueError('backend executable/source manifest mismatch')
    numerator_threshold = _numerator_division_threshold(backend, executable.read_bytes())
    denominator_bound = 6*float(np.max(abs(np.diag(integrals.fock_Ha))))
    # The legacy engine leaves a skipped numerator as the new amplitude.
    # In native atomic units, |N-D*t_old| <= |D|*update+(|D|+1)*threshold.
    # Thus a zero fixed-point update must not imply a zero physical residual.
    skip_floor = (denominator_bound+1.)*numerator_threshold
    if integral_storage == 'axial_compact' and (
            'axial_compact' not in backend.get('eri_storage_layouts', ()) or
            backend.get('source_sha256', {}).get('src/rcc/interfaces/pyscf_interface.c') !=
            _COMPACT_ERI_INTERFACE_SHA256 or b'ATOMX_AXIAL_ERI_V1' not in executable.read_bytes()):
        raise ValueError('compact axial integrals require source-bound storage/index capability')
    fov_max = float(np.max(abs(integrals.fock_Ha[:o, o:]), initial=0))
    if sector == '0h1p' and fov_max != 0 and (
            backend.get('noncanonical_attachment_equation') != 'ATOMX_FOV_T2_001' or
            backend.get('source_sha256', {}).get('src/rcc/models/sector01.c') !=
            _NONCANONICAL_01_SOURCE_SHA256):
        raise ValueError(
            'noncanonical 0h1p requires the source-bound ATOMX_FOV_T2_001 '
            'attachment equation repair; original/unverified equations rejected'
        )
    if sector == '0h1p':
        _validate_attachment_heff_source(backend)
    if tensor_symmetry == 'axial' and (
            'axial' not in backend.get('tensor_symmetries', ()) or
            backend.get('axial_max_omega_x2') not in (62, 126) or
            b'ATOMX_AXIAL_V1' not in executable.read_bytes()):
        raise ValueError('backend lacks the bound exact axial/parity capability')
    axial_range = backend.get('axial_max_omega_x2', 62) if tensor_symmetry == 'axial' else 62
    if tensor_symmetry == 'axial':
        if axial_range == 126 and backend.get('source_sha256', {}).get(
                'src/rcc/include/comdef.h') != '18b40af2a5911d49d815988dc1954e2ee40b2e52428b5ab9b785616c679c839b':
            raise ValueError('expanded axial range requires source-bound512-irrep allocation')
        estimate = resource_estimate(integrals, model=model, sector=sector,
            active_count=len(active), tensor_symmetry='axial', active_indices=active,
            axial_max_omega_x2=axial_range, integral_storage=integral_storage)
        if estimate['estimated_working_bytes'] > memory_bytes:
            raise MemoryError(f'complete exact axial tensors exceed memory cap: {estimate}')
    directory = Path(directory).resolve()
    manifest = export_integrals(integrals, directory, tensor_symmetry=tensor_symmetry,
                                max_integral_file_bytes=max_integral_file_bytes,
                                axial_max_omega_x2=axial_range, integral_storage=integral_storage)
    symmetry = (None if tensor_symmetry == 'C1' else
                np.asarray(manifest['tensor_symmetry']['spinors'], dtype=np.int32))
    flags = ' '.join(str(int(i in active)) for i in range(n))
    inp = '\n'.join(['title "Atomx signed native integrals"', 'interface pyscf',
        'integrals ATOMX_ONE ATOMX_TWO', f'model {model.lower()}', f'sector {sector}',
        f'active {flags}', f'maxiter {maxiter}', f'conv {tolerance:.16e}', 'diis 8',
        *(['diis triples'] if model == 'CCSDT' else []),
        f'memory {memory_bytes//1024**2} mb', f'tilesize {tile_size}',
        f'nthreads {threads}', 'disk_usage 0', ''])
    (directory/'cc.inp').write_text(inp)
    provenance = dict(backend='EXP-T Atomx interface', executable_sha256=_sha(executable),
        backend_source_manifest_sha256=_sha(executable.parent/'atomx_backend.json'),
        upstream_revision=backend['upstream_revision'], backend_patch_sha256=backend['patch_sha256'],
        atomx_solver_sha256=_sha(__file__), model=model, sector=sector, active_indices=active,
        model_matrix_order='ascending global active indices, original EXP-T one-particle/hole determinant basis',
        effective_matrices_convention='raw EXP-T formatted row-major bra-first blocks',
        model_space_matrix_convention=('source-bound 0h1p physical ket matrix = raw adjoint'
            if sector == '0h1p' else ('raw 1h0p; physical ket orientation unqualified'
            if sector == '1h0p' else 'no model-space matrix')),
        attachment_heff_source_sha256=(dict(_ATTACHMENT_HEFF_SOURCE_SHA256)
            if sector == '0h1p' else None),
        tensor_symmetry=tensor_symmetry,
        occupied_virtual_fock_max_Ha=fov_max,
        noncanonical_attachment_equation=backend.get('noncanonical_attachment_equation'),
        numerator_division=backend.get('numerator_division', 'pinned legacy numerator skip'),
        numerator_skip_threshold_Ha=numerator_threshold,
        numerator_skip_residual_floor_Ha=skip_floor,
        denominator_kernel_source_sha256=backend['source_sha256']['src/rcc/engine/diveps.c'],
        residual_scope='backend raw Jacobi update bound including the source numerator skip floor; independent retained/Bloch '
                       'and omitted-rank wave residuals are separate validations',
        resources=estimate, tolerance=tolerance, residual_bound_tolerance_Ha=residual_bound_tolerance_Ha,
        memory_bytes=memory_bytes, tile_size=tile_size, threads=threads,
        integral_manifest_sha256=_sha(directory/'integrals.json'), input_sha256=_sha(directory/'cc.inp'),
        integrals=manifest)
    (directory/'request.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with (directory/'cc.log').open('w') as log:
        result = subprocess.run([str(executable), '--no-clean', 'cc.inp'], cwd=directory, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    log = (directory/'cc.log').read_text()
    if ('ATOMX_AXIAL_V1 exact M/parity' in log) != (tensor_symmetry == 'axial'):
        raise RuntimeError('backend did not consume requested tensor symmetry')
    statuses = {s: (int(ok), int(it), *(float(v) for v in (a, b, c)))
        for s, ok, it, a, b, c in re.findall(
            r'ATOMX_CC_STATUS (\dh\dp) (\d+) (\d+) (\S+) (\S+) (\S+)', log)}
    needed = {'0h0p', sector}
    if result.returncode or not needed.issubset(statuses) or any(statuses[s][0] != 1 for s in needed):
        raise RuntimeError(f'EXP-T amplitude solve failed; inspect {directory}/cc.log')
    update = max(max(statuses[s][2:]) for s in needed)
    # With no shifts, residual = diagonal denominator * raw Jacobi update.
    # Six orbital energies bounds all rank <=3 denominators, including valence.
    residual_bound = denominator_bound*update+skip_floor
    if update > tolerance or residual_bound > residual_bound_tolerance_Ha:
        raise RuntimeError(f'raw amplitude residual bound failed: {residual_bound} Ha')
    heff_path = directory/'scratch/HEFF'
    blocks = read_heff(heff_path)
    matrix = _model_matrix(blocks, sector, active, symmetry, axial_max_omega_x2=axial_range)
    if sector == '0h1p':
        matrix = _attachment_ket_matrix(matrix, backend)
    core = blocks['0h0p', 1][0, 0]
    if abs(core.imag) > 1e-10:
        raise RuntimeError('complex closed-shell energy')
    energies = []
    if sector != '0h0p':
        for (s, _), block_matrix in blocks.items():
            if s == sector:
                energies.extend(np.linalg.eigvals(block_matrix)+core)
        if len(energies) != len(active) or max(abs(e.imag) for e in energies) > 1e-8:
            raise RuntimeError('invalid or incomplete model-space energy spectrum')
    else:
        energies = [core]
    provenance.update(statuses=statuses, log_sha256=_sha(directory/'cc.log'),
                      heff_sha256=_sha(heff_path))
    if matrix is not None:
        np.savez_compressed(directory/'model_space_matrix.npz', matrix_Ha=matrix,
                            global_active_indices=np.asarray(active, dtype=np.int32))
        provenance['model_space_matrix_sha256'] = _sha(directory/'model_space_matrix.npz')
    solved = dict(core_energy_Ha=float(core.real), sector_energies_Ha=sorted(float(e.real) for e in energies),
                  amplitude_update_max=update, equation_residual_bound_Ha=residual_bound, provenance=provenance)
    (directory/'result.json').write_text(json.dumps(solved, indent=2)+'\n')
    return FSCCResult(model, sector, solved['core_energy_Ha'], tuple(solved['sector_energies_Ha']),
                      MappingProxyType(blocks), update, residual_bound, MappingProxyType(provenance), matrix)
