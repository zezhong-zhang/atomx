"""Integral-driven all-order CCSD / FS-CCSDT through the EXP-T backend.

Uses the Atomx ATXCC001 interface patch, not upstream's unfinished PySCF
reader. This is a genuine external tensor amplitude solver, distinct from
the native finite determinant CCSD proof engine. No properties, CI+all-order
effective operator or resonance certificate are implied by an energy result.
"""
import hashlib
import json
import re
import struct
import subprocess
import sys
from copy import copy
from dataclasses import dataclass
from math import comb
from pathlib import Path
from types import MappingProxyType

import numpy as np


def atomic_integrals(core, *, virtual_lmax, virtual_energy_max_Ha, frozen_keys=(),
                     max_integral_bytes=2*1024**3, representation='packed', radial_pair_cache_size=1024):
    """Construct every positive-energy virtual in a declared band and l range.

    Includes bound and pseudo-continuum roots; no per-kappa count truncation.
    All stored core subshells must be closed and converged. Only explicitly
    named frozen subshells are excluded from the amplitude equations.
    Returns native integrals with the energy window and radial identities.
    """
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.physics.csf.spinor_integrals import build_spinor_integrals
    if not core.converged or any(s.occupation != 2*abs(s.kappa) for s in core.subshells):
        raise ValueError('tensor atomic reference must be converged and closed shell')
    if not isinstance(virtual_lmax, int) or virtual_lmax < 0 or not np.isfinite(virtual_energy_max_Ha):
        raise ValueError('require nonnegative lmax and finite energy upper bound')
    od = {key: value[:2] for key, value in core.orbitals_by_subshell().items()}
    frozen = tuple(tuple(k) for k in frozen_keys)
    if len(set(frozen)) != len(frozen) or any(k not in od for k in frozen):
        raise ValueError('invalid frozen core partition')
    occupied = tuple(k for k in od if k not in frozen)
    virtual = []
    energies = {}
    for l in range(virtual_lmax+1):
        for k in ((-1,) if l == 0 else (l, -l-1)):
            _, states = build_virtual_basis(core, k, e_max=virtual_energy_max_Ha,
                potential_mode='VN', orthogonal_to_occupied=True)
            n = max([l]+[key[0] for key in od if key[1] == k])+1
            for P, Q, eps in states:
                key = n, k
                od[key] = P, Q
                virtual.append(key)
                energies[str(key)] = eps
                n += 1
    a = build_spinor_integrals(od, core._bsp_data, core.Z, core._bsp_data['ns'],
        occupied_keys=occupied, virtual_keys=virtual, frozen_keys=frozen,
        nuclear_potential=core._nuclear_potential, max_integral_bytes=max_integral_bytes,
        representation=representation, radial_pair_cache_size=radial_pair_cache_size)
    identity = dict(a.identity, virtual_lmax=virtual_lmax,
        virtual_energy_max_Ha=virtual_energy_max_Ha, virtual_orbital_energies_Ha=energies,
        positive_energy_projector='native Dirac VN complement, complete retained band')
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


def resource_estimate(integrals, *, model='CCSD', sector='0h0p', active_count=0):
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
    return dict(occupied=o, virtual=v, core_independent_amplitudes=independent,
                raw_tensor_coordinates=core+valence,
                estimated_working_bytes=32*16*(core+valence)+32*(o**4+v**4)+integrals.dense_eri_bytes,
                estimate_scope='32 uncompressed amplitude-sized arrays, two hhhh/pppp tensors, packed ERI; backend cap also enforced')


def export_integrals(integrals, directory, *, max_integral_file_bytes=2*1024**3,
                     max_chunk_bytes=4*1024**2):
    """Write versioned, signed Fock/ERI files and source/model identity."""
    if sys.byteorder != 'little':
        raise RuntimeError('ATXCC001 currently supports little-endian hosts')
    n = len(integrals.labels)
    required = 28+12*n+16*n*n+integrals.dense_eri_bytes
    if required > max_integral_file_bytes:
        raise MemoryError(f'complete integral files need {required} bytes; cap {max_integral_file_bytes}')
    if max_chunk_bytes < 16:
        raise ValueError('integral chunk must hold at least one complex value')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory/'ATOMX_ONE').open('wb') as f:
        f.write(b'ATXCC001')
        f.write(struct.pack('<idd', n, 0., integrals.reference_energy_Ha))
        for i, eps in enumerate(np.diag(integrals.fock_Ha).real):
            f.write(struct.pack('<id', int(i < integrals.noccupied), float(eps)))
        integrals.fock_Ha.astype('<c16', copy=False).tofile(f)
    with (directory/'ATOMX_TWO').open('wb') as stream:
        for chunk in integrals.iter_packed_eri(max_chunk_bytes=max_chunk_bytes):
            chunk.astype('<c16', copy=False).tofile(stream)
    if (directory/'ATOMX_TWO').stat().st_size != integrals.dense_eri_bytes:
        raise ValueError('exported complete integral length failed')
    manifest = dict(abi='ATXCC001', labels=integrals.labels, noccupied=integrals.noccupied,
                    reference_energy_Ha=integrals.reference_energy_Ha,
                    frozen_energy_Ha=integrals.frozen_energy_Ha,
                    identity=dict(integrals.identity),
                    export=dict(logical_bytes=required, max_integral_file_bytes=max_integral_file_bytes,
                                max_chunk_bytes=max_chunk_bytes, producer=integrals.diagnostics()),
                    files={name: _sha(directory/name) for name in ('ATOMX_ONE', 'ATOMX_TWO')})
    (directory/'integrals.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_integrals(directory, *, max_integral_bytes=2*1024**3):
    """Replay signed exported inputs after byte hashes, sizes and ABI checks."""
    from atomx.physics.csf.spinor_integrals import (
        SpinorIntegrals,
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
    if (directory/'ATOMX_ONE').stat().st_size != 28+12*n+16*n*n or (directory/'ATOMX_TWO').stat().st_size != 16*packed_size(n):
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
    packed = np.fromfile(directory/'ATOMX_TWO', dtype='<c16')
    h = normal_ordered_one_body(fock, packed, o)
    if np.max(abs(h-h.conj().T)) > 1e-10:
        raise ValueError('saved normal-ordered one-body Hermitian gate failed')
    h = .5*(h+h.conj().T)
    return SpinorIntegrals(tuple(tuple(x) for x in manifest['labels']), o, h, fock, packed,
                          manifest['frozen_energy_Ha'], reference, manifest['identity'])


def read_heff(path):
    """Read the complete signed effective matrices, never just printed roots."""
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
    model: str
    sector: str
    core_energy_Ha: float
    sector_energies_Ha: tuple
    effective_matrices: object
    amplitude_update_max: float
    equation_residual_bound_Ha: float
    provenance: object


def compute(integrals, *, executable, directory, model='CCSD', sector='0h0p',
            active_indices=(), tolerance=1e-10, residual_bound_tolerance_Ha=1e-8,
            maxiter=200, memory_bytes=8*1024**3, tile_size=32, threads=2,
            timeout_seconds=600, environment=None):
    """Solve all retained amplitude equations, with explicit resource limits.

    ``active_indices`` supplies exact global model spinor indices. Complementary
    virtuals remain in the amplitudes. No intruder shifts or selections are
    added. Closed and requested valence sectors must both converge before a
    result is returned. On failure, the complete run log remains in directory.
    """
    active = tuple(sorted(active_indices))
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
    if estimate['estimated_working_bytes'] > memory_bytes:
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
    directory = Path(directory).resolve()
    manifest = export_integrals(integrals, directory)
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
        model_matrix_order='ascending global active indices, C1 one-particle/hole determinant basis',
        resources=estimate, tolerance=tolerance, residual_bound_tolerance_Ha=residual_bound_tolerance_Ha,
        memory_bytes=memory_bytes, tile_size=tile_size, threads=threads,
        integral_manifest_sha256=_sha(directory/'integrals.json'), input_sha256=_sha(directory/'cc.inp'),
        integrals=manifest)
    (directory/'request.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with (directory/'cc.log').open('w') as log:
        result = subprocess.run([str(executable), '--no-clean', 'cc.inp'], cwd=directory, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    log = (directory/'cc.log').read_text()
    statuses = {s: (int(ok), int(it), *(float(v) for v in (a, b, c)))
        for s, ok, it, a, b, c in re.findall(
            r'ATOMX_CC_STATUS (\dh\dp) (\d+) (\d+) (\S+) (\S+) (\S+)', log)}
    needed = {'0h0p', sector}
    if result.returncode or not needed.issubset(statuses) or any(statuses[s][0] != 1 for s in needed):
        raise RuntimeError(f'EXP-T amplitude solve failed; inspect {directory}/cc.log')
    update = max(max(statuses[s][2:]) for s in needed)
    # With no shifts, residual = diagonal denominator * raw Jacobi update.
    # Six orbital energies bounds all rank <=3 denominators, including valence.
    residual_bound = 6*float(np.max(abs(np.diag(integrals.fock_Ha))))*update
    if update > tolerance or residual_bound > residual_bound_tolerance_Ha:
        raise RuntimeError(f'raw amplitude residual bound failed: {residual_bound} Ha')
    heff_path = directory/'scratch/HEFF'
    blocks = read_heff(heff_path)
    core = blocks['0h0p', 1][0, 0]
    if abs(core.imag) > 1e-10:
        raise RuntimeError('complex closed-shell energy')
    energies = []
    if sector != '0h0p':
        for (s, _), matrix in blocks.items():
            if s == sector:
                energies.extend(np.linalg.eigvals(matrix)+core)
        if len(energies) != len(active) or max(abs(e.imag) for e in energies) > 1e-8:
            raise RuntimeError('invalid or incomplete model-space energy spectrum')
    else:
        energies = [core]
    provenance.update(statuses=statuses, log_sha256=_sha(directory/'cc.log'),
                      heff_sha256=_sha(heff_path))
    solved = dict(core_energy_Ha=float(core.real), sector_energies_Ha=sorted(float(e.real) for e in energies),
                  amplitude_update_max=update, equation_residual_bound_Ha=residual_bound, provenance=provenance)
    (directory/'result.json').write_text(json.dumps(solved, indent=2)+'\n')
    return FSCCResult(model, sector, solved['core_energy_Ha'], tuple(solved['sector_energies_Ha']),
                      MappingProxyType(blocks), update, residual_bound, MappingProxyType(provenance))
