"""Complete native CI spectra with separate J/parity matrix storage.

Every requested CSF and every root is retained. The physical builder and
native radial snapshot are shared with ordinary CI; only assembly/storage
are blocked. Numeric streamed angular assembly avoids coefficient tables.
"""
from collections import defaultdict
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import linalg

from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
from atomx.config.csf import get_shared_orbital_basis
from atomx.physics.csf.effective_one_body import CIReferenceBinding
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot


@dataclass(frozen=True)
class CIBlock:
    sector: tuple
    indices: np.ndarray
    hamiltonian_Ha: np.ndarray
    energies_Ha: np.ndarray
    coefficients: np.ndarray
    residual_Ha: float
    gram_defect: float


@dataclass(frozen=True)
class BlockedCISpectrum:
    blocks: tuple
    radial_binding: CIReferenceBinding
    csf_identities: tuple
    resource_plan: dict
    radial_diagnostics: dict


def _owned(values):
    a = np.asarray(values)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def plan_ci_blocks(csfs):
    """Bound dense matrices/eigensolver and sparse determinant scratch.

    The sparse graph is bounded by the smaller of all determinant pairs and
    the complete single/double substitution degree after exact common-core
    contraction. Bytes include retained H/C, construction/eigensolve scratch,
    sparse entries and a512MiB angular/radial allowance, not a measured RSS.
    """
    csfs = tuple(csfs)
    if not csfs:
        raise ValueError('nonempty complete CSF space required')
    signatures = tuple(csf_signature(c) for c in csfs)
    if len({s.key for s in signatures}) != len(csfs):
        raise ValueError('distinct CSFs required')
    if len({c.n_electrons for c in csfs}) != 1:
        raise ValueError('one electron count required')
    groups = defaultdict(list)
    for i, c in enumerate(csfs):
        if c.parity not in (-1, 1) or c.parity != (-1)**sum(
                (s.kappa if s.kappa > 0 else -s.kappa-1)*s.occupation for s in c.subshells):
            raise ValueError('explicit consistent CSF parity required')
        groups[c.total_J2, c.parity].append(i)
    rows = []
    for sector, indices in sorted(groups.items()):
        patterns = {signatures[i].pattern for i in indices}
        raw = sum(dict(pattern_m_counts(p)).get(sector[0], 0) for p in patterns)
        keys = {(n, k) for p in patterns for n, k, _ in p}
        frozen = sum(2*abs(k) for n, k in keys
                     if all((n, k, 2*abs(k)) in p for p in patterns))
        active = csfs[0].n_electrons-frozen
        virtuals = sum(2*abs(k) for n, k in keys)-frozen-active
        degree = 1+active*virtuals+comb(active, 2)*comb(virtuals, 2)
        rows.append(dict(J2=sector[0], parity=sector[1], size=len(indices),
                         raw_determinants=raw, graph_entry_bound=min(raw*raw, raw*degree),
                         indices=indices))
    retained = 16*sum(r['size']**2 for r in rows)
    working = max(96*r['size']**2+32*r['graph_entry_bound']+256*r['raw_determinants'] for r in rows)
    return dict(blocks=rows, levels=len(csfs), retained_H_C_bytes=retained,
                estimated_peak_bytes=retained+working+512*1024**2,
                avoided_global_H_bytes=8*len(csfs)**2)


def solve_ci_blocks(csfs, orbitals, workspace, Z, ns, *, nuclear_potential=None,
                    max_determinants=2000, memory_bytes=8*1024**3, progress=None,
                    engine='streamed', spinor_reference_keys=None):
    """Build and solve complete real DC blocks, rejecting resource excess first."""
    for name, value in [('max_determinants', max_determinants), ('memory_bytes', memory_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'positive integer {name} required')
    csfs = tuple(csfs)
    if engine not in ('streamed', 'spinor'):
        raise ValueError('explicit streamed or spinor CI engine required')
    plan = plan_ci_blocks(csfs)
    if max(r['raw_determinants'] for r in plan['blocks']) > max_determinants:
        raise ValueError('complete blocked determinant space exceeds cap')
    if plan['estimated_peak_bytes'] > memory_bytes:
        raise MemoryError(f"blocked CI bound {plan['estimated_peak_bytes']} bytes exceeds cap {memory_bytes}")
    keys = get_shared_orbital_basis(csfs)
    binding = CIReferenceBinding.capture([(*orbitals[k], k[1]) for k in keys], {},
                                         workspace, Z, ns, nuclear_potential)
    if engine == 'streamed':
        radial = CIRadialSnapshot(orbitals, workspace, Z, ns, nuclear_potential=nuclear_potential)
    else:
        from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian
        if spinor_reference_keys is None:
            raise ValueError('explicit closed reference keys required for spinor CI')
        radial = SpinorCIHamiltonian(csfs, orbitals, workspace, Z, ns,
                   reference_keys=spinor_reference_keys, nuclear_potential=nuclear_potential,
                   memory_bytes=min(memory_bytes//4, 512*1024**2))
    blocks = []
    for row in plan['blocks']:
        sector, indices = (row['J2'], row['parity']), row['indices']
        subset = [csfs[i] for i in indices]
        if progress is not None:
            progress('CI_BLOCK_START', sector, row['size'], row['raw_determinants'])
        if engine == 'streamed':
            h = build_ci_matrix(subset, orbitals, keys, workspace, Z, ns,
                                nuclear_potential=nuclear_potential, max_determinants=max_determinants,
                                radial_snapshot=radial, angular_storage='streamed')
        else:
            h = radial.block_matrix(subset, max_determinants=max_determinants)
        origin = float(np.rint(h[0, 0]))
        shifted = h.copy()
        shifted[np.diag_indices(len(h))] -= origin
        w, c = linalg.eigh(shifted, driver='evd')
        w += origin
        residual = float(np.max(np.abs(h@c-c*w)))
        gram = float(np.max(np.abs(c.T@c-np.eye(len(h)))))
        if residual > 1e-8 or gram > 1e-10:
            raise RuntimeError(f'complete blocked CI residual/Gram failed: {residual}, {gram}')
        blocks.append(CIBlock(sector, _owned(np.asarray(indices)), _owned(h), _owned(w),
                              _owned(c), residual, gram))
        if progress is not None:
            progress('CI_BLOCK_DONE', sector, residual, gram)
    binding.validate(keys, orbitals, workspace, Z, ns, nuclear_potential)
    diagnostics = (radial.diagnostics() if engine == 'streamed' else
                   dict(radial.integrals.diagnostics(), engine='shared native h_eff/g/E_frozen',
                        export_reference_keys=tuple(spinor_reference_keys),
                        Fock_and_reference_energy_consumed=False))
    return BlockedCISpectrum(tuple(blocks), binding, tuple(csf_signature(c).key for c in csfs),
                             plan, diagnostics)
