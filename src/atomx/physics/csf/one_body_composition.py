"""Bound scalar one-body additions to an already frozen-core normal-ordered H."""

import hashlib
from pathlib import Path

import numpy as np

from atomx.physics.csf.effective_one_body import CIEffectiveOneBody, CIReferenceBinding
from atomx.physics.csf.spinor_integrals import OnDemandSpinorIntegrals, SpinorIntegrals


def add_bound_one_body(integrals, operator, orbitals, workspace, Z, ns, *,
                       nuclear_potential=None, max_added_bytes=4*1024**3):
    """Add one explicit all-electron scalar matrix, including its core ledger.

    Both producers must match the same native radial/nuclear point. Every
    occupied, virtual and frozen radial key must occur in the supplied bound
    operator; missing entries are not inferred as zero. Its producer declares
    physics and provenance. This consumer does not generate QED/MBPT data,
    reoptimize orbitals or allow a correlated-core self-energy to be counted
    again as an all-electron correction. Packed or native on-demand storage
    is supported; the original complete two-body operator is unchanged.

    ``max_added_bytes`` admits new owned ERI/matrix payload, including the
    constructor's copy; it is not a process-RSS cap or backend tensor budget.
    """
    if (not isinstance(integrals, (SpinorIntegrals, OnDemandSpinorIntegrals))
            or not isinstance(operator, CIEffectiveOneBody)):
        raise TypeError('native spinor integrals and an explicit bound scalar operator required')
    if (isinstance(max_added_bytes, (bool, np.bool_))
            or not isinstance(max_added_bytes, (int, np.integer)) or max_added_bytes < 1):
        raise ValueError('positive integer additional-payload cap required')
    n = len(integrals.labels)
    packed = isinstance(integrals, SpinorIntegrals)
    needed = (integrals.packed_eri_Ha.nbytes if packed else 0)+6*16*n*n
    if needed > max_added_bytes:
        raise MemoryError('complete one-body composition exceeds additional payload cap')
    identity = integrals.identity
    occ, vir, frozen = (tuple(tuple(k) for k in identity[name])
                        for name in ('occupied_keys', 'virtual_keys', 'frozen_keys'))
    if set(operator.orbital_keys) != set(occ+vir+frozen) or operator.correlated_core:
        raise ValueError('complete all-electron scalar coverage without a repeated correlated core required')
    binding = CIReferenceBinding.capture([(*orbitals[k], k[1]) for k in occ+vir],
                {k: orbitals[k] for k in frozen+occ}, workspace, Z, ns, nuclear_potential)
    if repr(binding) != identity['binding']:
        raise ValueError('base integral producer radial/core/nuclear point mismatch')
    evaluate = operator.evaluator_for((), orbitals, workspace, Z, ns,
                                       nuclear_potential=nuclear_potential)
    additions = tuple(identity.get('one_body_additions', ()))
    if any(item['reference_identity'] == operator.reference_identity for item in additions):
        raise ValueError('one-body operator already included')
    correction = np.zeros((n, n), dtype=np.complex128)
    for i, (na, k, m) in enumerate(integrals.labels):
        for j, (nb, kb, mb) in enumerate(integrals.labels):
            if (kb, mb) == (k, m):
                correction[i, j] = evaluate(('h', na, k, nb))
    frozen_change = sum(2*abs(k)*evaluate(('h', na, k, na)) for na, k in frozen)
    reference_change = frozen_change+np.trace(correction[:integrals.noccupied, :integrals.noccupied]).real
    record = dict(operator.provenance(), frozen_energy_change_Ha=float(frozen_change),
                  reference_energy_change_Ha=float(reference_change),
                  composition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    updated_identity = dict(identity, hamiltonian=identity['hamiltonian']+' + '+operator.theory,
                            one_body_additions=additions+(record,))
    if not packed:
        return OnDemandSpinorIntegrals(integrals.labels, integrals.noccupied,
            integrals.one_body_Ha+correction, integrals.fock_Ha+correction,
            integrals.frozen_energy_Ha+frozen_change,
            integrals.reference_energy_Ha+reference_change, updated_identity,
            integrals._evaluate, integrals._diagnostics)
    return SpinorIntegrals(integrals.labels, integrals.noccupied,
        integrals.one_body_Ha+correction, integrals.fock_Ha+correction,
        integrals.packed_eri_Ha, integrals.frozen_energy_Ha+frozen_change,
        integrals.reference_energy_Ha+reference_change, updated_identity)
