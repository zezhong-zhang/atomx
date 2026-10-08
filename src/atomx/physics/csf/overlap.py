"""Signed overlaps of resolved CI levels in different common radial bases.

Each determinant overlap is det(S_bra,ket); no inverse or division by S is
used, so singular cross-basis overlaps remain valid. Only structural angular
zeros are skipped. Radial inputs share a B-spline metric; this module does
not optimize orbitals, select roots by experimental energy, or track a pole.
"""
from collections import defaultdict

import numpy as np

from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
from atomx.physics.csf.levels import CILevel, ci_level_blocks


def _projection(csfs, keys, symmetry):
    rows = [i for i,c in enumerate(csfs) if (c.total_J2,c.parity) == symmetry]
    vectors = [csf_magnetic_vector(csf_signature(csfs[i]), symmetry[0], keys) for i in rows]
    masks = sorted({mask for vector in vectors for mask, amplitude in vector})
    positions = {mask:i for i,mask in enumerate(masks)}
    c = np.zeros((len(masks),len(rows)))
    for j,vector in enumerate(vectors):
        for mask,amplitude in vector:
            c[positions[mask],j] += amplitude
    labels = [(n,k,m) for n,k in keys for m in range(-2*abs(k)+1,2*abs(k),2)]
    determinants = [tuple(i for i in range(len(labels)) if mask & (1 << i)) for mask in masks]
    groups = defaultdict(list)
    for row,occupied in enumerate(determinants):
        # Same (kappa,m) multiset is necessary for a scalar radial overlap.
        groups[tuple(sorted((labels[i][1],labels[i][2]) for i in occupied))].append(row)
    return rows,c,labels,np.asarray(determinants,dtype=int),groups


def ci_level_overlaps(bra_csfs, bra_ci, bra_orbitals, bra_level,
                      ket_csfs, ket_ci, ket_orbitals, ket_levels, metric,
                      *, max_determinant_pairs=1_000_000):
    """Return <bra level, M=J | ket level, M=J> with signed CI coefficients.

    ``bra_level`` and every entry of ``ket_levels`` are explicit CILevel
    selectors. The metric maps real P/Q coefficient vectors of the same
    radial geometry. Inputs must be orthonormal within each basis. Different
    J/parity overlaps are exactly zero. A pair budget is checked before dense
    determinant batches; exceeding it refuses instead of discarding terms.
    This is a basis-aware comparison, not energy-order root tracking.
    """
    return _overlaps(bra_csfs, bra_ci, bra_orbitals, (bra_level,),
                     ket_csfs, ket_ci, ket_orbitals, tuple(ket_levels), metric,
                     max_determinant_pairs=max_determinant_pairs,
                     unique_roots=True)[0]


def ci_subspace_overlaps(bra_csfs, bra_ci, bra_orbitals, bra_levels,
                         ket_csfs, ket_ci, ket_orbitals, ket_levels, metric,
                         *, max_determinant_pairs=1_000_000):
    """Signed overlap matrix between complete selected CI eigensubspaces.

    Each side must contain an entire exact-degenerate same-symmetry cluster
    when any member is selected. Individual entries depend on the basis in
    a degenerate cluster; singular values and projector traces do not. The
    determinant contractions are batched across all selected roots of a
    symmetry, so a larger target does not repeat the expensive minors.
    """
    return _overlaps(bra_csfs, bra_ci, bra_orbitals, tuple(bra_levels),
                     ket_csfs, ket_ci, ket_orbitals, tuple(ket_levels), metric,
                     max_determinant_pairs=max_determinant_pairs,
                     unique_roots=False)


def ci_subspace_overlap_gradient(bra_csfs, bra_ci, bra_orbitals, bra_levels,
                                ket_csfs, ket_ci, ket_orbitals, ket_levels, metric,
                                cotangent, *, max_determinant_pairs=1_000_000):
    """Radial derivative of sum(cotangent * overlap) at fixed CI amplitudes.

    Returns (signed overlap matrix, ket radial gradient). All selected exact
    clusters must be complete. Cofactors avoid inverses of singular minors.
    CI eigenvector response is a separate contribution.
    """
    return _overlaps(bra_csfs, bra_ci, bra_orbitals, tuple(bra_levels),
                     ket_csfs, ket_ci, ket_orbitals, tuple(ket_levels), metric,
                     max_determinant_pairs=max_determinant_pairs,
                     unique_roots=False, cotangent=cotangent)


def _overlaps(bra_csfs, bra_ci, bra_orbitals, bra_levels,
              ket_csfs, ket_ci, ket_orbitals, ket_levels, metric,
              *, max_determinant_pairs, unique_roots, cotangent=None):
    bra_levels, ket_levels = tuple(bra_levels), tuple(ket_levels)
    if not bra_levels or not ket_levels:
        raise ValueError('nonempty CI subspaces required')
    if any(not isinstance(s,CILevel) for s in bra_levels+ket_levels):
        raise TypeError('CILevel selectors required for basis-aware overlaps')
    if len(set(bra_levels)) != len(bra_levels) or len(set(ket_levels)) != len(ket_levels):
        raise ValueError('duplicate CI subspace selector')
    if cotangent is not None:
        cotangent = np.asarray(cotangent)
        if (cotangent.shape != (len(bra_levels), len(ket_levels))
                or np.iscomplexobj(cotangent) or not np.isfinite(cotangent).all()):
            raise ValueError('overlap cotangent requires a finite real subspace matrix')
    if (isinstance(max_determinant_pairs,(bool,np.bool_)) or
            not isinstance(max_determinant_pairs,(int,np.integer)) or max_determinant_pairs < 1):
        raise ValueError('positive integer determinant-pair budget required')
    _,bb = ci_level_blocks(bra_csfs,bra_ci)
    _,kb = ci_level_blocks(ket_csfs,ket_ci)
    def resolve(selector,blocks):
        columns = blocks.get((selector.J2,selector.parity),())
        if selector.root >= len(columns):
            raise ValueError('CI level selector outside overlap spectrum')
        return int(columns[selector.root])
    ib = [resolve(s,bb) for s in bra_levels]
    ik = [resolve(s,kb) for s in ket_levels]
    for selectors,indices,blocks,result in ((bra_levels,ib,bb,bra_ci),(ket_levels,ik,kb,ket_ci)):
        for selector,index in zip(selectors,indices,strict=True):
            columns = blocks[selector.J2,selector.parity]
            same = columns[abs(result.state_energies[columns]-result.state_energies[index])<1e-10]
            if unique_roots and len(same)>1:
                raise ValueError('an individual degenerate CI root has no unique overlap; compare its subspace')
            if not set(same).issubset(indices):
                raise ValueError('CI subspace must include the complete degenerate cluster')
    bn = {sum(s.occupation for s in c.subshells) for c in bra_csfs}
    kn = {sum(s.occupation for s in c.subshells) for c in ket_csfs}
    if len(bn)!=1 or len(kn)!=1 or bn != kn or next(iter(bn)) < 1:
        raise ValueError('one shared positive fixed electron number required for CI overlaps')
    bk,kk = tuple(sorted(bra_orbitals)),tuple(sorted(ket_orbitals))
    radial = {}
    for waves in (bra_orbitals,ket_orbitals):
        for kappa in {k for n,k in waves}:
            keys = [key for key in sorted(waves) if key[1]==kappa]
            if any(np.iscomplexobj(a) for key in keys for a in waves[key]):
                raise ValueError('real radial coefficient vectors required')
            v = np.column_stack([np.concatenate(waves[key]) for key in keys])
            s = np.asarray(metric[kappa])
            if (s.shape != (v.shape[0],v.shape[0]) or np.iscomplexobj(s)
                    or not np.isfinite(s).all()
                    or not np.allclose(s,s.T,atol=1e-12,rtol=0)):
                raise ValueError('one finite real symmetric radial metric required')
            if not np.isfinite(v).all() or not np.allclose(v.T @ s @ v,np.eye(len(keys)),atol=1e-10,rtol=0):
                raise ValueError('each radial basis must be finite and metric-orthonormal')
    for a in bk:
        for b in kk:
            if a[1]==b[1]:
                radial[a,b] = float(np.concatenate(bra_orbitals[a]) @ metric[a[1]] @ np.concatenate(ket_orbitals[b]))
    output = np.zeros((len(bra_levels),len(ket_levels)))
    gradient = ({key: np.zeros(len(np.concatenate(pair)))
                 for key, pair in ket_orbitals.items()} if cotangent is not None else None)
    projections = []
    pairs = 0
    for symmetry in sorted({(s.J2,s.parity) for s in bra_levels}):
        bselect = [i for i,s in enumerate(bra_levels) if (s.J2,s.parity)==symmetry]
        kselect = [i for i,s in enumerate(ket_levels) if (s.J2,s.parity)==symmetry]
        if not kselect:
            continue
        bp = _projection(bra_csfs,bk,symmetry)
        kp = _projection(ket_csfs,kk,symmetry)
        pairs += sum(len(rows)*len(kp[4].get(group,())) for group,rows in bp[4].items())
        projections.append((bselect,kselect,bp,kp))
    if pairs > max_determinant_pairs:
        raise ValueError(f'CI overlap needs {pairs} determinant pairs; cap={max_determinant_pairs}')
    for bselect,kselect,(br,bc,bl,bd,bg),(kr,kc,kl,kd,kg) in projections:
        single = np.zeros((len(bl),len(kl)))
        for i,(n,k,m) in enumerate(bl):
            for j,(nn,kkk,mm) in enumerate(kl):
                if (k,m)==(kkk,mm):
                    single[i,j] = radial[(n,k),(nn,kkk)]
        vb = bc @ bra_ci.mixing_coefficients[np.ix_(br,[ib[i] for i in bselect])]
        vk = kc @ ket_ci.mixing_coefficients[np.ix_(kr,[ik[i] for i in kselect])]
        value = np.zeros((len(bselect),len(kselect)))
        adjoint = np.zeros_like(single) if gradient is not None else None
        for group,brows in bg.items():
            krows = kg.get(group,())
            if not krows:
                continue
            batch = max(1,2_000_000//(len(krows)*len(bd[0])**2))
            for start in range(0,len(brows),batch):
                rows = brows[start:start+batch]
                minors = single[bd[rows][:,None,:,None],kd[krows][None,:,None,:]]
                value += vb[rows].T @ np.linalg.det(minors) @ vk[krows]
                if adjoint is not None:
                    amplitudes = vb[rows] @ cotangent[np.ix_(bselect, kselect)] @ vk[krows].T
                    order = minors.shape[-1]
                    for i in range(order):
                        for j in range(order):
                            small = np.delete(np.delete(minors, i, axis=-2), j, axis=-1)
                            cofactors = (-1)**(i+j) * np.linalg.det(small)
                            ii = np.broadcast_to(bd[rows][:, None, i], amplitudes.shape)
                            jj = np.broadcast_to(kd[krows][None, :, j], amplitudes.shape)
                            np.add.at(adjoint, (ii.ravel(), jj.ravel()),
                                      (amplitudes * cofactors).ravel())
        output[np.ix_(bselect,kselect)] = value
        if adjoint is not None:
            for i, (n, k, m) in enumerate(bl):
                source = metric[k] @ np.concatenate(bra_orbitals[n, k])
                for j, (nn, kkk, mm) in enumerate(kl):
                    if (k, m) == (kkk, mm):
                        gradient[nn, kkk] += adjoint[i, j] * source
    return output if gradient is None else (output, gradient)
