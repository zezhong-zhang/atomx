"""Signed electron-removal amplitudes in an explicitly ordered spinor basis."""

import numpy as np

from atomx.angular.clebsch import clebsch_gordan


def dyson_annihilation(bra, ket, n_spinors, *, engine='python'):
    """Return ``<bra|a_p|ket>`` for real sparse determinant vectors.

    The least significant bit is spinor zero. No small-amplitude screening,
    normalization, occupation factor or statistical average is applied.
    Consumers bind the vectors to their physical states and orbital layout.
    The explicit compiled engine uses the identical ordered finite bitset
    action for1..63 spinors, with numerical payload/work caps. Python remains
    the default and supports arbitrary-length integer determinant masks.
    """
    if isinstance(n_spinors, bool) or not isinstance(n_spinors, (int, np.integer)) or n_spinors < 1:
        raise ValueError("positive integer spinor count required")
    if engine not in ('python', 'compiled'):
        raise ValueError('explicit python or compiled removal engine required')
    if engine == 'compiled':
        from atomx.numerics.fock_removal import fock_removal_amplitudes

        if any(not isinstance(mask, (int, np.integer)) or mask < 0 or int(mask).bit_length() > n_spinors
               for vector in (bra, ket) for mask in vector):
            raise ValueError('integer sparse vectors in the declared spinor layout required')
        return fock_removal_amplitudes(np.asarray(tuple(bra), dtype=np.uint64), np.asarray(tuple(bra.values())),
                                      np.asarray(tuple(ket), dtype=np.uint64), np.asarray(tuple(ket.values())),
                                      n_spinors)
    for vector in (bra, ket):
        if any(not isinstance(mask, (int, np.integer)) or mask < 0 or int(mask).bit_length() > n_spinors
               or not np.isrealobj(value) or not np.isfinite(value) for mask, value in vector.items()):
            raise ValueError("finite real sparse vectors in the declared spinor layout required")
    out = np.zeros(n_spinors)
    for mask, ci in ket.items():
        rest, position = int(mask), 0
        while rest:
            bit = rest & -rest
            cf = bra.get(int(mask) ^ bit, 0.)
            out[bit.bit_length() - 1] += cf * ci * (-1)**position
            rest ^= bit
            position += 1
    return out


def reduced_dyson_annihilation(bra, ket, spinors, initial_J2, final_J2, *, engine='python'):
    """Signed radial removal coefficients from highest components of pure J.

    The returned D_(n,k) reconstructs every removal matrix element as
    D_(n,k) * CG(Jf Mf, j_k m | Ji Mi), with m=Mi-Mf. Supplied vectors
    must be the highest components in the same ordered spinor basis and
    Condon--Shortley convention. Consumers certify their pure-J identity;
    this function checks magnetic support, without normalizing the vectors.
    No magnetic statistical factor or small-amplitude cutoff is applied.
    The explicit compiled engine also checks magnetic support by exact
    integer bit-linear forms; neither engine certifies pure-J provenance.
    """
    spinors = tuple(spinors)
    if (any(isinstance(j, bool) or not isinstance(j, (int, np.integer)) or j < 0
            for j in (initial_J2, final_J2)) or len(set(spinors)) != len(spinors)
            or any(k == 0 or abs(m) > 2*abs(k)-1 or (m+2*abs(k)-1) % 2
                   for _, k, m in spinors)):
        raise ValueError('valid pure-J magnetic spinor layout required')
    raw = dyson_annihilation(bra, ket, len(spinors), engine=engine)
    for vector, angular in ((bra, final_J2), (ket, initial_J2)):
        if engine == 'compiled':
            from atomx.numerics.fock_removal import mask_linear_forms

            sums = mask_linear_forms(np.asarray(tuple(vector), dtype=np.uint64), [s[2] for s in spinors])
            wrong = np.any((sums != angular)&(np.asarray(tuple(vector.values())) != 0))
        else:
            wrong = any(sum(spinors[p][2] for p in range(len(spinors)) if int(mask)&(1 << p)) != angular
                        for mask, value in vector.items() if value != 0)
        if wrong:
            raise ValueError('highest magnetic component required')
    keys = tuple(sorted({(n, k) for n, k, _ in spinors}))
    lookup = {spinor: i for i, spinor in enumerate(spinors)}
    coefficients = np.zeros(len(keys))
    m = initial_J2-final_J2
    for i, (n, k) in enumerate(keys):
        j = 2*abs(k)-1
        if not abs(final_J2-j) <= initial_J2 <= final_J2+j or (final_J2+j+initial_J2) % 2:
            continue
        cg = float(clebsch_gordan(final_J2, final_J2, j, m, initial_J2, initial_J2))
        if cg == 0 or (n, k, m) not in lookup:
            raise ValueError('complete allowed highest-component spinor required')
        coefficients[i] = raw[lookup[n, k, m]]/cg
    return keys, coefficients


def scalar_radial_density(vector, spinors):
    """Angle-integrated radial one-body density of a real magnetic CI vector.

    Sum <a+_(n,k,m) a_(n',k,m)> over m, retaining every same-kappa radial
    coherence. For a pure J multiplet the resulting scalar is independent of
    M. Sparse single replacements avoid a quadratic determinant-pair scan.
    Consumers supply normalized states and a physically bound orbital layout.
    """
    spinors = tuple(spinors)
    if len(set(spinors)) != len(spinors):
        raise ValueError('distinct magnetic spinors required')
    keys = tuple(sorted({(n, k) for n, k, _ in spinors}))
    index = {key: i for i, key in enumerate(keys)}
    groups = {}
    for bit, (n, k, m) in enumerate(spinors):
        groups.setdefault((k, m), []).append((bit, index[n, k]))
    density = np.zeros((len(keys), len(keys)))
    for mask, coefficient in vector.items():
        if (not isinstance(mask, (int, np.integer)) or mask < 0 or int(mask).bit_length() > len(spinors)
                or not np.isrealobj(coefficient) or not np.isfinite(coefficient)):
            raise ValueError('finite real sparse vector in the declared spinor layout required')
        rest = int(mask)
        while rest:
            bit = rest & -rest
            q = bit.bit_length()-1
            n, k, m = spinors[q]
            spectator = int(mask)^bit
            sign_q = -1 if (int(mask)&(bit-1)).bit_count()%2 else 1
            for p, row in groups[k, m]:
                target_bit = 1 << p
                if spectator&target_bit:
                    continue
                bra = vector.get(spectator|target_bit, 0.)
                if bra != 0:
                    sign_p = -1 if (spectator&(target_bit-1)).bit_count()%2 else 1
                    density[row, index[n, k]] += bra*coefficient*sign_p*sign_q
            rest ^= bit
    return keys, density
