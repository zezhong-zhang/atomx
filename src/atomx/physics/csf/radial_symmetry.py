"""Exact key symmetries of real-or-c-product static radial bilinears.

This contract is explicit: it does not apply to arbitrary supplied numbers,
sesquilinear complex orbitals, frequency-dependent or nonreciprocal kernels.
The sign belongs to the angular coefficient, never to a complex conjugate.
"""

from functools import lru_cache


@lru_cache(maxsize=16384)
def canonical_static_radial_key(key):
    """Return ``(representative, sign)`` under exact product/pair exchange.

    Coulomb/Gaunt products commute within each electron coordinate and the
    radial kernel is reciprocal between coordinates (eight permutations).
    Static-retardation currents instead reverse sign within a pair. A
    diagonal transition current vanishes identically. One-body matrices
    are transpose-symmetric in this explicitly declared representation.
    Only exact algebraic zeros are removed; no small value is truncated.
    """
    kind=key[0]
    if kind=='h' and len(key)==4:
        _,a,kappa,c=key
        return ('h',min(a,c),kappa,max(a,c)),1
    if kind not in ('r','g','t') or len(key)!=6:
        raise ValueError('static radial symmetry requires h/r/g/t keys')
    _,rank,a,c,b,d=key
    if kind=='t' and (a==c or b==d):
        return key,0
    sign=(-1 if a>c else 1)*(-1 if b>d else 1) if kind=='t' else 1
    pairs=sorted((tuple(sorted((a,c))),tuple(sorted((b,d)))))
    return (kind,rank,*pairs[0],*pairs[1]),sign


__all__=['canonical_static_radial_key']
