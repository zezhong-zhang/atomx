"""Finite signed Fock actions; no integral or physical-model construction."""

from itertools import combinations
from math import comb

import numpy as np

from atomx.angular.csf_hamiltonian import _ops_sign


def apply_terms(terms, vector):
    """Apply creation/annihilation strings, rightmost first, without screening.

    Operators are ``(coefficient, ((index, create), ...))``. Determinants
    use ascending spinor bit order. The caller must preflight the complete
    intermediate space and work; this small dictionary engine is an anchor.
    """
    result = {}
    for mask, coefficient in vector.items():
        for value, operations in terms:
            sign = _ops_sign(mask, operations)
            if sign:
                target = mask
                for bit, create in reversed(operations):
                    target = target | (1 << bit) if create else target & ~(1 << bit)
                result[target] = result.get(target, 0.0) + coefficient * value * sign
    return {mask: value for mask, value in result.items() if value != 0}


def _canonical_inputs(masks, coefficients, spinor_count, max_work, memory_bytes):
    """Admit the shared finite/sparse canonical coefficient convention."""
    if (isinstance(spinor_count, (bool, np.bool_))
            or not isinstance(spinor_count, (int, np.integer)) or spinor_count < 1):
        raise ValueError('positive integer spinor count required')
    for value in (max_work, memory_bytes):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError('positive integer finite projection budgets required')
    masks = tuple(masks)
    if (not masks or any(isinstance(m, (bool, np.bool_))
                         or not isinstance(m, (int, np.integer)) or m < 0
                         or int(m).bit_length() > spinor_count for m in masks)
            or len(set(masks)) != len(masks)):
        raise ValueError('distinct valid finite determinant masks required')
    masks = tuple(map(int, masks))
    number = masks[0].bit_count()
    if any(m.bit_count() != number for m in masks):
        raise ValueError('one fixed electron-number sector required')
    coefficients = tuple(np.asarray(c) for c in coefficients)
    if not 1 <= len(coefficients) <= 4:
        raise ValueError('canonical ranks zero through at most three required')
    for rank, c in enumerate(coefficients):
        size = comb(spinor_count, rank)
        if (c.shape != (size, size) or c.dtype.kind not in 'iufc'
                or c.dtype.itemsize > 16 or not np.all(np.isfinite(c))):
            raise ValueError('finite numeric canonical body coefficient matrices required')
    return masks, coefficients, number


def canonical_body_matrix(masks, coefficients, spinor_count, *,
                          max_work=2_000_000, memory_bytes=64*1024**2):
    """Project supplied canonical zero/one/two/three-body words exactly.

    ``coefficients[r]`` is a square matrix in lexicographic combinations of
    ``r`` spinors. Each entry multiplies creators in increasing order and
    annihilators in decreasing order; no factorial prefactors remain.
    All masks belong to one declared fixed-N sector. This bounded finite
    anchor uses exact spectator sums, including diagonal and interference
    terms. It does not select physical states, construct MBPT, screen small
    coefficients or certify a large-space implementation.
    """
    masks, coefficients, number = _canonical_inputs(
        masks, coefficients, spinor_count, max_work, memory_bytes)
    rank_max = min(number, len(coefficients)-1)
    work = len(masks)**2*sum(comb(number, r) for r in range(rank_max+1))
    # Include the owned output and a conservative Python tuple/dict index
    # envelope. Supplied coefficient matrices remain caller-owned inputs.
    required = 16*len(masks)**2+256*sum(comb(spinor_count, r)
                                       for r in range(rank_max+1))
    if work > max_work or required > memory_bytes:
        raise MemoryError('complete finite canonical-body projection exceeds budget')
    indices = [{row: i for i, row in enumerate(combinations(range(spinor_count), r))}
               for r in range(rank_max+1)]
    occupations = [tuple(i for i in range(spinor_count) if m & (1 << i)) for m in masks]
    matrix = np.zeros((len(masks), len(masks)), dtype=np.result_type(np.float64, *coefficients))
    for bra, left in enumerate(masks):
        for ket, right in enumerate(masks):
            order = (left ^ right).bit_count()//2
            if order > rank_max:
                continue
            removed = tuple(i for i in occupations[ket] if not left & (1 << i))
            added = tuple(i for i in occupations[bra] if not right & (1 << i))
            common = tuple(i for i in occupations[ket] if left & (1 << i))
            for rank in range(order, rank_max+1):
                for spectators in combinations(common, rank-order):
                    creators = tuple(sorted(added+spectators))
                    annihilators = tuple(sorted(removed+spectators))
                    operations = (tuple((i, True) for i in creators)
                                  +tuple((i, False) for i in reversed(annihilators)))
                    sign = _ops_sign(right, operations)
                    matrix[bra, ket] += sign*coefficients[rank][
                        indices[rank][creators], indices[rank][annihilators]]
    return matrix
