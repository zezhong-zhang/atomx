"""Exact sparse action of supplied canonical body words in fixed-N spaces."""

from itertools import combinations
from math import comb

import numpy as np
from numba import njit
from scipy.sparse import csc_matrix

from atomx.numerics.fermion import _canonical_inputs

_ZERO_PREFIX = np.uint64(0)
_ZERO_PREFIX_SUPPORT = np.uint64(1)


@njit(cache=True)
def _popcount(value):
    count = 0
    while value:
        value &= value-np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def _colex(mask, lookup, counts):
    rank, previous = 0, 0
    for tile in range(lookup.shape[0]):
        part = int(mask & np.uint64(1023))
        rank += lookup[tile, previous, part]
        previous += counts[part]
        mask >>= np.uint64(10)
    return rank


@njit(cache=True)
def _hash_slot(mask, modulus):
    value = mask ^ (mask >> np.uint64(30))
    value *= np.uint64(0xbf58476d1ce4e5b9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94d049bb133111eb)
    value ^= value >> np.uint64(31)
    return np.int64(value & np.uint64(modulus))


@njit(cache=True)
def _hash_index(masks, slots):
    keys = np.zeros(slots, dtype=np.uint64)
    rows = np.full(slots, -1, dtype=np.int32)
    for row, mask in enumerate(masks):
        at = _hash_slot(mask, slots-1)
        while rows[at] >= 0:
            at = (at+1) & (slots-1)
        keys[at], rows[at] = mask, row
    return keys, rows


@njit(cache=True)
def _lookup_row(mask, lookup, counts, rowmap, keys,
                prefix_mask=_ZERO_PREFIX, count_support=_ZERO_PREFIX_SUPPORT):
    if keys.size:
        population = _popcount(mask & prefix_mask)
        if not (count_support & (np.uint64(1) << np.uint64(population))):
            return -1
        at = _hash_slot(mask, rowmap.size-1)
        while rowmap[at] >= 0:
            if keys[at] == mask:
                return rowmap[at]
            at = (at+1) & (rowmap.size-1)
        return -1
    return rowmap[_colex(mask, lookup, counts)]


@njit(cache=True)
def _prefix_populations(masks, n):
    support = np.zeros(n+1, dtype=np.uint64)
    for mask in masks:
        population = 0
        for bit in range(n):
            population += np.int64((mask >> np.uint64(bit)) & np.uint64(1))
            support[bit+1] |= np.uint64(1) << np.uint64(population)
    return support


def _population_prefix(masks, n, number):
    """An exact necessary support condition derived from every declared row.

    Choose deterministically by the count of compatible fixed-N masks, with
    no physical model, amplitude, coefficient or floating-point threshold.
    """
    supports = _prefix_populations(masks, n)
    best, prefix, support = comb(n, number), 0, 1
    for width in range(1, n):
        allowed = int(supports[width])
        compatible = sum(comb(width, q)*comb(n-width, number-q)
                         for q in range(max(0, number-(n-width)), min(width, number)+1)
                         if allowed & (1 << q))
        if compatible < best:
            best, prefix, support = compatible, (1 << width)-1, allowed
    return np.uint64(prefix), np.uint64(support)


@njit(cache=True)
def _rank_tables(n, number, binomial):
    counts = np.zeros(1024, dtype=np.int64)
    for value in range(1024):
        counts[value] = _popcount(np.uint64(value))
    lookup = np.zeros(((n+9)//10, number+1, 1024), dtype=np.int64)
    for tile in range(lookup.shape[0]):
        for before in range(number+1):
            for part in range(1024):
                order, value = before, 0
                for bit in range(10):
                    if part & (1 << bit):
                        order += 1
                        position = tile*10+bit
                        if position < n and order <= number:
                            value += binomial[position, order]
                lookup[tile, before, part] = value
    return lookup, counts


@njit(cache=True)
def _annihilator_columns(masks, n, number, maximum, binomial, offsets):
    count = 1
    for rank in range(1, maximum+1):
        count += binomial[number, rank]
    columns = np.empty((len(masks), count), dtype=np.int32)
    occupation = np.empty(number, dtype=np.int64)
    for ket, mask in enumerate(masks):
        used = 0
        for bit in range(n):
            if mask & (np.uint64(1) << np.uint64(bit)):
                occupation[used] = bit
                used += 1
        columns[ket, 0] = 0
        at = 1
        if maximum >= 1:
            for i in occupation:
                columns[ket, at] = offsets[1]+i
                at += 1
        if maximum >= 2:
            for a in range(number):
                i = occupation[a]
                for b in range(a+1, number):
                    j = occupation[b]
                    index = binomial[n, 2]-1-binomial[n-1-i, 2]-binomial[n-1-j, 1]
                    columns[ket, at] = offsets[2]+index
                    at += 1
        if maximum >= 3:
            for a in range(number):
                i = occupation[a]
                for b in range(a+1, number):
                    j = occupation[b]
                    for c in range(b+1, number):
                        k = occupation[c]
                        index = (binomial[n, 3]-1-binomial[n-1-i, 3]
                                 -binomial[n-1-j, 2]-binomial[n-1-k, 1])
                        columns[ket, at] = offsets[3]+index
                        at += 1
    return columns


@njit(cache=True)
def _column_frequencies(columns, count):
    frequencies = np.zeros(count, dtype=np.int64)
    for row in columns:
        for column in row:
            frequencies[column] += 1
    return frequencies


def _csc_row_dtype(nonzero):
    # SciPy promotes both CSC index arrays when indptr[-1] exceeds int32.
    # Allocate the matching width directly, and admit its actual byte count.
    return np.int64 if nonzero > np.iinfo(np.int32).max else np.int32


@njit(cache=True)
def _assemble(masks, columns, pointers, annihilators, creators, signs, constants,
              values, lookup, counts, rowmap, row_keys, prefix_mask, count_support,
              marker, scratch, targets,
              output_ptr, output_rows, output_values, count_only):
    total = 0
    for ket, mask in enumerate(masks):
        used = 0
        for column in columns[ket]:
            remaining = mask ^ annihilators[column]
            for entry in range(pointers[column], pointers[column+1]):
                creator = creators[entry]
                if remaining & creator:
                    continue
                target = remaining | creator
                row = _lookup_row(target, lookup, counts, rowmap, row_keys,
                                  prefix_mask, count_support)
                if row < 0:
                    continue
                if marker[row] != ket:
                    marker[row] = ket
                    targets[used] = row
                    used += 1
                    if not count_only:
                        scratch[row] = 0
                if not count_only:
                    parity = (_popcount(mask & signs[entry])+constants[entry]) & 1
                    scratch[row] += (-1 if parity else 1)*values[entry]
        if not count_only:
            ordered = np.sort(targets[:used])
            for i, row in enumerate(ordered):
                output_rows[total+i] = row
                output_values[total+i] = scratch[row]
        total += used
        output_ptr[ket+1] = total
    return total


def canonical_body_sparse(masks, coefficients, spinor_count, *,
                          max_work=2_000_000, memory_bytes=64*1024**2,
                          row_index='colex'):
    """Return every directed canonical0..3-body matrix element as CSC.

    Coefficients and masks follow :func:`canonical_body_matrix`. Exact-zero
    words alone are absent. Occupied annihilator subsets select all allowed
    creators; colex ranks map the declared fixed-N space without a 2**n map.
    A count pass admits the complete output before allocation. Candidate work,
    index tables, conversion scratch and owned matrix bytes are bounded.
    Input dense coefficient matrices remain caller-owned. No Hermitian average,
    small-value cutoff, physical-space selection or MBPT construction is used.
    The default ``row_index='colex'`` retains the full combinatorial row map.
    Explicit ``row_index='hash'`` instead indexes only the declared masks,
    with exact key comparisons and linear probing at load <=1/2. Both modes
    preserve the declared row order, directed entries and accumulation order.
    Hash lookup first checks a necessary prefix-population support condition
    derived from every declared mask; only provably absent rows are rejected.
    """
    if not isinstance(row_index, str) or row_index not in ('colex', 'hash'):
        raise ValueError('row_index must be colex or hash')
    masks, coefficients, number = _canonical_inputs(
        masks, coefficients, spinor_count, max_work, memory_bytes)
    n = int(spinor_count)
    if n > 63 or len(masks) >= 2**31:
        raise MemoryError('canonical sparse action requires <=63 spinors and int32 rows')
    maximum = min(number, len(coefficients)-1)
    coefficients = coefficients[:maximum+1]
    dtype = np.result_type(np.float64, *coefficients)
    size = dtype.itemsize
    dimensions = [comb(n, rank) for rank in range(maximum+1)]
    words = sum(int(np.count_nonzero(c)) for c in coefficients)
    occupied_columns = sum(comb(number, rank) for rank in range(maximum+1))
    d = len(masks)
    hash_slots = 1 << max(1, (2*d-1).bit_length())
    row_index_bytes = comb(n, number)*4 if row_index == 'colex' else hash_slots*12
    # Conservative envelope: coefficient conversion, full row map, lookup,
    # annihilator columns, markers/workspaces, sorting and CSC pointer copies.
    base = (words*(size+32)+sum(dimensions)*40+row_index_bytes
            +((n+9)//10)*(number+1)*1024*8+(n+1)*(max(number, 3)+1)*8
            +d*(4*occupied_columns+size+64)+32768)
    base += max(c.size for c in coefficients)  # exact-zero/finite check scratch
    if base > memory_bytes:
        raise MemoryError('complete canonical sparse index/conversion exceeds budget')
    binomial = np.zeros((n+1, max(number, 3)+1), dtype=np.int64)
    for upper in range(n+1):
        for lower in range(min(upper, binomial.shape[1]-1)+1):
            binomial[upper, lower] = comb(upper, lower)
    offsets = np.array([0]+list(np.cumsum(dimensions)), dtype=np.int64)
    annihilators = np.empty(sum(dimensions), dtype=np.uint64)
    prefixes = np.empty_like(annihilators)
    for rank in range(maximum+1):
        for index, row in enumerate(combinations(range(n), rank)):
            at = offsets[rank]+index
            annihilators[at] = sum(1 << bit for bit in row)
            prefix = 0
            for bit in row:
                prefix ^= (1 << bit)-1
            prefixes[at] = prefix
    pointers = np.zeros(sum(dimensions)+1, dtype=np.int64)
    creators = np.empty(words, dtype=np.uint64)
    signs = np.empty(words, dtype=np.uint64)
    constants = np.empty(words, dtype=np.int8)
    values = np.empty(words, dtype=dtype)
    at = 0
    for rank, c in enumerate(coefficients):
        for col in range(dimensions[rank]):
            column = offsets[rank]+col
            rows = np.flatnonzero(c[:, col])
            end = at+len(rows)
            indices = offsets[rank]+rows
            creators[at:end] = annihilators[indices]
            signs[at:end] = prefixes[column] ^ prefixes[indices]
            constants[at:end] = [(rank*(rank-1)//2+int(
                annihilators[column] & prefixes[i]).bit_count()) & 1 for i in indices]
            values[at:end] = c[rows, col]
            pointers[column+1] = end
            at = end
    assert at == words
    masks_array = np.array(masks, dtype=np.uint64)
    columns = _annihilator_columns(masks_array, n, number, maximum, binomial, offsets)
    frequencies = _column_frequencies(columns, len(pointers)-1)
    work = sum(int(count)*int(length) for count, length
               in zip(frequencies, np.diff(pointers), strict=True))
    if work > max_work:
        raise MemoryError('complete canonical sparse candidate work exceeds budget')
    lookup, counts = _rank_tables(n, number, binomial)
    if row_index == 'colex':
        prefix_mask, count_support = np.uint64(0), np.uint64(1)
        row_keys = np.empty(0, dtype=np.uint64)
        rowmap = np.full(comb(n, number), -1, dtype=np.int32)
        for row, mask in enumerate(masks_array):
            rowmap[_colex(mask, lookup, counts)] = row
    else:
        prefix_mask, count_support = _population_prefix(masks_array, n, number)
        row_keys, rowmap = _hash_index(masks_array, hash_slots)
    marker = np.full(d, -1, dtype=np.int64)
    scratch = np.empty(d, dtype=dtype)
    targets = np.empty(d, dtype=np.int32)
    output_ptr = np.zeros(d+1, dtype=np.int64)
    output_rows = np.empty(0, dtype=np.int32)
    output_values = np.empty(0, dtype=dtype)
    args = (masks_array, columns, pointers, annihilators, creators, signs, constants,
            values, lookup, counts, rowmap, row_keys, prefix_mask, count_support,
            marker, scratch, targets, output_ptr)
    nonzero = _assemble(*args, output_rows, output_values, True)
    row_dtype = _csc_row_dtype(nonzero)
    required = base+nonzero*(size+np.dtype(row_dtype).itemsize)+(d+1)*8
    if required > memory_bytes:
        raise MemoryError('complete counted canonical sparse matrix exceeds budget')
    output_rows = np.empty(nonzero, dtype=row_dtype)
    output_values = np.empty(nonzero, dtype=dtype)
    marker.fill(-1)
    assert _assemble(*args, output_rows, output_values, False) == nonzero
    matrix = csc_matrix((output_values, output_rows, output_ptr), shape=(d, d))
    return matrix
