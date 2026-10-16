"""Signed tensor contractions with explicit free-index permutations.

The caller supplies equations, occupied/virtual ranges and owned arrays.
Only contraction paths depend on shapes; no physical values are cached.
"""
import re
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4096)
def _parse(term):
    coefficient, *factors = term
    permutations, operands = [], []
    for factor in factors:
        if factor.startswith('P('):
            permutations.append(tuple(factor[2:-1].split(',')))
        elif factor.startswith('<'):
            operands.append(('g', factor[1:-1].replace('||', '').replace(',', '')))
        else:
            match = re.fullmatch(r'(f|t[123]|r[123])\(([a-z,]+)\)', factor)
            if match is None:
                raise ValueError(f'unsupported tensor factor: {factor}')
            operands.append((match[1], match[2].replace(',', '')))
    return float(coefficient), tuple(permutations), tuple(operands)


@lru_cache(maxsize=4096)
def _path(equation, shapes, maximum_elements):
    # Shape-only views: einsum_path inspects dimensions, never array contents.
    arrays = [np.lib.stride_tricks.as_strided(
        np.zeros(1), shape=shape, strides=(0,)*len(shape), writeable=False)
        for shape in shapes]
    return np.einsum_path(equation, *arrays,
                          optimize=('greedy', maximum_elements))[0]


def _projected(value, coefficient, permutations, output, indices):
    if not permutations:
        return coefficient*value[indices]
    first, second = permutations[-1]
    a, b = output.index(first), output.index(second)
    swapped = list(indices)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    # Preserve nested subtraction, including exact zero cancellations.
    return (_projected(value, coefficient, permutations[:-1], output, indices)
            -_projected(value, coefficient, permutations[:-1], output, tuple(swapped)))


def contract(terms, output, arrays, noccupied, *, intermediate_bytes,
             attachment_batch=False, output_indices=None):
    """Evaluate complete recipes; never screen terms or symmetrize matrices.

    Integral indices a..f are virtual and i..n occupied. Amplitude arrays
    already use local indices. Attachment arrays may carry one final batch
    axis, retained through every contraction. The caller accounts for owned
    arrays, output and permutation temporaries in its whole-call budget.
    Explicit output_indices gather requested coordinates before accumulation
    and free-index antisymmetrization. The contraction itself remains complete;
    no amplitude or integral is screened. A final attachment batch axis stays.
    """
    if intermediate_bytes < 16:
        raise MemoryError('tensor intermediate budget is too small')
    result = None
    slices = {'o': slice(0, noccupied), 'v': slice(noccupied, None)}
    for term in terms:
        coefficient, permutations, operands = _parse(term)
        arguments, indices = [], []
        for name, labels in operands:
            value = arrays[name]
            if name in ('f', 'g'):
                value = value[tuple(slices['o' if x in 'ijklmn' else 'v']
                                    for x in labels)]
            elif attachment_batch and name.startswith('r'):
                labels += 'z'
            indices.append(labels)
            arguments.append(value)
        target = output+('z' if attachment_batch else '')
        equation = ','.join(indices)+'->'+target
        dtype = np.result_type(*arguments)
        path = _path(equation, tuple(a.shape for a in arguments),
                     intermediate_bytes//dtype.itemsize)
        value = np.einsum(equation, *arguments, optimize=path)
        if output_indices is not None:
            if len(output_indices) != len(output):
                raise ValueError('one coordinate index per free output axis required')
            contribution = _projected(value, coefficient, permutations, output,
                                      tuple(output_indices))
            if result is None:
                result = contribution
            else:
                result += contribution
            continue
        value = coefficient*value
        for first, second in permutations:
            value = value-value.swapaxes(output.index(first), output.index(second))
        if result is None:
            result = value
        else:
            result += value
    return result
