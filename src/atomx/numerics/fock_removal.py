"""Exact signed removal and integer linear forms of finite occupied bitsets."""

import numpy as np
from numba import njit, types
from numba.typed import Dict


def _masks(values, n_spinors):
    if (isinstance(n_spinors, (bool, np.bool_))
            or not isinstance(n_spinors, (int, np.integer)) or not 1 <= n_spinors <= 63):
        raise ValueError('explicit1..63-spinor bit layout required')
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError('one-dimensional integer determinant masks required')
    if np.any(values < 0) or np.any(values.astype(np.uint64) >> np.uint64(n_spinors)):
        raise ValueError('determinant mask outside declared bit layout')
    if len(np.unique(values)) != len(values):
        raise ValueError('distinct determinant masks required')
    return np.ascontiguousarray(values, dtype=np.uint64)


def _caps(memory_bytes, max_steps):
    for value in (memory_bytes, max_steps):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('positive integer numerical resource caps required')


@njit(cache=True)
def _removal(bra_masks, bra_values, ket_masks, ket_values, n):
    lookup = Dict.empty(key_type=types.uint64, value_type=types.float64)
    for i in range(len(bra_masks)):
        lookup[bra_masks[i]] = bra_values[i]
    result = np.zeros(n)
    for i in range(len(ket_masks)):
        mask, phase = ket_masks[i], 1.
        for p in range(n):
            bit = np.uint64(1) << np.uint64(p)
            if mask & bit:
                result[p] += lookup.get(mask ^ bit, 0.) * ket_values[i] * phase
                phase = -phase
    return result


def fock_removal_amplitudes(bra_masks, bra_values, ket_masks, ket_values, n_spinors, *,
                            memory_bytes=128*1024**2, max_steps=500000000):
    """Return <bra|a_p|ket> in supplied real finite determinant coordinates.

    No particle-sector completion, normalization or small-amplitude screening.
    The ordered accumulation matches ascending annihilation bits per ket.
    Resource accounting includes conservative lookup storage and input copies;
    process RSS and compilation memory are distinct from this payload cap.
    """
    _caps(memory_bytes, max_steps)
    bra_masks, ket_masks = _masks(bra_masks, n_spinors), _masks(ket_masks, n_spinors)
    arrays = []
    for masks, coefficients in ((bra_masks, bra_values), (ket_masks, ket_values)):
        coefficients = np.asarray(coefficients)
        if (coefficients.shape != masks.shape or not np.isrealobj(coefficients)
                or coefficients.dtype.kind not in 'biuf' or not np.isfinite(coefficients).all()):
            raise ValueError('finite real coefficients matching complete mask arrays required')
        arrays.append(np.ascontiguousarray(coefficients, dtype=float))
    if 96*len(bra_masks)+32*len(ket_masks)+8*n_spinors > memory_bytes:
        raise MemoryError('complete signed removal exceeds numerical payload cap')
    if (len(bra_masks)+len(ket_masks))*n_spinors > max_steps:
        raise MemoryError('complete signed removal exceeds numerical work cap')
    out = _removal(bra_masks, arrays[0], ket_masks, arrays[1], n_spinors)
    if not np.isfinite(out).all():
        raise FloatingPointError('signed removal accumulation overflow')
    return out


@njit(cache=True)
def _linear_forms(masks, weights):
    out = np.zeros(len(masks), dtype=np.int64)
    for i in range(len(masks)):
        for p in range(len(weights)):
            if masks[i] & (np.uint64(1) << np.uint64(p)):
                out[i] += weights[p]
    return out


def mask_linear_forms(masks, weights, *, memory_bytes=128*1024**2, max_steps=500000000):
    """Exact integer sum of supplied per-bit weights for each occupied mask.

    This is a bitset numerical operation, with no interpretation of the weights
    as angular momenta, charges or other physical policy.
    """
    _caps(memory_bytes, max_steps)
    weights = np.asarray(weights)
    if weights.ndim != 1 or weights.dtype.kind not in 'iu' or not 1 <= len(weights) <= 63:
        raise ValueError('integer weights in a1..63-bit layout required')
    if sum(abs(int(x)) for x in weights) > np.iinfo(np.int64).max:
        raise ValueError('integer linear-form sum can overflow int64')
    masks = _masks(masks, len(weights))
    if 32*len(masks)+16*len(weights) > memory_bytes or len(masks)*len(weights) > max_steps:
        raise MemoryError('complete mask linear forms exceed numerical payload/work cap')
    return _linear_forms(masks, np.ascontiguousarray(weights, dtype=np.int64))
