"""Exact real single-continuum Fock action on supplied finite vectors."""

import numba as nb
import numpy as np
from numba.typed import Dict

from atomx.numerics.slater_values import _count, _first, _phase


@nb.njit(cache=True)
def _amplitudes(initial, coefficients, final, h, core, g, nspinors):
    # g[p,q,r] = <epsilon p || q r>; epsilon is appended last.
    ah, av = 0., 0.
    one = np.uint64(1)
    occupied = _count(initial[0])
    single_creation = -1. if (occupied-1) % 2 else 1.
    double_creation = -1. if (occupied-2) % 2 else 1.
    for index in range(len(initial)):
        mask, ci = initial[index], coefficients[index]
        remaining_q = mask
        while remaining_q:
            q = _first(remaining_q)
            qbit = one << np.uint64(q)
            remaining_q ^= qbit
            reduced = mask ^ qbit
            cf = final.get(reduced, 0.)
            if cf != 0:
                sign = _phase(mask, q)*single_creation
                value = core[q]
                spectators = reduced
                while spectators:
                    p = _first(spectators)
                    spectators ^= one << np.uint64(p)
                    value += g[p, q, p]
                ah += ci*cf*sign*h[q]
                av += ci*cf*sign*value
            remaining_r = remaining_q
            while remaining_r:
                r = _first(remaining_r)
                rbit = one << np.uint64(r)
                remaining_r ^= rbit
                spectator = reduced ^ rbit
                sign_remove = _phase(mask, q)*_phase(reduced, r)*double_creation
                for p in range(nspinors):
                    pbit = one << np.uint64(p)
                    if mask&pbit or g[p, q, r] == 0:
                        continue
                    cf = final.get(spectator|pbit, 0.)
                    if cf != 0:
                        # <p epsilon||q r> = -<epsilon p||q r>.
                        sign = sign_remove*_phase(spectator, p)
                        av -= ci*cf*sign*g[p, q, r]
    return ah, av


class ContinuumFockAction:
    """Caller-owned finite vectors, with exact one-/two-substitution action.

    The physical consumer owns frozen-core contraction, all integrals, angular
    normalization, states and resource admission. No physical truncation or
    magnitude screening occurs here. Masks have fewer than64 active spinors.
    """

    def __init__(self, initial_masks, initial_coefficients, final_masks, final_coefficients, nspinors):
        if not isinstance(nspinors, int) or not 1 <= nspinors <= 63:
            raise ValueError('explicit1..63 active spinors required')
        initial, final = np.asarray(initial_masks), np.asarray(final_masks)
        ci, cf = np.asarray(initial_coefficients), np.asarray(final_coefficients)
        if (initial.ndim != 1 or final.ndim != 1 or initial.size == 0 or final.size == 0
                or initial.dtype.kind not in 'iu' or final.dtype.kind not in 'iu'
                or len(set(map(int, initial))) != len(initial) or len(set(map(int, final))) != len(final)
                or any(int(m) < 0 or int(m).bit_length() > nspinors for m in (*initial, *final))
                or ci.shape != initial.shape or cf.shape != final.shape
                or not np.isrealobj(ci) or not np.isrealobj(cf) or not np.isfinite(ci).all() or not np.isfinite(cf).all()):
            raise ValueError('unique finite real vectors in the supplied spinor layout required')
        counts_i, counts_f = {int(m).bit_count() for m in initial}, {int(m).bit_count() for m in final}
        if len(counts_i) != 1 or len(counts_f) != 1 or next(iter(counts_i)) != next(iter(counts_f))+1:
            raise ValueError('fixed adjacent particle sectors required')
        self.nspinors = nspinors
        self.initial = np.asarray(initial, dtype=np.uint64).copy()
        self.coefficients = np.asarray(ci, dtype=float).copy()
        self._final = Dict.empty(key_type=nb.types.uint64, value_type=nb.types.float64)
        for mask, c in zip(final, cf, strict=True):
            self._final[np.uint64(mask)] = float(c)
        n = next(iter(counts_i))
        self.step_bound = len(initial)*(n+n*(n-1)//2*nspinors)

    def amplitudes(self, h_cross, frozen_core_cross, g_cross):
        h, core, g = (np.asarray(x) for x in (h_cross, frozen_core_cross, g_cross))
        n = self.nspinors
        if (h.shape != (n,) or core.shape != (n,) or g.shape != (n, n, n)
                or any(not np.isrealobj(x) or not np.isfinite(x).all() for x in (h, core, g))
                or np.max(abs(g+g.swapaxes(1, 2))) > 1e-12):
            raise ValueError('finite real one-body rows and exactly antisymmetric Coulomb row required')
        return tuple(float(x) for x in _amplitudes(self.initial, self.coefficients, self._final,
                     np.ascontiguousarray(h), np.ascontiguousarray(core), np.ascontiguousarray(g), n))
