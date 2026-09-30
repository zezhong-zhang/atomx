"""Radial energy contractions with a declared set of changing orbitals.

The caller supplies one-body weights and signed Coulomb coefficients. No
configuration, averaging prescription or SCF policy is selected here.
"""
import numpy as np
from numba import njit

from .coulomb import _yk_over_r
from .operators import one_body_element
from .quadrature import slag6


def _readonly(values, dtype, shape=None):
    array = np.array(values, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    array.setflags(write=False)
    return array


@njit(cache=True)
def _refresh_rotation_integrals(pa, qa, r, difr, pairs, pair_changing,
                                 requests, densities, potentials, changing):
    """Refresh only the declared fixed or variable part of one pair context."""
    n = len(r)
    for index in range(len(pairs)):
        if pair_changing[index] == changing:
            a, b = pairs[index]
            for i in range(n):
                densities[index, i] = pa[a, i]*pa[b, i]+qa[a, i]*qa[b, i]
    for index in range(len(requests)):
        rank, pair = requests[index]
        if pair_changing[pair] == changing:
            potentials[index] = _yk_over_r(densities[pair], rank, r, difr, n)


@njit(cache=True)
def _contract_rotation_energy(initial, changing, pa, qa, r, difr, rv,
                               occupations, kappas, orbital_changing,
                               terms, coefficients, term_changing,
                               densities, potentials):
    """Preserve the ordered one-body/Coulomb contraction and SLAG6 sums."""
    energy = initial
    for a in range(len(occupations)):
        if orbital_changing[a] == changing:
            energy += occupations[a]*one_body_element(
                pa[a], qa[a], pa[a], qa[a], kappas[a], r, difr, rv)
    for index in range(len(terms)):
        if term_changing[index] == changing:
            potential, pair = terms[index]
            energy += coefficients[index]*slag6(
                1., potentials[potential]*densities[pair]*difr, len(r))[-1]
    return float(energy)


class RadialRotationEnergy:
    """Fixed scalar Hamiltonian graph with live, owned-by-caller orbital buffers.

    Construct once for an occupied rotation sweep with declared one-body and
    signed Coulomb coefficients. Graph arrays are read-only and contain no
    radial-state cache. Each ``for_pair`` creates a new independent radial
    context; all changing pairs are refreshed on every callback evaluation.
    Rebuild the sweep object after a Hamiltonian/state-weight update.
    """

    def __init__(self, pa, qa, r, difr, rv, one_body, coulomb):
        n = len(r)
        if (pa.ndim != 2 or qa.shape != pa.shape or pa.shape[1] != n
                or len(one_body) != len(pa) or difr.shape != (n,) or rv.shape != (n,)):
            raise ValueError('rotation energy requires aligned orbital and radial arrays')
        occupations = _readonly([weight for weight, _ in one_body], float)
        kappas = _readonly([kappa for _, kappa in one_body], np.int64)
        pair_lookup, request_lookup = {}, {}
        pairs, requests, terms, coefficients = [], [], [], []
        for coefficient, rank, a, b, c, d in coulomb:
            if (not isinstance(rank, (int, np.integer)) or rank < 0
                    or any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(pa)
                           for i in (a, b, c, d))):
                raise ValueError('invalid Coulomb rank or orbital index')
            left, right = tuple(sorted((a, b))), tuple(sorted((c, d)))
            for pair in (left, right):
                if pair not in pair_lookup:
                    pair_lookup[pair] = len(pairs)
                    pairs.append(pair)
            left_id, right_id = pair_lookup[left], pair_lookup[right]
            request = (rank, left_id)
            if request not in request_lookup:
                request_lookup[request] = len(requests)
                requests.append(request)
            terms.append((request_lookup[request], right_id))
            coefficients.append(coefficient)
        pairs = _readonly(pairs, np.int64, (-1, 2))
        requests = _readonly(requests, np.int64, (-1, 2))
        terms = _readonly(terms, np.int64, (-1, 2))
        coefficients = _readonly(coefficients, float)
        self.pa, self.qa = pa, qa
        self.r, self.difr, self.rv = r, difr, rv
        self.occupations, self.kappas = occupations, kappas
        self.pairs, self.requests = pairs, requests
        self.terms, self.coefficients = terms, coefficients

    def for_pair(self, a, b):
        """Freeze the unaffected contributions for one pair, retaining order.

        Only orbitals a,b may mutate while this callback is in use. The
        callback is not reentrant, like the orbital buffers it operates on.
        Left/right Coulomb densities are never interchanged: that symmetry
        is not exact for the discrete quadrature map.
        """
        pa, qa, r, difr, rv = self.pa, self.qa, self.r, self.difr, self.rv
        if any(not isinstance(i, (int, np.integer)) or not 0 <= i < len(pa)
               for i in (a, b)):
            raise ValueError('changing orbital index is outside the supplied orbital set')
        orbital_changing = np.zeros(len(pa), dtype=bool)
        orbital_changing[[a, b]] = True
        orbital_changing.setflags(write=False)
        pairs, requests = self.pairs, self.requests
        terms, coefficients = self.terms, self.coefficients
        occupations, kappas = self.occupations, self.kappas
        pair_changing = np.any(orbital_changing[pairs], axis=1)
        term_changing = (pair_changing[requests[terms[:, 0], 1]]
                         | pair_changing[terms[:, 1]])
        pair_changing.setflags(write=False)
        term_changing.setflags(write=False)
        densities = np.empty((len(pairs), len(r)))
        potentials = np.empty((len(requests), len(r)))
        _refresh_rotation_integrals(pa, qa, r, difr, pairs, pair_changing,
                                     requests, densities, potentials, False)
        fixed = _contract_rotation_energy(0., False, pa, qa, r, difr, rv,
            occupations, kappas, orbital_changing, terms, coefficients,
            term_changing, densities, potentials)

        def energy():
            _refresh_rotation_integrals(pa, qa, r, difr, pairs, pair_changing,
                                         requests, densities, potentials, True)
            return _contract_rotation_energy(fixed, True, pa, qa, r, difr, rv,
                occupations, kappas, orbital_changing, terms, coefficients,
                term_changing, densities, potentials)
        return energy
