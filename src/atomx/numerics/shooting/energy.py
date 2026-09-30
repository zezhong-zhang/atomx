"""Radial energy contractions with a declared set of changing orbitals.

The caller supplies one-body weights and signed Coulomb coefficients. No
configuration, averaging prescription or SCF policy is selected here.
"""
from .coulomb import _yk_over_r
from .operators import one_body_element
from .quadrature import slag6


def radial_rotation_energy(pa, qa, r, difr, rv, one_body, coulomb, changing):
    """Build an energy callback for one pair rotation at fixed state weights.

    ``one_body`` holds (occupation, kappa) per orbital; ``coulomb`` holds
    (coefficient, rank, a, b, c, d) for R^rank(ab,cd). Only orbitals in
    ``changing`` may mutate while the callback is in use. The caller creates
    a fresh callback for each pair, so no caches survive a state/SCF update.
    """
    changing = frozenset(changing)
    fixed_densities, fixed_potentials = {}, {}
    n = len(r)

    def density(pair, cache):
        if pair not in cache:
            i, j = pair
            cache[pair] = pa[i]*pa[j]+qa[i]*qa[j]
        return cache[pair]

    def potential(rank, pair, densities, cache):
        key = (rank, pair)
        if key not in cache:
            cache[key] = _yk_over_r(density(pair, densities), rank, r, difr, n)
        return cache[key]

    fixed = 0.
    variable_one_body = []
    for i, (weight, kappa) in enumerate(one_body):
        if i in changing:
            variable_one_body.append((i, weight, kappa))
        else:
            fixed += weight*one_body_element(pa[i], qa[i], pa[i], qa[i],
                                            kappa, r, difr, rv)
    variable_coulomb = []
    for coefficient, rank, a, b, c, d in coulomb:
        left, right = tuple(sorted((a,b))), tuple(sorted((c,d)))
        if changing.isdisjoint(left+right):
            y = potential(rank, left, fixed_densities, fixed_potentials)
            fixed += coefficient*slag6(1., y*density(right, fixed_densities)*difr, n)[-1]
        else:
            variable_coulomb.append((coefficient, rank, left, right))

    def energy():
        result = fixed
        densities, potentials = {}, {}
        for i, weight, kappa in variable_one_body:
            result += weight*one_body_element(pa[i], qa[i], pa[i], qa[i],
                                             kappa, r, difr, rv)
        for coefficient, rank, left, right in variable_coulomb:
            ld = fixed_densities if changing.isdisjoint(left) else densities
            lp = fixed_potentials if changing.isdisjoint(left) else potentials
            rd = fixed_densities if changing.isdisjoint(right) else densities
            y = potential(rank, left, ld, lp)
            result += coefficient*slag6(1., y*density(right, rd)*difr, n)[-1]
        return float(result)
    return energy
