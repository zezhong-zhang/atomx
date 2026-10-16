"""Independent full magnetic and fermion two-valence Fov screening anchors."""

import itertools

import numpy as np

from atomx.angular.csf_hamiltonian import _ops_sign


def action(fock, eps, sigmas, energies, nc, *, primes=None):
    n = len(eps)
    if sigmas.shape != (n, n, nc, n) or not np.all(np.isfinite(sigmas)):
        raise ValueError("complete finite magnetic numerator columns required")
    if primes is not None and (
        primes.shape != sigmas.shape or not np.all(np.isfinite(primes))
    ):
        raise ValueError("complete finite magnetic response columns required")
    o, v = slice(0, nc), slice(nc, n)
    result = np.zeros((n - nc,) * 4, dtype=np.result_type(fock, eps, sigmas, energies))
    for a, b in itertools.product(range(nc, n), repeat=2):
        da = (
            energies[a]
            + energies[b]
            + eps[o][None, None, :]
            - eps[b]
            - eps[v][:, None, None]
            - eps[v][None, :, None]
        )
        db = da + eps[b] - eps[a]
        if any(np.any(d == 0) or not np.all(np.isfinite(d)) for d in (da, db)):
            raise ValueError("singular/nonfinite complete joint screening denominator")
        ta = sigmas[v, v, :, a] / da
        tb = sigmas[v, v, :, b] / db
        if primes is not None:
            ta = primes[v, v, :, a] / da - 2 * ta / da
            tb = primes[v, v, :, b] / db - 2 * tb / db
        result[:, :, a - nc, b - nc] = -np.einsum(
            "c,mnc->mn", fock[o, b], ta
        ) - np.einsum("c,nmc->mn", fock[o, a], tb)
    return result


def fermion_projection(fock, eps, sigmas, energies, nc):
    """Actual irreducible P[F,S_v+S_w]P after independent one-body subtraction.

    The full pair projection also embeds the one-valence Sigma1 operator.
    Evaluate that part with actual operators on the singly attached core,
    using the same pair-dependent denominators, before extracting Sigma2.
    """
    n = len(eps)
    pairs = tuple(itertools.combinations(range(nc, n), 2))
    core = (1 << nc) - 1
    masks = {core | (1 << m) | (1 << nn): i for i, (m, nn) in enumerate(pairs)}
    fterms = [
        (float(fock[p, q]), ((p, True), (q, False)))
        for p, q in itertools.product(range(n), repeat=2)
        if p != q and fock[p, q] != 0
    ]

    def apply(terms, vector):
        result = {}
        for mask, coefficient in vector.items():
            for value, ops in terms:
                phase = _ops_sign(mask, ops)
                if not phase:
                    continue
                target = mask
                for bit, create in reversed(ops):
                    target = target | (1 << bit) if create else target & ~(1 << bit)
                result[target] = result.get(target, 0.0) + coefficient * value * phase
        return {m: value for m, value in result.items() if value != 0}

    output = np.zeros((len(pairs), len(pairs)))
    reverse_projection = np.zeros_like(output)
    embedded_single = np.zeros_like(output)
    for j, (a, b) in enumerate(pairs):
        terms = []
        single = {}
        for column, other in ((a, b), (b, a)):
            column_terms = []
            for m, nn, c in itertools.product(range(nc, n), range(nc, n), range(nc)):
                den = (
                    energies[column]
                    + energies[other]
                    + eps[c]
                    - eps[other]
                    - eps[m]
                    - eps[nn]
                )
                value = sigmas[m, nn, c, column] / den
                if value != 0:
                    column_terms.append(
                        (value, ((m, True), (nn, True), (c, False), (column, False)))
                    )
            terms.extend(column_terms)
            attached = {core | (1 << column): 1.0}
            one_fs = apply(fterms, apply(column_terms, attached))
            one_sf = apply(column_terms, apply(fterms, attached))
            single[column] = {
                m: one_fs.get(core | (1 << m), 0) - one_sf.get(core | (1 << m), 0)
                for m in range(nc, n)
            }
        vector = {core | (1 << a) | (1 << b): 1.0}
        fs = apply(fterms, apply(terms, vector))
        sf = apply(terms, apply(fterms, vector))
        for mask, i in masks.items():
            output[i, j] = fs.get(mask, 0) - sf.get(mask, 0)
            reverse_projection[i, j] = sf.get(mask, 0)
            m, nn = pairs[i]
            embedded_single[i, j] = (
                single[a][m] * (nn == b)
                + single[b][nn] * (m == a)
                - single[a][nn] * (m == b)
                - single[b][m] * (nn == a)
            )
    return pairs, output - embedded_single, reverse_projection, embedded_single
