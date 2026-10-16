"""Native long-wavelength multipoles in a complete finite determinant order."""

import numpy as np

from atomx.angular.wigner import wigner_3j
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import radial_eL, radial_mL


def single_particle(labels, orbitals, workspace, kind, rank, q):
    """Grant/FAC public primitives, signed spherical component (not reduced)."""
    if kind not in ("E", "M") or rank < 1 or abs(q) > rank:
        raise ValueError("explicit electric/magnetic spherical component required")
    angular = angular_eL if kind == "E" else angular_mL
    radial = radial_eL if kind == "E" else radial_mL
    matrix = np.zeros((len(labels), len(labels)))
    cache = {}
    for i, (nf, kf, mf) in enumerate(labels):
        for j, (ni, ki, mi) in enumerate(labels):
            if mf != mi + 2 * q:
                continue
            factor = angular(ki, kf, rank)
            if factor == 0:
                continue
            pair = ((nf, kf), (ni, ki))
            if pair not in cache:
                cache[pair] = factor * radial(
                    *orbitals[ni, ki], *orbitals[nf, kf], workspace, rank
                )
            jf, ji = 2 * abs(kf) - 1, 2 * abs(ki) - 1
            matrix[i, j] = (
                (-1) ** ((jf - mf) // 2)
                * wigner_3j(jf, 2 * rank, ji, -mf, 2 * q, mi)
                * cache[pair]
            )
    return matrix


def determinant_operator(single, masks):
    """Independent occupation-list remove/insert signs, no CC oracle action."""
    index = {int(mask): i for i, mask in enumerate(masks)}
    result = np.zeros((len(masks), len(masks)), dtype=single.dtype)
    for column, mask in enumerate(masks):
        occupied = [bit for bit in range(len(single)) if int(mask) & (1 << bit)]
        for position, removed in enumerate(occupied):
            remaining = occupied[:position] + occupied[position + 1 :]
            for created in range(len(single)):
                if created in remaining or single[created, removed] == 0:
                    continue
                insert = sum(bit < created for bit in remaining)
                target = sum(1 << bit for bit in remaining) | (1 << created)
                result[index[target], column] += (-1) ** (position + insert) * single[
                    created, removed
                ]
    return result
