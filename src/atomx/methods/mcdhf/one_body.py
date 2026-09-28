"""Spherical one-body operators in the exact jj-coupled CSF basis."""
from __future__ import annotations

import numpy as np

from atomx.methods.mcdhf.exact_ci import csf_signature, one_body_tensor_terms


def scalar_one_body_matrix(csfs, keys, radial):
    """Lift radial <a|v|b> to CSFs, retaining same-κ radial coherence.

    The radial scalar has no reduced-angular normalization. The factor
    sqrt((2j_a+1)/(2J+1)) converts the cached reduced tensor coefficients
    to ordinary matrix elements of the sum over electrons.
    """
    index = {key: i for i, key in enumerate(keys)}
    signatures = [csf_signature(csf) for csf in csfs]
    matrix = np.zeros((len(csfs), len(csfs)))
    for a, bra in enumerate(signatures):
        for b in range(a, len(signatures)):
            ket = signatures[b]
            if bra.total_J2 != ket.total_J2 or csfs[a].parity != csfs[b].parity:
                continue
            value = 0.0
            for dest, source, angular in one_body_tensor_terms(bra, ket, 0):
                if dest[1] == source[1]:
                    value += (angular * np.sqrt(2 * abs(dest[1]) / (bra.total_J2 + 1))
                              * radial[index[dest], index[source]])
            matrix[a, b] = matrix[b, a] = value
    return matrix


def scalar_one_body_expectations(csfs, mixing, keys, radial):
    """First-order expectation in each column of the real CI mixing matrix."""
    matrix = scalar_one_body_matrix(csfs, keys, radial)
    return np.einsum("ik,ij,jk->k", mixing, matrix, mixing, optimize=True)
