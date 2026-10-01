"""Common-core normal ordering preserves exact determinant matrix elements."""
from itertools import combinations

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import _core_contracted_pair_terms, _pair_terms
from atomx.physics.csf.breit_ci import breit_terms


def collect(terms):
    result = {}
    for key, value in terms:
        result[key] = result.get(key, 0.)+value
    return result


@pytest.mark.parametrize('operator', [None, breit_terms])
@pytest.mark.parametrize('core_bits', [(0, 1), (2, 4)])
def test_common_core_matches_uncontracted_slater_condon(operator, core_bits):
    labels = ((1, -1, -1), (1, -1, 1), (2, 1, -1), (2, 1, 1),
              (2, -2, -3), (2, -2, -1), (2, -2, 1), (2, -2, 3))
    core = sum(1 << i for i in core_bits)
    dets = [core | sum(1 << i for i in bits) for bits in
            combinations([i for i in range(len(labels)) if i not in core_bits], 2)]
    contracted = _core_contracted_pair_terms(core, labels, operator)
    for bra in dets:
        for ket in dets:
            actual = collect(contracted(bra, ket))
            expected = collect(_pair_terms(bra, ket, labels, term_fn=operator))
            keys = sorted(set(actual) | set(expected))
            np.testing.assert_allclose([actual.get(k, 0.) for k in keys],
                                       [expected.get(k, 0.) for k in keys], atol=2e-14)
