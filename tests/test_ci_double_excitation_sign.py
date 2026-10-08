"""Double excitation signs against occupation-list second quantisation."""
from bisect import bisect_left
from itertools import combinations

import pytest

from atomx.angular import csf_hamiltonian as ci


def _occupation_sign(occupied, created, removed):
    # Independent ordered occupation-list action, without bit parity formulas.
    occupied = sorted(occupied)
    sign = 1
    for site in removed:
        index = occupied.index(site)
        sign *= (-1) ** index
        occupied.pop(index)
    for site in reversed(created):
        index = bisect_left(occupied, site)
        assert site not in occupied
        sign *= (-1) ** index
        occupied.insert(index, site)
    return sign


@pytest.mark.parametrize('number', [2, 3, 4, 5])
@pytest.mark.parametrize('sites', [tuple(range(8)), (0, 2, 63, 64, 65, 127, 150, 511)])
def test_all_double_excitation_signs_against_occupation_lists(number, sites):
    occupations = list(combinations(sites, number))
    labels = [(i + 1, -1, 1) for i in range(max(sites) + 1)]

    def tagged_operator(a, b, c, d, coefficient, _labels):
        return (((a, b, c, d), coefficient),)

    tested = 0
    for a in occupations:
        mask_a = sum(1 << site for site in a)
        for b in occupations:
            created, removed = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            if len(created) != 2 or len(removed) != 2:
                continue
            sign = _occupation_sign(b, created, removed)
            mask_b = sum(1 << site for site in b)
            p, q = created
            r, s = removed
            assert ci._pair_terms(mask_a, mask_b, labels, tagged_operator) == (
                ((p, q, r, s), float(sign)), ((p, q, s, r), -float(sign)))
            tested += 1
    assert tested > 0
