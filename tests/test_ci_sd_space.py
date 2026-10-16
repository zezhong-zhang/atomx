"""Complete NR SD spaces independently checked against occupation distances."""
import itertools
from unittest.mock import patch

import pytest

from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.config.parsing import parse_nr_shells
from atomx.methods.ci_sd import plan_sd_ci_space


def _occupation(config):
    return {(n, l): q for n, l, q in parse_nr_shells(config)}


@pytest.mark.parametrize('rank', (1, 2))
def test_reference_union_complete_occupations_and_jj_parentage(rank):
    refs = ('1s2 2s1', '2s1 1s2', '1s2 2p1')
    targets = ('1s', '2s', '2p')
    keys = ((1, 0), (2, 0), (2, 1))
    reference = [_occupation(c) for c in refs]
    # Independent fixed-N occupation distance: minimum number of moved
    # electrons, including occupied targets and same-shell double moves.
    expected = set()
    for occ in itertools.product(range(3), range(3), range(4)):
        if sum(occ) == 3 and min(sum(max(q-r.get(k, 0), 0) for k, q in zip(keys, occ, strict=True))
                                for r in reference) <= rank:
            expected.add(tuple(occ))
    for j, parity in ((1, 1), (1, -1), (3, -1)):
        plan = plan_sd_ci_space(refs, targets, J2=j, parity=parity, max_excitations=rank,
                                dimension_cap=500, max_determinants=500)
        actual = {tuple(_occupation(c).get(k, 0) for k in keys) for c in plan.configurations}
        assert actual == expected
        assert len(plan.reference_configurations) == 2
        csfs, p = plan.materialize()
        independent = enumerate_jj_csfs_many([' '.join(f'{n}{"sp"[l]}{q}' for (n, l), q in zip(keys, occ, strict=True) if q)
                                               for occ in expected])
        signatures = {csf_signature(c).key for c in independent
                      if (c.total_J2, c.parity) == (j, parity)}
        assert {csf_signature(c).key for c in csfs} == signatures
        assert len(csfs) == plan.n_csfs
        patterns = {csf_signature(c).pattern for c in csfs}
        assert sum(dict(pattern_m_counts(pat)).get(j, 0) for pat in patterns) <= plan.determinant_bound
        assert p


def test_frozen_shell_exclusion_and_preallocation_caps():
    plan = plan_sd_ci_space(('1s2 2s2',), ('2s', '2p', '3s'), frozen_orbitals=('1s',),
                            J2=0, parity=1, dimension_cap=100, max_determinants=100)
    assert all(_occupation(c)[1, 0] == 2 for c in plan.configurations)
    assert any(_occupation(c).get((2, 1)) == 2 for c in plan.configurations)
    with patch('atomx.methods.ci_sd.enumerate_jj_csfs_many') as materializer:
        with pytest.raises(ValueError, match='caps'):
            plan_sd_ci_space(('1s2 2s2',), ('2s', '2p', '3s'), J2=0, parity=1, dimension_cap=1)
        materializer.assert_not_called()


@pytest.mark.parametrize('options,match', [
    ({'max_excitations': 3}, 'singles'),
    ({'max_excitations': True}, 'singles'),
    ({'parity': 1.0}, 'integer'),
    ({'frozen_orbitals': ('3s',)}, 'frozen'),
    ({'frozen_orbitals': ('2s',)}, 'targets'),
    ({'target_orbitals': ('2s', '2s')}, 'duplicate'),
    ({'target_orbitals': ('2s1',)}, 'occupations'),
    ({'J2': 0}, 'electron-number'),
    ({'reference_configurations': ('1s2', '1s2 2s1')}, 'electron count'),
    ({'reference_configurations': ('1s1 1s2',)}, 'duplicate'),
])
def test_invalid_space_requests_are_explicit(options, match):
    args = dict(reference_configurations=('1s2 2s1',), target_orbitals=('2s', '2p'), J2=1, parity=1)
    args.update(options)
    with pytest.raises(ValueError, match=match):
        plan_sd_ci_space(**args)
