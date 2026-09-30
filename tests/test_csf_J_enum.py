"""J-resolved jj-CSF enumeration checks.

The multi-digit-n regressions (HYP-1788) are pinned through the shared
parser :func:`atomx.config.parsing.parse_nr_shells` — the single-character
slicing it replaced (``n = int(token[0])`` …) crashed on Rydberg configs
(``enumerate_jj_csfs("1s2 10s1")`` → ``KeyError: '0'``).  The malformed-token
assertions are implementation-agnostic: a ``ValueError`` naming the token,
not a ``KeyError``.  Ported from PR #85 (HYP-1788), whose code change is
subsumed by the shared parser.
"""

import re
from collections import Counter

import pytest

from atomx.config.csf import enumerate_jj_csfs


def _J2_multiset(config):
    return Counter(csf.total_J2 for csf in enumerate_jj_csfs(config))


def test_p5_matches_grant_appendix_a_terms():
    assert _J2_multiset("1s2 2s2 2p5") == Counter({1: 1, 3: 1})


def test_p4_3d1_matches_grant_appendix_a_coupling():
    assert _J2_multiset("1s2 2s2 2p4 3d1") == Counter({
        1: 5,
        3: 8,
        5: 8,
        7: 5,
        9: 2,
    })


def test_closed_shell_stays_single_J0_csf():
    csfs = enumerate_jj_csfs("1s2 2s2 2p6")

    assert len(csfs) == 1
    assert csfs[0].total_J2 == 0
    assert csfs[0].parity == 1


def test_fe17_benchmark_config_is_J_resolved():
    configs = [
        "1s2 2s2 2p6",
        "1s2 2s2 2p5 3s1",
        "1s2 2s2 2p5 3p1",
        "1s2 2s2 2p5 3d1",
    ]
    csfs = [csf for cfg in configs for csf in enumerate_jj_csfs(cfg)]

    assert len(csfs) == 27
    assert any(csf.total_J2 != 0 for csf in csfs)


def test_multi_digit_n_s_shell_no_keyerror():
    # HYP-1788: single-character parsing crashed on n>=10 with KeyError: '0'.
    csfs = enumerate_jj_csfs("1s2 10s1")

    assert len(csfs) == 1
    assert csfs[0].total_J2 == 1
    assert csfs[0].parity == 1
    rydberg = [sh for sh in csfs[0].subshells if sh.occupation == 1]
    assert [(sh.n, sh.kappa) for sh in rydberg] == [(10, -1)]


def test_multi_digit_n_p_shell_matches_low_n_coupling():
    # Same jj-coupling structure as 1s2 2p1, only the principal number differs.
    high = _J2_multiset("1s2 10p1")
    low = _J2_multiset("1s2 2p1")

    assert high == low == Counter({1: 1, 3: 1})
    csfs = enumerate_jj_csfs("1s2 10p1")
    assert {csf.parity for csf in csfs} == {-1}
    for csf in csfs:
        rydberg = [sh for sh in csf.subshells if sh.n == 10]
        assert len(rydberg) == 1
        assert rydberg[0].kappa in (1, -2)


def test_multi_digit_n_d_shell_J_resolved():
    # 10d1 splits into j=3/2 (kappa=+2) and j=5/2 (kappa=-3).
    csfs = enumerate_jj_csfs("1s2 2s2 2p6 10d1")

    assert Counter(csf.total_J2 for csf in csfs) == Counter({3: 1, 5: 1})
    assert {csf.parity for csf in csfs} == {1}


def test_high_l_orbital_letters_supported():
    # L_MAP covers i (l=6) like the central parser: kappa=+6 -> 2J=11,
    # kappa=-7 -> 2J=13.
    csfs = enumerate_jj_csfs("10i1")

    assert Counter(csf.total_J2 for csf in csfs) == Counter({11: 1, 13: 1})


def test_implicit_unit_occupation():
    # The shared parser accepts a missing occupation as 1 (review finding M2).
    csfs = enumerate_jj_csfs("1s2 11s")
    assert [(sh.n, sh.kappa) for sh in csfs[0].subshells if sh.n == 11] == [(11, -1)]


@pytest.mark.parametrize("bad", ["1s2 2x1", "10"])
def test_malformed_token_raises_value_error(bad):
    """Malformed tokens raise ValueError naming the token, not KeyError."""
    with pytest.raises(ValueError, match=re.escape(bad.split()[-1])):
        enumerate_jj_csfs(bad)
