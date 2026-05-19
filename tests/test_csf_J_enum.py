"""J-resolved jj-CSF enumeration checks."""

from collections import Counter

from atomx.methods.mcdhf.config import enumerate_jj_csfs


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
