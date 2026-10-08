"""Unit gates for the v4 table infrastructure (layers 0a/0b/1a).

Physics-free checks of the graph mechanics introduced with
build_rate_table_parallel v4 (2026-10-03 work order):

* shell-support skeleton sharing — every configuration of the same
  occupied-(n,l) set reuses ONE solve (Ne: 134 jj configs -> 7 supports);
* the jj-corrected hole gate in ``auger_triples`` (2|kappa| capacity);
* the F*sigma subtree prune — frozen terminals keep their row, PI
  subtrees below threshold are not expanded, the Auger closure never
  drops a reachable configuration;
* grouped/split pool jobs preserve every config exactly once.

The measured live gates (solves <= 8, pruned-table MC equivalence vs
the full table, anchor ratios) land with the v4 measurement fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.xfel.cascade import (  # noqa: E402
    RateProvider,
    _daughter_after_auger,
    _daughter_after_pi,
    _group_by_support,
    _key,
    _prune_keep_set,
    _split_for_pool,
    auger_triples,
    enumerate_configs,
    shell_support,
    skeleton_config_for,
)

NE_CONFIG = {(1, -1): 2.0, (2, -1): 2.0, (2, 1): 2.0, (2, -2): 4.0}


# -- layer 0a: support key + skeleton construction -------------------------


def test_shell_support_merges_j_partners():
    assert shell_support(NE_CONFIG) == frozenset({(1, 0), (2, 0), (2, 1)})
    # 2s-stripped config: different support -> different skeleton family
    two_s_gone = dict(NE_CONFIG)
    del two_s_gone[(2, -1)]
    assert shell_support(two_s_gone) == frozenset({(1, 0), (2, 1)})


def test_skeleton_fills_support_to_closed_shell():
    # l = 0 shell has no j-partner: {1s1} -> 1s2
    assert skeleton_config_for({(1, -1): 1.0}) == {(1, -1): 2.0}
    # l > 0: both j-partners filled to 2|kappa|
    assert skeleton_config_for({(2, -2): 3.0}) == {
        (2, 1): 2.0, (2, -2): 4.0}
    # the neutral's own support maps to the neutral (maximal) config
    assert skeleton_config_for(NE_CONFIG) == NE_CONFIG


def test_ne_enumeration_collapses_to_seven_supports():
    configs = enumerate_configs(NE_CONFIG, max_charge=10)
    supports = {shell_support(c) for c in configs}
    assert len(configs) == 134           # integer jj tuples, excluding bare ion
    assert len(supports) == 7            # non-empty subsets of {1s,2s,2p}
    assert frozenset() not in supports
    assert shell_support(NE_CONFIG) in supports


def test_provider_solves_once_per_support():
    solved_labels: list[str] = []

    def factory(label):
        solved_labels.append(label)
        return object()

    prov = RateProvider(factory)
    for cfg in enumerate_configs(NE_CONFIG, max_charge=10):
        prov._solve(cfg)
    # every enumerated config, ONE solve per support family
    assert prov.n_solves == 7
    assert len(solved_labels) == 7
    # skeletons are closed shells of their own support
    assert sorted(solved_labels) == [
        "1s2", "1s2 2p6", "1s2 2s2", "1s2 2s2 2p6",
        "2p6", "2s2", "2s2 2p6",
    ]


def test_forced_skeleton_overrides_sharing():
    def factory(label):
        return label

    prov = RateProvider(factory, skeleton_config=NE_CONFIG)
    cfg = {(1, -1): 1.0}
    assert prov._solve(cfg) == "1s2 2s2 2p6"     # the forced skeleton
    assert prov.n_solves == 0                    # sharing never engaged


# -- the jj-corrected hole gate ---------------------------------------------


def test_auger_triples_respect_jj_capacity():
    # 2p1/2 FULL (2 electrons = 2|kappa|) with electrons in 2p3/2 above:
    # no 2p1/2-hole channel.  The pre-v4 merged gate (4l+2 = 6) admitted
    # one here; only the negative-energy check caught it downstream.
    cfg = {(1, -1): 2.0, (2, 1): 2.0, (2, -2): 3.0}
    holes = {t[0] for t in auger_triples(cfg)}
    assert (2, 1) not in holes
    # the genuine 2p3/2 vacancy (3 < 4) with nothing above it -> closed
    assert holes == set()


def test_auger_triples_enumerate_k_hole_channels():
    channels = auger_triples(_daughter_after_pi(NE_CONFIG, (1, -1)))
    holes = {c[0] for c in channels}
    sources = {(c[1], c[2]) for c in channels}
    assert holes == {(1, -1)}            # the K vacancy
    assert ((2, -1), (2, -1)) in sources  # 2s2s KLL channel
    assert ((2, -2), (2, -2)) in sources  # 2p2p KLL channel


# -- layer 0b: the F*sigma prune --------------------------------------------


def _mini_model():
    """1s2 2s2 world: base + enumerated configs, synthetic sigma."""
    base = {(1, -1): 2.0, (2, -1): 2.0}
    configs = enumerate_configs(base, max_charge=4)
    sigma = {}
    for c in configs:
        k = _key(c)
        if c.get((1, -1), 0.0) > 0.5:
            sigma[k] = 1e-18           # K-bearing: absorbs strongly
        elif c.get((2, -1), 0.0) > 0.5:
            sigma[k] = 1e-24           # L-only: F*sigma at the prune edge
        else:
            sigma[k] = 0.0
    return base, configs, sigma


def test_prune_freezes_weak_subtrees_but_keeps_terminals():
    base, configs, sigma = _mini_model()
    fluence = 1e18                      # K: 1.0, L-only: 1e-6 < 1e-3
    keep = _prune_keep_set(base, configs, sigma, fluence, 1e-3)
    # base and its K-hole daughters (and their Auger refills) survive
    assert _key(base) in keep
    k_hole = _key(_daughter_after_pi(base, (1, -1)))
    assert k_hole in keep
    # the Auger closure refills the K hole from 2s: 1s2 (frozen terminal
    # with sigma 0) is KEPT — it holds population
    refilled = _key(_daughter_after_auger(
        _daughter_after_pi(base, (1, -1)), (1, -1), (2, -1), (2, -1)))
    assert refilled in keep
    # but the L-only configs' own PI subtrees are frozen: 2s2 (sigma at
    # the edge) does not expand to 2s1
    lone_2s2 = {(2, -1): 2.0}
    lone_2s1 = {(2, -1): 1.0}
    # Both remain reachable by removing 1s from distinct K-bearing
    # parents. Pruning one weak edge must not delete another strong
    # path to its daughter. Test the weak root in isolation instead.
    assert _key(lone_2s2) in keep and _key(lone_2s1) in keep
    weak_keep = _prune_keep_set(lone_2s2, configs, sigma, fluence, 1e-3)
    assert _key(lone_2s2) in weak_keep and _key(lone_2s1) not in weak_keep


def test_prune_closure_never_drops_auger_reachables():
    base, configs, sigma = _mini_model()
    keep = _prune_keep_set(base, configs, sigma, 1e18, 1e-3)
    # every kept config's hole-gate Auger daughters are kept or outside
    # the enumerated universe
    by_key = {_key(c): c for c in configs}
    for k in keep:
        cfg = by_key[k]
        for hole, s1, s2 in auger_triples(cfg):
            d = _key(_daughter_after_auger(cfg, hole, s1, s2))
            assert d in keep or d not in by_key


def test_higher_fluence_keeps_superset():
    base, configs, sigma = _mini_model()
    lo = _prune_keep_set(base, configs, sigma, 1e15, 1e-3)
    hi = _prune_keep_set(base, configs, sigma, 1e21, 1e-3)
    assert lo <= hi, "keep sets must grow monotonically with fluence"


# -- layer 1a: grouped pool jobs --------------------------------------------


def test_group_by_support_buckets_and_orders():
    configs = enumerate_configs(NE_CONFIG, max_charge=10)
    cfg_dicts = [{f"{n},{k}": q for (n, k), q in c.items()} for c in configs]
    groups = _group_by_support(cfg_dicts)
    assert sum(len(g) for g in groups) == len(configs)
    sizes = [len(g) for g in groups]
    assert sizes == sorted(sizes, reverse=True)
    # every group shares one support
    from atomx.xfel.cascade import _support_of_serialized
    for g in groups:
        assert len({_support_of_serialized(c) for c in g}) == 1


def test_split_for_pool_preserves_all_configs():
    configs = enumerate_configs(NE_CONFIG, max_charge=10)
    cfg_dicts = [{f"{n},{k}": q for (n, k), q in c.items()} for c in configs]
    groups = _group_by_support(cfg_dicts)
    jobs = _split_for_pool(groups, 16)
    assert sum(len(j) for j in jobs) == len(configs)
    assert all(j for j in jobs)
    assert len(jobs) >= min(16, len(configs))
    # split pieces stay support-pure (one skeleton solve per job)
    from atomx.xfel.cascade import _support_of_serialized
    for j in jobs:
        assert len({_support_of_serialized(c) for c in j}) == 1


def test_split_noop_when_pool_already_full():
    groups = [[{"1,-1": 2.0}], [{"2,-1": 2.0}]]
    assert _split_for_pool(groups, 1) == groups
    assert _split_for_pool(groups, 2) == groups
