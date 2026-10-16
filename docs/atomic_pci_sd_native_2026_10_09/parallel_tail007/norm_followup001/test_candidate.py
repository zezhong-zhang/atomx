"""Independent contracts for the isolated complete-gram cache proposal."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OWNER = HERE.parents[1]
ROOT = OWNER.parents[1]


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def candidate():
    return load(HERE / "candidate.py", "owned_gram_candidate")


@pytest.fixture(scope="module")
def original():
    return load(
        HERE
        / "run002/source/docs/atomic_pci_sd_native_2026_10_09/monovalent_norm_control.py",
        "original_gram_control",
    )


@pytest.mark.parametrize(
    "entries, cap", [(0, 4096), (True, 4096), (2, 0), (2, -1), (2, 4.0)]
)
def test_cache_rejects_invalid_budget(candidate, original, entries, cap):
    with pytest.raises(ValueError, match="positive integer"):
        candidate.BoundGrams(
            original.grams.__wrapped__, max_entries=entries, max_bytes=cap
        )


def test_complete_signed_gram_survives_hits_evictions_and_reconstruction(
    candidate, original
):
    keys = (
        (-1, -1, -1, -1, (0, 1), (0, 1), "none", 0),
        (-5, -5, -4, -4, (0, 1), (0, 1), "none", 0),
        (4, -5, -1, -1, (0, 1), (0, 1), "hole", 1),
    )
    cache = candidate.BoundGrams(
        original.grams.__wrapped__, max_entries=2, max_bytes=192
    )
    first = cache(*keys[0])
    assert cache(*keys[0]) is first
    assert cache.hits == 1
    for key in (keys[1], keys[2], keys[0]):
        actual = cache(*key)
        expected = original.grams.__wrapped__(*key)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
            assert not a.flags.writeable
        assert len(cache.cache) <= 2
        assert cache.resident_bytes <= 192
    assert cache.evictions >= 2
    assert cache.peak_bytes <= 192


def test_one_complete_gram_larger_than_cap_is_rejected_without_pruning(
    candidate, original
):
    cache = candidate.BoundGrams(original.grams.__wrapped__, max_bytes=95)
    key = (-1, -1, -1, -1, (0, 1), (0, 1), "none", 0)
    assert sum(a.nbytes for a in original.grams.__wrapped__(*key)) == 96
    with pytest.raises(MemoryError, match="complete Gram matrix"):
        cache(*key)
    assert cache.resident_bytes == 0
    assert not cache.cache


def test_original_64MiB_fails_and_candidate_cap_failure_precedes_decoding(
    candidate, original
):
    from atomx.methods.core_sd import plan_core_sd
    from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd

    feature = dict(
        auxiliary_nodes=20,
        quadrature_nodes=(2, 3),
        same_cell_edges=12,
        coefficient_sizes=(4, 4),
        component_orders=(3, 3),
        payload_estimate_bytes=10000,
    )
    virtual = (
        (2, -1),
        (3, -1),
        (2, 1),
        (2, -2),
        (3, 2),
        (3, -3),
        (4, 3),
        (4, -4),
        (5, 4),
        (5, -5),
    )
    core = plan_core_sd(((1, -1),), virtual, feature)
    valence = plan_reduced_monovalent_sd(core, virtual)
    namespace = {"__name__": "self_contained_candidate"}
    exec(
        compile(
            candidate.specialize(Path(original.__file__).read_text()),
            "candidate_norm",
            "exec",
        ),
        namespace,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("budget failure must precede complete rho decoding")

    namespace["decode"] = forbidden
    required = candidate.plan_scratch(core, valence, virtual[0])["required_bytes"]
    assert required < 64 * 1024**2
    with pytest.raises(MemoryError, match="extra scratch"):
        original.norm_components(core, valence, None, None, virtual[0])
    with pytest.raises(MemoryError, match="extra scratch"):
        namespace["norm_components"](
            core, valence, None, None, virtual[0], memory_bytes=required - 1
        )


def test_live_signed_complete_magnetic_oracle_and_nonzero_interference():
    result = json.loads((HERE / "run004/result.json").read_text())
    assert result["status"] == "pass" and result["cases"] == 104
    assert all(result["gates"].values())
    rows = result["rows"]
    assert len({(tuple(row["key"]), row["twoM"], row["sign"]) for row in rows}) == 104
    pairs = {}
    for row in rows:
        actual, expected = row["actual_components"], row["independent_components"]
        assert len(expected) == 8
        assert max(abs(actual[k] - expected[k]) for k in expected) <= 1e-11
        pairs.setdefault((tuple(row["key"]), row["twoM"]), {})[row["sign"]] = actual
    nonzero = 0
    interference = "core_singles_valence_doubles_interference"
    for signed in pairs.values():
        positive, negative = signed[1], signed[-1]
        if abs(positive[interference]) > 1e-14:
            nonzero += 1
        assert abs(positive[interference] + negative[interference]) <= 1e-11
        assert (
            abs(
                positive["norm_squared"]
                - negative["norm_squared"]
                - 2 * positive[interference]
            )
            <= 1e-11
        )
    assert nonzero > 0
    dense = result["complete_dense_angular_transient"]
    assert dense["all_rows_and_ranks_retained"]
    assert dense["traced_peak_bytes"] <= dense["declared_transient_bound_bytes"]


def test_all_four_real_Yb_spaces_keep_five_states_and_frozen_physical_identity():
    expected_keys = {(6, -1), (6, 1), (6, -2), (5, 2), (5, -3)}
    current = json.loads((HERE / "run004/result.json").read_text())
    actual = json.loads((OWNER / "parallel_tail007/run004/result.json").read_text())
    identities = {
        task["task_id"]: task["physical_identity"] for task in actual["tasks"]
    }
    assert {
        (task["lmax"], task["emax_Ha"]) for task in current["yb_counting_only"]
    } == {(3, 5.0), (3, 10.0), (4, 5.0), (4, 10.0)}
    for task in current["yb_counting_only"]:
        assert task["original_physical_identity"] == identities[task["task_id"]]
        assert {
            tuple(state["key"]) for state in task["all_five_states"]
        } == expected_keys
        assert task["no_large_norm_executed"]
        for state in task["all_five_states"]:
            assert state["required_bytes"] <= 64 * 1024**2
            assert (
                state["cached_Gram_array_bound_bytes"]
                <= state["angular_cache_array_cap_bytes"]
            )
    for label in ("source", "input"):
        hashes = json.loads((HERE / "run004" / f"{label}_sha256.json").read_text())
        for name, expected in hashes.items():
            bound = HERE / "run004/source" / name if label == "source" else ROOT / name
            assert hashlib.sha256(bound.read_bytes()).hexdigest() == expected
    # Candidate controls stay bound to the source actually run, even when
    # the parent later adopts the proposal into the shared live helper.
    sources = json.loads((HERE / "run004/source_sha256.json").read_text())
    for name in ("candidate.py", "probe.py"):
        path = HERE / name
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == sources[str(path.relative_to(ROOT))]
        )
    assert (
        not current["shared_helper_modified"] and not current["highest_level_reached"]
    )
