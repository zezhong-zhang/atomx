"""Complete-space planning anchored to saved real native/GPU allocations."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import atomic_tail_preflight as tail  # noqa: E402

from atomx.methods.atomic_space import AtomicVirtualSpace  # noqa: E402

OWNER = ROOT / "docs/atomic_pci_sd_native_2026_10_09"


def space_from_band(band):
    occupied = tuple(map(tuple, band["occupied_keys"]))
    virtual = tuple(map(tuple, band["virtual_keys"]))
    # Counting requires radial identities only; no physical wave array is
    # loaded and no fake orbitals are consumed by a correlation calculation.
    return AtomicVirtualSpace(
        {key: (np.empty(0), np.empty(0)) for key in occupied + virtual},
        occupied,
        virtual,
        {
            "virtual_lmax": band["lmax"],
            "virtual_energy_max_Ha": band["energy_max_Ha"],
            "virtual_orbital_energies_Ha": band["energies_Ha"],
        },
    )


def test_owned_norm_planner_preserves_complete_all_five_state_space(
    measured, feature_identity
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bounded_complete_norm", OWNER / "monovalent_norm_control.py"
    )
    norm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(norm)
    for task in measured["tasks"]:
        band = json.loads(
            (
                OWNER / "parallel_tail007/run004" / (task["task_id"] + "_band.json")
            ).read_text()
        )
        space = space_from_band(band)
        legacy = tail.estimate_task(space, feature_identity)
        owned = tail.estimate_task(
            space, feature_identity, norm_planner=norm.plan_scratch
        )
        assert owned["state_amplitudes"] == legacy["state_amplitudes"]
        assert owned["core_raw_amplitudes"] == legacy["core_raw_amplitudes"]
        assert set(owned["norm_state_scratch_plans"]) == {
            str(key) for key in tail.YB_STATES
        }
        assert owned["norm_scratch_cap_bytes"] == 64 * 1024**2
        assert owned["norm_scratch_required_bytes"] <= 18538304
        assert owned["norm_scratch_within_original_cap"]
        assert owned["norm_replay_bytes"] >= 64 * 1024**2
        assert (
            owned["maximum_legacy_numeric64_host_payload_bytes"]
            == legacy["maximum_legacy_numeric64_host_payload_bytes"]
        )
    with pytest.raises(ValueError, match="positive integer"):
        tail.estimate_task(
            space, feature_identity, norm_planner=lambda *a: dict(required_bytes=True)
        )


@pytest.fixture(scope="module")
def measured():
    return json.loads((OWNER / "parallel_tail007/run004/result.json").read_text())


@pytest.fixture(scope="module")
def feature_identity():
    replay = json.loads((OWNER / "yb_full_monovalent006/core_replay.json").read_text())
    return replay["provenance"]["feature_identity"]


def test_real_baseline_counts_and_gpu_bound_match_actual_independent_receipts(
    measured, feature_identity, monkeypatch
):
    # A constructor call would allocate native COMMON/correlation storage;
    # proving this entry point avoids it is part of its memory contract.
    from atomx.methods.core_sd import CoreSD

    def forbidden(*args, **kwargs):
        raise AssertionError("counting called native constructor")

    monkeypatch.setattr(CoreSD, "__init__", forbidden)
    first = measured["tasks"][0]
    band = json.loads(
        (
            OWNER / "parallel_tail007/run004" / (first["task_id"] + "_band.json")
        ).read_text()
    )
    resource = tail.estimate_task(
        space_from_band(band), feature_identity, worker_threads=8
    )
    actual = json.loads((OWNER / "yb_full_monovalent006/resource.json").read_text())
    actual_numeric64 = json.loads(
        (OWNER / "yb_full_monovalent006/numeric64_resource.json").read_text()
    )
    actual_device = json.loads(
        (
            OWNER / "yb_full_monovalent006/cuda/context_000005/cuda_full_plan_5_-3.json"
        ).read_text()
    )
    assert resource["state_amplitudes"] == actual["standalone_plan"]["amplitudes"]
    assert resource["core_raw_amplitudes"] == actual["core_plan"]["raw_amplitudes"]
    assert resource["virtual_radials"] == 414
    assert resource["occupied_electrons"] == 68
    assert (
        resource["maximum_legacy_numeric64_host_payload_bytes"]
        == actual_numeric64["maximum_bound_bytes"]
    )
    assert resource["device_payload_bound_bytes"] == actual_device["device_bound_bytes"]
    actual_norm = json.loads((OWNER / "yb_monovalent_norm001/result.json").read_text())
    assert (
        resource["norm_scratch_required_bytes"]
        == actual_norm["rows"][0]["components"][0]["extra_scratch_bound_bytes"]
    )
    assert resource["maximum_complete_host_payload_bytes"] > 16629201972


def test_complete_panel_counts_expand_joint_states_without_default_pruning(
    measured, feature_identity
):
    rows = []
    for task in measured["tasks"]:
        band = json.loads(
            (
                OWNER / "parallel_tail007/run004" / (task["task_id"] + "_band.json")
            ).read_text()
        )
        resource = tail.estimate_task(space_from_band(band), feature_identity)
        assert set(resource["state_amplitudes"]) == {str(key) for key in tail.YB_STATES}
        assert resource["worker_threads"] == 1
        assert not resource["native_cuda_matches_worker_policy"]
        rows.append(resource)
    assert [row["virtual_radials"] for row in rows] == [414, 441, 532, 567]
    assert rows[-1]["core_raw_amplitudes"] == 50351931
    assert rows[-1]["state_amplitudes"]["(5, -3)"] == 7239518
    assert all(
        rows[0]["state_amplitudes"][key] < rows[-1]["state_amplitudes"][key]
        for key in rows[0]["state_amplitudes"]
    )


def test_exact_declared_energy_edge_and_complete_immutable_subspace():
    occupied = ((1, -1),)
    virtual = ((2, -1), (3, -1), (5, 4))
    source = AtomicVirtualSpace(
        {key: (np.array([1.0]), np.array([0.0])) for key in occupied + virtual},
        occupied,
        virtual,
        {
            "virtual_lmax": 4,
            "virtual_energy_max_Ha": 10.0,
            "virtual_orbital_energies_Ha": {
                "(2, -1)": -1.0,
                "(3, -1)": 5.0,
                "(5, 4)": 1.0,
            },
        },
    )
    subset = tail.retained_band(source, lmax=3, energy_max_Ha=5.0)
    assert subset.virtual_keys == ((2, -1),)
    assert subset.occupied_keys == occupied
    assert set(subset.orbitals) == {occupied[0], virtual[0]}
    with pytest.raises(ValueError):
        subset.orbitals[virtual[0]][0][0] = 0.0
    with pytest.raises(ValueError, match="contained"):
        tail.retained_band(source, lmax=5, energy_max_Ha=5)


def test_missing_complete_energy_identity_is_rejected():
    source = AtomicVirtualSpace(
        {(1, -1): (np.empty(0), np.empty(0)), (2, -1): (np.empty(0), np.empty(0))},
        ((1, -1),),
        ((2, -1),),
        {
            "virtual_lmax": 0,
            "virtual_energy_max_Ha": 5.0,
            "virtual_orbital_energies_Ha": {},
        },
    )
    with pytest.raises(ValueError, match="complete virtual"):
        tail.retained_band(source, lmax=0, energy_max_Ha=5)


def test_missing_joint_state_is_rejected_before_native_allocation(
    measured, feature_identity
):
    first = measured["tasks"][0]
    band = json.loads(
        (
            OWNER / "parallel_tail007/run004" / (first["task_id"] + "_band.json")
        ).read_text()
    )
    band["virtual_keys"].remove([5, -3])
    with pytest.raises(ValueError, match="distinct valence keys"):
        tail.estimate_task(space_from_band(band), feature_identity)


def test_memory_disk_device_failure_reports_without_reducing_model(measured):
    resource = measured["tasks"][-1]["existing_gpu8_resource"]
    before = json.dumps(resource, sort_keys=True)
    admission = tail.admit(
        resource,
        {"available_host_bytes": 1, "available_disk_bytes": 1, "gpu_free_bytes": 1},
        host_cap_bytes=1,
        device_cap_bytes=1,
    )
    assert {
        "complete_host_payload_exceeds_original_cap",
        "insufficient_live_host_memory",
        "insufficient_live_disk_reservation",
        "complete_device_payload_exceeds_original_cap",
        "insufficient_live_device_memory",
    } <= set(admission["reasons"])
    assert not admission["physical_model_reduced"]
    assert not admission["solve_authorized"]
    assert json.dumps(resource, sort_keys=True) == before


def test_memory_pass_cannot_waive_executor_angular_or_live_device_gate(measured):
    resource = measured["tasks"][0]["existing_gpu8_resource"]
    admission = tail.admit(
        resource,
        {
            "available_host_bytes": 64 * tail.GIB,
            "available_disk_bytes": 64 * tail.GIB,
            "gpu_free_bytes": None,
        },
    )
    assert set(admission["reasons"]) == {
        "live_device_free_memory_unverified",
        "source_bound_tail_executor_required",
        "native_angular_plan_count_required",
    }
    assert admission["status"] == "deferred"
    assert not admission["solve_authorized"]


def test_expanded_l4_norm_cap_fails_even_with_large_host_device_budget(
    measured, feature_identity
):
    task = measured["tasks"][-1]
    band = json.loads(
        (
            OWNER / "parallel_tail007/run004" / (task["task_id"] + "_band.json")
        ).read_text()
    )
    resource = tail.estimate_task(
        space_from_band(band), feature_identity, worker_threads=8
    )
    assert resource["norm_scratch_required_bytes"] > 64 * tail.MIB
    assert not resource["norm_scratch_within_original_cap"]
    admission = tail.admit(
        resource,
        {
            "available_host_bytes": 64 * tail.GIB,
            "available_disk_bytes": 64 * tail.GIB,
            "gpu_free_bytes": 8 * tail.GIB,
        },
    )
    assert "complete_norm_scratch_exceeds_original_cap" in admission["reasons"]
    assert resource["maximum_complete_host_payload_bytes"] < 32 * tail.GIB
    assert not admission["physical_model_reduced"]


def test_live_probe_receipt_schema_is_card_ready_and_does_not_claim_tail_roots(
    measured,
):
    assert measured["status"] == "pass"
    assert isinstance(measured["gates"], dict) and all(measured["gates"].values())
    assert isinstance(measured["card_summary"], str) and measured["card_summary"]
    assert "deferred" in measured["card_summary"]
    assert measured["highest_level_reached"] is False
    assert len(measured["tasks"]) == 4
    assert all(not task["correlation_execution_started"] for task in measured["tasks"])
