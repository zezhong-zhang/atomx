"""Independent adaptive-selection and complete signed-action controls."""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "docs/atomx_hourly_progress_2026_10_10"
sys.path.insert(0, str(HERE))
import iteration_planner as planner  # noqa: E402

spec = importlib.util.spec_from_file_location("gap_action", HERE / "run_gap_step.py")
action = importlib.util.module_from_spec(spec)
spec.loader.exec_module(action)
POLICY = json.loads((HERE / "iteration_policy.json").read_text())


def case(ns, error=8.75e-10, mode="dynamic", source="same", Z=4):
    return dict(
        job=dict(kind="finite_transverse_bloch", ns=ns, mode=mode),
        output=f"docs/atomic_pci_sd_native_2026_10_09/hourly_iterations/ns{ns}_{mode}",
        result=dict(backend_heff_max_Ha=error),
        result_sha256=f"result{ns}{mode}",
        arrays_sha256=f"arrays{ns}{mode}",
        model_identity=dict(Z=Z, nuclear="fixed", backend_sha256="same"),
        meta=dict(
            labels=[[1, -1, -1], [1, -1, 1]],
            noccupied=2,
            abi="ATXCC001",
            identity=dict(physical_source_sha256=dict(operator=source)),
        ),
    )


def choose(cases, records=None):
    return planner.decide(cases, records or {}, POLICY, "runner", 0.04)


def test_platform_switches_away_from_low_information_grid_scan():
    decision = choose([case(42), case(60, 8.7498e-10), case(80)])
    assert decision["job"]["kind"] == "bloch_partition"
    assert decision["gap"]["ns"] == [42, 60, 80]
    assert decision["gap"]["strict_gate_Ha"] == 1e-10
    assert decision["deferred"] and not decision["highest_level_reached"]
    assert decision["frontier"][1]["observed_max_relative_error"] == 0.04


@pytest.mark.parametrize(
    "cases",
    [
        [case(42), case(60), case(80, source="different")],
        [case(42), case(60), case(80, Z=5)],
        [case(42), case(42), case(42)],
        [case(42), case(60, 3e-10), case(80, 1.5e-10)],
        [case(42, 1e-14), case(60, 1e-14), case(80, 1e-14)],
    ],
)
def test_nonmatching_or_improving_or_passed_results_do_not_make_a_plateau(cases):
    assert choose(cases)["status"] == "no_plateau"


def test_results_select_controls_then_explicit_frontier_without_repeats():
    cases = [case(n) for n in (42, 60, 80)] + [case(80, 1e-14, "DC")]
    records = {}
    jobs = []
    for _ in range(3):
        d = choose(cases, records)
        jobs.append(d["job"])
        records[d["job"]["id"]] = dict(status="completed")
    assert [j["kind"] for j in jobs] == [
        "bloch_partition",
        "whole_transverse_control",
        "whole_transverse_control",
    ]
    assert [j["scale"] for j in jobs[1:]] == [0.5, 1.5]
    assert len({j["id"] for j in jobs}) == 3
    final = choose(cases, records)
    assert final["status"] == "implementation_frontier" and final["job"] is None
    assert all(not x["runnable"] for x in final["frontier"])


def test_failed_diagnostic_is_not_silently_replayed_or_bypassed():
    cases = [case(n) for n in (42, 60, 80)]
    job = choose(cases)["job"]
    d = choose(cases, {job["id"]: dict(status="failed")})
    assert d["job"] is None and d["status"] == "implementation_frontier"


@pytest.mark.parametrize(
    "ratios,word", [([0.5031, 1.4907], "线性依赖"), ([0.2, 3.0], "非线性耦合")]
)
def test_actual_response_changes_next_equation_audit_direction(ratios, word):
    cases = [case(n) for n in (42, 60, 80)] + [case(80, 1e-14, "DC")]
    records = {}
    job = choose(cases)["job"]
    records[job["id"]] = dict(status="completed")
    for ratio in ratios:
        job = choose(cases, records)["job"]
        records[job["id"]] = dict(
            status="completed",
            result=dict(
                backend_heff_max_Ha=8.75e-10 * ratio, defect_response_ratio=ratio
            ),
        )
    result = choose(cases, records)
    assert result["status"] == "implementation_frontier" and word in result["direction"]
    assert result["gap"]["whole_transverse_response"][0]["ratio"] == ratios[0]


def test_ledger_binding_rejects_modified_reference_arrays(tmp_path):
    repo, state = tmp_path / "repo", tmp_path / "state"
    directory = repo / planner.OWNER / "one"
    (directory / "fscc").mkdir(parents=True)
    (state / "iteration_ledger").mkdir(parents=True)
    job = dict(kind="finite_transverse_bloch", ns=42, mode="dynamic")
    array_raw = b"immutable-array-fixture"
    result = dict(
        job=job,
        source_and_inputs_unchanged=True,
        oracle_gate_pass=True,
        arrays_sha256=planner.digest(array_raw),
    )
    raw = json.dumps(result).encode()
    (directory / "result.json").write_bytes(raw)
    (directory / "full_arrays.npz").write_bytes(array_raw)
    (directory / "fscc/integrals.json").write_text(
        json.dumps(dict(identity=dict(binding="Z=4, ns=42, nuclear='fixed'")))
    )
    (directory / "contract.json").write_text(json.dumps(dict(backend_sha256="same")))
    record = dict(
        status="completed",
        job=job,
        output=str(directory.relative_to(repo)),
        result_sha256=planner.digest(raw),
    )
    (state / "iteration_ledger/one.json").write_text(json.dumps(record))
    assert len(planner.collect_cases(repo, state)[0]) == 1
    (directory / "full_arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="array hash mismatch"):
        planner.collect_cases(repo, state)


def test_complete_complex_partition_matches_direct_unsplit_operator():
    rng = np.random.default_rng(93011)
    masks = [sum(1 << i for i in c) for c in itertools.combinations(range(10), 3)]
    p = [i for i, mask in enumerate(masks) if mask & 3 == 3]
    hbar = rng.normal(size=(120, 120)) + 1j * rng.normal(size=(120, 120))
    omega = rng.normal(size=(120, 8)) + 1j * rng.normal(size=(120, 8))
    omega[p, :] = np.eye(8)
    h2 = np.diag(np.linspace(-9, -1, 45)).astype(complex)
    exact = (hbar @ omega)[p, :] + 9 * np.eye(8)
    defect = np.diag([8.75e-10] * 2 + [0] * 6)
    arrays = dict(
        hbar3_Ha=hbar,
        omega=omega,
        full_h2_Ha=h2,
        exact_heff_Ha=exact,
        model_determinant_indices=np.asarray(p),
        backend_minus_exact_heff_Ha=defect,
    )
    metrics, payload = action.partition(arrays)
    np.testing.assert_allclose(
        payload["model_block_Ha"]
        + payload["one_core_hole_Ha"]
        + payload["two_core_hole_Ha"],
        exact,
        atol=2e-13,
        rtol=0,
    )
    assert (
        metrics["partition_gate_pass"] and not metrics["backend_strict_1e10_gate_pass"]
    )
    assert [s["determinant_rows"] for s in metrics["sector_contributions"]] == [56, 56]
    assert np.max(abs(payload["two_core_hole_Ha"].imag)) > 1
    arrays["model_determinant_indices"] = np.asarray(p[::-1])
    with pytest.raises(ValueError, match="determinant order"):
        action.partition(arrays)


def test_full_operator_scaling_matches_signed_hamiltonian_and_fock_ledger(native):
    from test_spinor_integrals_fscc import integrals
    from test_transverse_spinor_integrals import determinant_matrix, energies

    dc = integrals(native)
    transverse = integrals(
        native, include_breit=True, breit_orbital_energies=energies(native)
    )
    scaled = action.scaled_interaction(dc, transverse, 0.5)
    _, hdc = determinant_matrix(dc, 3)
    _, ht = determinant_matrix(transverse, 3)
    _, hs = determinant_matrix(scaled, 3)
    np.testing.assert_allclose(hs, hdc + 0.5 * (ht - hdc), atol=2e-14, rtol=0)
    assert scaled.identity["diagnostic_changed_H"]
    assert np.max(abs(scaled.fock_Ha[:2, 2:])) > 0
    with pytest.raises(ValueError, match="declared"):
        action.scaled_interaction(dc, transverse, 0)


from test_spinor_integrals_fscc import native as native_fixture  # noqa: E402

native = native_fixture
