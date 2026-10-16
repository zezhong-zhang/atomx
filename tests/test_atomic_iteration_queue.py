"""Receipt queue durability: failures, duplicates, crashes and exhaustion."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "docs/atomx_hourly_progress_2026_10_10"
spec = importlib.util.spec_from_file_location(
    "atomic_worker", HERE / "iteration_worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    state, repo, code = (tmp_path / name for name in ("state", "repo", "code"))
    for p in (state / "iteration_requests", code, repo):
        p.mkdir(parents=True)
    (code / "iteration_queue.json").write_text(
        json.dumps(dict(jobs=[dict(id="first"), dict(id="second")]))
    )
    monkeypatch.setattr(worker, "HERE", code)
    monkeypatch.setattr(worker, "REPO", repo)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[-1])
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(dict(test_fixture=True)))
        return 0

    monkeypatch.setattr(worker, "run_process", run)

    def request(key):
        p = state / "iteration_requests" / (key + ".json")
        worker.save(
            p, dict(key=key, cycle=key, message_id="receipt-" + key, status="queued")
        )
        p.with_suffix(".ready").touch()
        return p

    return state, calls, request, dict(state_directory=str(state))


def test_duplicate_receipt_never_repeats_scientific_job(harness):
    state, calls, request, config = harness
    p = request("one")
    worker.drain(config)
    worker.drain(config)
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 600
    assert calls[0][1]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert json.loads(p.read_text())["status"] == "completed"
    assert not p.with_suffix(".ready").exists()
    assert (
        json.loads((state / "iteration_status.json").read_text())["message_id"]
        == "receipt-one"
    )


def test_failure_is_retained_next_card_advances_not_retries(harness, monkeypatch):
    state, calls, request, config = harness

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 600)

    monkeypatch.setattr(worker, "run_process", fail)
    p = request("one")
    worker.drain(config)
    assert json.loads(p.read_text())["status"] == "failed"
    request("two")
    worker.drain(config)
    assert set(p.stem for p in (state / "iteration_ledger").glob("*.json")) == {
        "first",
        "second",
    }
    request("three")
    worker.drain(config)
    assert (
        json.loads((state / "iteration_status.json").read_text())["status"]
        == "waiting_for_queue_expansion"
    )


def test_interrupted_claim_has_visible_ledger_and_no_silent_replay(harness):
    state, calls, request, config = harness
    p = request("one")
    worker.save(
        p,
        dict(
            key="one",
            cycle="one",
            message_id="receipt-one",
            status="running",
            job_id="first",
        ),
    )
    worker.save(
        state / "iteration_ledger/first.json", dict(job_id="first", status="running")
    )
    worker.drain(config)
    assert not calls
    assert json.loads(p.read_text())["status"] == "interrupted"
    assert (
        json.loads((state / "iteration_ledger/first.json").read_text())["status"]
        == "interrupted"
    )
    assert not p.with_suffix(".ready").exists()


def test_adaptive_selection_replaces_queue_and_is_saved_with_real_receipt(
    harness, monkeypatch
):
    state, calls, request, config = harness
    (worker.HERE / "iteration_policy.json").write_text("{}")
    selection = dict(
        status="selected",
        job=dict(id="adaptive_diagnostic", kind="bloch_partition"),
        reason="stable operator gap",
        caps=dict(threads=1),
    )
    monkeypatch.setattr(worker, "plan", lambda *args, **kwargs: selection)
    monkeypatch.setattr(worker, "resource_preflight", lambda *args: dict(passed=True))
    request("adaptive")
    worker.drain(config)
    assert len(calls) == 1 and calls[0][0][1].endswith("run_gap_step.py")
    actual = json.loads((state / "iteration_status.json").read_text())["decision"]
    assert actual["job"] == selection["job"]
    assert Path(actual["execution_sources"]["archive"]).exists()
    assert (state / "iteration_decisions/adaptive.json").exists()


def test_resource_failure_keeps_direction_visible_without_claiming_work(
    harness, monkeypatch
):
    state, calls, request, config = harness
    (worker.HERE / "iteration_policy.json").write_text("{}")
    selection = dict(
        status="selected",
        job=dict(id="adaptive_diagnostic", kind="bloch_partition"),
        reason="stable operator gap",
        caps=dict(threads=1),
    )
    monkeypatch.setattr(worker, "plan", lambda *args, **kwargs: selection)
    monkeypatch.setattr(worker, "resource_preflight", lambda *args: dict(passed=False))
    p = request("low_memory")
    worker.drain(config)
    assert not calls and not list((state / "iteration_ledger").glob("*.json"))
    assert json.loads(p.read_text())["status"] == "deferred_resources"
    assert not p.with_suffix(".ready").exists()
