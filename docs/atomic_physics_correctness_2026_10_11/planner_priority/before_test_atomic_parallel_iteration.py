"""Batch admission, true overlap, sibling durability and whole-lane timeout."""

import importlib.util
import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1] / "docs/atomx_hourly_progress_2026_10_10"
sys.path.insert(0, str(HERE))
import parallel_planner as planner  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "parallel_worker", HERE / "iteration_worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_card_rejects_failed_bounded_norm_evidence(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import sync_card

    monkeypatch.setattr(sync_card, "REPO", tmp_path)
    (tmp_path / "comparison.json").write_text(
        json.dumps(dict(rows=[], splittings=[], computed_run="fixture"))
    )
    result = dict(
        status="pass", gates=dict(original_cap=True), card_summary="complete budget"
    )
    path = tmp_path / "norm.json"
    path.write_text(json.dumps(result))
    progress = dict(
        yb_comparison="comparison.json",
        norm_memory_result="norm.json",
        tasks=[],
        spectra=[],
        updated_at="fixture",
    )
    card, citations = sync_card.render(progress, datetime.now(timezone.utc))
    assert any(citation["path"] == "norm.json" for citation in citations)
    assert any(
        "complete budget" in element["content"]
        for element in card["body"]["elements"]
    )
    result["gates"]["original_cap"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="bounded norm evidence"):
        sync_card.render(progress, datetime.now(timezone.utc))


def registry_task(name):
    entry = f"docs/atomic_pci_sd_native_2026_10_09/parallel_{name}/probe.py"
    return dict(
        name=name,
        title=name,
        gap=name,
        enabled=True,
        entry_point=entry,
        arguments=[],
        source_files=[entry],
        input_files=["input.json"],
        threads=1,
        timeout_s=120,
        reserved_bytes=4 * 1024**3,
        address_bytes=4 * 1024**3,
        hypothesis="fixed hypothesis",
        fixed="same space",
        distinguishing_outcomes="retained negative result",
        scientific_gates=dict(target_Ha=1e-10),
    )


@pytest.fixture
def batch_harness(tmp_path, monkeypatch):
    state, repo, code = (tmp_path / name for name in ("state", "repo", "code"))
    for p in (state / "iteration_requests", state / "iteration_ledger", code, repo):
        p.mkdir(parents=True)
    monkeypatch.setattr(worker, "HERE", code)
    monkeypatch.setattr(worker, "REPO", repo)
    (code / "iteration_queue.json").write_text('{"jobs": []}')
    (code / "iteration_policy.json").write_text("{}")
    caps = json.loads((HERE / "iteration_policy.json").read_text())["caps"]
    parallel = json.loads((HERE / "parallel_registry.json").read_text())["caps"]
    selection = dict(
        status="selected",
        reason="independent",
        job=None,
        jobs=[dict(id="lane_a"), dict(id="lane_b")],
        caps=caps,
        parallel_caps=parallel,
    )
    monkeypatch.setattr(worker, "plan", lambda *a, **kw: selection)
    monkeypatch.setattr(worker, "resource_preflight", lambda *a: dict(passed=True))
    path = state / "iteration_requests/receipt.json"
    worker.save(
        path,
        dict(key="receipt", cycle="one", message_id="actual-receipt", status="queued"),
    )
    path.with_suffix(".ready").touch()
    return state, repo, selection, path, dict(state_directory=str(state))


def test_two_lanes_overlap_and_one_failure_does_not_lose_sibling(
    batch_harness, monkeypatch
):
    state, repo, selection, request, config = batch_harness
    barrier = threading.Barrier(2)
    calls = []

    def run(command, **kwargs):
        identifier = Path(command[-1]).name
        calls.append((identifier, kwargs))
        # Sequential scheduling cannot satisfy this barrier; both independent
        # subprocess invocations must be active before either completes.
        barrier.wait(timeout=3)
        if identifier == "lane_a":
            raise subprocess.TimeoutExpired(command, 120)
        output = Path(command[-1])
        output.mkdir(parents=True)
        (output / "result.json").write_text('{"test_fixture": true}')
        return 0

    monkeypatch.setattr(worker, "run_process", run)
    worker.drain(config)
    ledger = {
        p.stem: json.loads(p.read_bytes())
        for p in (state / "iteration_ledger").glob("*.json")
    }
    assert {key: value["status"] for key, value in ledger.items()} == {
        "lane_a": "failed",
        "lane_b": "completed",
    }
    assert json.loads(request.read_text())["status"] == "partially_completed"
    assert all(kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1" for _, kwargs in calls)
    worker.drain(config)
    assert len(calls) == 2


def test_crash_marks_only_unfinished_claims_interrupted(batch_harness):
    state, repo, selection, request, config = batch_harness
    worker.save(
        state / "iteration_ledger/lane_a.json",
        dict(job_id="lane_a", status="completed"),
    )
    worker.save(
        state / "iteration_ledger/lane_b.json", dict(job_id="lane_b", status="running")
    )
    data = json.loads(request.read_bytes())
    data.update(status="running", job_ids=["lane_a", "lane_b"])
    worker.save(request, data)
    worker.drain(config)
    assert (
        json.loads((state / "iteration_ledger/lane_a.json").read_text())["status"]
        == "completed"
    )
    assert (
        json.loads((state / "iteration_ledger/lane_b.json").read_text())["status"]
        == "interrupted"
    )
    assert not request.with_suffix(".ready").exists()


def test_whole_batch_resources_are_checked_before_any_claim(batch_harness, monkeypatch):
    state, repo, selection, request, config = batch_harness
    observed = []

    def preflight(state, caps):
        observed.append(caps)
        return dict(passed=False)

    monkeypatch.setattr(worker, "resource_preflight", preflight)
    worker.drain(config)
    assert observed[0]["reserved_bytes"] == 8 * 1024**3
    assert observed[0]["total_threads"] == 2
    assert not list((state / "iteration_ledger").glob("*.json"))
    assert json.loads(request.read_text())["status"] == "deferred_resources"


@pytest.mark.parametrize("width", [3, 6])
def test_registry_jobs_bind_physics_sources_inputs_and_original_gates(
    tmp_path, monkeypatch, width
):
    repo, state, here = (
        tmp_path / "repo",
        tmp_path / "state",
        tmp_path / "repo/controller",
    )
    here.mkdir(parents=True)
    (state / "iteration_ledger").mkdir(parents=True)
    registry = json.loads((HERE / "parallel_registry.json").read_bytes())
    registry["caps"].update(
        maximum_concurrent_jobs=width,
        maximum_threads=width,
        maximum_reserved_bytes=4 * width * 1024**3,
        service_memory_bytes=(4 * width + 2) * 1024**3,
    )
    registry["tasks"] = [
        registry_task(f"lane_{i}") for i in range(width + 1)
    ]
    for task in registry["tasks"]:
        path = repo / task["entry_point"]
        path.parent.mkdir(parents=True)
        path.write_text("# exact reviewed fixture source\n")
    (repo / "input.json").write_text("{}")
    (here / "run_registered_step.py").write_text("# registered runner\n")
    (here / "parallel_registry.json").write_text(json.dumps(registry))
    primary = dict(job=None, status="implementation_frontier", evidence=[])
    monkeypatch.setattr(planner, "primary_plan", lambda *a, **kw: primary)
    decision = planner.plan_batch(repo, state, here)
    assert len(decision["jobs"]) == width
    assert decision["parallel_lanes"][-1]["status"] == "deferred_batch_budget"
    before = decision["jobs"][0]
    assert before["scientific_contract"]["scientific_gates"]["target_Ha"] == 1e-10
    (repo / "input.json").write_text('{"changed_H": true}')
    after = planner.plan_batch(repo, state, here)["jobs"][0]
    assert after["id"] != before["id"]
    registry["tasks"][0]["threads"] = 2
    with pytest.raises(ValueError, match="probe contract"):
        planner.validate_registry(registry)


def test_six_actual_child_processes_overlap(batch_harness):
    state, repo, selection, request, config = batch_harness
    selection["jobs"] = [dict(id=f"lane_{i}") for i in range(6)]
    selection["parallel_caps"].update(
        maximum_concurrent_jobs=6,
        maximum_threads=6,
        maximum_reserved_bytes=24 * 1024**3,
        service_memory_bytes=26 * 1024**3,
    )
    root = worker.HERE
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").symlink_to(sys.executable)
    # Each real subprocess waits for all six siblings. Sequential dispatch or
    # a hidden pool smaller than six cannot complete this unchanged barrier.
    (root / "run_iteration_step.py").write_text(
        "import argparse,json,time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--output');a=p.parse_args()\n"
        "out=Path(a.output);out.mkdir(parents=True);start=out.parent/'started';start.mkdir(exist_ok=True)\n"
        "(start/out.name).touch();deadline=time.monotonic()+4\n"
        "while len(list(start.iterdir()))<6:\n"
        " if time.monotonic()>deadline:raise RuntimeError('six-process barrier failed')\n"
        " time.sleep(.01)\n"
        "time.sleep(.1);(out/'result.json').write_text(json.dumps({'test_fixture':True}))\n"
    )
    worker.drain(config)
    records = [
        json.loads(p.read_text()) for p in (state / "iteration_ledger").glob("*.json")
    ]
    assert len(records) == 6
    assert all(r["status"] == "completed" for r in records)
    assert len({r["process_pid"] for r in records}) == 6
    latest_start = max(datetime.fromisoformat(r["process_spawned_at"]) for r in records)
    earliest_finish = min(
        datetime.fromisoformat(r["process_finished_at"]) for r in records
    )
    assert (earliest_finish - latest_start).total_seconds() > 0
    assert json.loads(request.read_text())["status"] == "completed"


@pytest.mark.parametrize("width", [True, 1, 4, 7])
def test_undeclared_concurrency_caps_are_rejected(width):
    registry = json.loads((HERE / "parallel_registry.json").read_bytes())
    registry["caps"]["maximum_concurrent_jobs"] = width
    with pytest.raises(ValueError, match="resource contract"):
        planner.validate_registry(registry)


def test_six_lane_admission_includes_aggregate_memory_and_live_load(
    tmp_path, monkeypatch
):
    gib = 1024**3
    live = dict(available=31 * gib, load=4.0)
    monkeypatch.setattr(
        worker.os,
        "sysconf",
        lambda name: 4096 if name == "SC_PAGE_SIZE" else live["available"] // 4096,
    )
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: set(range(32)))
    monkeypatch.setattr(worker.os, "getloadavg", lambda: (live["load"],) * 3)
    monkeypatch.setattr(
        worker.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024 * gib)
    )
    parallel = json.loads((HERE / "parallel_registry.json").read_text())["caps"]
    parallel.update(
        maximum_concurrent_jobs=6,
        maximum_threads=6,
        maximum_reserved_bytes=24 * gib,
        service_memory_bytes=26 * gib,
    )
    caps = dict(
        threads=1,
        minimum_available_bytes=1,
        minimum_free_disk_bytes=1,
        parallel=parallel,
        reserved_bytes=24 * gib,
        total_threads=6,
    )
    assert not worker.resource_preflight(tmp_path, caps)["passed"]
    live["available"] = 32 * gib
    assert worker.resource_preflight(tmp_path, caps)["passed"]
    live["load"] = 27.0
    assert not worker.resource_preflight(tmp_path, caps)["passed"]


def test_completed_probe_payload_corruption_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    output = planner.OWNER + "/saved"
    directory = repo / output
    (directory / "probe").mkdir(parents=True)
    job = dict(id="saved", kind="registered_probe")
    probe = directory / "probe/result.json"
    probe.write_text('{"gates": {"production": false}}')
    result = dict(
        job=job,
        sources_and_inputs_unchanged=True,
        execution_gate_pass=True,
        highest_level_reached=False,
        probe_result_sha256=planner.sha(probe),
        probe_outputs_sha256={"result.json": planner.sha(probe)},
    )
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    record = dict(output=output, result_sha256=planner.sha(path))
    citation, _ = planner.validate_completed_probe(repo, record, job)
    assert citation == dict(path=output + "/result.json", sha256=planner.sha(path))
    probe.write_text('{"gates": {"production": true}}')
    with pytest.raises(ValueError, match="payload changed"):
        planner.validate_completed_probe(repo, record, job)


def test_timeout_terminates_descendant_backend(tmp_path):
    pid_path = tmp_path / "backend.pid"
    parent = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    with (tmp_path / "log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        worker.run_process(
            [sys.executable, "-c", parent], cwd=tmp_path, env=None, log=log, timeout=0.5
        )
    pid = int(pid_path.read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] == "Z"


def test_finite_repair_changes_direction_without_admitting_production(tmp_path):
    repo = tmp_path
    here = repo / "controller"
    case = repo / "candidate"
    here.mkdir()
    case.mkdir()
    (case / "full_arrays.npz").write_bytes(b"complete signed fixture")
    (case / "source_inputs.tar.gz").write_bytes(b"exact source fixture")
    data = dict(
        complete_candidate_gate_pass=True,
        inputs_unchanged=True,
        backend_strict_heff_max_Ha=4e-15,
        backend_wave_bloch_residual_max_Ha=2e-15,
        arrays_sha256=planner.sha(case / "full_arrays.npz"),
        archive_sha256=planner.sha(case / "source_inputs.tar.gz"),
    )
    (case / "result.json").write_text(json.dumps(data))
    (here / "progress.json").write_text(
        json.dumps(dict(fscc_equation_fix_result="candidate/result.json"))
    )
    decision = dict(
        status="implementation_frontier",
        job=None,
        jobs=[],
        evidence=[],
        highest_level_reached=False,
        frontier=[dict(gap="noncanonical_fscc_operator", runnable=False)],
    )
    planner.annotate_finite_repair(repo, here, decision)
    assert "Fov×T2" in decision["direction"]
    assert decision["job"] is None and decision["jobs"] == []
    assert not decision["highest_level_reached"]
    assert not decision["frontier"][0]["runnable"]
    (case / "full_arrays.npz").write_bytes(b"corrupt signed payload")
    with pytest.raises(ValueError, match="original complete gates"):
        planner.annotate_finite_repair(repo, here, decision)
