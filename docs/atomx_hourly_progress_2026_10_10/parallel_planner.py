"""Select independent, source-bound probes without inventing physical methods."""

import hashlib
import json
import re
from pathlib import Path

from iteration_planner import card_summary as primary_summary
from iteration_planner import plan as primary_plan

HERE = Path(__file__).resolve().parent
OWNER = "docs/atomic_pci_sd_native_2026_10_09/hourly_iterations"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bound_file(repo, relative):
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo.resolve()) or not path.is_file():
        raise ValueError("registered source/input must be a repository file")
    return path


def validate_registry(registry):
    caps = registry["caps"]
    count = caps["maximum_concurrent_jobs"]
    if (
        type(count) is not int
        or count not in (3, 6)
        or caps["maximum_threads"] != count
        or caps["maximum_reserved_bytes"] != 4 * count * 1024**3
        or caps["service_memory_bytes"] != (4 * count + 2) * 1024**3
        or caps["host_headroom_bytes"] != 8 * 1024**3
        or caps["maximum_load_per_cpu"] != 1.0
    ):
        raise ValueError(
            "parallel registry differs from the declared resource contract"
        )
    names = set()
    for task in registry["tasks"]:
        name = task["name"]
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
            raise ValueError("duplicate or invalid registered lane")
        names.add(name)
        if not task["enabled"]:
            continue
        if (
            task["threads"] != 1
            or not 0 < task["timeout_s"] <= 120
            or not 0 < task["reserved_bytes"] <= 4 * 1024**3
            or task["address_bytes"] != 4 * 1024**3
        ):
            raise ValueError("registered lane exceeds its bounded probe contract")
        if not task["entry_point"].startswith(
            "docs/atomic_pci_sd_native_2026_10_09/parallel_"
        ) or not task["entry_point"].endswith("/probe.py"):
            raise ValueError("only reviewed probe entry points can be registered")
        if not all(isinstance(a, str) for a in task["arguments"]):
            raise ValueError("probe arguments must be an explicit string list")
        if any(a == "--output" or a.startswith("--output=") for a in task["arguments"]):
            raise ValueError("each probe output is owned by the receipt worker")
        if task["entry_point"] not in task["source_files"]:
            raise ValueError("registered probe source must be bound")


def registered_job(repo, here, task):
    sources = {p: sha(bound_file(repo, p)) for p in sorted(set(task["source_files"]))}
    inputs = {p: sha(bound_file(repo, p)) for p in sorted(set(task["input_files"]))}
    payload = dict(
        kind="registered_probe",
        lane=task["name"],
        gap=task["gap"],
        entry_point=task["entry_point"],
        arguments=task["arguments"],
        scientific_contract={
            k: task[k]
            for k in (
                "hypothesis",
                "fixed",
                "distinguishing_outcomes",
                "scientific_gates",
            )
        },
        sources=sources,
        inputs=inputs,
        resources={
            k: task[k]
            for k in ("threads", "timeout_s", "reserved_bytes", "address_bytes")
        },
        runner_sha256=sha(here / "run_registered_step.py"),
    )
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return dict(id=f"parallel_{task['name']}_{identity}", **payload)


def validate_completed_probe(repo, record, job):
    result_relative = record["output"] + "/result.json"
    path = bound_file(repo, result_relative)
    data = json.loads(path.read_bytes())
    if (
        not record["output"].startswith(OWNER + "/")
        or sha(path) != record["result_sha256"]
        or data["job"] != job
        or not data["sources_and_inputs_unchanged"]
        or not data["execution_gate_pass"]
        or data["highest_level_reached"]
    ):
        raise ValueError("completed probe has invalid execution/result identity")
    result_path = bound_file(repo, record["output"] + "/probe/result.json")
    if sha(result_path) != data["probe_result_sha256"]:
        raise ValueError("completed scientific probe payload changed")
    if data["probe_outputs_sha256"].get("result.json") != data["probe_result_sha256"]:
        raise ValueError("completed probe lacks its full output identity manifest")
    for relative, value in data["probe_outputs_sha256"].items():
        path = bound_file(repo, record["output"] + "/probe/" + relative)
        if (
            not path.is_relative_to((repo / record["output"] / "probe").resolve())
            or sha(path) != value
        ):
            raise ValueError("completed scientific array/archive identity changed")
    # A successful bounded audit is not a passed production precision target.
    return dict(
        path=result_relative, sha256=sha(bound_file(repo, result_relative))
    ), data


def annotate_finite_repair(repo, here, decision):
    """Update the next direction without admitting a new solver or changing gates."""
    progress = here / "progress.json"
    if not progress.exists():
        return
    record = json.loads(progress.read_bytes())
    relative = record.get("fscc_equation_fix_result")
    if not relative:
        return
    path = bound_file(repo, relative)
    data = json.loads(path.read_bytes())
    if (
        not data["complete_candidate_gate_pass"]
        or not data["inputs_unchanged"]
        or data["backend_strict_heff_max_Ha"] > 1e-10
        or data["backend_wave_bloch_residual_max_Ha"] > 1e-11
        or sha(path.parent / "full_arrays.npz") != data["arrays_sha256"]
        or sha(path.parent / "source_inputs.tar.gz") != data["archive_sha256"]
    ):
        raise ValueError("finite repair evidence failed its original complete gates")
    decision["evidence"].append(dict(path=relative, sha256=sha(path)))
    direction = (
        "有限 0h1p 重复 Fov×T2 排列修复已通过；下一步在更大核／价空间及其他扇区"
        "验证候选，同时补齐生产约化／分块收缩。"
    )
    for item in decision.get("frontier", []):
        if item["gap"] == "noncanonical_fscc_operator":
            item.update(
                direction=direction,
                dependency="有限同 H 修复已验证；更大空间／其他扇区生产任务尚未准入。",
            )
    if decision["status"] == "implementation_frontier":
        decision.update(
            direction=direction,
            reason="原后端失败记录保留；有限候选已闭合严格门，当前缺口是生产范围扩展与独立验证。",
        )


def annotate_physics_priority(repo, here, decision):
    """Select a source-bound physical hypothesis without inventing an executor."""
    progress = here / "progress.json"
    if not progress.exists():
        return
    record = json.loads(progress.read_bytes())
    relative = record.get("physics_budget_result")
    if not relative:
        return
    path = bound_file(repo, relative)
    data = json.loads(path.read_bytes())
    if data["fitted_corrections"] or data["highest_level_reached"]:
        raise ValueError("physics estimates cannot certify fitted spectra")
    if record.get("yb_mcdhf_result"):
        current = data["inputs"].get("current_mcdhf")
        if current is None or current["path"] != record["yb_mcdhf_result"]:
            raise ValueError("physics priority uses a different current MCDHF result")
    else:
        current = data["inputs"]["current_five_states"]
        if current["path"] != record["yb_comparison"]:
            raise ValueError("physics priority uses a different five-state comparison")
    for binding in data["inputs"].values():
        if sha(bound_file(repo, binding["path"])) != binding["sha256"]:
            raise ValueError("physics priority input identity changed")
    decision["evidence"].append(dict(path=relative, sha256=sha(path)))
    direction = data["card_summary"]
    for item in decision.get("frontier", []):
        if item["gap"] == "yb_physical_space_and_experiment":
            item.update(direction=direction, dependency=data["next_execution_status"])
    if decision["status"] == "implementation_frontier":
        if decision["jobs"]:
            direction = "先完成已准入的源变化有限复验；生产推进方向：\n" + direction
        decision.update(direction=direction, reason=data["selection_reason"])


def plan_batch(repo, state, here=HERE):
    primary = primary_plan(repo, state, here=here)
    registry_path = here / "parallel_registry.json"
    if not registry_path.exists():
        return primary
    registry = json.loads(registry_path.read_bytes())
    validate_registry(registry)
    caps = registry["caps"]
    jobs = [primary["job"]] if primary["job"] else []
    reserved = len(jobs) * 4 * 1024**3
    lanes, citations = [], []
    records = {
        p.stem: json.loads(p.read_bytes())
        for p in (state / "iteration_ledger").glob("*.json")
    }
    for task in registry["tasks"]:
        lane = dict(
            name=task["name"],
            title=task["title"],
            gap=task["gap"],
            next_direction=task.get("next_direction"),
        )
        if not task["enabled"]:
            lanes.append(
                dict(lane, status="implementation_frontier", reason=task["reason"])
            )
            continue
        job = registered_job(repo, here, task)
        prior = records.get(job["id"])
        if prior is not None:
            if prior["status"] == "completed":
                citation, data = validate_completed_probe(repo, prior, job)
                citations.append(citation)
                lanes.append(
                    dict(lane, status="completed", job_id=job["id"], result=data)
                )
            else:
                lanes.append(
                    dict(
                        lane,
                        status=prior["status"],
                        job_id=job["id"],
                        reason="已领取任务保留原状态，失败或中断不静默重跑。",
                    )
                )
            continue
        if (
            len(jobs) >= caps["maximum_concurrent_jobs"]
            or reserved + task["reserved_bytes"] > caps["maximum_reserved_bytes"]
        ):
            lanes.append(
                dict(
                    lane,
                    status="deferred_batch_budget",
                    reason="下一张卡重新准入，物理空间不缩小。",
                )
            )
            continue
        reserved += task["reserved_bytes"]
        jobs.append(job)
        lanes.append(dict(lane, status="selected", job_id=job["id"]))
    result = dict(primary)
    result.update(
        jobs=jobs,
        parallel_lanes=lanes,
        parallel_caps=caps,
        parallel_registry_sha256=sha(registry_path),
        parallel_planner_sha256=sha(Path(__file__)),
        evidence=primary["evidence"]
        + citations
        + [dict(path=str(registry_path.relative_to(repo)), sha256=sha(registry_path))],
    )
    annotate_finite_repair(repo, here, result)
    annotate_physics_priority(repo, here, result)
    return result


def card_summary(decision):
    text = primary_summary(decision)
    if "parallel_lanes" not in decision:
        return text
    lines = [text, "", "**独立方向并行推进**"]
    for lane in decision["parallel_lanes"]:
        summary = f"• {lane['title']}：{lane['status']}"
        if lane.get("result"):
            summary += "\n" + lane["result"]["card_summary"]
        if lane.get("reason"):
            summary += " · " + lane["reason"]
        if lane.get("next_direction"):
            summary += "\n下一步：" + lane["next_direction"]
        lines.append(summary)
    count = len(decision["jobs"])
    caps = decision["parallel_caps"]
    lines.append(
        f"成功发卡后重新核验，准入 {count} 项；最多 {caps['maximum_concurrent_jobs']} 项"
        f"同时运行、各 1 CPU 线程，总服务内存上限 {caps['service_memory_bytes'] // 1024**3} GiB。"
        "审计和资源预检完成不代表 SD／CCSDT 精度缺口已闭合。"
    )
    return "\n".join(lines)
