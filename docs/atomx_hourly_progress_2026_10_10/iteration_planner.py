"""Evidence-guided scientific selection, independent of Lark and execution.

Decisions are deterministic and replayable. Capabilities are explicit: this
selector can launch registered diagnostics, not invent an unimplemented method.
"""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OWNER = "docs/atomic_pci_sd_native_2026_10_09/hourly_iterations"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(repo, relative):
    path = (repo / relative).resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError("scientific evidence must remain in the repository")
    raw = path.read_bytes()
    return json.loads(raw), dict(path=relative, sha256=digest(raw))


def collect_cases(repo, state):
    """Validate completed receipts against actual saved result and array bytes."""
    cases, citations = [], []
    for path in sorted((state / "iteration_ledger").glob("*.json")):
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            continue
        job = record.get("job", {})
        if job.get("kind") != "finite_transverse_bloch":
            continue
        directory = record["output"]
        if not directory.startswith(OWNER + "/"):
            raise ValueError("unexpected finite result owner")
        result, citation = read_bound(repo, directory + "/result.json")
        if citation["sha256"] != record["result_sha256"]:
            raise ValueError("completed result differs from its receipt ledger")
        if not result["source_and_inputs_unchanged"] or not result["oracle_gate_pass"]:
            raise ValueError("unverified finite result cannot guide a new calculation")
        if result["job"] != job:
            raise ValueError("result and receipt job identity disagree")
        array_path = (repo / directory / "full_arrays.npz").resolve()
        if not array_path.is_relative_to(repo.resolve()):
            raise ValueError("array evidence leaves repository")
        array_hash = digest(array_path.read_bytes())
        if array_hash != result["arrays_sha256"]:
            raise ValueError("scientific array hash mismatch")
        meta, meta_citation = read_bound(repo, directory + "/fscc/integrals.json")
        contract, contract_citation = read_bound(repo, directory + "/contract.json")
        # The current exporter archives this binding as a representation string.
        # Require its explicit nuclear/Z fields instead of grouping only labels.
        physical = re.search(
            r"Z=(\d+), ns=(\d+), nuclear='([^']+)'", meta["identity"]["binding"]
        )
        if not physical or int(physical[2]) != job["ns"]:
            raise ValueError("finite binding lacks the declared Z/ns/nuclear identity")
        model_identity = dict(
            Z=int(physical[1]),
            nuclear=physical[3],
            backend_sha256=contract["backend_sha256"],
        )
        cases.append(
            dict(
                job=job,
                output=directory,
                result=result,
                result_sha256=citation["sha256"],
                arrays_sha256=array_hash,
                meta=meta,
                model_identity=model_identity,
            )
        )
        citations.extend(
            [
                citation,
                meta_citation,
                contract_citation,
                dict(
                    path=str(array_path.relative_to(repo.resolve())), sha256=array_hash
                ),
            ]
        )
    return cases, citations


def frontier(experimental):
    return [
        dict(
            gap="noncanonical_fscc_operator",
            direction="推导并实现完整非 canonical 0h1p 图项；逐项对照独立 Bloch 算符后验证候选后端。",
            runnable=False,
            dependency="完整图项候选实现尚未注册；诊断不等于已修复方程。",
        ),
        dict(
            gap="yb_physical_space_and_experiment",
            direction="五态共同空间 lmax=3/4 × Emax=5/10 Ha 的尾部收敛，随后核响应与 dressed E1/E2/M1。",
            runnable=False,
            observed_max_relative_error=experimental,
            dependency="需先完成完整空间的资源预估和有结果契约的生产任务接入。",
        ),
        dict(
            gap="native_sd_transverse",
            direction="复用 signed 横向积分与约化角代数，补齐原生 SD 横向图项；不把冻结核能移拼入旧 SD。",
            runnable=False,
            dependency="现有原生 J 约化 SD 消费器仍只覆盖 DC。",
        ),
        dict(
            gap="performance_certification",
            direction="保持同 H／空间／门限，测主耗时与数据搬运，做冷热 CPU/GPU 成对端到端计时。",
            runnable=False,
            dependency="当前有限诊断不支持大规模 GPU 性能认证；不得由小矩阵计时外推。",
        ),
    ]


def job_identity(job, policy, runner_hash):
    raw = json.dumps(
        dict(job=job, policy=policy, runner_sha256=runner_hash),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "adaptive_" + job["kind"] + "_" + digest(raw)[:20]


def decide(cases, records, policy, runner_hash, experimental=None):
    """Pure rules: scientific blockers precede cheapness; plateau stops scans."""
    strict = policy["backend_strict_Ha"]
    groups = {}
    for case in cases:
        mode = case["job"]["mode"]
        meta = case["meta"]
        group = json.dumps(
            dict(
                mode=mode,
                labels=meta["labels"],
                occupied=meta["noccupied"],
                abi=meta["abi"],
                source=meta["identity"]["physical_source_sha256"],
                frequency=meta["identity"].get("frequency_prescription"),
                model_identity=case["model_identity"],
            ),
            sort_keys=True,
        )
        if case["result"]["backend_heff_max_Ha"] > strict:
            groups.setdefault(group, {})[case["job"]["ns"]] = case
    plateaus = []
    for group in groups.values():
        errors = [x["result"]["backend_heff_max_Ha"] for x in group.values()]
        if len(group) >= policy["plateau_min_cases"]:
            spread = (max(errors) - min(errors)) / max(errors)
            if spread <= policy["plateau_relative_spread"]:
                plateaus.append((group, spread))
    result = dict(
        version=policy["version"],
        highest_level_reached=False,
        requirements=policy["invariants"],
        frontier=frontier(experimental),
        job=None,
        deferred=[],
    )
    # Prefer the complete dynamic model when both static and dynamic plateau.
    plateaus.sort(
        key=lambda x: (x[0][max(x[0])]["job"]["mode"] != "dynamic", -max(x[0]))
    )
    for group, spread in plateaus:
        case = group[max(group)]
        result["gap"] = dict(
            kind="stable_operator_defect",
            mode=case["job"]["mode"],
            ns=sorted(group),
            maximum_Ha=max(x["result"]["backend_heff_max_Ha"] for x in group.values()),
            relative_spread=spread,
            strict_gate_Ha=strict,
        )
        result["deferred"] = [
            dict(
                direction="继续径向细化扫描",
                reason="同模型多点误差平台已确认，先转向完整算符诊断。",
            )
        ]
        payload = dict(
            kind="bloch_partition",
            source=case["output"],
            result_sha256=case["result_sha256"],
            arrays_sha256=case["arrays_sha256"],
        )
        identifier = job_identity(payload, policy, runner_hash)
        prior = records.get(identifier)
        if prior is None:
            result.update(
                status="selected",
                direction="分解完整 signed Bloch 有效算符，测一核孔／二核孔贡献与矩阵闭合。",
                reason="多个径向分辨率下更严门持续失败；直接复用已保存完整算符，增加可区分信息。",
                job=dict(id=identifier, **payload),
                estimate=dict(
                    determinants=120,
                    model_roots=8,
                    retained_dense_payload_bytes=2 * 1024**2,
                    seconds_upper_budget=600,
                    threads=1,
                ),
            )
            return result
        if prior["status"] != "completed":
            result.update(
                status="implementation_frontier",
                direction=result["frontier"][0]["direction"],
                reason="已领取的诊断未完成；保留失败，先修复执行原因，不静默重跑。",
            )
            return result
        # A complete matched DC/transverse radial pair permits new H controls.
        pairs = [
            (t, d)
            for t in group.values()
            for d in cases
            if d["job"]["mode"] == "DC"
            and d["job"]["ns"] == t["job"]["ns"]
            and d["meta"]["labels"] == t["meta"]["labels"]
            and d["model_identity"] == t["model_identity"]
        ]
        responses = []
        if pairs:
            transverse, dc = max(pairs, key=lambda x: x[0]["job"]["ns"])
            for scale in policy["whole_transverse_scales"]:
                payload = dict(
                    kind="whole_transverse_control",
                    scale=scale,
                    source=transverse["output"],
                    dc_source=dc["output"],
                    result_sha256=transverse["result_sha256"],
                    dc_result_sha256=dc["result_sha256"],
                )
                identifier = job_identity(payload, policy, runner_hash)
                prior = records.get(identifier)
                if prior is None:
                    result.update(
                        status="selected",
                        direction=f"全横向相互作用强度 λ={scale:g} 的完整 H／FCI／CCSDT 对照。",
                        reason="算符分解已完成；受控改变完整横向相互作用，检验误差响应阶次，每个 H 对照自己的 FCI。",
                        job=dict(id=identifier, **payload),
                        estimate=dict(
                            determinants=120,
                            model_roots=8,
                            retained_dense_payload_bytes=2 * 1024**2,
                            backend_memory_bytes=512 * 1024**2,
                            seconds_upper_budget=600,
                            threads=1,
                        ),
                    )
                    return result
                if prior["status"] != "completed":
                    result.update(
                        status="implementation_frontier",
                        direction=result["frontier"][0]["direction"],
                        reason="完整相互作用对照失败或中断；保留证据，先定位原因。",
                    )
                    return result
                measured = prior.get("result")
                if measured:
                    responses.append(
                        dict(
                            scale=scale,
                            defect_Ha=measured["backend_heff_max_Ha"],
                            ratio=measured["defect_response_ratio"],
                        )
                    )
        if len(responses) == len(policy["whole_transverse_scales"]):
            result["gap"]["whole_transverse_response"] = responses
            deviation = max(abs(row["ratio"] / row["scale"] - 1) for row in responses)
            result["gap"]["relative_linear_response_deviation"] = deviation
            # This is a direction heuristic, not a new scientific pass gate.
            if deviation <= 0.02:
                result["frontier"][0]["direction"] = (
                    "优先审计非 canonical 核—价 Fock 耦合的线性依赖图项，推导完整 0h1p 方程并验证候选后端。"
                )
            else:
                result["frontier"][0]["direction"] = (
                    "优先审计完整 Tcore／附着簇的非线性耦合与反馈项，逐项对照 Bloch 有效算符。"
                )
        result.update(
            status="implementation_frontier",
            direction=result["frontier"][0]["direction"],
            reason="此平台的可执行区分实验已完成；下一步需要新图项实现，继续旧网格扫描不能代替修复。",
        )
        if responses:
            result["reason"] += (
                " 实测全横向响应："
                + "；".join(
                    f"λ={r['scale']:g}，差 {r['defect_Ha']:.3e} Ha，相对原差 {r['ratio']:.4f}"
                    for r in responses
                )
                + "。这些变化只支持审计方向，尚未确定缺失图项。"
            )
        return result
    result.update(
        status="no_plateau",
        direction="补充同模型对照，再按实际缺口选择算符或收敛任务。",
        reason="尚无足够一致的有效算符平台证据。",
    )
    return result


def plan(repo, state, here=HERE):
    policy_raw = (here / "iteration_policy.json").read_bytes()
    policy = json.loads(policy_raw)
    if policy["backend_strict_Ha"] != 1e-10 or policy["oracle_Ha"] != 1e-11:
        raise ValueError("adaptive policy must preserve declared scientific gates")
    if (
        policy["caps"]["threads"] != 1
        or policy["caps"]["timeout_s"] != 600
        or policy["caps"]["process_address_bytes"] != 4 * 1024**3
        or policy["caps"]["service_memory_bytes"] != 6 * 1024**3
    ):
        raise ValueError(
            "adaptive policy must preserve the installed resource contract"
        )
    cases, citations = collect_cases(repo, state)
    records = {
        p.stem: json.loads(p.read_text())
        for p in (state / "iteration_ledger").glob("*.json")
    }
    for record in records.values():
        if record["status"] != "completed" or record.get("job", {}).get("kind") not in (
            "bloch_partition",
            "whole_transverse_control",
        ):
            continue
        data, citation = read_bound(repo, record["output"] + "/result.json")
        if (
            citation["sha256"] != record["result_sha256"]
            or not data["sources_and_inputs_unchanged"]
            or not data["oracle_gate_pass"]
            or data["job"] != record["job"]
        ):
            raise ValueError(
                "adaptive completed result failed its source/result/oracle identity"
            )
        arrays = (repo / record["output"] / "full_arrays.npz").resolve()
        if (
            not arrays.is_relative_to(repo.resolve())
            or digest(arrays.read_bytes()) != data["arrays_sha256"]
        ):
            raise ValueError("adaptive scientific array hash mismatch")
        if data["job"]["kind"] == "bloch_partition" and not data["partition_gate_pass"]:
            raise ValueError("adaptive partition gate failed")
        record["result"] = data
        citations.extend(
            [
                citation,
                dict(
                    path=str(arrays.relative_to(repo.resolve())),
                    sha256=data["arrays_sha256"],
                ),
            ]
        )
    registry, citation = read_bound(
        repo, str((here / "progress.json").relative_to(repo))
    )
    citations.append(citation)
    comparison, citation = read_bound(repo, registry["yb_comparison"])
    citations.append(citation)
    experimental = max(
        abs(row["relative_error"])
        for row in comparison["rows"]
        if row["relative_error"] is not None
    )
    runner_hash = digest((here / "run_gap_step.py").read_bytes())
    result = decide(cases, records, policy, runner_hash, experimental)
    if result["status"] == "no_plateau":
        queue = json.loads((here / "iteration_queue.json").read_bytes())
        job = next((j for j in queue["jobs"] if j["id"] not in records), None)
        result.update(status="selected" if job else "implementation_frontier", job=job)
    result.update(
        evidence=citations,
        policy_sha256=digest(policy_raw),
        planner_sha256=digest(Path(__file__).read_bytes()),
        runner_sha256=runner_hash,
        caps=policy["caps"],
    )
    return result


def card_summary(decision):
    lines = [
        "**按 gap 自主选择迭代**",
        decision["direction"],
        "依据：" + decision["reason"],
    ]
    if decision.get("gap"):
        gap = decision["gap"]
        lines.append(
            f"同模型 ns={gap['ns']}；算符差最大 {gap['maximum_Ha']:.3e} Ha；门 {gap['strict_gate_Ha']:.1e} Ha。"
        )
    lines.append(
        "准则：完整精度门优先；复用不可变输入；模块分工与 H 身份一致；执行前核对资源。"
    )
    lines.append(
        "本卡成功送达后重新核验并执行一项可用任务；完成结果在后续卡片同步。"
        if decision["job"]
        else "当前方向需要新增实现或修复失败原因；明确保留实现前沿，不重复扫描充当进展。"
    )
    return "\n".join(lines)
