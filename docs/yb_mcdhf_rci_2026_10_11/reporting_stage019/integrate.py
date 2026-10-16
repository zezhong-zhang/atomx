"""Publish the closed fixed-CI scale experiment without changing older gates."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
REPO = HERE.parents[2]
REGISTRY = REPO / "docs/atomx_hourly_progress_2026_10_10/progress.json"
RUN = TOP / "rank_aware_cell_discriminator_preparation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def bound(path):
    return {"path": str(path.relative_to(REPO)), "sha256": sha(path)}


def once(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    contract = read(HERE / "contract.json")
    assert contract["driver_sha256"] == sha(Path(__file__))
    assert contract["before_registry_sha256"] == sha(REGISTRY)
    assert contract["previous_stage_sha256"] == sha(TOP / "results_stage018.json")
    previous = read(TOP / "results_stage018.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise ValueError("Reporting gate failed: " + name)

    for name, row in previous["inputs"].items():
        check("previous/" + name, sha(REPO / row["path"]) == row["sha256"])
    for name, digest in contract["evidence_sha256"].items():
        check("evidence/" + name, sha(REPO / name) == digest)
    actual, receipt = read(RUN / "actual/result.json"), read(RUN / "root_execution.json")
    completion = read(RUN / "completion.json")
    check("foreground_closed", receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0)
    check("fixed_CI_scope", actual["fixed_CI_expectation_only"]
          and not actual["new_spectrum_or_experiment_qualification"]
          and not actual["uniform_radial_error_bound"]
          and actual["new_H_CI_gradient_optimizer_GPU"] == 0)
    check("complete_affected_scope", actual["affected_radial_keys"] == 7398
          and actual["full_table_scope"] == 12403)
    check("original_gates_preserved", actual["original_anchor_and_derivative_failures_preserved"]
          and actual["original_stationarity_not_changed"])
    check("declared_controls_passed", actual["all_declared_discriminator_controls_passed"])
    for phase in ("anchors", "ordinary", "candidate"):
        worker = read(RUN / "actual" / phase / "result.json")
        check("worker/" + phase, worker["all_numeric_controls_passed"]
              and all(c["passed"] for c in worker["controls"])
              and worker["started"] == worker["completed"]
              and actual["worker_results_sha256"][phase] == sha(RUN / "actual" / phase / "result.json"))
    declared = read(RUN / "contract.json")["max_all_calls"]
    check("all_real_call_counts", all(actual["calls_completed"].get(k, 0) == v
          for k, v in declared.items()) and actual["calls_started"] == actual["calls_completed"])
    rows = {r["label"]: r for r in actual["targets"]}
    check("five_original_states", set(rows) == {"6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"}
          and len({tuple(r["UID"]) for r in rows.values()}) == 5
          and rows["6s_1/2"]["UID"] == [1, 1, 0])
    delivery = read(TOP / "delivery_stage018/delivery_proof.json")
    check("actual_card_identity", "om_x100b638da67954a4c4a755da13ea1d9" in json.dumps(delivery)
          and "oc_fb2fc627b3e98dc9555de31580878758" in json.dumps(delivery))
    # Producer result and completion are bound separately; no second numerical consumer.
    check("completion_bound", "ff6551b49ebc59ecccf399e7f1d33c9669a06b579611d6df2ff336066b10fe65" in json.dumps(completion))
    result = copy.deepcopy(previous)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["historical_stage018_card_summary"] = previous["card_summary"]
    result["inputs"]["previous_stage018"] = bound(TOP / "results_stage018.json")
    for path, digest in contract["evidence_sha256"].items():
        result["inputs"][path.removeprefix(str(TOP.relative_to(REPO)) + "/")] = {"path": path, "sha256": digest}
    result["fixed_CI_inner_quadrature_correction"] = {
        "result": bound(RUN / "actual/result.json"), "execution": bound(RUN / "root_execution.json"),
        "completion": bound(RUN / "completion.json"), "source_maps": actual["source_maps"],
        "targets": actual["targets"], "calls_completed": actual["calls_completed"],
        "elapsed_uncontrolled_seconds": receipt["elapsed_uncontrolled_seconds"],
        "scope": actual["interpretation"],
        "interpretation": "Nominal fixed-CI operator changes are near float64 rounding; they do not explain the measured ~2000 cm−1 gap. This is not an error bound or a corrected CI spectrum.",
        "shared_source_applied": False, "performance_certified": False}
    result["latest_hourly_delivery"] = bound(TOP / "delivery_stage018/delivery_proof.json")
    result["expanded29_optimizer_status"] = "Unexecuted. Qualify the corrected f043 source with fresh H/CI/three gradients and four fixed CI-only direction endpoints before bounded native optimization."
    result["selection_reason"] = (
        "Independent signed ordinary/RKB analytical cell anchors now pass. Complete all7398 affected multipole keys for each source epoch, "
        "with original29 orbitals and five targetwise full signed densities, give only ~1e−11 cm−1 nominal transition changes. "
        "Keep the numerical correction, but target the unresolved original3451 constrained residual and balanced correlated orbital optimization to reduce the physical gap. "
        "New source requires fresh complete H/roots/signed-C/owner/gradient rather than reusing an old physical-input cache identity.")
    result["next_execution_status"] = result["expanded29_optimizer_status"]
    result["card_summary"] = (
        "原生MCDHF→RCI主线。内积分阶数修复通过独立有符号解析锚；同一Yb五态全部7398受影响多极项逐源实算，"
        "名义跃迁变化最大7.68e−12cm⁻¹，接近浮点舍入，无法解释约2000cm⁻¹实验gap；并非新CI谱或误差界。"
        "实际14h/S、14796径向求值，前台17.23s，零H/CI/梯度/优化器/GPU，没有新加速比。"
        "下一步验证修正源的新完整基线和原方向导数，再优化29共同轨道；原驻值、实验门和所有失败保留。"
        "本次卡片已实际送达读回并紧接这次有界物理迭代；远程周期表与Ne/Eu历史保留。")
    check("old_science_unchanged", all(result[k] == previous[k] for k in (
        "stationarity", "full_normalized_gradient_control", "experimental_gate_passed", "highest_level_reached")))
    registry = read(REGISTRY)
    registry["updated_at"] = result["created_at"]
    registry["physics_budget_result"] = str((TOP / "results_stage019.json").relative_to(REPO))
    registry["tasks"][0].update(status="积分修复量级判别通过；准备修正源MCDHF完整基线", detail=result["card_summary"])
    registry["tasks"][1].update(status="29共同轨道/1046扇区仍待自洽优化", detail=result["selection_reason"])
    registry["tasks"][5].update(status="解析核修复通过；无新端到端/并行/GPU加速认证", detail=result["card_summary"])
    registry["physics_next_setups"][1] = result["selection_reason"]
    registry["physics_next_setups"][5] = result["next_execution_status"]
    with (HERE / "before_registry.json").open("xb") as handle:
        handle.write(REGISTRY.read_bytes())
    once(TOP / "results_stage019.json", result)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    proof = {"result": bound(TOP / "results_stage019.json"), "registry": bound(REGISTRY),
             "driver": bound(Path(__file__)), "contract": bound(HERE / "contract.json"),
             "checks_count": len(checks), "checks": checks, "all_passed": True,
             "new_science_calls": 0, "new_optimizer_calls": 0, "cards_sent": 0}
    once(HERE / "integration_proof.json", proof)
    print(json.dumps({k: proof[k] for k in ("result", "registry", "checks_count", "all_passed")}))


if __name__ == "__main__":
    main()
