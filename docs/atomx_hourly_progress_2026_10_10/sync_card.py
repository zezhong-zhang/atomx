"""Render source-backed progress and send it through the current bridge profile.

The user timer runs independently of an agent turn. A successful card receipt
durably requests evidence-guided selection and an admitted independent batch.
An unavailable implementation frontier is reported explicitly.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from parallel_planner import card_summary
from parallel_planner import plan_batch as plan

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CHANNEL_ENV = (
    "LARK_CHANNEL",
    "LARK_CHANNEL_HOME",
    "LARK_CHANNEL_PROFILE",
    "LARK_CHANNEL_CONFIG",
    "LARKSUITE_CLI_CONFIG_DIR",
)


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def evidence(relative):
    path = (REPO / relative).resolve()
    if not path.is_relative_to(REPO.resolve()):
        raise ValueError("evidence must be within the repository")
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def mcdhf_summary(data, citations):
    """Keep genuine orbital stationarity beside the new Yb model's energies."""
    if (data["method"] != "MCDHF -> RCI" or data["fitted_corrections"]
            or data["highest_level_reached"]):
        raise ValueError("MCDHF panel has an unsupported method or certification")
    for binding in data["inputs"].values():
        path = (REPO / binding["path"]).resolve()
        if (not path.is_relative_to(REPO.resolve()) or
                hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]):
            raise ValueError("MCDHF input identity changed")
        citations.append(dict(path=binding["path"], sha256=binding["sha256"]))
    gate = data["stationarity"]
    values = [gate[key] for key in ("orbital_residual", "orbital_tolerance",
                                  "estimated_energy_error", "energy_tolerance")]
    if (not all(math.isfinite(value) and value >= 0 for value in values)
            or gate["orbital_tolerance"] <= 0 or gate["energy_tolerance"] <= 0):
        raise ValueError("MCDHF stationarity metrics require finite residuals and positive gates")
    passed = (gate["orbital_residual"] < gate["orbital_tolerance"]
              and gate["estimated_energy_error"] < gate["energy_tolerance"])
    if ((gate["converged"] and not passed) or
            (data["production_certified"] and not gate["converged"])):
        raise ValueError("MCDHF stationarity certification contradicts its original gates")
    expected = {("6s_1/2", 1, 1), ("5d_3/2", 3, 1), ("5d_5/2", 5, 1),
                ("6p_1/2", 1, -1), ("6p_3/2", 3, -1)}
    rows = data["rows"]
    if len(rows) != 5 or {(r["label"], r["twoJ"], r["parity"]) for r in rows} != expected:
        raise ValueError("MCDHF five-target state identity changed")
    lines, experiment_gates = [], []
    for row in rows:
        error = row["computed_cm1"] - row["experimental_cm1"]
        if not math.isclose(error, row["error_cm1"], rel_tol=0, abs_tol=1e-6):
            raise ValueError("MCDHF position/reference arithmetic mismatch")
        if row["experimental_cm1"]:
            relative = error / row["experimental_cm1"]
            if (not math.isclose(relative, row["relative_error"], rel_tol=0, abs_tol=1e-12)
                    or row["gate_pass"] != (abs(relative) <= .01)):
                raise ValueError("MCDHF experimental gate changed")
            lines.append(f"• {row['label']}：{row['computed_cm1']:.2f} / "
                         f"{row['experimental_cm1']:.2f} cm⁻¹；差 {error:+.2f}")
            experiment_gates.append(row["gate_pass"])
    if data["experimental_gate_passed"] != all(experiment_gates):
        raise ValueError("MCDHF overall experimental gate contradicts its rows")
    status = "原驻值门通过" if gate["converged"] else "驻值未通过，能级仅用于诊断"
    return (
        f"Atomx 原生 {data['computed_run']}；{data['CSFs']} CSF / "
        f"{data['radial_orbitals']} 径向轨道 / {data['electrons']} 电子。\n"
        f"{status}：残差 {gate['orbital_residual']:.3e} / 门 "
        f"{gate['orbital_tolerance']:.1e}；估计能量误差 "
        f"{gate['estimated_energy_error']:.3e} Ha / 门 {gate['energy_tolerance']:.1e}。\n"
        + "\n".join(lines)
        + f"\n{data['card_summary']}\n实验门整体通过={data['experimental_gate_passed']}。"
    )


def render(registry, now):
    citations = []
    comparison, citation = evidence(registry["yb_comparison"])
    citations.append(citation)
    excitation = []
    for row in comparison["rows"]:
        if row["relative_error"] is not None:
            name = f"{row['key'][0]}{'spdf'[abs(row['key'][1]) - (row['key'][1] < 0)]}"
            excitation.append(
                f"• {name} (2J={row['twoJ']})：计算 {row['computed_cm1']:.2f} / "
                f"实验 {row['experimental_cm1']:.2f} cm⁻¹；"
                f"差 {row['error_cm1']:+.2f}；{100 * row['relative_error']:+.3f}% · "
                + ("通过" if row["gate_pass"] else "未通过")
            )
    # The saved SD comparator remains distinct from the new MCDHF example.
    sections = [
        "**当前判断**\n尚未达到最高水准。数值、物理空间收敛和实验精度分别判断。"
        "当前优先原子级。每次成功发卡触发按 gap 选择任务；可执行任务有资源上限，"
        "实现缺口或资源不足则明确列出。计算结果在后续卡片同步，未通过的精度门保留。",
        "**待办与进度**\n"
        + "\n".join(
            f"{i}. {item['status']} · {item['title']}\n{item['detail']}"
            for i, item in enumerate(registry["tasks"], 1)
        ),
    ]
    physics_path = registry.get("physics_budget_result")
    if physics_path:
        physics, citation = evidence(physics_path)
        if physics["fitted_corrections"] or physics["highest_level_reached"]:
            raise ValueError("physics-scale estimates cannot promote fitted spectra or certification")
        citations.append(citation)
        sections.insert(1, "**物理量级与推进顺序**\n" + physics["card_summary"])
    # Names of archived comparator fields are read explicitly below; schema
    # drift fails the report rather than silently displaying old cached values.
    mcdhf_path = registry.get("yb_mcdhf_result")
    if mcdhf_path:
        mcdhf, citation = evidence(mcdhf_path)
        citations.append(citation)
        sections.append(f"**Yb⁺ 主线：原生 MCDHF→RCI · {mcdhf['CSFs']} CSF**\n"
                        + mcdhf_summary(mcdhf, citations))
    sections.append(
        ("**Yb⁺ 历史 SD 对照 · 共同 6s 零点**\n" if mcdhf_path else
         "**Yb⁺ 相关能级与实验 · 共同 6s 零点**\n")
        + "\n".join(excitation)
        + "\n"
        + "\n".join(
            f"• {row['label']} 精细分裂：{100 * row['relative_error']:+.3f}% · "
            + ("通过" if row["gate_pass"] else "未通过")
            for row in comparison["splittings"]
        )
        + "\n本页数值来源每次重新读取并记录 SHA256；相对误差门限 1%。"
    )
    spectrum_lines = []
    for item in registry["spectra"]:
        comparison_text = item["comparison"]
        artifact = item.get("source_artifact")
        if artifact:
            data, citation = evidence(artifact)
            citations.append(citation)
            if item.get("reader") == "rare_earth_lifetimes":
                comparison_text = "；".join(
                    f"{row['state']}：{row['calc_lifetime']:.4f} vs "
                    f"{row['experiment_lifetime']}±{row['experiment_uncertainty']} "
                    f"{row['unit']} ({row['relative_error_percent']:+.2f}%)"
                    for row in data["lifetimes"]
                )
            elif item.get("reader") == "ne_xray_positions":
                rows = [
                    row
                    for row in data["models"]["full_n12_joint"]
                    if "energy_error_eV" in row
                ]
                errors = [row["energy_error_eV"] for row in rows]
                comparison_text = (
                    f"{len(rows)} 条匹配线：联合弛豫 CI 与实验评估差 "
                    f"{min(errors):+.3f} 至 {max(errors):+.3f} eV。"
                )
            elif item.get("reader") == "native_ne_positions":
                positions = []
                for line in ("q", "r"):
                    row = data["transitions"][line]
                    calculated = row["energy_eV"]
                    reference = data["experiment"][line + "_eV"]
                    difference = row["minus_experiment_meV"]
                    if not all(math.isfinite(value) for value in
                               (calculated, reference, difference)) or not math.isclose(
                        difference, 1000 * (calculated - reference),
                        rel_tol=0, abs_tol=1e-6,
                    ):
                        raise ValueError("native Ne position/reference identity mismatch")
                    positions.append(
                        f"{line}：{calculated:.9f} / {reference:.4f} eV；"
                        f"计算−实验 {difference:+.6f} meV"
                    )
                comparison_text = "；".join(positions)
                comparison_text += (
                    f"。实验来源：{data['experiment']['source']}；"
                    f"完整相关收敛={data['full_correlation_converged']}，"
                    f"物理极点认证={data['physical_pole_certified']}。"
                )
            elif item.get("reader") == "native_eu_baseline":
                # The historical calibrated target is distinct from the
                # current NIST semiempirical rows. Verify the audit inputs
                # before presenting its source/classification-aware summary.
                for binding in data["inputs"].values():
                    path = (REPO / binding["path"]).resolve()
                    if (not path.is_relative_to(REPO.resolve()) or
                        hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]):
                        raise ValueError("native Eu baseline input identity changed")
                    citations.append(dict(path=binding["path"], sha256=binding["sha256"]))
                comparison_text = data["card_summary"]
        spectrum_lines.append(
            f"• **{item['title']}**：{comparison_text}\n{item['status']}"
        )
    sections.append("**各类谱与实验对照**\n" + "\n".join(spectrum_lines))
    atlas = registry.get("atlas")
    if atlas:
        sections.append(
            "**NIST 元素周期表**\n"
            + f"[远程打开原子谱学图谱]({atlas['url']})\n"
            + atlas["scope"]
        )
    dynamic = registry.get("dynamic_breit_result")
    if dynamic and (REPO / dynamic).exists():
        result, citation = evidence(dynamic)
        citations.append(citation)
        sections.append("**动态 Breit 实算更新**\n" + result["card_summary"])
    else:
        sections.append(
            "**动态 Breit**\n已有实频 Coulomb 规范横向交换算符及 FAC "
            "矩阵元对照。正在推进 Yb⁺ 同轨道 DC / 静态 / 动态配对实验；"
            "完整原生 SD/FS-CCSDT 算符接入与大空间验证仍待完成。"
        )
    precision_path = registry.get("fscc_precision_result")
    if precision_path:
        precision, citation = evidence(precision_path)
        citations.append(citation)
        defect = max(abs(value) for value in precision["tighter_root_defects_Ha"])
        sections[-1] += (
            "\n\n**原后端 FS-CCSDT 严格失败记录**\n"
            f"收紧阈值后方程残差界 {precision['tighter_equation_residual_bound_Ha']:.3e} Ha；"
            f"完整 CI 八根最大差仍为 {defect:.3e} Ha。"
            "原 1e−9 门通过；更严 1e−10 门未通过。"
            "这是原后端的保留失败；当前隔离候选修复另列完整门，旧结果不改写。"
        )
    target_path = registry.get("fscc_strict_target")
    if target_path:
        target, citation = evidence(target_path)
        citations.append(citation)
        checks = target["checks"]
        sections[-1] += (
            f"\n当前专门面板：{checks['passed']} 通过／"
            f"{checks['expected_failures']} 个严格已知目标失败／"
            f"{checks['skipped']} 跳过。已知失败保留为 strict xfail。"
        )
    candidate_path = registry.get("fscc_equation_fix_result")
    if candidate_path:
        candidate, citation = evidence(candidate_path)
        citations.append(citation)
        if (
            not candidate["complete_candidate_gate_pass"]
            or not candidate["inputs_unchanged"]
        ):
            raise ValueError("candidate equation fix lacks its declared complete gates")
        panel, citation = evidence(registry["fscc_equation_fix_panel"])
        citations.append(citation)
        controls = [
            item
            for item in panel.values()
            if item.get("complete_candidate_gate_pass")
            and not item.get("label_correction")
        ]
        sections.append(
            "**非 canonical 方程修复 · 有限严格门已闭合**\n"
            "open Fov*T2 项重复反对称化已在隔离候选修复，保留原 Hamiltonian。"
            f"完整 HEFF 差 {candidate['backend_strict_heff_max_Ha']:.3e} Ha；"
            f"完整实际幅度 Bloch 残差 {candidate['backend_wave_bloch_residual_max_Ha']:.3e} Ha；"
            f"八根完整 CI 差 {candidate['candidate_energy_root_max_Ha']:.3e} Ha。\n"
            f"{len(controls)} 项候选控制保留各自 H；原 1e−10 Ha 门未放宽。"
            "大规模重原子、完整原生横向 SD 和实验精度继续分别验证；默认旧后端保留。"
        )
    norm_path = registry.get("norm_memory_result")
    if norm_path:
        norm, citation = evidence(norm_path)
        if norm["status"] != "pass" or not all(norm["gates"].values()):
            raise ValueError("bounded norm evidence failed its declared gates")
        citations.append(citation)
        sections.append("**完整空间范数预算更新**\n" + norm["card_summary"])
    benchmark_path = registry.get("external_benchmark_metrics")
    if benchmark_path:
        benchmark, citation = evidence(benchmark_path)
        citations.append(citation)
        lines = []
        for row in benchmark["cpu"]:
            if row["workers"] == 1:
                continue
            measurement = (
                f"S={row['speedup']:.3f}× / E={100 * row['efficiency']:.1f}%"
                if row["controlled"]
                else "准入门未通过；受控加速比/效率不报告"
            )
            lines.append(f"• {row['workload']} · {row['workers']}线程：{measurement}")
        matched = benchmark["matched_AMBiT"]
        sections.append(
            "**软件精度与并行效率 · 实测**\n"
            f"完整 Ne Σ₁ 对 AMBiT 差 {matched['max_error_Ha']:.3e} Ha；"
            f"Na⁺ 完整 SD action 差 {benchmark['finite_SD']['maximum_signed_error_Ha']:.3e} Ha。\n"
            f"固定能量 Σ₁ warm：Atomx {1000 * matched['statistics']['left_median_seconds']:.2f} / "
            f"AMBiT {1000 * matched['statistics']['right_median_seconds']:.2f} ms；"
            "Atomx 额外构造 poles/binding，非全程序排名。\n"
            + "\n".join(lines)
            + "\nS 为逐轮配对中位数，E=S/请求线程数；各行是不同的固定工作量。"
            "Yb 仅完整 term2 数值及响应阶段，采用封存 signed stress 输入，"
            "非已收敛 SD 波函数；AMBiT 是完整 BeII 程序；"
            "SD 是 pCI-derived 图式对照，FSCC 继承 EXP-T，并采用私有低开销等待器计时。"
            "EXP-T 请求4/8线程超过资源上限，效率不报告。"
            "该轮 GPU 有其他计算进程，仅保留观察计时；实验1%门仍未通过。"
            f"\n本次重新读取 benchmark SHA256：{citation['sha256'][:16]}。"
        )
        if "FAC" in benchmark:
            fac = benchmark["FAC"]
            precision = fac["precision"]
            common = []
            spinors = []
            for case in precision["cases"]:
                gate = case["gates"]["common_integrals"]
                state = "通过" if gate["strict_pass"] else "未通过"
                common.append(
                    f"Z={case['z']}：{gate['max_absolute_Ha']:.3e} Ha（{state}）"
                )
                aligned = case["gates"]["4000"]["c_aligned_eigenvalues"]
                state = "通过" if aligned["strict_pass"] else "未通过"
                spinors.append(
                    f"Z={case['z']}：{aligned['max_absolute_Ha']:.3e} Ha（{state}）"
                )
            costs = [
                f"{row['workload'].split()[0]} {row['seconds_median']:.3f}s"
                for row in benchmark["cpu"]
                if row["workers"] == 1 and row["controlled"]
                and "four-ion" in row["workload"]
            ]
            model = fac["model_spread"]["cases"]
            level_spread = max(
                case["maximum_excitation_energy_relative_difference"]
                for case in model
            )
            rate_spread = max(
                case["maximum_line_A_relative_difference"] for case in model
            )
            sections.append(
                "**FAC 新增完整对照**\n"
                f"{precision['total_levels']} 能级／{precision['total_integrals']} 径向积分，"
                "保留全部 J／宇称矩阵与有符号数组。共同积分能级差：\n"
                + "；".join(common)
                + "\n共同 FAC 轨道、4000 点并解析对齐 c 后：\n"
                + "；".join(spinors)
                + "\n原绝对1e−11 Ha／相对1e−10门均未放宽。"
                "四离子分组独立的冻结CI流程，不含3s/3d组间耦合；"
                "共68能级／20条物理E1跃迁×2光子模式（40条记录）；"
                + ("单线程子进程总耗时：" + "／".join(costs) + "。" if costs else "")
                + "各自优化轨道、诊断导出负担不同，不能据耗时排列相同实验精度下的速度。"
                + f"全部64激发最大模型差{100 * level_spread:.4f}%，"
                + f"全部40线A最大模型差{100 * rate_spread:.3f}%；按配置/J/宇称及能量排序对应，非NIST误差。"
                + f"单线程SCF相关阶段：FAC {fac['stages']['medians']['FAC']['physical_settings_grid_and_ground_scf']:.3f}s"
                + f"／Atomx {fac['stages']['medians']['Atomx']['ground_scf_seconds']:.3f}s，边界不同；"
                "实验谱排名未建立；共同轨道积分差距和小流程并行粒度仍需改进。"
            )
    sections.append(
        "**口径与来源**\nYb 位置锚："
        "[NIST Yb II 历史能级表](https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6.htm)。"
        "此表未提供能级不确定度；真空 Ritz 波长不是独立实测波长。"
        "不同模型的强度/寿命不拼接。\n"
        f"进度登记更新：{registry['updated_at']}\n"
        f"本次读取：{now.astimezone(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds')}\n"
        f"证据：{comparison['computed_run']} / {citations[0]['sha256'][:12]}"
    )
    card = {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "title": {"tag": "plain_text", "content": "Atomx · 任务与谱学实验对照"},
            "template": "blue",
        },
        "body": {"elements": [{"tag": "markdown", "content": s} for s in sections]},
    }
    return card, citations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--cycle", help="manual update name; default UTC hour")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    state = Path(config["state_directory"])
    state.mkdir(parents=True, exist_ok=True)
    os.chmod(state, 0o700)
    with (state / "sync.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another hourly synchronization is running; skipped.")
            return
        now = datetime.now(timezone.utc)
        raw_registry = (HERE / "progress.json").read_bytes()
        registry = json.loads(raw_registry)
        card, citations = render(registry, now)
        try:
            decision = plan(REPO, state, here=HERE)
        except Exception as error:
            # Invalid science evidence must remain visible in the hourly card.
            decision = dict(
                status="waiting_for_evidence_repair",
                job=None,
                direction="修复证据身份或结果契约后再执行科学任务。",
                reason=type(error).__name__ + ": " + str(error),
                evidence=[],
            )
        card["body"]["elements"].insert(
            1, {"tag": "markdown", "content": card_summary(decision)}
        )
        citations.extend(decision["evidence"])
        worker_status = state / "iteration_status.json"
        if worker_status.exists():
            raw = worker_status.read_bytes()
            previous = json.loads(raw)
            summary = f"**上轮实际迭代**\n{previous.get('job_id', '队列')} · {previous['status']}"
            result = previous.get("result")
            if result:
                if result.get("card_summary"):
                    summary += "\n" + result["card_summary"]
                else:
                    summary += (
                        f"\n独立 Bloch 方程差 {result['bloch_equation_max_Ha']:.3e} Ha；"
                        f"后端完整有效矩阵差 {result['backend_heff_max_Ha']:.3e} Ha；"
                        + (
                            "更严 1e−10 门通过。"
                            if result["backend_strict_1e10_gate_pass"]
                            else "更严 1e−10 门未通过。"
                        )
                    )
            if previous.get("error"):
                summary += "\n" + previous["error"]
            if len(previous.get("jobs", [])) > 1:
                for record in previous["jobs"]:
                    summary += f"\n• {record['job_id']}：{record['status']}"
                    if record.get("result", {}).get("card_summary"):
                        summary += "\n" + record["result"]["card_summary"]
                    if record.get("error"):
                        summary += "\n" + record["error"]
            card["body"]["elements"].append({"tag": "markdown", "content": summary})
            citations.append(
                {
                    "path": "private iteration_status.json",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        save(state / "latest_card.json", card)
        if args.render_only:
            print(
                json.dumps(
                    {
                        "rendered": True,
                        "elements": len(card["body"]["elements"]),
                        "evidence": citations,
                    },
                    ensure_ascii=False,
                )
            )
            return
        env = os.environ.copy()
        for name in CHANNEL_ENV:
            env[name] = config["channel_environment"][name]
        if env["LARK_CHANNEL"] != "1":
            raise ValueError("hourly synchronization requires bridge-bound mode")
        cycle = args.cycle or now.strftime("%Y-%m-%dT%H")
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, config["chat_id"] + ":atomx:" + cycle))
        previews = state / "iteration_previews"
        previews.mkdir(exist_ok=True)
        preview_path = previews / (key + ".json")
        if not preview_path.exists():
            save(preview_path, decision)
        preview_sha256 = hashlib.sha256(preview_path.read_bytes()).hexdigest()
        command = [
            config["lark_cli"],
            "im",
            "+messages-send",
            "--chat-id",
            config["chat_id"],
            "--msg-type",
            "interactive",
            "--content",
            json.dumps(card, ensure_ascii=False),
            "--idempotency-key",
            key,
        ]
        record = {
            "started_at": now.isoformat(),
            "cycle": cycle,
            "idempotency_key": key,
            "evidence": citations,
            "registry_sha256": hashlib.sha256(raw_registry).hexdigest(),
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "passed": False,
            "iteration_preview_sha256": preview_sha256,
            "iteration_preview": str(preview_path),
        }
        try:
            # Verify bound identity before any send, including after restarts.
            preview = subprocess.run(
                command + ["--dry-run"],
                env=env,
                text=True,
                capture_output=True,
                timeout=45,
                check=True,
            )
            dry = json.loads(preview.stdout)
            if dry.get("identity") != "bot" or not dry.get("ok"):
                raise RuntimeError(
                    "configured bridge did not resolve a usable bot identity"
                )
            response = subprocess.run(
                command, env=env, text=True, capture_output=True, timeout=90, check=True
            )
            data = json.loads(response.stdout)
            if not data.get("ok"):
                raise RuntimeError("Lark reported an unsuccessful card send")
            message = data.get("data", {})
            record.update(
                passed=True,
                message_id=message.get("message_id"),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            save(state / ("receipt_" + key + ".json"), data)
        except Exception as error:
            record["error"] = type(error).__name__ + ": " + str(error)
            if isinstance(error, subprocess.CalledProcessError):
                record["cli_diagnostic"] = (error.stderr or "")[-4000:]
            save(state / "last_status.json", record)
            raise
        save(state / "last_status.json", record)
        save(state / "last_success.json", record)
        # Receipt-backed durable outbox. Duplicate cycle keys never repeat a job.
        if config.get("iteration_service"):
            requests = state / "iteration_requests"
            requests.mkdir(exist_ok=True)
            request_path = requests / (key + ".json")
            if not request_path.exists():
                if not record.get("message_id"):
                    raise RuntimeError(
                        "cannot start iteration without a real card receipt"
                    )
                save(
                    request_path,
                    dict(
                        key=key,
                        cycle=cycle,
                        message_id=record["message_id"],
                        status="queued",
                        requested_at=datetime.now(timezone.utc).isoformat(),
                        preview_sha256=preview_sha256,
                    ),
                )
                (requests / (key + ".ready")).touch()
            record["iteration_request"] = str(request_path)
            dispatch = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "start",
                    "--no-block",
                    config["iteration_service"],
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            record["iteration_dispatch_ok"] = dispatch.returncode == 0
            if dispatch.returncode:
                record["iteration_dispatch_error"] = dispatch.stderr[-2000:]
            save(state / "last_success.json", record)
            save(state / "last_status.json", record)
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
