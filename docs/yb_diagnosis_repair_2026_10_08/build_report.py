"""Build the dated continuation from retained measurements, never best-fit selection."""
import csv
import json
from pathlib import Path
from common import ROOT, OLD, save


def table(headers, rows):
    return "| "+" | ".join(headers)+" |\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)


def main():
    audit=json.loads((ROOT/"baseline_audit_v3/results.json").read_text())
    gauge=json.loads((ROOT/"gauge_audit_v3/results.json").read_text())
    tags=["l2_n4","l3_n8","l4_n24","l4_n48","l4_e50_all","l6_e50_all"]
    runs={t:json.loads((ROOT/"results"/t/"results.json").read_text()) for t in tags if (ROOT/"results"/t/"results.json").exists()}
    best_tag="l6_e50_all" if "l6_e50_all" in runs else "l4_e50_all"
    best=runs[best_tag]["records"]
    if best_tag!="l6_e50_all":raise ValueError("Do not publish an incomplete intended final control")
    csvrows=[]
    for tag,run in runs.items():
        for r in run["records"]:
            csvrows.append(dict(run=tag,label=r["label"],Sigma2_cm1=r["Sigma2_cm1"],
                excitation_DF_cm1=r["excitation_DF_cm1"],excitation_Sigma2_cm1=r["excitation_Sigma2_cm1"],
                experiment_cm1=r["experiment_cm1"],error_cm1=r["excitation_Sigma2_cm1"]-r["experiment_cm1"],
                virtual_orbitals=len(r.get("virtual_manifest",{})),
                diagrams_2p1h_Ha=r["diagrams_Ha"]["2p1h"],diagrams_2h1p_Ha=r["diagrams_Ha"]["2h1p"]))
    with (ROOT/"correlation_comparison.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    names={"6s_1/2":"6s₁/₂","5d_3/2":"5d₃/₂","5d_5/2":"5d₅/₂","6p_1/2":"6p₁/₂","6p_3/2":"6p₃/₂"}
    pub=audit["baseline_replays"][0]
    dfrows=[[names[r["label"]],f'{r["excitation_cm1"]:.6f}',str(r["paper_DF_excitation_cm1"]),f'{r["excitation_difference_cm1"]:+.6f}'] for r in pub["energies"] if r["label"]!="6s_1/2"]
    grows=[[names[r["lower"]]+" → "+names[r["upper"]],f'{r["length_RME_au"]:+.6f}',f'{r["velocity_RME_au"]:+.6f}',f'{r["closure_radial_au"]:.2e}'] for r in gauge["records"]]
    mainrows=[[names[r["label"]],f'{r["excitation_DF_cm1"]:.2f}',f'{r["excitation_Sigma2_cm1"]:.2f}',f'{r["experiment_cm1"]:.2f}',f'{r["excitation_Sigma2_cm1"]-r["experiment_cm1"]:+.2f}'] for r in best if r["label"]!="6s_1/2"]
    published_E2={"6s_1/2":-8288,"5d_3/2":-9003,"5d_5/2":-8001,"6p_1/2":-4344,"6p_3/2":-3606}
    absolute=[[names[r["label"]],f'{r["Sigma2_cm1"]:.2f}',str(published_E2[r["label"]]),f'{r["diagrams_Ha"]["2p1h"]:.8f}',f'{r["diagrams_Ha"]["2h1p"]:.8f}'] for r in best]
    scan=[]
    for t in tags:
        run=runs[t]; rs={r["label"]:r for r in run["records"]}
        scan.append([t]+[f'{rs[k]["excitation_Sigma2_cm1"]-rs[k]["experiment_cm1"]:+.1f}' for k in ("5d_3/2","5d_5/2","6p_1/2","6p_3/2")])
    a={r["label"]:r for r in runs["l4_e50_all"]["records"]}
    tailrows=[[names[r["label"]],f'{r["Sigma2_cm1"]-a[r["label"]]["Sigma2_cm1"]:+.2f}',
        f'{r["excitation_Sigma2_cm1"]-a[r["label"]]["excitation_Sigma2_cm1"]:+.2f}'] for r in best]
    max_tail=max(abs(r["excitation_Sigma2_cm1"]-a[r["label"]]["excitation_Sigma2_cm1"]) for r in best)
    warnings=json.loads((ROOT/"qed_coverage_after.json").read_text())
    injection=json.loads((ROOT/"results"/best_tag/"ci_injection/results.json").read_text())
    goal=all(abs(r["excitation_Sigma2_cm1"]-r["experiment_cm1"])/r["experiment_cm1"]<.05 for r in best if r["experiment_cm1"])
    text=f'''# Yb⁺ 谱线偏差的定位与修复续报

日期：2026-10-08。计算软件：Atomx。本文延续[原始五能级、七谱线与两寿命报告](../rare_earth_yb_spectrum_2026_10_08/README.md)，保留原始数据与失败记录。

**判断：冻结核 Dirac–Fock 数值与偶极算符已获得独立对照，当前主要问题是相关物理及其收敛。已补齐外部单价电子的核–价二阶自能接口，并阻止未覆盖轨道的 QED 解析估算静默冒充表格结果。实验精度仍须单独判断。**

## 1. 为什么能排查到这里

沿用 Z=70、68 电子 [Xe]4f¹⁴ 闭核与一个外部价电子。主计算只有五个单维 J/宇称块，包含冻结核的直接库仑和非局域交换，没有核–价虚激发。原子核是既有经验 A=175 参数的 Fermi 电荷分布；没有同位素特异反冲、超精细或晶体场。

最新源码重放粗、细两个旧径向空间，原 69 电子 Dirac–Coulomb 能量及本征残差都通过 10⁻⁸ Ha 门限。独立的 [Porsev 等 2012 年论文](https://arxiv.org/html/1205.5895v2) 表 I 给出了同一 V⁽ᴺ⁻¹⁾ 冻结核模型的四个激发能；打印精度内与 Atomx 一致，能级排序也一致：

{table(["态","Atomx / cm⁻¹","论文 DF / cm⁻¹","差 / cm⁻¹"],dfrows)}

闭核参考自身有 N=68 个电子，所以原生 `potential_mode="VN"` 保持完整 68 电子场；相对于目标 N=69，它就是 V⁽ᴺ⁻¹⁾。这里再选 `VN-1` 会错误地去掉一个核电子。

## 2. 长度—速度差异不是可以随意归一化的错误

五条 E1 长度矩阵元在 [Safronova 等 2009 年论文](https://doi.org/10.1103/PhysRevA.79.022512) 表 II 的四位小数精度内复现。闭壳层旁观者的原生 CSF 张量收缩，与单电子解析角因子乘径向积分的幅度差小于 10⁻¹⁰ au。

进一步直接计算静态偶极的投影 Fock 对易子。对于 h_F=h_D+U_F：

`ω⟨f|r|i⟩ = ⟨f|[h_D,r]|i⟩ + ⟨f|[U_F,r]|i⟩`。

在原生 B-spline 度量 S 中，第二项由 `U_f S⁻¹ T − T S⁻¹ U_i` 独立计算。以下长度与速度保留同一个原生轨道相位，速度是领先光子阶的裸电流；闭合残差为径向 L−V−相互作用对易子：

{table(["跃迁","长度 RME / au","裸速度 RME / au","闭合残差 / au"],grows)}

五条线都通过预先设定的 10⁻⁶ au 闭合门限。特别是 5d₃/₂→6p₁/₂ 的巨大速度幅度、以及 5d₃/₂→6p₃/₂ 极小的裸速度幅度，独立论文表 III 也有对应现象。跨论文比较采用相位无关的 V/L 相对符号，以及能量加权电流 ωV；论文轨道能打印到整数 cm⁻¹，不把这些整数当成无限精确的很小跃迁分母。

这验证的是冻结核模型中的算符和表示关系，**没有实现 RPA/TDHF 的辐射顶角，也没有因此获得可靠相关寿命**。原始不一致可以被解释，不应强行令两种裸规范数值相等。

## 3. 修复了什么计算通路

原有 `mbpt2_orbital_energy_shift` 要求目标标签已存在于闭核参考，无法直接计算纯 68 电子核以外的 6s、6p、5d 单价电子。现在接受显式 `(P,Q,κ,ε)` 外部轨道，检查原生径向归一化与闭核正交性；保持参考占据数恰好 68，外部价电子既不伪装成核占据，也不从内部虚轨道求和中随意删除。

同时增加显式 `virtual_kappas` 与逐项诊断接口。完整 ℓ≤2 必须包含 `(-1,1,-2,2,-3)`；仅设置 `kappa_max=2` 不包含 d₅/₂。当前计算均保留每个声明 ℓ 的两个自旋轨道分支、全部 19 个占据核轨道、所有允许库仑多极，以及束缚虚态和正能伪连续态。

使用同一个原生反对称 J 耦合库仑核，包含两类二阶图：

`Σ_a(E)=½∑_brs |⟨ab∥rs⟩|²/(E+ε_b−ε_r−ε_s)`

`　　　　+½∑_bcr |⟨ar∥bc⟩|²/(E+ε_r−ε_b−ε_c)`。

这里 b,c 是占据自旋轨道，r,s 是虚自旋轨道；对 a 的磁态平均，并取 E=ε_a，即在壳二阶近似。非零耦合遇到小于 10⁻⁸ Ha 的分母会报错，未删去近共振项。

这是一体自能的**二阶阶数**；注入 CI 时参数名是 `sigma1`。它不是尚未实现的两价电子二体屏蔽 `sigma2`。所有五个态在同一核、同一空间同时修正；跃迁能移为 ΔΣ=Σ_upper−Σ_ground。共同的闭核相关能零点在激发能中相消。

## 4. 实算结果与不能掩盖的窗口依赖

最新完整控制：ℓ≤6，虚态能量 ε<50 Ha，保留原 ns=123、ks=9、R≈60 的径向工作区；根数上限 128 大于此基空间可提供的物理根数，因此这个窗口没有根数裁剪。实际虚轨道数为 {len(best[0]["virtual_manifest"])}。这只表示**声明窗口完整**，不代表无限虚空间、径向基或所有分波已经收敛。

{table(["态","冻结核 DF / cm⁻¹","加 Σ⁽²⁾ / cm⁻¹","实验 / cm⁻¹","修正后误差 / cm⁻¹"],mainrows)}

实验激发能沿用先前保存的 [NIST Yb II 能级](https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6.htm)。以上是在壳二阶、无 Breit/QED 的同一模型对照，不把文献相关值或实验能级加入哈密顿量。原先预设的四个激发能相对误差均小于 5% 的目标：**{'通过，但不是收敛或精密谱学认证' if goal else '未通过'}**。

绝对附着自能分项：

{table(["态","Σ / cm⁻¹","论文二阶 / cm⁻¹","2p1h / Ha","2h1p / Ha"],absolute)}

论文二阶值来自上述 2009 年论文表 I，是独立的数量级和态依赖核对。该文使用 200 au 盒、二阶分波至 ℓ=8 并外推更高分波；与本次 ns=123、约 60 au、ℓ≤6、50 Ha 空间不同，未设定两者必须在打印精度内相等的门限，也未用论文值补偿本次尾部。

所有窗口的实验残差都保留：

{table(["空间","5d₃/₂","5d₅/₂","6p₁/₂","6p₃/₂"],scan)}

表中前两档同时改变分波和根数，仅作小空间探针。`l4_n24`、`l4_n48`、`l4_e50_all` 固定 ℓ≤4，逐步补齐尾部。两条 6p 在 n24 时一度接近实验，补齐之后却出现上千 cm⁻¹ 正偏差，明确排除了“挑一个最接近的有限窗口”作为验证。

保持同一 50 Ha 窗口，ℓ≤4→ℓ≤6 的能移为：

{table(["态","自能增量 / cm⁻¹","激发能增量 / cm⁻¹"],tailrows)}

最大激发能增量仍有 {max_tail:.2f} cm⁻¹，{'超过' if max_tail>1 else '未超过'}预设 1 cm⁻¹ 收敛目标。能量窗 >50 Ha、更高分波、Σ 的径向收敛、能量依赖与高阶相关尚未完成。不能把这项有限窗口增量当成无条件尾部误差界。

原生 `ci_mbpt_solve(..., sigma1=...)` 已在五个态上实际运行。一体密度注入的最大能量核对误差为 {max(abs(x) for x in injection["matched_energy_error_Ha"]):.2e} Ha，本征残差及态匹配门限通过。没有仅在 CSV 中加一个数来冒充哈密顿量修复。

## 5. Breit、QED 与“加入后再迭代”的判断

此前完整静态 Breit 对四个激发能的增量约为 +27.59、−14.24、−37.23、−65.54 cm⁻¹。2009 年论文表 I 的第一阶 Breit 能差在整数打印精度内一致；该文第一阶项还含频率依赖，因此这里只是静态实现的数量级与约定交叉核对，不是所有 Breit 物理的验证。

已定位另一个具体缺陷：默认 `auto` 在 Z=70 选择 Indelicato/Mohr 表格后端，但它对 6s/6p 实际落入未覆盖轨道的领先 Zα 解析估算，原来标量入口不暴露这一覆盖变化。新增回归检验在修复前 3 项失败；修复后 `auto`、`mohr`、`indelicato` 均对解析回退发出明确的非精密 QED 警告，已覆盖的 5d 表格轨道不触发该警告。保留原来数学表达式，没有拟合新 QED 偏移或根据期望正负号改数。

{warnings["scope"]}

先前 Yb 6s 的默认估算为负，与不同辐射势模型的正值不同；这种后端敏感性是适用范围问题的实测证据。它不能通过机械地“再迭代一次”变成可信的全阶、屏蔽 QED。

从驻定原理出发，若未扰动态确实对所选哈密顿量驻定，一阶能移是 ⟨ψ|δH|ψ⟩，轨道响应首先进入更高阶；高精度仍需一致的算符与响应求解。这里尚未进行全 QED 模型算符的自洽重算，也不声称 QED 迭代绝对可忽略。证据支持先完成核–价相关及其辐射响应，而不是把当前上千 cm⁻¹ 偏差归因于漏跑一次 QED 迭代。

## 6. 验证、复现与边界

独立显式 m-scheme 自旋轨道求和覆盖占据目标及外部 s₁/₂、p₃/₂、d₅/₂，与原生 J 耦合自能在 rel=10⁻¹¹、abs=10⁻¹³ Ha 内相符。原有一体 CI 注入门限也通过。完整测试命令、结果和 QED 回退修复前后的记录见本目录日志及 [verification.json](verification.json)。

实际源代码身份见各运行的 `identity.json`；旧源与新源的显式链、修改前后差异见 [source_chain.json](source_chain.json) 与 `source_changes.patch`。所有计算固定一个 BLAS 线程，时间不是受控性能测量。每次二阶求和限制 900 秒，没有放宽残差、删减声明空间或使用经验比例系数。

```bash
.venv/bin/python docs/yb_diagnosis_repair_2026_10_08/run_sigma2.py --label NEW_L6 --lmax 6 --nvirt 128
.venv/bin/python docs/yb_diagnosis_repair_2026_10_08/inject_sigma1.py --run NEW_L6
```

脚本通过原子核/轨道哈希读取既有 `v68_default` 工作区；复现数据包包含该基线的输入及数值工作区。输出标签须新建，避免覆盖历史证据。两个早期基线审计适配错误、一个跨论文原始带符号速度门限失败都保留；后者改用匹配能量精度和相位约定的电流观测量，未改原生闭合门限。

下一项有判断力的工作是同一完整窗口的径向和高能尾部控制，再引入能量依赖的 Brueckner/更高阶相关，并以一致的 RPA/TDHF 加顶角、归一化算符计算强度。现有七条 DF 谱线和两寿命报告继续作为旧模型证据；本次没有把未经算符修正的强度包装成新相关寿命。
'''
    (ROOT/"README.md").write_text(text)
    save(ROOT/"report_summary.json",dict(final_control=best_tag,experimental_5_percent_goal_passed=goal,
        radial_or_correlation_converged=False,largest_l4_to_l6_transition_increment_cm1=max_tail,
        old_lifetimes_repaired=False, fitted_parameters=0))


if __name__=="__main__":main()
