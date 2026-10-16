"""Build tables, standalone plots and the current scientific account from measurements."""
from pathlib import Path
import csv
import json
import hashlib
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LABELS=["5d_3/2","5d_5/2","6p_1/2","6p_3/2"]
SHORT=[r"$5d_{3/2}$",r"$5d_{5/2}$",r"$6p_{1/2}$",r"$6p_{3/2}$"]


def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n|"+"|".join("---" for _ in headers)+"|\n"+"\n".join("| "+" | ".join(str(x) for x in row)+" |" for row in rows)+"\n"


def main():
    data=json.loads((ROOT/"analysis/results.json").read_text())
    audit=json.loads((ROOT/"representation_audit_v4/results.json").read_text())
    capture=json.loads((ROOT/"capture_status.json").read_text())
    coarse=data["primaries"]["coarse_l8_e200"]
    fine=data["primaries"]["fine_l6_e200"]
    def variant(case,l,e):return next(v for v in data["variants"][case] if v["lmax"]==l and v["emax_Ha"]==e)
    def control(kind,oldcase,newcase,lo_l,hi_l,lo_e,hi_e):
        return next(c for c in data["controls"] if c["kind"]==kind and c["old"]==dict(case=oldcase,lmax=lo_l,emax_Ha=lo_e) and c["new"]==dict(case=newcase,lmax=hi_l,emax_Ha=hi_e))
    cc="coarse_l8_e200";ff="fine_l6_e200"
    angular=control("angular",cc,cc,6,8,200.,200.)
    last_l=control("angular",cc,cc,7,8,200.,200.)
    coarse_uv=control("energy_ceiling",cc,cc,8,8,100.,200.)
    fine_uv=control("energy_ceiling",ff,ff,6,6,100.,200.)
    radial50=control("radial_reference",cc,ff,6,6,50.,50.)
    radial200=control("radial_reference",cc,ff,6,6,200.,200.)
    fine_angular=control("angular",ff,ff,4,6,200.,200.)
    selected=[("粗基 ℓ6→8，200 Ha",angular),("粗基 ℓ7→8，200 Ha",last_l),
        ("粗基 ℓ8，100→200 Ha",coarse_uv),("细基 ℓ6，100→200 Ha",fine_uv),
        ("径向加密，ℓ6/50 Ha",radial50),("径向加密，ℓ6/200 Ha",radial200)]
    controls_table=table(["控制","5d₃/₂","5d₅/₂","6p₁/₂","6p₃/₂","1 cm⁻¹ 门"],
        [[name]+[f"{r['excitation_change_cm1']:+.3f}" for r in c["records"][1:]]+
         ["通过" if c["max_absolute_excitation_change_cm1"]<1 else "未通过"] for name,c in selected])
    counts=[]
    for e in (5.,20.,50.,100.,200.):
        row=[f"{e:g}"]
        for case,l in ((cc,6),(cc,8),(ff,6)):
            row.append(sum(b["counts"][str(e)] for b in audit["blocks"] if b["case"]==case and b["l"]<=l))
        counts.append(row)
    counts_table=table(["正能上限/Ha","粗基 ℓ≤6","粗基 ℓ≤8","细基 ℓ≤6"],counts)
    exp_table=table(["上态","实验","旧粗基 ℓ6/50","粗基 ℓ8/200","细基 ℓ6/200","细基−实验"],
        [[r["label"],f"{r['experimental_excitation_cm1']:.2f}",
          f"{variant(cc,6,50.)['records'][i+1]['excitation_dyson_cm1']:.2f}",
          f"{coarse['records'][i+1]['excitation_dyson_cm1']:.2f}",
          f"{r['excitation_dyson_cm1']:.2f}",f"{r['experiment_residual_cm1']:+.2f}"]
         for i,r in enumerate(fine["records"][1:])])
    shifts_table=table(["态","粗基 ℓ8/200 能移","细基 ℓ6/200 能移","细基 Z"],
        [[r["label"],f"{coarse['records'][i]['dyson_shift_cm1']:.3f}",f"{r['dyson_shift_cm1']:.3f}",f"{r['residue']:.6f}"]
         for i,r in enumerate(fine["records"])])
    literature=json.loads((ROOT/"literature_check/results.json").read_text())
    literature_table=table(["态","论文在壳 E₂","粗基在壳 E₂","粗基−论文","细基在壳 E₂"],
        [[r["label"],r["paper_printed_E2_cm1"],f"{r['coarse_l8_e200']['on_shell_cm1']:.2f}",
          f"{r['coarse_l8_e200']['native_minus_paper_E2_cm1']:+.2f}",f"{r['fine_l6_e200']['on_shell_cm1']:.2f}"] for r in literature["records"]])
    anchor=json.loads((ROOT/"absolute_anchor/results.json").read_text())
    anchor_contract=json.loads((ROOT/"absolute_anchor/contract.json").read_text())
    absolute_rows=anchor["cases"][ff]
    absolute_table=table(["态","实验附着能","细基附着能","计算−实验"],
        [[r["label"],f"{r['experimental_attachment_cm1']:.2f}",f"{r['computed_attachment_cm1']:.2f}",
          f"{r['absolute_residual_cm1']:+.2f}"] for r in absolute_rows])
    radial_table=table(["态","DF 附着能变化","相关能移变化","总附着能变化","跃迁变化"],
        [[r["label"]]+[f"{r[k]:+.4f}" for k in ("DF_attachment_change_cm1","correlation_shift_change_cm1","individual_attachment_change_cm1","excitation_change_cm1")]
         for r in radial200["records"]])
    roots=[r for vv in data["variants"].values() for v in vv for r in v["records"]]
    max_res=max(abs(r["root_residual_Ha"]) for r in roots)
    max_ci=max(abs(x) for ci in data["ci_injection"] for x in ci["matched_energy_error_Ha"])
    max_ci_res=max(abs(x) for ci in data["ci_injection"] for x in ci["residuals_Ha"])
    metric=max(b["virtual_Gram_max_error"] for b in audit["blocks"])
    overlap=max(b["max_occupied_overlap"] for b in audit["blocks"])
    old_replay=max(abs(r["old_on_shell_error_Ha"]) for r in roots if r["old_on_shell_error_Ha"] is not None)
    jobs=[]
    for job in capture["jobs"]:
        row=json.loads((ROOT/"results"/job["case"]/job["label"].replace("/","_")/"results.json").read_text())
        jobs.append(row)
    source=json.loads((ROOT/"capture_source_sha256.json").read_text())
    previous=json.loads((ROOT.parent/"yb_energy_dependent_sigma_2026_10_08/final_source_sha256.json").read_text())
    previous_production={name:digest for name,digest in previous.items() if name.startswith("src/atomx/")}
    if source!=previous_production:raise ValueError("Previous 137-test production identity does not match current capture")
    repo=ROOT.parents[1]
    for name,digest in previous.items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Previous validated dependency changed: {name}")
    with zipfile.ZipFile(ROOT.parent/"yb_energy_dependent_sigma_2026_10_08/Affected_tests_source.zip") as tests:
        for name in tests.namelist():
            if tests.read(name)!=(repo/name).read_bytes():
                raise ValueError(f"Previous affected test changed: {name}")
    feedback=max(abs(r["excitation_dyson_cm1"]-r["excitation_on_shell_cm1"]) for r in fine["records"][1:])
    with (ROOT/"all_models.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["case","lmax","emax_Ha","label","epsilon_DF_Ha","dyson_attachment_Ha","dyson_shift_cm1","excitation_dyson_cm1","experiment_residual_cm1","residue"])
        writer.writeheader()
        for case,vs in data["variants"].items():
            for v in vs:
                for r in v["records"]:
                    writer.writerow({**{k:v[k] for k in ("case","lmax","emax_Ha")},**{k:r[k] for k in writer.fieldnames if k not in ("case","lmax","emax_Ha")}})
    plt.rcParams.update({"font.size":10,"figure.dpi":160})
    fig,axes=plt.subplots(1,2,figsize=(11.2,3.9),constrained_layout=True)
    es=np.asarray([5.,20.,50.,100.,200.])
    axes[0].plot(es,[r[1] for r in counts],"o-",label="ns=123, l<=6")
    axes[0].plot(es,[r[3] for r in counts],"s-",label="ns=230, l<=6")
    axes[0].set(xscale="log",xlabel="Virtual energy ceiling (Ha)",ylabel="Actual radial virtual orbitals",title="Representation capacity")
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.25)
    for i,label in enumerate(LABELS):
        ys=[]
        for e in es:
            c=control("radial_reference",cc,ff,6,6,float(e),float(e))
            ys.append(c["records"][i+1]["excitation_change_cm1"])
        axes[1].plot(es,ys,"o-",label=SHORT[i])
    axes[1].axhline(0,color="black",lw=.6)
    axes[1].axhspan(-1,1,color="gray",alpha=.13,label="+/-1 cm-1")
    axes[1].set(xscale="log",xlabel="Common energy ceiling (Ha)",ylabel="Refined - coarse excitation (cm$^{-1}$)",title="Measured radial response, l<=6")
    axes[1].legend(fontsize=8,ncol=2);axes[1].grid(alpha=.25)
    fig.savefig(ROOT/"representation_and_response.png");fig.savefig(ROOT/"representation_and_response.svg");plt.close(fig)
    summary=dict(max_root_residual_Ha=max_res,max_CI_energy_error_Ha=max_ci,max_CI_residual_Ha=max_ci_res,
        max_virtual_Gram_error=metric,max_occupied_overlap=overlap,old_on_shell_replay_error_Ha=old_replay,
        current_source_matches_previous_137_test_panel=True,
        captures_wall_seconds_uncontrolled=capture["elapsed_seconds_uncontrolled"],
        max_process_RSS_MiB=max(j["peak_process_RSS_MiB"] for j in jobs),
        fine_max_excitation_energy_feedback_cm1=feedback,
        last_angular_increment_cm1=last_l["max_absolute_excitation_change_cm1"],
        fine_last_UV_increment_cm1=fine_uv["max_absolute_excitation_change_cm1"],
        radial200_max_increment_cm1=radial200["max_absolute_excitation_change_cm1"],
        selected_control_gate_passed={name:c["max_absolute_excitation_change_cm1"]<1 for name,c in selected},
        absolute_ionization_anchor_cm1=anchor["ionization_cm1"],
        fine_absolute_attachment_residuals_cm1={r["label"]:r["absolute_residual_cm1"] for r in absolute_rows},
        absolute_relative_closure_error_cm1=anchor["maximum_closure_error_cm1"])
    (ROOT/"report_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    report=f'''# Yb⁺ 谱线继续定位：相关尾部与径向表示

2026-10-08，Atomx 原生实算。承接[能量自洽与二阶自能修复](../yb_energy_dependent_sigma_2026_10_08/README.md)。本报告是当前尾部诊断；前两轮源码、输入、失败及结果保持封存。

## 有判断力的结论

本轮完成两组完整五态计算和共同空间对照：粗径向基扩至 ℓ≤8、200 Ha，加密径向基计算 ℓ≤6、200 Ha。核、全部占据轨道、反对称 Coulomb 算符和 Dyson 分支规则保持一致，未按实验改选根或调参。

**高分波和正能尾部仍未收敛。**粗基最后 ℓ7→8 的最大跃迁增量为 **{last_l['max_absolute_excitation_change_cm1']:.3f} cm⁻¹**；细基 100→200 Ha 的最大增量为 **{fine_uv['max_absolute_excitation_change_cm1']:.3f} cm⁻¹**。共同 ℓ≤6/200 Ha 的径向加密响应最高 **{radial200['max_absolute_excitation_change_cm1']:.3f} cm⁻¹**。这些是实际已算增量，均不能当作未算无限尾部的误差界。

粗基 ℓ≤8/200 Ha 的实验最大相对偏差为 **{100*coarse['max_experimental_relative_error']:.3f}%**，细基 ℓ≤6/200 Ha 为 **{100*fine['max_experimental_relative_error']:.3f}%**，四能级 5% 目标分别为 **{'通过' if coarse['max_experimental_relative_error']<.05 else '未通过'} / {'通过' if fine['max_experimental_relative_error']<.05 else '未通过'}**。扩空间对 5d 与 6p 的作用不同，不能用统一能量偏移解释剩余误差。数值闭合和实验准确度必须分别报告。

独立电离阈值锚进一步表明：细基模型的基态过束缚 **{-absolute_rows[0]['absolute_residual_cm1']:.2f} cm⁻¹**，5d 上态偏浅 **{absolute_rows[1]['absolute_residual_cm1']:.2f}/{absolute_rows[2]['absolute_residual_cm1']:.2f} cm⁻¹**，6p 上态偏浅 **{absolute_rows[3]['absolute_residual_cm1']:.2f}/{absolute_rows[4]['absolute_residual_cm1']:.2f} cm⁻¹**。所以剩余跃迁偏差同时包含基态和上态的系统偏差，不能唯一归给基态。详见后面的 [NIST ASD 阈值对照]({anchor_contract['sources']['ionization']})；这是当前纯 DC 模型与物理实验的比较，还不是遗漏相关、Breit/QED 等分项的精密误差预算。

本轮修正了诊断 setup 的空间覆盖：完整保留更高能量、更高分波及加密径向表示中的虚轨道，同时减去各自空间的基态。**生产物理源码没有新增修改**；没有发现需据此改符号或改系数的 Coulomb 算符缺陷。不能把空间扩展、标量能量自洽或不同模型的文献修正相加，伪装成已收敛预测。

## 第一性原理与固定 setup

求解 Eₐ=εₐ+Σ₂,ₐ(Eₐ)，其中 Σ₂(w)=Σᵢ Wᵢ/(w−pᵢ)，原生反对称耦合 Wᵢ≥0；2p1h 与 2h1p 两类图全部保留。跃迁 Eγ=E上−E基，所以必须看 ΔE上−ΔE基。固定一套完整极点后，按虚轨道 κ 和能量精确筛选各子空间，每个空间重新求五态根；只去掉严格零耦合，没有幅度阈值或根数截断。

Z=70，经验 Fermi 核；68 电子 [Xe]4f¹⁴ 闭核，19 个占据径向轨道全部参加；五个外部态为 6s₁/₂、5d₃/₂、5d₅/₂、6p₁/₂、6p₃/₂。虚轨道在闭核完整 VN 场中产生，即相对 69 电子 Yb⁺ 的 V^(N−1)，不得再从 68 核移走一电子。两套参考均为已独立通过数值控制的冻结核 DF：粗基 ns=123、细基 ns=230，阶数 ks=9、外盒 R≈60 a₀；细基同时加密实际径向网格，不能将本对照归成单纯 ns 参数误差。

能级与能移表使用 cm⁻¹，参数、残差及权重另标单位。所有计算为纯 Dirac–Coulomb，没有添加 Breit、QED、反冲或超精细项。标量 Dyson 仅迭代外部能量，未重新优化轨道、密度或核 Fock 场，未实现完整 Brueckner/GF2/高阶屏蔽。这些实数附着根不认证此前 Ne 的复自电离共振或宽度。

## 虚轨道数与表示覆盖

下表包含束缚虚轨道及窗口内全部正能虚轨道；数量为实际构造值，不是默认上限。

{counts_table}
在共同 ℓ≤6/50 Ha，径向加密使虚轨道从 861 增至 1616；粗基提高上限至 200 Ha 仅增至 897。这提示能量窗与有限径向谱必须一起检查。**数量增加本身不能证明能量误差大或小，下一节以实际耦合响应判定。**初次细基预检仅构造 <50 Ha 谱，其文件里的“200 Ha 数量”受这个前置过滤限制；当前表采用另行完整构造的 200 Ha 快照及完整捕获，原预检保留并注明范围。

![虚空间容量与共同空间径向响应](representation_and_response.png)

## 三因素的同空间控制

每一行均同时更新全部五态；“径向加密”对照使用各自 DF 核、五态与度量，不混用粗基基态和细基激发态。1 cm⁻¹ 门沿用预声明值，不因结果改变。

{controls_table}
这些是分别固定另两个因素后的条件响应，**不能相加成为误差预算**。高分波增量在不同能量窗有不同值，径向响应也随能量窗变化；只在一个低能窗观察平台会遗漏这种交互。细基 ℓ4→6 的最大跃迁增量仍为 {fine_angular['max_absolute_excitation_change_cm1']:.3f} cm⁻¹，尚未在细基测 ℓ7/8，因此粗基高分波结果不能移植为细基收敛证书。

共同 ℓ≤6/200 Ha 的单态变化如下。DF 与相关项分别列出，避免把参考变化错误归为相关；激发能明确减去同一模型基态。

{radial_table}
## 能级、劈裂与实验

{exp_table}
实验沿用封存的 [NIST Yb II 能级表](https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable6.htm)，不做能量平移或参数拟合。粗基 ℓ8/200 的 5d、6p 劈裂为 {coarse['D_splitting_cm1']:.3f}、{coarse['P_splitting_cm1']:.3f}；细基 ℓ6/200 为 {fine['D_splitting_cm1']:.3f}、{fine['P_splitting_cm1']:.3f}；对应实验为 1371.89、3330.41。基态共同误差不影响这些同多重态劈裂，剩余劈裂误差须在上态相关和算符层面解释。

{shifts_table}
细基 ℓ6/200 中，能量求根相对直接相加自能的最大跃迁响应为 {feedback:.3f} cm⁻¹。它与第三阶图、非对角轨道响应和辐射顶角不是同一修正。文献中的三阶/全阶结果只能帮助选择下一项控制，不能未经同模型验证加到本表。QED 加入后的完整轨道 SCF 本轮没有测量；先验纯 DC 相关与表示，是避免用小修正补偿未闭合主模型的必要步骤。

## 独立文献量级核对

重新读取已封存的 [Safronova 等 2009 年论文 Table I](https://doi.org/10.1103/PhysRevA.79.022512)，比较相同附着相关量的**在壳二阶值**，不将非线性 Dyson 能移混充论文 E₂。

{literature_table}
论文采用 70 个九阶 B 样条、R=200 a₀，二阶 ℓ≤8 并外推更高分波；本轮为有限 R≈60、无外推。上述 17–104 cm⁻¹ 粗基差值支持二阶量级与约定，不能精确归为本模型的缺失尾项。

论文印刷三阶项相对基态的激发能差依次为 −2043、−2317、−681、−850 cm⁻¹。它们说明高阶相关可在当前残差尺度起作用，但直接移植会让 5d 过度修正，且模型并不匹配。仅再次迭代 Σ₂ 也不会生成完整三阶图：冻结参考下使全部 W→λ²W，Dyson 展开为 δE=λ²Σ₂(ε)+λ⁴Σ₂′(ε)Σ₂(ε)+…，只是特定能量反馈项。下一步需原生同模型高阶/非对角响应，不能靠经验常数或文献表值补齐。

## 独立绝对能量锚：区分基态与上态

[NIST ASD 5.12 电离数据]({anchor_contract['sources']['ionization']})给出 Yb II → Yb III 闭核基态的电离能 12.179185 eV、所列不确定度 0.000025 eV（L12907）。沿用 Atomx 单位转换，I={anchor['ionization_cm1']:.5f} cm⁻¹，所列阈值不确定度约 {anchor['quoted_ionization_uncertainty_cm1']:.3f} cm⁻¹；[当前 ASD 能级表]({anchor_contract['sources']['levels']})的四个激发能与本轮封存值一致。将实验和理论都放在 Yb III [Xe]4f¹⁴ ¹S₀ 零点：E基,exp=−I，E上,exp=Eγ,exp−I。没有平移理论能量、修改激发能或拟合阈值。

{absolute_table}
相对误差严格满足 δEγ=δE上−δE基，最大记账闭合误差为 {anchor['maximum_closure_error_cm1']:.3e} cm⁻¹。基态过束缚约 600 cm⁻¹ 是共同贡献，6p 上态还各有约 839/1036 cm⁻¹ 偏浅，故单独修正基态不足以解释当前四条激发能。

这次核对还发现：旧 Handbook 的阈值为 98207 cm⁻¹，2009 论文 Table I 引用的基态束缚锚为 98269 cm⁻¹，分别与当前 ASD 差 {anchor['source_zero_discrepancies']['handbook_minus_current_ASD_cm1']:+.2f}、{anchor['source_zero_discrepancies']['paper_minus_current_ASD_cm1']:+.2f} cm⁻¹。它们不改变原有激发能差或纯二阶增量对照，但不能混为同一个绝对能量基准。旧证据不重写；新绝对诊断明确使用当前 ASD。阈值不确定度不是完整实验能级、同位素/超精细或理论算符误差界。

## 验证、边界与下一项判别

两个完整空间及所有子空间的 Dyson 根均通过预声明无加权极点穿越、DF 连续分支与残差检查；根最大残差 **{max_res:.3e} Ha**。旧粗基在壳重放误差最高 **{old_replay:.3e} Ha**，旧 Dyson 子空间重放误差最高 **{data['gates']['max_Dyson_replay_error_Ha']:.3e} Ha**，门均为 1e−10 Ha。完整虚轨道 Gram 最大误差 **{metric:.3e}**，核空间最大重叠 **{overlap:.3e}**，门均为 1e−7。

两组五态能移均实际注入各自原生 CI；能量匹配误差最高 **{max_ci:.3e} Ha**，CI 残差最高 **{max_ci_res:.3e} Ha**，门 1e−8；所有态匹配权重为 1。该门验证能量记账，不认证新的跃迁矩阵元、寿命或辐射率。生产源码哈希与上一轮 **137 项通过的测试版本完全相同**；本轮未重复未受影响的单元测试，新增证据是两组完整 live 捕获、子空间重放、径向度量和 CI 注入。

后续顺序由剩余门决定：在细径向表示继续扩高分波及正能上限，并检查外盒/有限谱密度；随后用原生同模型计算判别非对角轨道响应和高阶屏蔽，与同 Hamiltonian 的独立方法比较。现有两套径向基、有限能窗和分波上限不足以认证渐近外推、总理论误差或精密实验一致性。

## 复现、资源与文档处置

完整十态捕获耗时 {capture['elapsed_seconds_uncontrolled']:.2f} s，峰值单进程 RSS {summary['max_process_RSS_MiB']:.1f} MiB。六个独立计算工作进程，每进程 BLAS/OMP 一线程；粗基单态上限 1800 s、细基 5400 s，均未超限。记录了实际机器负载，这些时间不是受控性能倍数。原候选项数预估有 1000 项算术笔误，已在 workload_count_check.json 更正；未改变模型、计算窗口或资源上限。

本目录保留契约、捕获源码、全量极点、轨道快照、原始日志、all_models.csv、每态身份、CI 和表示审计。前三次表示审计驱动分别遇到原生二维求积数组与两代 manifest 格式的读取错误，均发生在诊断脚本；失败契约、源码与日志保留，v4 用相同物理输入和容差通过。前两轮封存 manifest 的 **93 / 52 项哈希全部复核未变**。

交付复现包包含完整 Atomx 捕获源码、coarse/fine DF 输入、控制驱动和结果汇总；大体积全量耦合极点 NPZ 留在本地并封存哈希，不混称“可离线即时重放全极点”。从包中源码可重新捕获，按 REPLAY.txt 在独立目录运行，驱动拒绝覆盖旧结果。运行主顺序为 run_controls.py → analyze_controls.py → build_report.py；audit_representation.py 是已封存快照的补充审计。

当前文档入口和验证状态更新到本报告；上一轮自能 API、测试说明及独立 Breit/QED 对照保留为阶段证据，能力边界不变。洞察并入已有“跃迁反馈优先”规则：窗口扫描必须核对实际有限谱，并以完整上下态的共同空间响应判断；没有新增根规则。
'''
    (ROOT/"README.md").write_text(report)
    print("REPORT BUILT",summary)


if __name__=="__main__":main()
