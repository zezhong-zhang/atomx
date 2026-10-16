"""Generate measured tables/figures and replace the running report status."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT/"analysis.json").read_text())
    fine = data["cases"]["fine_l6_e200"]
    coarse = data["cases"]["coarse_l6_e200"]
    labels = ["5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2"]
    all_labels = ["6s_1/2"]+labels
    exp = {s: fine["models"]["3"]["excitation_cm1"][s]-fine["models"]["3"]["experiment_residual_cm1"][s] for s in labels}
    exp["6s_1/2"] = 0.
    anchor = json.loads((ROOT.parent/"yb_correlation_tail_2026_10_08/absolute_anchor/results.json").read_text())
    threshold = anchor["ionization_cm1"]
    absolute = {s: fine["models"]["3"]["attachment_cm1"][s]-(exp[s]-threshold) for s in all_labels}
    results = dict(absolute_residual_cm1=absolute, threshold_anchor=anchor,
                   primary_fine_max_relative_percent=fine["models"]["3"]["max_abs_relative_percent"],
                   source="analysis.json; full ten matched native captures", fitted=False)
    (ROOT/"report_summary.json").write_text(json.dumps(results, indent=2)+"\n")
    # Seven original Yb II branches; energy-derived wavelengths only.
    branches = [("6s_1/2", "5d_3/2", "E2"), ("6s_1/2", "5d_5/2", "E2"),
                ("6s_1/2", "6p_1/2", "E1"), ("6s_1/2", "6p_3/2", "E1"),
                ("5d_3/2", "6p_1/2", "E1"), ("5d_3/2", "6p_3/2", "E1"),
                ("5d_5/2", "6p_3/2", "E1")]
    line_rows = []
    for lower, upper, kind in branches:
        measured = exp[upper]-exp[lower]
        predicted = fine["models"]["3"]["excitation_cm1"][upper]-fine["models"]["3"]["excitation_cm1"][lower]
        line_rows.append(dict(lower=lower, upper=upper, multipole=kind,
                             experiment_wavenumber_cm1=measured, matrix_wavenumber_cm1=predicted,
                             experiment_energy_derived_vacuum_nm=1e7/measured,
                             matrix_energy_derived_vacuum_nm=1e7/predicted))
    with (ROOT/"seven_lines.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(line_rows[0]))
        writer.writeheader(); writer.writerows(line_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
    names = ["5d3/2", "5d5/2", "6p1/2", "6p3/2"]
    x = np.arange(4)
    for n, offset, color in ((1, -.22, "#a2acb8"), (3, .22, "#147eaa")):
        values = [100*fine["models"][str(n)]["experiment_residual_cm1"][s]/exp[s] for s in labels]
        axes[0].bar(x+offset, values, width=.4, color=color, label=f"P={n}")
    axes[0].axhline(5., color="#be4c42", ls="--", lw=1.2, label="5% target")
    axes[0].set(xticks=x, xticklabels=names, ylabel="Excitation residual / experiment (%)",
                title="Matched fine l6 / 200 Ha")
    axes[0].legend(frameon=False)
    response = [fine["attachment_response_from_diagonal_cm1"][s] for s in all_labels]
    axes[1].bar(np.arange(5), response, color="#285573")
    axes[1].set(xticks=np.arange(5), xticklabels=["6s1/2"]+names,
                ylabel="Attachment shift P3-P1 (cm-1)", title="Ground and upper-state response")
    for i, val in enumerate(response):
        axes[1].text(i, val-12, f"{val:.1f}", ha="center", va="top", fontsize=8)
    axes[1].set_ylim(-380, 0)
    fig.savefig(ROOT/"matrix_response.png", dpi=230)
    fig.savefig(ROOT/"matrix_response.svg")
    plt.close(fig)
    text = (ROOT/"README.md").read_text()
    a = text.index("**状态：") if "**状态：" in text else text.index("**已加入")
    b = text.index("## 物理问题", a)
    lead = f'''**已加入完整带符号矩阵 Σ₂，并完成两套完整五态计算及保存矩阵重放。结论是：冻结核轨道响应不能同时修好当前绝对束缚能与谱位偏差。** 细基最大激发能偏差由标量模型的 5.384% 增至矩阵模型的 **{fine["models"]["3"]["max_abs_relative_percent"]:.3f}%**；四能级 5% 目标仍未通过。99 项受影响测试通过，计数为最终物理/架构面板 76 项与 pair/ladder 回归 23 项，不重复计算重检。

6s 附着根下降 **265.428 cm⁻¹**，相对 ASD 阈值的过束缚由约 600 cm⁻¹ 增至 **865.740 cm⁻¹**。上态各自也下降：5d₃/₂、5d₅/₂、6p₁/₂、6p₃/₂ 为 **316.644、253.139、154.487、118.250 cm⁻¹**。减去同模型基态后，四激发能分别移动 **−51.216、+12.289、+110.941、+147.178 cm⁻¹**；只有第一项的谱位偏差减小。数值闭合通过不等于实验改善。

在同一最低分支、共同无极点能区与不变核/Σ₂ 的条件下，外部空间继续扩大只能进一步降低 6s 根，不能将它的过束缚往上修正。下文给出这个 Rayleigh–Ritz/Σ′ 推导。当前优先应改变高阶相关覆盖，随后测试互补粒子空穴与 Coulomb–Breit 交叉项；继续同一冻结核上的低信息迭代不足以回答这些问题。

## 实测五态、谱位与空间门

所有能量及差值单位为 cm⁻¹。

| 上态 | 实验激发能 | 同参考标量 P1 | 矩阵 P3 | P3−实验 | P3−P1 |
|---|---:|---:|---:|---:|---:|
'''
    for s in labels:
        one, three = fine["models"]["1"]["excitation_cm1"][s], fine["models"]["3"]["excitation_cm1"][s]
        lead += f"| {s} | {exp[s]:.2f} | {one:.3f} | {three:.3f} | {three-exp[s]:+.3f} | {three-one:+.3f} |\n"
    lead += "\n![完整五态轨道响应与实验偏差](matrix_response.png)\n\n"
    lead += "| 附着态 | P3 附着能 | 与独立阈值/实验的绝对偏差 |\n|---|---:|---:|\n"
    for s in all_labels:
        lead += f"| {s} | {fine['models']['3']['attachment_cm1'][s]:.3f} | {absolute[s]:+.3f} |\n"
    lead += f'''
绝对比较沿用已核对的 [ASD 5.12 Yb II→Yb III 阈值](https://physics.nist.gov/cgi-bin/ASD/ie.pl?at_num_out=1&biblio=1&e_out=0&el_name_out=1&ion_charge_out=1&spectra=ytterbium&unc_out=1&units=1)：12.179185 eV，即 {threshold:.5f} cm⁻¹；未做实验平移或拟合。上态绝对附着误差均变小，而基态过束缚加重，因此谱线差值未同步改善。这些仍是纯 DC 模型与物理实验之差，不是单项遗漏物理的精密预算。

| 控制/门 | 5d₃/₂ | 5d₅/₂ | 6p₁/₂ | 6p₃/₂ | 1 cm⁻¹ 门 |
|---|---:|---:|---:|---:|---|
'''
    increments = fine["P2_to_P3_excitation_cm1"]
    radial = data["radial_response_P3_excitation_cm1"]
    lead += "| 细基 P2→P3 | "+" | ".join(f"{increments[s]:+.3f}" for s in labels)+" | 未通过 |\n"
    lead += "| P3 粗→细径向表示 | "+" | ".join(f"{radial[s]:+.3f}" for s in labels)+" | 未通过 |\n"
    split = fine["models"]["3"]["splitting_cm1"]
    lead += f'''
细基 5d/6p 劈裂为 **{split["d"]:.3f}/{split["p"]:.3f}**，实验为 1371.89/3330.41。基态共同偏差无法解释这些劈裂。粗基同 ℓ6/200 矩阵最大实验偏差为 {coarse["models"]["3"]["max_abs_relative_percent"]:.3f}%；粗细结果一致不认证完整外部函数、高分波/正能或盒尾收敛。P2→P3 的最大激发能增量仍为 **32.274 cm⁻¹**，不能当作无限尾部误差界。

七条原始 Yb II 分支的能量与真空波长算术对照另见 [seven_lines.csv](seven_lines.csv)。波长由对应能级差计算，未计算新的相关辐射矩阵元、强度或寿命；E1/E2 选择规则保持原分支声明。

## 数值、保存矩阵与封存控制

完整十态捕获耗时 **{data["elapsed_capture_seconds"]:.2f} s**，峰值单态 RSS **{data["peak_state_RSS_MiB"]:.1f} MiB**，均低于声明上限。五个计算进程、每个 OMP/BLAS 一线程；实际负载保留，未声明受控的端到端加速倍数。首次未完成逐对计算的证据另外保留。

| 控制 | 最大实际误差 | 预设门 |
|---|---:|---:|
| 旧对角在壳/根重放 | {data["maximum_old_diagonal_replay_error_Ha"]:.3e} Ha | 1e−10 Ha |
| 原生矩阵本征残差 | {data["maximum_matrix_eigen_residual_Ha"]:.3e} Ha | 1e−10 Ha |
| 保存极点矩阵的向量残差重放 | {data["maximum_saved_matrix_vector_residual_Ha"]:.3e} Ha | 1e−10 Ha |
| 保存矩阵的 Z 重放 | {data["maximum_saved_matrix_residue_replay_error"]:.3e} | 1e−10 |
| 混合径向函数度量 | {data["maximum_mixed_orbital_metric_error"]:.3e} | 1e−7 |
| 混合函数与闭核重叠 | {data["maximum_core_overlap"]:.3e} | 1e−7 |

两套每个实际虚块均重新用原生提供器生成，逐系数与预检一致。所有候选项数核对通过；两类图没有幅度裁剪。前四轮 **93/52/127/21** 个封存文件的 SHA 全部保持，生产源码在最终捕获中不变。[analysis.json](analysis.json) 是本轮数值和验证依据，[comparison.csv](comparison.csv) 包含 30 个匹配态/外部空间记录。

捕获后恢复了旧标量入口原有的 `np.asarray/int/float` 输入转换，避免 NumPy float32 外部能量改变旧默认精度。捕获时的 30 个实际外部函数均已是 ndarray/float64、Python float 与 int，这些转换逐一核实为恒等；两图、积分、虚态、矩阵与求根实现没有变化。生产与最终源码 SHA 分开保存于 [源码身份链](final_source_chain.json)，不声称两份源码逐字相同。最终源码通过 76 项面板及全部保存矩阵的根向量/残重重放。

'''
    text = text[:a]+lead+text[b:]
    (ROOT/"README.md").write_text(text)


if __name__ == "__main__":
    main()
