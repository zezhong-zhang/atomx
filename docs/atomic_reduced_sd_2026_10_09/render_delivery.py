"""Concise standalone scientific artifact from final recorded measurements."""
import json
from pathlib import Path
import markdown
from weasyprint import HTML
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'final_yb_scalar_plan.json').read_text())
a=json.loads((HERE/'final_pci/pci_angular_validation.json').read_text())
n=json.loads((HERE/'final_native/native_results.json').read_text())
text=f'''# Atomx：原生角动量约化核 SD 实现与验收

2026-10-09 · 参照 pCI / AMBiT / EXP-T 的分阶段提升记录

**已完成有限原生 J=0 核的 linked 线性化 SD 方程与完整径向池的精确标量预检。大空间直接收缩、价电子有效算符和 Yb 光学精度仍未完成，尚不能称最高水准。**

## 实际新增

`methods.linearized_sd` 求解 `Q(H+[H,T1+T2])|Φ₀⟩=0`，保留非正则轨道的单激发乘积减项，复用共享原生 DC / 完整静态 DCB 算符；冻结 NR 壳仍贡献 Hamiltonian 和能量零点。独立角动量计数核对完整 J=0 SD 空间，超限整体拒绝。

这是独立的线性化核 SD 入口，和 CISD 对角化、EXP-T 非线性 CCSD/FS-CCSDT 各有明确方程与适用范围。默认有限锚为512 CSFs、2000磁行列式、方程矩阵工作估算128 MiB，不自动接入 Atom.solve()。

## 完整 Yb 轨道池规模

保留原生 Fermi V68 / ns123、ℓ≤3、虚能量<5 Ha：68核电子、19占据径向键、414虚径向键、1894虚磁自旋轨道。

| 表示 | 独立核 SD 坐标 |
|---|---:|
| 未利用磁/旋转对称性 | {p['unrestricted_coordinates']:,} |
| 只利用总 M 与宇称 | {p['magnetic_symmetry_coordinates']:,} |
| 完整旋转标量 | {p['scalar_amplitudes']:,} |

约化坐标约310倍；单个实振幅向量约100.44 MiB。**这不是速度提升或迭代峰值内存证明。** 稠密方程仍不可行，后续必须直接约化张量收缩，保留全部坐标并逐项回到独立小空间基准。

## 正确性证据

相关面板 **56项通过**，lint通过。独立完整 Fock 交换子覆盖三/四电子、非正则轨道、随机同秩旋转和参考相位；非相互作用片段能量可加性与弱耦合 MP2 极限通过；完整 jj-CSF 枚举与径向对独立计数一致。

原始 pCI Fortran 角向函数实跑 {a['rows']} 个组合，最大差 {a['maximum_error']:.3e}，通过1e−12门。相位映射：`S_pCI=(−1)^(|κa|+|κb|+k−1) C_Atomx`。此证据只核对角向约定，没有认证完整 pCI 积分/方程适配。

| 原生有限模型 | 总能量 / Ha | 方程残差 / Ha |
|---|---:|---:|
'''
for row in n:
    text+=f"| {row['atom']} {row['tag']} | {row['total_energy_Ha']:.12f} | {row['equation_residual_Ha']:.2e} |\n"
text+='''
Be 使用1s闭核→2s/2p、点核ns30；Ne使用点核ns40、每κ第一虚根。Ne池是扩散盒态，相关能极小，仅检验消费/冻结覆盖，不是相关收敛或天体谱线基准。数值残差不代替空间收敛和实验精度；本轮没有新 Yb 光学谱位，前序实验偏差结论保持。

## 推向高标准的验收路径

直接约化核 SD 收缩 → 同原生积分的实际 pCI 核 SD 再现 → 明确 P/Q 的Σ₁/Σ₂及平衡4f空穴 → 同模型上下态、Breit/QED/反冲、dressed跃迁与物理极点。逐项验证相位、对归一化和真实振幅残差；保持1 cm⁻¹空间与1%光学精度门，不用实验移位替代缺失物理。

原始失败日志保留：不成立的 Be 最小方法差值断言、零占据通道元数据、已有 Numba 缓存弱引用失效均有具体处置。独立新缓存完成回归，没有降低物理门限。源码身份、完整方程、独立角向结果和重放入口随源码包提供，旧封存不改写。

参考：pCI，CPC 308 (2025) 109463；Porsev–Safronova–Kozlov，PRA 86 (2012) 022504。当前 API：docs/ATOMIC_LINEARIZED_SD.md；完整证据：docs/atomic_reduced_sd_2026_10_09/README.md。
'''
(HERE/'delivery_report.md').write_text(text)
css='''@page{size:A4;margin:15mm 16mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.48;color:#142333}h1{font-size:19pt;color:#12324c}h2{font-size:12.5pt;margin:12pt 0 5pt}table{border-collapse:collapse;width:100%;font-size:9pt;margin:8pt 0}th,td{padding:5pt;border-bottom:1px solid #dae2e8}th{background:#eaf0f4}code{font-size:8.5pt;overflow-wrap:anywhere}p{margin:6pt 0}strong{color:#124461}'''
html='<meta charset="utf-8"><style>'+css+'</style>'+markdown.markdown(text,extensions=['tables','fenced_code'])
(HERE/'Atomx_原生约化核SD实现与验收.html').write_text(html)
HTML(string=html,base_url=str(HERE)).write_pdf(HERE/'Atomx_原生约化核SD实现与验收.pdf')
