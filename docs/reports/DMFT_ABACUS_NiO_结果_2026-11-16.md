# ABACUS + atomx 自洽 DMFT：NiO 计算结果与方法（2026-11-16）

> 本文档对应代码：atomx main `5609cf9a`（`embedding/abacus_hr.py`、`embedding/dmft.py`、
> `embedding/ed_dmft.py`、`embedding/afm_dmft.py`；测试 `tests/test_dmft_abacus.py`
> A1–A4 + `tests/test_ed_dmft.py` B1–B5，共 29 项全绿）。
> 计算数据：`~/data/dmft/nio_pm/`（顺磁）与 `~/data/dmft/nio_afm/`（反铁磁）。
> 设计层面的定位讨论见 `docs/QUANTUM_EMBEDDING_DESIGN.md` §8 与
> `docs/reports/调研_DMFT生态克隆对比_2026-11-08.md`。

## 1. 一句话

ABACUS v3.7.0 的 LCAO 哈密顿量 H(R) 经 Löwdin 正交化与 d 轨道投影进入 atomx 杂质
求解器，完成 **Hubbard-I 与团簇 ED 两级**的电荷自洽 DMFT；NiO 顺磁基态从 PBE 金属
（带隙 0）变为 ~2.3–3.7 eV 电荷转移绝缘体，AFM-II 解自洽给出 1.80 μB 磁矩。

## 2. 计算链

```
ABACUS SCF (PBE / PBE+U, LCAO)
  └─ data-HR-sparse_SPIN0.csr / data-SR-...  (abacus_hr.read_csr)
  └─ Löwdin 正交化 H(k)                     (abacus_hr.lowdin_hk, 8³ k 网格)
  └─ 每磁量子数 m 取 ζ1/ζ2 自然轨道          → 5 个立方谐函数关联轨道 W
杂质求解（二选一，共享接口 sigma_inf + sigma_poles）：
  ① Hubbard-I：Σ 精确极点展开 → 辅助轨道扩展哈密顿   (dmft.py)
  ② 团簇 ED：d + 投影 bath 轨道（20 自旋轨道），
     N0±1 扇区对角化，Lehmann 表示               (ed_dmft.py)
自洽循环：μ–n_d–E_dc(FLL) 三方自洽 + bath 混合；
  电子计数 N0 精确守恒，GAP = 能带占据数二分处的 vbm/cbm
```

关键实现事实：

- **Hubbard-I（①）**：T=0 精确计数 + 实频谱；非粒子数守恒（见 §5）。
- **ED（②）**：bath 由自洽过程中晶格杂化的**实频投影**拟合
  （`project_bath`，窗口默认 [μ−10, μ] eV），即 `c + V²/(z−ε_b)` 单极点 +
  静态余项；拟合必须加物理边界并用**固定参考 μ_PBE 的 Matsubara 网格**
  （以能隙中点为参考会振荡）。
- **自旋分辨**（`solve_ed(..., spins=(0,1))`）：Lehmann 极点/Σ 极点按 (a, s) 键存，
  是 AFM 解的基础；对称性锚 = test B5（自旋对称输入 → 零磁矩、双通道全同；
  交错劈裂 → ~2 μB 且通道可区分）。
- **AFM-II**（`afm_dmft.py`）：双子格 A/B 由自旋翻转联系，只解一个杂质，
  Σ_B,s = Σ_A,−s；交错种子场只喂前 `seed_its`（=3）轮，撤场后磁矩必须自持续
  （否则会弛豫回顺磁）。

## 3. 结果

### 3.1 顺磁 Hubbard-I（U=8, J=1）

| 量 | 值 | 备注 |
|---|---|---|
| PBE 基态 | 金属（gap 0） | 起点 |
| DMFT 能隙 | **3.68 eV**（实验 3.8–4.3） | μ 位于能隙中时电子数多 0.27（HI 非守恒） |
| U 扫描 | U=6→9：3.40→3.77 eV | 变化平缓 = 电荷转移型（Zaanen-Sawatzky-Allen） |
| k 网格收敛 | 8³ ≈ 11³ | |
| CBM | Ni 4s 带（Γ 点上方 1.8 eV） | PBE 本征缺陷，DMFT 未修正 |
| O p → UHB | 4.82 eV | 与实验电荷转移激发同量级 |

### 3.2 顺磁 ED-DMFT（U=8, J=1，δ = E_dc − FLL 扫描）

收敛判据：N=24 精确守恒（N(mid)=24.001）；各点 11–33 轮收敛。

| δ | E_dc (eV) | 能隙 (eV) | 顶下 2 eV：eg/t2g/Op | VBM 带：eg/t2g/Op | n_d | d⁸/d⁹/d¹⁰ |
|---|---|---|---|---|---|---|
| −2 | 53.12 | 1.04 | 23.7/61.0/10.2 % | 9/65/24 % | 7.815 | 93/7/0 % |
| −1 | 54.76 | 1.83 | 13.6/66.4/14.9 % | 20/25/51 % | 7.901 | 91/9/0 % |
| 0 | 56.14 | 2.33 | 14.1/49.2/29.9 % | 18/14/63 % | 7.952 | 89/11/0 % |
| +1 | 57.47 | 2.70 | 21.3/8.3/60.1 % | 13/10/71 % | 7.995 | 85/14/0 % |
| +2 | 58.87 | 3.01 | 16.2/6.6/66.6 % | 10/8/76 % | 8.050 | 80/19/1 % |

物理读法：

- **价顶 t2g 权重是 DC 模糊的人造特征**：δ 从 −2 到 +2，t2g 61%→7%。单极点
  bath 描述不了 t2g 的真实杂化（其权重 75% 在 +8~+25 eV 的 Ni 4s/4p 高位，
  Matsubara 拟合会把 bath 极点拖到 −10 eV 卡下界——实频投影后 V_t2g=1.20 才物理）。
  δ≥+1 时价顶变为 Op 主导的电荷转移图像（d⁸ 80–85%，能隙 ~3 eV），与实验
  谱学图像一致。
- bath 参数（δ=0，实频投影）：ε_b−μ：eg −2.33 / t2g 高位；V：eg 2.21、
  t2g 1.20 eV（eg 与文献 2.0–2.2 一致）；d⁸/d⁹L = 89/11。
- VBM→UHB 激发 4.62 eV（实验 4.3）；价顶出现 d 主峰（Hubbard-I 缺失此特征）。
- Ni 4s 带悬在价顶上 2.31 eV —— PBE 位置未修正，非 DMFT 问题。

### 3.3 AFM-II ED-DMFT（U=8, J=1）

| 量 | 值 |
|---|---|
| 磁矩（晶格 / 杂质） | **1.80 / 1.89 μB**（实验 1.6–1.9） |
| 自持续性 | 种子场 3 轮后撤场，21 轮收敛磁矩不衰减 |
| d⁸ 占比 | 89.5 %，n_d = 7.91 |

## 4. 复现

```bash
# 数据与运行脚本
~/data/dmft/nio_pm/{run_dc.py,analyze_dc.py,eddmft_dc±N.pkl,dc_scan_summary.pkl}
~/data/dmft/nio_afm/{run_afm.py,afm_U8J1.pkl}

# 测试（必须限 OpenBLAS 线程，否则批量 eigh 慢 ~200×）
OPENBLAS_NUM_THREADS=4 python -m pytest \
  tests/test_dmft_abacus.py tests/test_ed_dmft.py \
  tests/test_embedding_impurity.py tests/test_embedding_bath_e2.py \
  tests/test_embedding_native.py tests/test_abacus_cf.py tests/test_abacus_dftu.py

# ABACUS 侧：qe conda 环境 + mpirun（本机 libmpi 损坏只影响 EDRIXS，不影响此处）
```

已知坑备忘：OpenBLAS 默认 32 线程使批量 eigh 慢 200×；E_dc 扫描 job 需限
`OMP_NUM_THREADS` 否则内存打满 swap 触发 systemd 回收；`analyze_dc.py` 全程
保持 k 分辨 `(k,n)` 形状到最后再展平（布尔掩码轴对齐）。

## 5. 诚实边界

1. **非电荷自洽**：ABACUS 3.7 的 dft_plus_dmft 只解析输入、源码零消费者，
   密度无法回读——DMFT 不修正 DFT 电荷密度，只在其上重算谱。
2. **单极点 bath 的模型局限**：无 Zhang-Rice 单态物理、无双计数内部一致
   方案、无长程序；t2g 价顶成分结论是模型层面的，不是 NiO 的定论。
3. **Hubbard-I 非粒子数守恒**：能隙中点定 μ 时电子数偏差 0.27。
4. **CBM 不可信**：是 PBE 的 Ni 4s 带，任何"带隙内 4s"陈述都来自 DFT 起点。
5. **DC 扫描不是拟合**：δ 是理论不确定度展示，我们没有用实验谱反推 δ 的
   客观判据——给出的是"图像随 δ 的敏感性"，不是"最佳 δ"。

## 6. 开放下一步

- 每立方轨道 2 个 bath 极点（解决 t2g 高位杂化与 ZRS 物理）
- DMFT 自能 → L 边 XAS/RIXS 谱（对接 RIXS 主线的天然出口）
- E3 v2：CT-QMC 杂质求解器（TRIQS 依赖待装；三方对账见 DMFT 生态备忘录）
- PM vs AFM 总能量比较（判断基态相）
