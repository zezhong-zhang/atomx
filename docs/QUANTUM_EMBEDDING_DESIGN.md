# Quantum Embedding in atomx — Design (v1 草案)

> 定位：atomx 成为生态中的**多重态级杂质求解器**（multiplet-grade impurity solver）。
> 生态分工：ABACUS（DFT/+U/DMFT 工具链）出晶格侧与杂化函数 Δ(iω)；atomx 出杂质侧——
> 用其理论阶梯（Dirac-Fock → MCDHF/CI → MBPT/全阶）控制杂质壳层的原子物理精度，
> 输出光谱级（RIXS/XAS 可用）的杂质格林函数与自能。DMFT 自洽循环驱动器在两码之外，
> 用薄脚本粘合；v1 先交付**无循环的一击式（Hubbard-I / DFT+atomic）模式**，立即有价值。

## 1. 为什么是 atomx（差异点）

DMFT 求解器生态有 CT-QMC/NRG/ED，但都把杂质壳层当"Kanamori + 微扰修正"处理；
光谱应用（L/X 边 RIXS、深层能谱）需要的是**完整多重态结构**（Slater 全套 + SOC + Breit
+ 相关修正）——这正是 atomx 的核心能力。Hubbard-I 极限 = 纯原子分辨，是 atomx 现有
机械（能级 + 跃迁算符矩阵元，`_sos_basis`/transitions 的同一套代数）的直接重排。
原子极限内的 Breit/QED/相关阶梯——别家求解器没有的维度——成为本方案的独特卖点。

## 2. 模块设计

```
src/atomx/embedding/
  impurity.py   # Lehmann 求解器：G_imp(iω/ω) = Σ_g w_g Σ_n |⟨n|ĉ|g⟩|² / (z − (E_n−E_g))
                # 原料 = 现有 CI 本征问题 + 跃迁算符矩阵元；输出含谱矩守恒校验
  bath.py       # 杂化表示：(a) 离散 bath 能级——CI 空间自然扩展（点模型/团簇/SIAM
                #   的统一超集，配体场团簇即其特例）；(b) 连续 Δ(iω) 表——v1 仅插值，
                #   杂化微扰修正走 MBPT 模式（后续）
  io.py         # 负载 schema `impurity-greens@1`（hyperx 契约已预留名）：
                #   G(iω)/Σ(iω)/A(ω) 网格 + provenance（阶梯层级、bath 规格、收敛态）
workflows/dmft_loop.py  # 薄驱动器（v1 只做一击式；自洽迭代为 v2）
```

关键不变量：杂质本征问题完全复用现有 CI 机械（CSF 空间、Slater/Breit 算符、
阶梯选择）；embedding 层只新增 bath 轨道簿记与 Lehmann 装配，不重造原子物理。

## 3. 与 hyperx/ABACUS 的接口

- 入：`cluster-parameters@1`（ABACUS 参数）或直接 Δ(iω) 表；
- 出：`impurity-greens@1`（按 hyperx 契约的负载规则：JSON 可序列化、provenance
  必填、禁止静默默认）；
- 消费：hyperx 的团簇/SIAM 谱链（同一求解器服务两条线：光谱端 RIXS、关联端
  DFT+DMFT 谱函数）。

## 4. 验证阶梯（anchor-first，先锚后扩）

| 级 | 锚 |
|---|---|
| L0 原子极限 | G 极点=能级；谱矩求和规则（粒子数、⟨H⟩）精确；与 transitions 线强一致 |
| L1 离散 bath | SIAM 原子极限解析式（电子-空穴对称特例闭式）；与 hyperx 自研 2v1c 链交叉互证（该链已对 EDRIXS r=0.9997） |
| L2 基准 AIM | 与公开 CT-QMC 基准（TRIQS benchmark 集，β/U 网格）在杂质可观测量上对账 |
| L3 物理 | NiO：HI 自能版 d 谱/XAS vs 单离子基线——可测改进主张 |

## 5. 非目标（v1）

- 不在 atomx 内做 CT-QMC（生态已有求解器；atomx 拥有原子/多重态+光谱生态位）
- 不在核内做全 DMFT 自洽循环（驱动器外置）
- 不做连续 bath 的精确对角化（v1 离散 bath + 插值连续）

## 6. 开放问题（设计即标注，不静默）

- 双计数（double counting）记账：v1 显式选择并在 provenance 声明，不声称解决
- bath 离散化对光谱的伪影：需要收敛性扫描工具（L1 级锚附带）
- 相对论杂质（Dirac CSF）× bath 的轨道对称簿记：v1 限定标量 bath 轨道

## 7. 里程碑

- **E1**：`impurity.py` 原子极限 + 谱矩校验 + `impurity-greens@1` 负载（小；全复用）✅ 已交付（含 E1.5 DF 原生参数：F2/F4 vs Cowan 0.04%）
- **E2**：离散 bath = 团簇统一超集 ✅ 已交付——hyperx NiO 电荷转移团簇作为 ClusterSpec 配置实例，**120 能级跨仓逐位一致（平移 0.0、残差 0.0）**；L1 非相互作用闭式（杂化二能级）通过；调试过程抓出镜像直项传参错与第三处旧自旋检查残留
- **E1.5+（CSF 原生装配，暂缓）**：验收标准已由 E2 锚定义——CSF/jj 装配必须复现同一 120 能级谱（当前 LS 装配已与 hyperx 机器精度互证）。相对论 CSF 空间的置换簿记是独立研究量级，且当前物理收益有限（3d 壳相对论修正小）；待有 jj 分辨可观测需求时启动
- **E3**：驱动器 + L2 CT-QMC 基准对账 → 一击式 DFT+atomic 光谱成品
