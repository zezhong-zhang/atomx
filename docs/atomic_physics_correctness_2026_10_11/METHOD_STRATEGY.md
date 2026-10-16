# 方法定位：以 Atomx 原生实现为主线

2026-10-11 用户明确裁定：原子物理和求解器在 Atomx 内实现；其他软件用于
参考方程、比较物理、精度与效率，不能以外部后端接入作为完成目标。
当前主线按用户随后裁定为
[Atomx 原生 MCDHF→RCI 与 Yb⁺ 实算](../yb_mcdhf_rci_2026_10_11/README.md)。
Ne/Eu 成功基线和此前独立方法证据见
[原生原子物理路线](../atomic_native_roadmap_2026_10_11/README.md)。

本页上一版漏掉 Ne 后期完整 3200 Ha 原生 CI 和 Eu 原生标定多重态成功记录，
仅列早期有限实例不足以判断 Atomx 水平。修订前原文与源审计保存在
[前版快照](../atomic_native_roadmap_2026_10_11/previous_sources/manifest.json)；
早期数值仍是其原模型的证据，不作为最佳实践的代表。

当前大空间 Yb CoreSD → ReducedMonovalentSD 仍实际调用 pCI-derived 图项库，
fscc.compute 仍实际执行 EXP-T。既有比较与缺陷修复证据保留；这描述当前依赖，
不是未来目标，也不把依赖重新标为全链原生实现。

MCDHF→RCI 为原子结构开发主线，Yb⁺ 五态示例采用真实共同态轨道与 CI 优化；
先解决驻值、投影与平衡核价/空穴物理，再按谱线差分的重要性推进空间。
Ne 成功 RCI 与 Eu 多参考 CI 基础继续复用。约化 SD、非线性 CCSDT/FSCC
保留为独立方法对照及后续补充，不强制所有元素采用同一 CC 方案。共享径向、
角代数、signed 积分、态身份与有效算符；CPU/GPU 实现遵守相同物理合同。
原方法阶数、p 核低阶和非正则方程证据仍限定已记录模型，不能转移为新谱
或普适最优方法证明。
