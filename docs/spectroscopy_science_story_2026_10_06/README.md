# 原子、分子、晶体谱学与未来产业研究体系

此目录是本主题的当前框架文档：宏观科学问题 → 12 个相互承接的案例 → Atomx 软件基础 → 光源与产业验证。此前聚焦 Ne 的[参赛项目书](../agentic_science_beijing_light_source_2026_10_06/README.md)保留为专题支持材料；本框架取代其作为当前广义故事线入口，不改写它的历史证据。

- [完整文档](研究体系与案例设计.md)、[PDF](研究体系与案例设计.pdf)、[可编辑 Word](研究体系与案例设计.docx)。
- [故事线摘要](故事线摘要.md)、[摘要 PDF](故事线摘要.pdf)。
- [12 个案例登记表](case_registry.json)：由正文推导的六字段故事与 setup 参数、单位、拟议阈值。不是通用 Atomx 输入 API。
- [三个可运行基础例子](run_foundation_examples.py)、[预注册软件契约](foundation_results/contract.json)、[执行结果](foundation_results/checks.json)。
- [Ne X 输出](foundation_results/ne_x_ground.json)、[CO provider 输出](foundation_results/co_provider.json)、[MgO 嵌入输出](foundation_results/mgo_embedding.json)、[入口源码身份](foundation_results/source_identity.json)。
- [文档构建脚本](build_document.py)、[登记表构建](build_registry.py)、[交付核查](verify_document.py)。
- [离线参考快照说明](references/README.md)：所查能力指南与晶场约定的有来源副本；仓库原文件仍是当前权威入口。

## 科学与软件状态

12 个完整案例为计划，尚未计算完成；所有误差目标是拟议研究标准。三个基础例子只验证受声明约束的接口、电子数、度量与芯身份，没有完成 XAS、QED、周期材料或实验验证。文档中的旧 Ne 诊断链接指向独立封存档案；本文未修改封存源、输入或缓存。

本次运行用仓库现有 `.venv`，单 BLAS 线程。审查了实际入口源码与官方方法文档，未修改科学求解代码。所审入口哈希不代表完整依赖源码快照；生产回放需进一步封存全部源码、输入与外部数据。

## 源码与能力依据

| 基础 | 当前源码或规范 |
|---|---|
| 原子公共输入/求解 | [atom.py](../../src/atomx/atom.py)、[atom_inputs.py](../../src/atomx/atom_inputs.py) |
| provider 契约 | [contract.py](../../src/atomx/providers/contract.py)、[能力指南](../ARCHITECTURE_CAPABILITIES.md) |
| 分子数据与芯 ΔSCF | [pyscf_adapter.py](../../src/atomx/providers/pyscf_adapter.py) |
| 有限点电荷嵌入 | [pyscf_embedding.py](../../src/atomx/providers/pyscf_embedding.py) |
| 分子 Auger | [molecular.py](../../src/atomx/scattering/autoionization/molecular.py) |
| 晶场约定 | [CRYSTAL_FIELD_SPEC.md](../CRYSTAL_FIELD_SPEC.md) |
| 非共振原子散射 | [xray.py](../../src/atomx/scattering/xray.py) |

## 回放

在与对应源码一致的 Atomx 仓库根目录执行软件演示：

```bash
.venv/bin/python docs/spectroscopy_science_story_2026_10_06/run_foundation_examples.py
.venv/bin/python docs/spectroscopy_science_story_2026_10_06/build_registry.py
.venv/bin/python docs/spectroscopy_science_story_2026_10_06/verify_document.py
```

文档渲染使用独立工具环境，不要求把渲染依赖装进物理 `.venv`：

```bash
uv run --no-project --with markdown==3.10.3 --with weasyprint==70.0 --with python-docx==1.2.0 --with cairosvg==2.9.1 python docs/spectroscopy_science_story_2026_10_06/build_document.py
```

公开 ZIP 提供文档、登记表、图、脚本、实测软件 JSON 与来源信息；它不包含 Atomx 完整源码、第三方论文全文或未获得的实验数据。解压后可直接阅读及执行文档一致性检查；运行物理接口例子须另有匹配的 Atomx 源与依赖。正文中的 `../` 与 `../../` 链接保持仓库导航含义，离线 ZIP 使用 `references/` 中的有来源快照；封存 Ne 证据与源码需在原仓库查看。
