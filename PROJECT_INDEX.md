# Project index: atomx

自动生成的定位地图；先读此页，再查询相关符号/模块，最后读取命中的源码和测试。
不要把完整 JSON 或全部文档加载到上下文；索引不证明物理能力或测试覆盖。

## 按需查询

```sh
python scripts/build_project_index.py --query CILevel
python scripts/build_project_index.py --query numerics/bspline --limit 10
python scripts/build_project_index.py --tests src/atomx/atom.py --limit 10
```

查询只读 PROJECT_INDEX.json，返回路径/行号和有限条匹配；再用 rg 缩小读取范围。
行号属于建索引时的快照；动态导入、重导出及间接测试需另查，静态候选不是覆盖证明。

## 入口

- [AGENTS.md](AGENTS.md)
- [src/atomx/__init__.py](src/atomx/__init__.py)
- [src/atomx/atom.py](src/atomx/atom.py)
- [src/atomx/atom_inputs.py](src/atomx/atom_inputs.py)
- [src/atomx/calculation_spec.py](src/atomx/calculation_spec.py)
- [docs/README.md](docs/README.md)
- [docs/ARCHITECTURE_DESIGN.md](docs/ARCHITECTURE_DESIGN.md)
- [docs/ARCHITECTURE_CAPABILITIES.md](docs/ARCHITECTURE_CAPABILITIES.md)
- [docs/TESTING.md](docs/TESTING.md)
- [docs/AGENT_REFERENCE.md](docs/AGENT_REFERENCE.md)

## 源码分区

| 查询路径 | Python 文件数 |
|---|---:|
| [src/atomx/](src/atomx/) | 14 |
| [src/atomx/angular/](src/atomx/angular/) | 16 |
| [src/atomx/config/](src/atomx/config/) | 5 |
| [src/atomx/corrections/](src/atomx/corrections/) | 18 |
| [src/atomx/data/](src/atomx/data/) | 6 |
| [src/atomx/fitting/](src/atomx/fitting/) | 2 |
| [src/atomx/io/](src/atomx/io/) | 3 |
| [src/atomx/lineshapes/](src/atomx/lineshapes/) | 3 |
| [src/atomx/methods/](src/atomx/methods/) | 54 |
| [src/atomx/numerics/](src/atomx/numerics/) | 80 |
| [src/atomx/physics/](src/atomx/physics/) | 34 |
| [src/atomx/plasma/](src/atomx/plasma/) | 13 |
| [src/atomx/potentials/](src/atomx/potentials/) | 8 |
| [src/atomx/properties/](src/atomx/properties/) | 31 |
| [src/atomx/providers/](src/atomx/providers/) | 5 |
| [src/atomx/reference_data/](src/atomx/reference_data/) | 9 |
| [src/atomx/scattering/](src/atomx/scattering/) | 54 |
| [src/atomx/spectrum/](src/atomx/spectrum/) | 7 |
| [src/atomx/uta/](src/atomx/uta/) | 2 |
| [src/atomx/viz/](src/atomx/viz/) | 6 |
| [src/atomx/workflows/](src/atomx/workflows/) | 5 |
| [src/atomx/xfel/](src/atomx/xfel/) | 19 |

## 其他上下文

- 测试 411，工具/示例 100，文档 560，配置 4 个文件。
- 详细路径、AST 定义/类方法、静态内部导入和内容 SHA 存在 PROJECT_INDEX.json（schema 2）。
- 不含归档目录、source_* 源码快照和运行输出；保留的专题文档仍须按正文识别适用范围。
- 名称匹配无结果时直接 rg；只查询源代码结构，不推断实现状态、正确性或性能。

## 更新

```sh
python scripts/build_project_index.py                 # 当前工作树，含未忽略的新文件
python scripts/build_project_index.py --check         # 检查工作树索引是否过期
python scripts/build_project_index.py --staged --stage # 提交 hook：只读暂存内容，只暂存两份索引
```

提交前由 pre-commit 自动更新；新克隆先运行 .venv/bin/pre-commit install。
暂存模式不读取未暂存源码/文档，不导入 atomx、不运行 SCF；输出无时间戳，相同内容生成相同索引。
索引有未暂存改动时 hook 拒绝覆盖：先核对并暂存/保存这些改动，再重新提交。
文档修改/合并/删除仍按 AGENTS.md 执行，重建项目索引不代替文档语义维护。
