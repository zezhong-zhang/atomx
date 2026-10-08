# Agent 文档维护、记录与蒸馏机制

目标：让下一位 agent 能找到证据、复用经验、识别边界，并在新证据出现时修正行动。
本机制由根 [AGENTS.md](../AGENTS.md) §6 要求执行；文档语义维护与蒸馏由 agent 执行，
项目结构索引通过提交 hook 自动重建，二者的验证范围不同。
记录规模随风险和工作量变化，避免归档工作挤占判别实验。
每个代码变更增量都包含对应文档的影响检查和处置；新增报告不代替维护当前说明。

## 1. 三层各存什么

| 层 | 位置 | 内容与责任 |
|---|---|---|
| 证据 | 已有所属 `docs/<topic>/README.md` 与其运行附件；新主题按现有命名约定建目录 | 执行任务的 agent 记录问题、契约、原始结果、验证、失败和局限；这里是科学事实的来源 |
| 洞察 | [AGENT_INSIGHTS.md](AGENT_INSIGHTS.md) 的紧凑登记行，详细说明放所属专题 docs | 将证据转成有范围的行动建议，标明依据、状态、失效条件及规则去向 |
| 哲学与规则 | 根 [AGENTS.md](../AGENTS.md) | 跨任务的判断原则、必须做、禁止做；只保存会改变行动的短规则 |

[AGENT_REFERENCE.md](AGENT_REFERENCE.md) 负责按主题导航，不再复制完整结论。
`trace.md` 是被忽略的会话账本，记录一段摘要和持久证据链接，不是长期唯一存储。
科研封存目录保持只读；当前指南、活动主题说明和重复旧指南可修改、合并或删除。
对封存主题新增证据时建续篇，在索引和洞察中链接续篇与旧证据，勿覆盖旧文件。
当前用法和能力以 [文档总览](README.md) 的当前指南为入口；它们不应被逐轮报告替代。

## 2. 何时做、做到哪一步

| 触发时机 | 必须完成的动作 |
|---|---|
| 开始相关任务 | 搜索主题索引及洞察 ID，读当前结论、适用条件与勘误；核对依赖是否变化 |
| 每个代码变更增量 | 按文档总览的代码映射确定受影响说明，执行本页 §6 的修改/合并/删除/保留流程 |
| 开始判别实验 | 写假设、固定条件、对照、预设判据和停止条件；可先写入活动任务记录 |
| 获得改变判断的结果 | 及时记录实际测量与失败；避免只保留终端输出或最成功的一次运行 |
| 发现新约定、复现 bug、证伪归因、用户裁定 | 更新持久任务记录，并新增/修订候选洞察；未知原因明确写未知 |
| 阶段完成、提交准备或会话交接 | 蒸馏并决定晋升/保留/撤回；更新导航和受影响规则，核对证据可找到 |

“增量”是一组可审查的逻辑变更，不是每次保存文件；提交或交付前代码和文档必须一致。
修改代码的 agent 负责文档处置，不能把已知过期内容推给未指定的后续维护者。
不要求每次工具调用产生日志。小修订可在既有任务文档用一段完成记录；
纯格式/文案变更记录文件、检查及“无新增洞察”即可，不生成研究报告。
时间不足时先保存最小记录（契约、结果、证据、未决项）；候选可交接，不能假称验证。
上述维护在用户授权的仓库任务内直接执行，无需为每次记录另行请求批准。
未请求的提交、发布或改写用户物理目标不由本机制授权。

## 3. 最小证据记录模板

复制到所属任务文档并删去不适用项。使用稳定的记录 ID，例如 `2026-10-05-ci-cache-01`；
同主题并行工作选不同后缀，修改公共索引前重新读取，合并他人的条目。

```text
记录 ID / 日期 / 主题：
问题与假设：要区分什么？出现何种结果时停止？
契约：源版本或实际 diff/SHA、输入/seed、模型/态/单位/归一化、固定与改变项。
实验：实际命令或脚本路径；对照、预设判据、需要的环境与资源。
结果：测量值和单位、阈值、pass/fail；失败/负结果同列。未测写“未测”。
证据：可持久读取的输入、输出、日志和校验信息链接。
解释：观察到了什么；推断是什么；排除了什么；仍不能说明什么。
行动：改了什么、如何验证、回归范围、下一最小判别实验/停止原因。
文档处置：受影响代码/行为 → 已核对文档 → 修改/合并/删除/归档/保留及理由 → 验证。
蒸馏：洞察 ID 与状态/修订，或“无新增洞察：原因”。
```

脏工作树不能只记 HEAD 冒充实际源码。实验记录所消费的源与输入；重放敏感任务
保存实际数组、基底、算符和度量。普通文档任务无需保存整套生产源码。
证据先复用既有持久文件；大原始数据可用稳定存储位置加校验和，不能只链接 `/tmp`。
记录测量所用的真实判据；改变判据须另起显式契约，不能回写旧 pass/fail。

## 4. 蒸馏方法与状态

每次蒸馏先检索是否已有同义洞察，更新已有 ID，避免“每轮一个新原则”。
将结果写成一句行动规则，再补足适用范围与反例：

```text
洞察 ID / 标题：稳定语义 ID，例如 convention-first。
类型：实证经验 / 解析或实现契约 / 用户决策。
状态：candidate / validated / adopted / superseded / withdrawn。
触发条件 → 必须做或禁止做 → 原因：
适用边界 / 依赖 / 何时需重新验证：
依据：证据记录链接；实证列测量，决策列用户裁定日期及内容。
反例与局限：单面板、非独立对照、尚未解决的解释。
去向：专题指南 / 测试或契约 / AGENTS.md 对应章节；或暂不提升及原因。
修订：日期、原结论、修正理由、supersedes 链接（如适用）。
```

| 状态 | 含义及允许动作 |
|---|---|
| candidate | 只有观察或待验解释；可指导下一实验，不得表述成已验证事实 |
| validated | 声明范围内有可审查支持；仍可能只适合专题 docs |
| adopted | 已进入指定指南、测试契约或 AGENTS.md；必须记录具体去向 |
| superseded | 被更精确的结论替代；登记替代 ID/修订链接，当前入口指向新结论 |
| withdrawn | 原结论不再可依赖且无替代；保留证据与撤回原因，删除当前行动要求 |

常见路径为 candidate → validated → adopted；任何状态可被替代或撤回。
明确用户决策可直接 adopted，但只能作为决策依据，不伪装成物理或性能实测。
状态是维护标签，不增加科学认证范围。重复运行同一实现不算独立验证。

## 5. 何时提升到 AGENTS.md

同时满足以下条件才提升，否则留在专题 docs 或候选登记中：

1. **有依据**：有独立锚/机制解释/可重放判别证据，或用户明确裁定。
   一次严重失败也可支持有边界的防错规则，不用凑重复次数；不得外推单面板数值。
2. **有长期价值**：影响多个任务，或保护高成本的公共物理/数据契约。
   某次耗时、当前状态、下一轮队列及未归因偏差不满足条件。
3. **能执行**：写明触发条件及必须做/禁止做；哲学说明判断优先级，避免空泛口号。
4. **不重复、不冲突**：合并现有规则；不静默改用户目标、默认模型、门限或授权范围。
5. **可追溯且预算内**：洞察登记具体规则去向与证据；AGENTS.md 保持 ≤150 行、
   ≤12 KiB UTF-8。超限压缩同义项或下沉领域细节，不能通过超长行绕过预算。

当前 agent 负责依据审查并直接维护规则，不要求额外 agent 或例行用户审批。
若证据不足就保留 candidate；若与明确用户裁定冲突且无法据上下文解决，
记录冲突并提出聚焦问题，继续不依赖该裁定的工作。
可编码的科学不变量优先落成适当测试/接口校验，规则链接其契约；
不要仅为证明文档措辞或低影响可逆修改编写测试。

## 6. 代码变更驱动的文档维护

### 影响定位

1. 以本任务开始时的工作树为边界查看实际修改、删除、改名及新增文件。
   核对暂存/未暂存和本任务未跟踪文件；不把他人的在途改动混入本次维护。
2. 从 [代码→文档入口](README.md) 找候选指南，再用 `rg` 检索旧/新符号、
   CLI 参数、配置键、路径、默认值及相关行为描述；同时检查 docstring、注释、
   README、示例、能力/限制、设计/迁移说明和测试中的可执行声明。
3. 映射只是起点，搜索没有命中不代表没有影响。核对变更前后的可观察行为、
   数据格式、物理近似和用户步骤；未映射模块必须手工确定所属文档并补上入口。
4. 文档与代码不一致时，先判断是过期说明还是代码违反已批准契约。
   后者修代码或明确处理目标变更，不能通过改文档将 bug 合法化。

### 处置规则

| 发现的情况 | 同一增量中的动作 |
|---|---|
| 行为、接口、默认值、单位、范围或限制改变 | 原位修改对应当前说明和示例；涉及兼容性时更新迁移说明 |
| 已实现原 TODO/计划 | 将当前计划改为实际支持与边界，移除已失效待办，不留“仍未实现” |
| 多篇当前指南重复同一契约 | 选一个权威入口，合并仍有效且独有的内容，其他改为必要导航或删除 |
| 功能/命令被移除，旧教程已无独立价值 | 删除教程或失效章节，更新入口、入链、测试引用；必要时保留简短迁移说明 |
| 旧设计只剩历史价值 | 有独有决策依据时归档并注明源版本/替代入口，否则删除；不默认全部归档 |
| 独有科学结果、失败证据或可重放封存 | 保留原件，当前指南去掉适用性误导，勘误与新结果链接旧证据 |
| 内部重构且对文档所描述的契约确无影响 | 记录核对过的具体文档/声明、依据和验证，保留内容；不为满足流程改日期 |
| 确实没有覆盖新能力的合适文档 | 才新增指南，同时指定权威入口、依赖及验证方式，避免新的重复来源 |

普通过期指南可依赖 Git 历史追溯；未入版本控制的独有证据不能假设 Git 已保存。
只有明确封存的科研资料享有不可改写保护，文件较旧或带日期不自动成为保留理由。
删除/合并前读取内容和入链；改名、删除后搜索旧路径/符号，区分需要修复的
当前引用与合理保留的历史叙述。迁移说明应写可用替代或明确无替代，不无限保留空壳页。
自动生成文档修改生成源并重新生成，不手改产物；生成脚本变化也检查对应说明。

### 依赖和核对记录

每个当前主题指定一个权威入口，正文由它维护，其他页链接而不是复制。
本次触及的当前指南若缺依赖信息，补一个简短维护注释或记录：

```text
类型：current / proposal / historical（按内容用途，不按文件日期）。
依赖：源路径及关键符号/配置；验证：相关测试、可执行示例或检查命令。
本次核对：日期 + 实际源版本/diff/输入身份 + 核对了哪些声明及结果。
替代关系：合并/移除的旧入口及新入口（如有）。
```

依赖变化就重新核对对应声明；修改核对日期不是验证，历史测试通过也不保证新代码正确。
无需每次扫描所有归档或强制改无关文档；按变更影响向依赖和入链扩展。
同一范围只维护一份简短处置表，放在已有任务记录/变更说明中即可：

| 代码/行为变更 | 核对的文档/章节 | 动作与理由 | 验证结果 |
|---|---|---|---|
| 填实际改动 | 填实际路径及声明 | 修改 / 合并 / 删除 / 归档 / 保留 | 命令/测试/人工核对；未测项明确 |

该表允许“保留”，不接受无依据的“无文档影响”；所有受影响的当前说明必须闭合。
代码/配置/测试门变更应连同对应文档一起交付；小修订可以一行记录完成。
已知错误当前说明尚未修正时不能宣称任务完成。用户明确要求分阶段时，
在当前入口标明范围和待办，交接真实未完成状态，不冒充已同步。

### 修正与遗忘

- 发现反证时先标明当前结论的冲突/适用失效，勿等完整报告才修正行动指导。
  同一增量更新任务说明、洞察状态、索引及 AGENTS.md 中受影响的规则。
- 原始数据和已封存报告不变；新记录写出“此前判断 → 新证据 → 新结论”，
  给出旧记录链接。修正文档不是删除失败，也不表示所有旧测量无效。
- 新源/模型/输入只令相关证据需要重验，不自动推翻无关洞察。
  索引指向最新适用结论；不能仅按日期覆盖不同物理契约。
- 登记簿每条只保留紧凑当前摘要，详细修订放专题页。主题条目较多时按主题拆页，
  根登记簿保留导航；不要求 agent 每次读全库，也不全量回填旧历史。

### Project index：为 agent 按需读取上下文

[PROJECT_INDEX.md](../PROJECT_INDEX.md) 是精简代码地图，
[PROJECT_INDEX.json](../PROJECT_INDEX.json) 是程序查询的结构数据，不供全文加载。
它们替换旧的手写项目索引；保留原文件名，JSON 使用显式 schema 2。
生成器不导入 atomx；从源码 AST 提取定义、类方法、行号和内部静态导入，
索引测试、工具、当前可读取文档与配置。不索引 archive、source_* 快照及输出目录。
总览限制在 8 KiB；其余符号通过有数量上限的查询读取，无需为每个模块新增索引页。

```sh
.venv/bin/python scripts/build_project_index.py --query CILevel --limit 10
.venv/bin/python scripts/build_project_index.py --query numerics/bspline --limit 10
.venv/bin/python scripts/build_project_index.py --tests src/atomx/atom.py --limit 10
.venv/bin/python scripts/build_project_index.py          # 工作树预览，含未忽略的新文件
.venv/bin/python scripts/build_project_index.py --check  # 检查当前工作树索引
.venv/bin/pre-commit install                            # 每个克隆安装一次 hook
```

运行顺序：先读精简地图 → 限量查询 → 读取命中的源码/测试 → 用 `rg` 补查。
行号及 SHA 对应生成快照；后续在途改动可能使其过期。无匹配时回到源码搜索。
静态导入只提供测试候选，动态导入、重导出、间接调用需人工扩展，不能当完整覆盖图。

提交时现有 pre-commit 框架的 `project-index` hook `always_run`：

1. `--staged --stage` 读取 Git 暂存 blob，自动反映新增、修改、改名和删除；
   未暂存/未跟踪源码不会被夹带，部分暂存按已暂存版本索引。
2. 输出稳定排序，无时间戳；相同输入产生相同索引，避免无意义提交噪声。
3. 只更新并暂存这两份生成索引，不暂存源码、文档、配置或其他会话的内容。
   无法解析输入或生成失败时拒绝提交；输出有未暂存编辑时拒绝覆盖，先核对并
   暂存或另存该编辑。pre-commit 自身会暂存保护并恢复其他未暂存内容。
4. 本仓库不因安装 hook 而自动 commit。新克隆需安装；绕过 Git hook 的提交
   不受此本地机制保护。当前没有远端 CI 强制索引门，也没有自动文档语义判定。

维护入口：[生成器](../scripts/build_project_index.py)、
[hook 配置](../.pre-commit-config.yaml)、[独立 Git 仓库集成测试](../tests/test_project_index.py)。
修改生成器/查询 schema 时同步更新本节、使用入口及测试，重新生成索引。

## 7. 交付前完成检查

- 对照本任务代码 diff 检查文档处置覆盖；改名/删除后的当前引用和导航均已修复。
- 核对接口签名、默认值、单位/约定、能力限制与当前实现；可执行示例按影响范围验证。
  链接存在不能证明语义正确，只改日期或新增报告不算通过。
- 已失效的当前文档/章节已修改、合并或删除；独有历史证据及勘误关系完整。
- 有实质结论则有可持久读取的 docs 记录；事实、推断、决策和未测项区分明确。
- 新洞察有证据、适用边界、状态与去向；“无新增”是合法结果。
- 已推翻的规则在当前入口已修正，封存证据完整，链接能解析。
- AGENTS.md 行数/字节预算满足，改动经 diff 检查；只运行适合此次修改的验证。
- 最终回复简短链接记录/洞察，并说明规则是否提升或修订及验证局限。

文档语义由执行任务的 agent 在交付时检查；项目索引有本地提交 hook，当前没有自动 CI 文档门。
索引重建、链接或变更文件配对通过，均不能替代文档语义与代码的一致性核对。

## 8. 落地与修订记录

初版记录 ID：`2026-10-05-knowledge-workflow-01`。用户请求建立记录→蒸馏→规则的机制。
该次新增本指南与洞察登记簿，更新根 §6 和主题索引；未修改科学模型或历史证据。
预设维护预算为 150 行/12 KiB；根文件由 108 行变为 128 行，仍在预算内。
四份当前文档的 63 个本地链接、空白、预算及 `git diff --check` 检查通过，
旧 AGENTS 快照 SHA-256 保持原值。新登记的三条历史经验核对原记录，未重跑科学实验。
蒸馏去向：`knowledge-layers` 已按用户决策 adopted 到根 §6；
规则可执行性通过文档检查，实际遵守率和 agent 效率收益尚未测量。


## 9. README and root guidance refresh (2026-10-05)

Record ID: `2026-10-05-readme-principles-01`. User decision: document the current
project design in README, keep AGENTS.md in English with lasting philosophy and
principles, and keep detailed records in docs. This refines the existing
`knowledge-layers` decision; it is not a measured agent-efficiency improvement.

Source review: the current working tree, including its pre-existing changes;
public Atom/input exports, CalculationSpec, shared CSF solver/functionals,
level/average/tracking/snapshot modules, common-orbital continuum channels,
XFEL exports/drivers, complex CI, package metadata, and CI workflows.
No production or test code is changed by this documentation increment.

Documentation disposition:

- README: added design/data flow, independent state and optimization contracts,
  process/dynamics boundaries, and current implementation limits; retained the
  setup, examples, broad feature scope, and license established by earlier work.
- AGENTS.md: rewritten in English; retained cross-task obligations and six
  section numbers so existing insight destinations remain valid. Domain-specific
  formulas and conventions moved to the [reference guide](AGENT_REFERENCE.md#task-specific-scientific-contracts).
- Architecture, capabilities, atomic-state design, and docs navigation: updated
  current CI/XFEL scope and distinguished public sorted-root selectors from
  explicit lower-level tracking. Existing implementation evidence stays in its topics.
- Insight register: refined `knowledge-layers` with the user's language/scope
  decision; no new physical or performance rule promoted.
- Testing, verification, migration, NIST guides, and XFEL gap analysis: reviewed
  and retained. No code/API/default, reference-data, scientific acceptance, or
  numerical-model change is made here; README links their existing detailed contracts.
- Prior README maintenance record: moved intact below. It retains earlier
  licensing, execution checks, and failed-example evidence; those checks are
  not claimed as new measurements in this increment.

Validation for this increment: 85 existing focused tests passed in 1.38 s with
the existing `.venv`, `PYTHONPATH=src:.`, and serial BLAS/OMP/Numba threads.
The run covered `test_atom_input_surface.py`, `test_ci_level_selection.py`,
the three small selector/degeneracy checks in `test_ci_state_average.py`, the
invalid-objective and normalization checks in `test_ci_charge_state_average.py`,
and the physical-overlap/degenerate-projector checks in `test_ci_basis_tracking.py`.
All 204 local links/anchors in the nine touched documents and README module
paths resolved; all six Python example blocks compiled. Root guidance is
English, 132 lines / 8,993 UTF-8 bytes. Diff/whitespace checks passed. All 332
production Python source hashes and the original archived AGENTS SHA-256
(`498e469e963b5e6ff5d8c0c5460d6c2a64a9d00777b480836da055185936d9fe`)
were preserved. Earlier example execution evidence is copied below; those
solves were not repeated in this increment. No scientific campaign or full
test suite was run, and no new physical-accuracy or performance claim is made.

Distillation: existing `knowledge-layers` refined as a user decision. No new
empirical insight; detailed contracts retain their original scope and evidence.

### Earlier README and license verification

The following record was copied from README's maintenance comment during this
move. It records the earlier increment and its source-specific checks.

```text
Maintenance record: 2026-10-05, restored the earlier README scope with current API
and command corrections; GPL-3.0-only assigned at the user's explicit request.
Dependencies: pyproject.toml; Atom/input exports; ladder/compare; providers;
benchmark gates; architecture/capability/NIST guides; CI workflows.
Validation: all 6 Python example blocks passed convergence/finite-energy checks;
all 7 ladder rungs and 4 comparison results passed. All 40 local links/anchors
and architecture paths resolved; 69 focused tests passed in 0.75 s. Wheel metadata
and bundled GPL text verified; uv lock check passed; diff whitespace clean.
Example audit: Fe XVII Slater B-spline did not converge in the default 40 iterations
for ns80/rmax40 (-1148.2259590817914 Ha) or the auto grid (-1148.2259490074835 Ha).
Shooting with the same physical inputs converged (-1148.230506455547 Ha); the
example now uses that explicit representation. Solver cause remains unassigned;
no physical gate, tolerance, or iteration budget changed. These are execution
checks, not new external accuracy or performance measurements.
Documentation disposition: current guides retained; README and package metadata
updated together. No new reusable insight or root-rule change.
```
