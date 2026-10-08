# Atomx documentation

## Current guides

- [Architecture](ARCHITECTURE_DESIGN.md): ownership, dependency direction and remaining work.
- [Capabilities and limits](ARCHITECTURE_CAPABILITIES.md): implemented physics and supported interfaces.
- [Atomic states and optimization](ATOMIC_STATE_DESIGN.md): current selectors, common-orbital objectives, explicit basis tracking and remaining proposals.
- [XFEL design and evidence](XFEL_GAP_ANALYSIS.md): rate/coherent workflows, model boundaries and dated measurements.
- [Migration notes](MIGRATION.md): breaking imports, public API changes and saved-data compatibility.
- [Verification](VERIFICATION.md): scientific claims, reference sources and test gates.
- [Testing](TESTING.md): test organization and commands.
- [NIST reference data](NIST_SPECTRUM_ATLAS.md): supported parsing and the status of the retired atlas tools.
- [Theory–NIST comparison](THEORY_NIST_COMPARISON.md): state matching, uncertainty and interpretation.

## Design records and evidence

- [稀土 Yb⁺ 谱线实算与实验对照](rare_earth_yb_spectrum_2026_10_08/README.md)：5 个能级、7 条目标线和 2 个直接寿命对照；冻结核数值控制通过、实验精度目标未通过，保存 Breit、4f 空穴与 QED 后端敏感性及失败证据。
- [原子、分子、晶体谱学与未来产业研究体系](spectroscopy_science_story_2026_10_06/README.md)：当前宏观故事线、12 个案例 setup、QED/XAS/材料模型选择、Atomx 能力边界与三个已运行接口例子；完整科学案例仍为计划。此前 [Ne 精密谱学参赛项目书](agentic_science_beijing_light_source_2026_10_06/README.md)保留为专题支持材料。
- [Electronic-structure provider contract](adr/ADR-0001-electronic-structure-provider.md).
- [Historical reports](archive/README.md): dated physics investigations and campaign evidence.
- [Test physics audit](test_audit/README.md): reference independence and coverage assessment.
- [FAC benchmark report](../benchmarks/REPORT.md).
- [External verification](../benchmarks/EXTERNAL_VERIFICATION.md).

## Development references

- [Project index](../PROJECT_INDEX.md): compact code map; query symbols and static test candidates on demand. Rebuilt automatically from staged files before commits once the hook is installed.
- [Documentation maintenance and insight workflow](AGENT_KNOWLEDGE_WORKFLOW.md): update, merge or remove affected documentation with each code change.
- [Agent reference index](AGENT_REFERENCE.md) and [insight register](AGENT_INSIGHTS.md).
- [Repository working principles](../AGENTS.md): lasting English guidance; task-specific contracts and detailed records live in the reference guide and topic docs.
- [FAC parity plan](plans/FAC_PARITY_IMPLEMENTATION_PLAN.md).
- [Feature gaps](plans/gaps.md).
- [Book plan](book/PLAN.md).

Current guides describe supported code. Historical reports retain the state of
older source snapshots; their commands and database counts may no longer apply.

## Code-to-documentation entry points

For every logical code change, review these starting points and search for affected
symbols, paths and behavior in current guides, docstrings, examples and migration
notes. Update, merge or delete stale content in the same change. This table is a
routing aid, not proof that every affected document has been found.

| Changed code or contract | Documentation to review first |
|---|---|
| Public exports, `atom.py`, `atom_inputs.py`, `calculation_spec.py`, `config/` | [Project README](../README.md), [capabilities](ARCHITECTURE_CAPABILITIES.md), [state design](ATOMIC_STATE_DESIGN.md), affected API docstrings/examples |
| `methods/`, `numerics/`, `angular/`, `physics/`, `potentials/`, `corrections/` | [Architecture](ARCHITECTURE_DESIGN.md), [capabilities](ARCHITECTURE_CAPABILITIES.md), [verification](VERIFICATION.md), relevant topic via [reference index](AGENT_REFERENCE.md) |
| `xfel/`, `scattering/`, `properties/`, `plasma/`, `spectrum/`, `uta/`, `lineshapes/`, `fitting/` | [Capabilities](ARCHITECTURE_CAPABILITIES.md), [verification](VERIFICATION.md), [XFEL gaps](XFEL_GAP_ANALYSIS.md), owning topic and examples |
| `providers/`, `io/`, public interfaces or saved-data compatibility | [Provider contract](adr/ADR-0001-electronic-structure-provider.md), [migration](MIGRATION.md), [capabilities](ARCHITECTURE_CAPABILITIES.md) |
| `reference_data/`, `workflows/nist_comparison.py`, reference conventions | [NIST data](NIST_SPECTRUM_ATLAS.md), [theory comparison](THEORY_NIST_COMPARISON.md), affected fixture generation notes |
| `viz/`, scripts, examples, package/dependency/CI configuration | [Project README](../README.md), owning usage guide, [testing](TESTING.md), affected executable commands |
| `tests/`, `benchmarks/`, scientific acceptance gates, physical constants/data | [Testing](TESTING.md), [verification](VERIFICATION.md), owning measurement/claim and fixture notes |
| Agent instructions or documentation organization | [Knowledge workflow](AGENT_KNOWLEDGE_WORKFLOW.md), [reference index](AGENT_REFERENCE.md), [insights](AGENT_INSIGHTS.md), affected navigation |

Paths are under `src/atomx/` unless they identify a repository-level directory or
configuration. For unmapped code, locate the owning guide and extend this table
when needed. If a reviewed document needs no change, record the specific contract
checked and why it remains valid; do not edit dates merely to satisfy the process.
Preserve unique archived scientific evidence, while removing obsolete current
instructions and repairing incoming links. A new report does not fix an old guide.

The [2026-10-05 code review](code_review_2026_10_05/README.md) records current
fix dispositions, replayable validation and measured test hotspots.
