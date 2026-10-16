# Development and documentation maintenance


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
| `methods/core_sd.py`, `methods/valence_sd.py`, `methods/sigma2_sd.py`, `methods/_pci_*.f90`, `tools/build_pci_*` | [原生SD接口与近似](ATOMIC_LINEARIZED_SD.md)、[活动方程证据](atomic_pci_sd_native_2026_10_09/README.md)、[测试](TESTING.md)、[能力](ARCHITECTURE_CAPABILITIES.md)、[迁移](MIGRATION.md) |
| `methods/monovalent_sd.py`, `numerics/fermion.py` | [单价SD反馈与归一化态](ATOMIC_LINEARIZED_SD.md#finite-standalone-monovalent-sd)、[相关方法测试](CORRELATED_METHOD_TESTING.md)、[验证](VERIFICATION.md)、[活动证据](atomic_pci_sd_native_2026_10_09/README.md) |
| `methods/reduced_monovalent_sd.py`, `methods/_pci_monovalent.f90`, `tools/build_pci_monovalent_backend.py` | [约化单价SD与完整资源](ATOMIC_LINEARIZED_SD.md#j-reduced-standalone-monovalent-sd)、[独立磁态与J门](CORRELATED_METHOD_TESTING.md#j-reduced-standalone-monovalent-sd)、[验证](VERIFICATION.md)、[活动证据](atomic_pci_sd_native_2026_10_09/README.md) |
| `methods/ci_effective.py` | [有限Bloch波算符与真实度量](ATOMIC_LINEARIZED_SD.md#exact-finite-bloch-wave-operator-and-physical-metric)、[有限CI能力](ARCHITECTURE_CAPABILITIES.md)、[迁移](MIGRATION.md)、[验证](TESTING.md) |
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
