# Atomx documentation

Choose a guide by task. Current guides describe supported code; dated reports
bind claims to their original source and inputs.

## Current guides

- [Architecture](ARCHITECTURE_DESIGN.md): ownership, dependency direction and remaining work.
- [Capabilities and limits](ARCHITECTURE_CAPABILITIES.md): implemented physics and supported interfaces.
- [Atomic states and optimization](ATOMIC_STATE_DESIGN.md): current selectors, common-orbital objectives, explicit basis tracking and remaining proposals.
- [XFEL design and evidence](XFEL_GAP_ANALYSIS.md): rate/coherent workflows, model boundaries and dated measurements.
- [XFEL core modalities, SOTA, challenges and agent path](XFEL_CORE_MODALITIES_SOTA_AND_AGENT_PATH.md): 核心模态 / SOTA 代码全景 / 前沿挑战分级 / agentic-science 工作流与实验一致化路线.
- [Migration notes](MIGRATION.md): breaking imports, public API changes and saved-data compatibility.
- [Verification](VERIFICATION.md): scientific claims, reference sources and test gates.
- [Testing](TESTING.md): test organization and commands.
- [NIST reference data and periodic atlas](NIST_SPECTRUM_ATLAS.md): read-only export, state matching, remote snapshot and historical coverage limits.
- [Theory–NIST comparison](THEORY_NIST_COMPARISON.md): state matching, uncertainty and interpretation.
- [Laser/EUV source and experimental alignment](EUV_SOURCE_MODEL.md): explicit multi-charge EOS/continua, finite resonance/kinetic energy closure, native adjacent-CI PI/RR/Auger/direct EII, independently qualified bound model-QED composition, balanced Sn thresholds, layered radiation, Drude waves, smooth-flow dynamics and failed optical/EUV/CE experimental gates.
- [Native MCDHF→RCI and Yb⁺](yb_mcdhf_rci_2026_10_11/README.md): current atomic-structure direction, five-target stationarity, balanced core–valence physics and exact radial action reuse.
- [Native atomic roadmap](atomic_native_roadmap_2026_10_11/README.md): preserved Ne/Eu precision and earlier independent fixed-space controls.

## Atomic solvers and operators

| Task | Guide |
|---|---|
| DF solver controls, convergence and engine selection | [DF solver controls](DF_SOLVER_CONTROLS.md) |
| E1 and higher multipoles, gauge and frequency coverage | [Transition operators](TRANSITION_OPERATORS.md) |
| Static Breit and finite-frequency transverse interactions | [Transverse interaction](TRANSVERSE_INTERACTION.md) |
| Atomic integral-driven CCSD / FS-CCSDT | [Atomic FSCC](ATOMIC_FSCC.md) |
| Recoil operators and mass conventions | [Recoil operators](RECOIL_OPERATORS.md) |
| Crystal fields and radial scaling | [Crystal fields](CRYSTAL_FIELD_SPEC.md), [Slater scaling](SLATER_SCALING_SPEC.md) |
| Inner-region scattering | [R-matrix specification](RMATRIX_INNER_REGION_SPEC.md) |

## Design records and evidence

[Research and evidence](RESEARCH.md) separates dated campaigns, method surveys,
validation history and retained failures from current usage instructions.
[Hourly task and spectrum report](atomx_hourly_progress_2026_10_10/README.md)
records Feishu cards, the timer and evidence-guided selection after each receipt,
including admitted independent parallel probes, bounded scientific actions and explicit implementation frontiers.
[Historical reports](archive/README.md) preserve older investigations.

## Development references

- [Development and documentation maintenance](DEVELOPMENT.md): code-to-guide routing, hooks and change review.
- [Testing and GitHub Actions](TESTING.md): installation, commands and automated gates.
- [Zenodo deposit](ZENODO_RELEASE.md): source packaging, citation metadata and prepared upload page.
- [Project index](../PROJECT_INDEX.md): query code symbols and static test candidates.
- [Agent reference index](AGENT_REFERENCE.md) and [insight register](AGENT_INSIGHTS.md): task-specific contracts and evidence.

## Code-to-documentation entry points

The [code-to-documentation table](DEVELOPMENT.md#code-to-documentation-entry-points)
is maintained in the development guide. Update the owning guide and its incoming
links with each contract change; preserve unique archived scientific evidence.
