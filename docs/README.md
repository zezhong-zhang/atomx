# Atomx documentation

Choose a guide by task. Current guides describe supported code; dated reports
bind claims to their original source and inputs.

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

## Atomic solvers and operators

| Task | Guide |
|---|---|
| DF solver controls, convergence and engine selection | [DF solver controls](DF_SOLVER_CONTROLS.md) |
| E1 and higher multipoles, gauge and frequency coverage | [Transition operators](TRANSITION_OPERATORS.md) |
| Static Breit and finite-frequency transverse interactions | [Transverse interaction](TRANSVERSE_INTERACTION.md) |
| Recoil operators and mass conventions | [Recoil operators](RECOIL_OPERATORS.md) |
| Crystal fields and radial scaling | [Crystal fields](CRYSTAL_FIELD_SPEC.md), [Slater scaling](SLATER_SCALING_SPEC.md) |
| Inner-region scattering | [R-matrix specification](RMATRIX_INNER_REGION_SPEC.md) |

## Design records and evidence

[Research and evidence](RESEARCH.md) separates dated campaigns, method surveys,
validation history and retained failures from current usage instructions.
[Historical reports](archive/README.md) preserve older investigations.

## Development references

- [Development and documentation maintenance](DEVELOPMENT.md): code-to-guide routing, hooks and change review.
- [Testing and GitHub Actions](TESTING.md): installation, commands and automated gates.
- [Project index](../PROJECT_INDEX.md): query code symbols and static test candidates.
- [Agent reference index](AGENT_REFERENCE.md) and [insight register](AGENT_INSIGHTS.md): task-specific contracts and evidence.

## Code-to-documentation entry points

The [code-to-documentation table](DEVELOPMENT.md#code-to-documentation-entry-points)
is maintained in the development guide. Update the owning guide and its incoming
links with each contract change; preserve unique archived scientific evidence.
