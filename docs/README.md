# Atomx documentation

## Current guides

- [Architecture](ARCHITECTURE_DESIGN.md): ownership, dependency direction and remaining work.
- [Capabilities and limits](ARCHITECTURE_CAPABILITIES.md): implemented physics and supported interfaces.
- [Migration notes](MIGRATION.md): breaking imports, public API changes and saved-data compatibility.
- [Verification](VERIFICATION.md): scientific claims, reference sources and test gates.
- [Testing](TESTING.md): test organization and commands.
- [NIST reference data](NIST_SPECTRUM_ATLAS.md): supported parsing and the status of the retired atlas tools.
- [Theory–NIST comparison](THEORY_NIST_COMPARISON.md): state matching, uncertainty and interpretation.

## Design records and evidence

- [Electronic-structure provider contract](adr/ADR-0001-electronic-structure-provider.md).
- [Historical reports](archive/README.md): dated physics investigations and campaign evidence.
- [Test physics audit](test_audit/README.md): reference independence and coverage assessment.
- [FAC benchmark report](../benchmarks/REPORT.md).
- [External verification](../benchmarks/EXTERNAL_VERIFICATION.md).

## Development references

- [FAC parity plan](plans/FAC_PARITY_IMPLEMENTATION_PLAN.md).
- [Feature gaps](plans/gaps.md).
- [Book plan](book/PLAN.md).

Current guides describe supported code. Historical reports retain the state of
older source snapshots; their commands and database counts may no longer apply.

The [2026-10-05 code review](code_review_2026_10_05/README.md) records current
fix dispositions, replayable validation and measured test hotspots.
