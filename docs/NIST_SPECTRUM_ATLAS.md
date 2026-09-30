# NIST reference data and atlas status

## Supported package interfaces

`atomx.reference_data` parses caller-supplied NIST text and normalizes reference
records without implicit network access. Its modules provide:

- `nist`: levels, lines, configurations, classifications and units.
- `matching`: state identity, complete configuration centroids and wavelength media.
- `bibliography`, `annotations`, `spectrum_notes`: source citations and comments.
- `cowan`, `structure_outputs`: reference calculation output parsing.

Native campaign comparisons live in `atomx.workflows.nist_comparison`; see
[comparison conventions](THEORY_NIST_COMPARISON.md). These interfaces consume
caller-selected records and databases. They do not download a complete archive.

## Retired atlas commands

The legacy NIST archive builder, supplementary importers, interactive viewer
and campaign CLI scripts were removed during repository cleanup. There is
currently no supported website launch command or HTTP API in this checkout.
The retained parsing and comparison APIs do not establish archive completeness.

Historical viewer features, source manifests and campaign counts are documented
in [the atlas build history](archive/2026-09/ATLAS_BUILD_HISTORY_2026-09-29.md).
Their script contracts are preserved in
[the retired test archive](archive/2026-09/retired_script_tests/README.md).
These records describe previous snapshots and are not current run instructions.

## Reference interpretation

Keep raw responses, hashes and source manifests with any externally maintained
archive. Distinguish observed from Ritz wavelengths, experimental uncertainty
from numerical refinement, and resolved levels from configuration averages.
Missing records, ambiguous assignments and failed calculations remain visible.
An element's presence in a periodic table does not certify experimental data
for every ionization stage or completeness of a database.
