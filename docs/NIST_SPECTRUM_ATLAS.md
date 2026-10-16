# NIST reference data and periodic atlas

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

## Supported read-only periodic atlas

`tools/export_nist_atlas.py` exports a caller-selected local SQLite archive to
a static periodic-table viewer, using the package's `Level` identity and
configuration normalization. It does not calculate states, download data,
write to the archive, or revive retired campaign scripts.

The exporter creates an SQLite backup, indexed historical CSV ledger and
source hashes. It streams one ion at a time under a 4 GiB address-space limit;
allow at least 2 GiB free disk and limit BLAS threads before starting it.
Output and snapshot paths must be new so older evidence is retained.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 .venv/bin/python \
  tools/export_nist_atlas.py \
  --archive output/nist_spectra/spectra.sqlite \
  --output /tmp/atomx-atlas-new \
  --evidence /tmp/atomx-atlas-evidence-new \
  --snapshot /tmp/atomx-atlas-snapshot-new.sqlite
python3 -m http.server 8877 --bind 127.0.0.1 --directory /tmp/atomx-atlas-new
```

The viewer has all 118 elements, ion selection, paginated levels and lines,
observed/Ritz wavelengths, reference uncertainties, Aki, ionization values,
historical classification status, source URLs/hashes, and decoded-table JSON
downloads. Ion assets use lossless gzip and are fetched only when selected;
the browser requires `DecompressionStream`. Its decoded cache retains at most
six ion payloads. Empty and failed classifications remain visible. No element
is currently certified at the highest level.

Yb II's saved DC-SD and frozen-CI finite-frequency results are independent
models. Nominal-branch comparison requires a unique evaluated 6s zero,
configuration, J and parity; ambiguous/theoretical matches are unscored. This
does not certify large correlated wavefunction purity or dressed transition
rates. Historical CSV values retain their Hamiltonian and status; they have
not been recertified against current source.

The [export and deployment evidence](atomic_nist_atlas_2026_10_11/README.md)
binds the selected archive, coverage, source-count discrepancies, checks and
owner-private remote publication. The site is a published snapshot;
subsequent calculations are reported by the
[hourly card and calculation queue](atomx_hourly_progress_2026_10_10/README.md).

The current [native MCDHF→RCI Yb⁺ development](yb_mcdhf_rci_2026_10_11/README.md)
is shown in [the closed published version9 snapshot](yb_mcdhf_rci_2026_10_11/atlas_update008/PUBLICATION_CLOSURE.md):
23 Yb II models and 92 comparison rows. This retains the separate29-orbital
MCDHF endpoints, both history10/history40 branches and all277 trajectory records
with262 saved identities. Two fixed20831-orbital controls use nuclear and
closed68 screened radial constructors with the same Dirac–Coulomb operator and
complete1595 target-sector CSFs. Their expanded MCDHF stationarity was not
evaluated in these fixed-RCI models. The3789parent CSFs were enumerated; the
other2194 were not computed. All6019 NIST gzip payloads and118elements are
preserved. Original stationarity and experimental failures remain visible;
corrections from other points are not combined.

The [version10 native31 update](yb_mcdhf_rci_2026_10_11/atlas_update009/README.md)
is isolated source/JSON preparation, with build, browser and publication pending.
It proposes one literal terminal16 model and all17 chronological checkpoints
from a closed historical native Atomx f043 run:24models96rows. The original
3689metric residual and energy indicator fail at every point; finite stored-H
acceptance and saved scalar closure do not certify stationarity, experimental
accuracy, physical completeness, current source or GPU adoption.

## Retired atlas commands

The legacy NIST archive builder, supplementary importers, interactive viewer
and campaign CLI scripts were removed during repository cleanup. The new static
exporter does not reinstate their HTTP API or archive-ingestion contracts.
Neither the viewer nor the retained APIs establish archive completeness.

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

`tools.atomic_nist_audit.audit_spectrum` adds a reusable strict audit for an
explicit complete theoretical state panel. It checks common Hamiltonian,
nucleus, core, space, energy zero, units, metric convention and approximation
scope; requires unique species/configuration/J/parity matches; and keeps the
ground reference outside accuracy denominators. It distinguishes observed
vacuum wavelengths from Ritz and level-derived intervals and leaves missing
uncertainties/covariances unknown. Necessary multipole symmetry rules do not
establish nonzero matrix elements or rates. Returned whole-state iteration
requests describe experiments; this reporting interface does not dispatch them.

The [source-bound six-direction audit](atomx_hourly_progress_2026_10_10/parallel_six002/README.md)
uses the original complete Yb five-state wavefunctions and local ASD5.12
snapshot (349 levels/327 lines). Three of four excitations and both fine-structure
splittings still fail the original 1% gate; only two of four requested lines
have unique observed matches. It produces no new spectrum or experimental
accuracy improvement, and does not update the published remote snapshot.
