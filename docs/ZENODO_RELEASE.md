# Atomx Zenodo deposit

This software deposit preserves atomx version 0.2.0, including the current
uncommitted implementation, tests, and documentation. The source archive and
wheel are prepared locally. The creator list is provisional while the requested
additional creator details are pending. No online record or DOI has been created.

## Zenodo fields

| Field | Prepared value |
|---|---|
| Title | atomx: Relativistic Atomic Structure, Spectroscopy, and X-ray Dynamics |
| Resource type | Software |
| Version | 0.2.0 |
| Creator | Zhang, Zezhong; additional details pending |
| License | GNU General Public License v3.0 only, SPDX `GPL-3.0-only` |
| Visibility | Public |
| Language | English |
| Related work | https://github.com/zezhong-zhang/atomx; relation `isSupplementTo`, type Software |
| Publication date | Set to the actual publication date when submitting |
| DOI | Let Zenodo assign a DOI; reserve one in the draft if needed |

Keywords: atomic physics; relativistic atomic structure; Dirac–Fock;
configuration interaction; atomic spectroscopy; X-ray spectroscopy;
XFEL dynamics; B-splines; Python.

The machine-readable fields and HTML description are in
[.zenodo.json](../.zenodo.json). [CITATION.cff](../CITATION.cff) provides the
matching software citation. Add supplied creator details to both before
publication; do not invent an ORCID, affiliation, funder, or existing DOI.

## Description to paste into Zenodo

atomx is an open-source Python library for relativistic atomic structure,
spectroscopy, and X-ray dynamics. It computes atomic energies, radial orbitals,
radiative transitions, and other atomic properties using shared physical
operators and numerical engines.

The theory interfaces include hydrogenic and mean-field models, Hartree–Fock
and Dirac–Fock, multiconfiguration Dirac–Hartree–Fock and configuration
interaction, second-order many-body perturbation theory, and explicit
lower-level correlation solvers. B-spline Galerkin and ODE shooting engines
support method-dependent radial calculations. Additional modules address
scattering, recombination, spectrum synthesis, plasma kinetics, and pulse-driven
XFEL populations and coherent dynamics.

This deposit preserves a source snapshot of version 0.2.0 with tests, reference
fixtures, documentation, examples, and retained research reports. The release
manifest identifies the base Git commit, working-tree changes, included file
hashes, and excluded local artifacts. The Python wheel provides the installable
library.

Supported operators, approximations, result types, convergence controls, and
unfinished capabilities are documented in the included guides. Optional pCI
and EXP-T interfaces require separately obtained, pinned upstream sources and
suitable build tools. Passing software and internal consistency checks does
not establish experimental accuracy or convergence for every physical system.
Large local research arrays, compiler caches, and copied dependency/build trees
are outside this software deposit; retained reports identify their original
provenance.

## Upload files

The preparation command creates these files under its output directory:

| File | Purpose |
|---|---|
| `atomx-0.2.0-source.zip` | Curated repository snapshot with source, tests, fixtures, guides and retained reports |
| `atomx-0.2.0-py3-none-any.whl` | Installable Python library, including native Fortran source templates |
| `RELEASE_MANIFEST.json` | Exact source identity, per-file hashes, and complete list of omitted Git-visible artifacts |
| `SHA256SUMS` | SHA-256 checksums for upload artifacts and preparation files |
| `zenodo-metadata.json` | Zenodo metadata, also suitable for the GitHub integration |
| `zenodo-deposition.json` | The same metadata wrapped for the deposition API |
| `ZENODO_PAGE.md` | This page text and upload instructions |
| `build.log` | Wheel build result |

The completed release checks also provide `final-validation.json` and
`extracted-tests.log` as upload companions. The verified upload set for this
preparation is in `dist/zenodo-0.2.0-upload/`.

Upload the ZIP, wheel, manifest, and checksum file. The page and JSON files
can also be deposited as metadata companions. Historical attachments retain
their own attribution and license notices; the software's license is unchanged.

## Preparation and verification

From the repository root, use the existing environment:

```bash
uv pip install --python .venv/bin/python 'hatchling>=1.27' build
.venv/bin/python scripts/prepare_zenodo.py --output dist/zenodo-0.2.0
cd dist/zenodo-0.2.0
sha256sum -c SHA256SUMS
```

The [preparation script](../scripts/prepare_zenodo.py) refuses an existing
output directory. It never stages, commits, deletes original artifacts, uploads,
or publishes. It copies current tracked and nonignored untracked software files;
tracked numerical fixtures and historical evidence remain included. It excludes
untracked compiler/test scratch; explicitly tracked fixtures inside historical
scratch or source directories remain included with their original bytes. The
[cleanup selection check](repository_cleanup_2026_10_16/release-selection-check.json)
records the exact fixture-preservation and untracked-exclusion controls.
For other untracked files it excludes
compiler caches, build/dependency trees, and untracked large research arrays
and historical source copies. Omitted Git-visible paths and sizes are recorded
in the manifest. Ignored local directories such as `.venv`, `output`, `dataset`,
agent settings, and `trace.md` are outside the inventory.

Cleanup applies to the release copy. Original scientific evidence remains in
the working directory. The software deposit is not a complete archive of every
local experiment: some historical report links target omitted data or external
toolchains. Replaying those campaigns requires their recorded original inputs.
The included current guides and test fixtures must resolve within the snapshot.

The archive includes regenerated project navigation for its actual file set.
Original navigation hashes are retained in the manifest; the working tree's
navigation and staging area are untouched. Snapshot file identities include
uncommitted changes and must not be described as the base Git commit alone.
ZIP timestamps and permissions are normalized for consistent packaging.

Verification fixes the copied source and existing physical models. The
acceptance gates are exact copied-source hashes, intact ZIP/checksums, wheel
installation/import from outside the checkout, current-guide link closure,
lint for the preparation script, and the essential numerical test suite on
the extracted snapshot. Stop if source changes during packaging or a gate
fails; diagnose the failure without changing physical tolerances. Live optional
pCI/EXP-T campaigns and broad experimental validation are outside these release
checks. See [testing](TESTING.md) for their separate requirements.

Release checks and their source identities are recorded in
[the preparation record](zenodo_release_2026_10_10/README.md).

## Save the Zenodo draft

1. Open [New upload](https://zenodo.org/uploads/new) in the intended account.
   If this software already has a Zenodo record, create a new version from that
   record to preserve its version history.
2. Upload the prepared files and enter the fields and description above.
3. Add the supplied creator details and set the actual publication date.
   Reserve a DOI if it must appear in the files, then add it to the citation
   and metadata and rebuild the bundle before uploading the final files.
4. Select **Save draft** and use **Preview** to review the record.
5. Publish when the final creator list, metadata, and files have been reviewed.

The official [upload guide](https://help.zenodo.org/docs/deposit/create-new-upload/)
documents draft saving, DOI reservation, preview, and publication. The
[API guide](https://developers.zenodo.org/) documents `.zenodo.json` and
authenticated deposition. No Zenodo connector was found in the available
plugin catalog during this preparation.

## Documentation impact and distillation

The new release command affects packaging and citation metadata. The project
README and documentation index link here. Installation requirements, physical
methods, public APIs, and scientific acceptance criteria remain as documented
in their owning guides; they were retained because packaging does not change
those contracts. The commit-time project-index hook remains installed.

No new scientific insight. Operational observation: Git-visible experiment
outputs can greatly exceed the tracked repository size. For this software
deposit, explicitly enumerate omissions and preserve the actual dirty-source
identity; the existing evidence-preservation and source-identity rules apply.
