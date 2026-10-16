# Atomx Zenodo preparation record

The requested deliverable is an uploadable software snapshot and a prepared
Zenodo page. Packaging preserves the current implementation, tests, and
scientific criteria. It does not publish a record or delete original evidence.

The initial inventory contained 126,876 Git-visible candidates. Existing tracked
files occupied approximately 172 MB; untracked experiments, arrays, copied
dependencies, and build products made the candidate inventory approximately
47.5 GB. These are file-byte totals, not measured archive sizes. The base commit
was `233fec1`; many code and documentation changes were uncommitted.

The preparation policy and replay command are maintained in
[the Zenodo guide](../ZENODO_RELEASE.md). The generated `RELEASE_MANIFEST.json`
binds the final snapshot to exact included bytes and enumerates exclusions.

## Verification contract

Hypothesis: a curated copy can install and pass the essential suite without
depending on omitted local builds or changing any physical input or tolerance.
Fixed conditions: current source and fixtures, the existing `.venv`, serial
BLAS/OpenMP/Numba settings, and the reviewed default test selection.
Distinguishing outcomes: an omitted dependency, import from the original
checkout, a source hash mismatch, a broken current-guide target, or a failed
test rejects the package. Accept only exact identities, successful independent
wheel import, intact archive/checksums, guide closure, and a passing essential
suite. Stop and investigate any failed gate or concurrent source change.

## Candidate results

The isolated release copy passed **452 tests, with 25 warnings, in 63.84 s**
using Python 3.10.21 and the reviewed default selection. The
[test log](essential-tests.log) retains the warnings and durations.
[Validation identities](validation.json) bind this result to every copied
production, test, fixture, and backend-tool file; the combined SHA-256 is
`adbfacd586768b1aa562aaee5cb744e41c16a9363ba5b8c4580e40d8384b56d7`.
This was a correctness run, not a controlled timing comparison.

The built wheel installed into a separate target directory and imported from
`/tmp`, outside the checkout. All eight `_pci_*.f90` templates were present;
`Atom(1).solve()` converged to −0.5000054122283553 Ha. Full repository lint
and citation CFF schema validation passed. The Zenodo metadata's common fields
passed its official legacy schema. That schema omits `version` and `language`,
although the [current API documentation](https://developers.zenodo.org/)
explicitly supports both; those fields were checked against the documented
types and values separately. Remote server validation requires saving a draft.

## Negative controls and corrections

The first preparation rejected untracked experiment symlinks. Inspection found
40 local pytest `current` aliases; they are now omitted explicitly and recorded,
while tracked or software symlinks still reject packaging. The next candidate
rejected two newly appearing experiment JSON sidecars from concurrent work.
The final policy fixes the inventory-time snapshot, records later experiment
additions separately, rejects changes to the software file set, and still
checks the bytes and metadata of every included original file.

The guide-link gate found three references to one omitted evidence file,
`expt_compact_public_build001_source_proof.json`. The filter had matched
`source` in a filename; it now applies only to copied upstream directory names.
The final archive must pass this guide-link gate. A wheel smoke invocation
using retired `method='hydrogenic'` was correctly rejected; the successful
smoke uses the documented `Atom(1)` interface. No physical code or tolerance
was changed in response to these preparation failures.

Final ZIP integrity, checksums, extracted-source identities, guide closure,
and extracted-suite results are saved as validation companions with the upload
artifacts. They must agree with the candidate's production/test/tool identity
before delivery. Original evidence and failed candidate directories are kept.

## Extracted archive results

The ZIP extracted into a separate directory passed **452 tests, with 25 warnings,
in 63.79 s**. Its [test log](extracted-tests.log) and
[archive and source checks](extracted-validation.json) record the tested archive
hash, all 14,526 included file hashes, 354 wheel source files, regenerated
navigation, and 38 current-guide entry points with zero broken repository links.
Its production/test/tool identity exactly matches the candidate result above.

The live [Zenodo license vocabulary](zenodo-license-vocabulary.json), queried
from `https://zenodo.org/api/vocabularies/licenses?q=gpl-3.0&size=5`, confirms
`gpl-3.0-only`. Zenodo JSON uses that service identifier; the CFF citation retains
the equivalent SPDX spelling `GPL-3.0-only`. The older documented
`/api/licenses/gpl-3.0-only` endpoint returned HTTP 404; the current vocabulary
endpoint succeeded. No authenticated deposition or remote draft was created.

The upload set in `dist/zenodo-0.2.0-upload/` incorporates this validation record
and license-identifier correction. Final verification compares its production,
tests, fixtures, and backend tools to the tested extracted source and checks
the resulting archive independently. Publication metadata and validation
documentation changed; the tested physical source and inputs did not.

[Final upload verification](final-validation.json) confirms all 14,529 included
file hashes, the unchanged tested source identity, 38 current guides with no
broken links, and a wheel identical to the independently installed artifact.
This report accompanies the source ZIP, so its archive hash does not refer to
an archive containing the report itself. The final source ZIP is 220,693,268 B
and the wheel is 1,418,721 B. All upload-file checksums passed.

## Scope and knowledge disposition

No physical formulas, public numerical behavior, or acceptance thresholds were
changed for this task. Cleanup is confined to archive selection; local research
data and original source snapshots are retained. A passing essential suite
does not imply that every extended or optional scientific campaign passed.
Creator identifiers and affiliations remain pending the user's details.

No new scientific insight; the existing dirty-source identity, replayability,
and evidence-preservation rules cover this release task.
