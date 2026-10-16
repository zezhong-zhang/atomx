# Grouped repository cleanup, 16 October 2026

The user requested cleanup and commits by topic, then explicitly requested
removal of the unrelated, untracked photonics/RCWA addition. That source tree
and its ignored Reticolo benchmark copy were removed. Atomic research evidence
was preserved on disk. No remote push, upload or message was requested.

## Selection and preservation

The initial Git-visible untracked inventory contained 217,190 regular files,
78.231 GiB in file-byte totals. The full local `docs/` tree occupied 277 GiB,
including already ignored material. These are inventory observations, not
archive sizes. The starting commit was `233fec1` with the existing dirty source.

Commit reports, replay drivers, small inputs and bounded result summaries.
Root ignore rules exclude copied dependencies/source trees, extracted bundles,
compiler/test scratch, raw numerical payloads and checkpoint tables. Topic
ignore files identify remaining individual files exceeding the existing 2 MiB
pre-commit limit. None of these research files was deleted. Small scientific
fixtures required by executable tests are explicit exceptions, retaining their
original bytes and identities. Immutable dated evidence is excluded from
whitespace-fixing hooks; current guides and production code remain checked.

Historical report bodies may refer to local raw arrays, source snapshots or
compiled backends. Their presence in a report does not promise those assets in
a clean checkout. Original failures, archived source identities and acceptance
rules remain intact. The repository link gate checks current entry points;
it does not certify archived bodies, external URLs or scientific claims.

## Qualification contract

Fixed conditions: existing Python 3.10 `.venv`, unchanged physical models,
states, spaces, seeds and acceptance tolerances; serial BLAS/OpenMP/Numba.
Accept successful essential and affected checks, current-guide closure,
atomic-source lint, staged-index freshness and intact retained evidence.
Stop and investigate numerical failures rather than relaxing tolerances.
Optional external backends and sealed historical campaigns remain explicit.

The [first essential run](essential-tests.log) passed449 tests, failed2 and
had6 setup errors: generated Numba-cache serialization raised
`ReferenceError: underlying object has vanished`. With a fresh cache and no
source changes, the [11-check replay](cache-replay.log) passed, then the
[complete essential suite](essential-fresh-cache.log) passed457 tests.
These are correctness runs, not controlled timing comparisons.

The [initial affected run](affected-tests.log) passed1241, failed13, skipped158
and had33 setup errors. Twelve failures exposed inconsistent inner quadrature:
the shared native convolution had become rank-aware, while both auxiliary
representations still used the historical fixed order. The remaining failure
was a fake backend missing the earlier numerator-division capability; one
error was a relocated XFEL fixture missing its dependency. Thirty-two errors
required a sealed historical execution driver, not ordinary pytest invocation.

## Exact auxiliary quadrature repair

Hypothesis: matching the shared native triangular rule removes the signed
high-rank discrepancy without changing outer quadrature, physical spaces or
tolerances. For rank k>=0, the source polynomial degree requires the component
order plus floor(k/2) inner Gauss nodes. Fixed tagged polynomial density
coordinates resample these nodes exactly; they do not replace integration
nodes, fit a density or discard coordinates. Full P/Q terms and c-products
remain. Additional payload and per-rank scratch are admitted explicitly.

The [first distinguishing control](cell-replay-initial.log) measured
0.08307385555378886 versus the shared native0.08307389687977092Ha at rank3,
outside the existing1e-11 absolute gate. The original source and test are
retained as `coulomb_features_before.py` and
`test_cell_coulomb_features_before.py`. The first repair also exposed incorrect
cell grouping in the compressed transform; its [eight failures](cell-replay.log)
are retained. Correct cell grouping [passes29 checks](cell-replay-corrected.log).
The later [focused panel](focused-replay-final.log) passes56 with34 explicit
optional skips, including even-rank and preallocation guards. The initial
allocation-guard failure remains in `focused-replay.log`.

Fresh case-specific native and polynomial pCI builds both pass the finite
independent core-SD gate ([log](live-core-sd.log)). A separate complete finite
Be two-electron-core, one-virtual-per-partial-wave l<=3 kinetic-balance solve
passes its original1e-11 gate ([contract](live-be2-f/contract.json),
[resource plan](live-be2-f/plan.json), [result](live-be2-f/result.json),
[log](live-be2-f.log)). It retains51 complete M=0/even SD determinants;
the planned native workspace is14,731,840B. Maximum canonical/noncanonical
commutator differences are6.94e-17/1.94e-16Ha; solved-amplitude difference is
8.32e-16 and the raw equation residual7.04e-14Ha. This is a bounded equation
anchor, not full-Yb convergence, experiment or a performance comparison.
Changed node/edge counts require fresh planning and backend compilation.

The compact-storage fake backend now supplies its required denominator-engine
identity, so the test reaches its intended storage guard. The reservoir module
imports its dependent `neon3` fixture explicitly. The historical mixed-CI
fixture skips without its execution-driver environment; its original coarse
derivative gate and archived failed result are unchanged. No derivative step,
error budget, state identity or scientific acceptance rule was adjusted.

## Documentation impact and group disposition

| Group | Documentation disposition |
|---|---|
| Research records and local artifact hygiene | Retain dated evidence bytes and failures; add scoped ignore rules and this account; preserve current scientific limits |
| Atomic engines, API and operators | Update/retain architecture, capabilities, state design, solver controls, migration, SD/FSCC, transition, recoil and transverse guides alongside implementation; document new auxiliary geometry and backend rebuild requirement |
| EUV, plasma and continuum | Include the source-model guide and Sn baseline with the explicit workflows; retain experimental, resonance and convergence failures |
| NIST/planning tools | Include atlas, comparison and native-roadmap documentation with the tested audit/export/resource interfaces |
| CI and test organization | Include essential-suite selection, explicit campaign boundaries, cache replay guidance and archived removed tests; keep live scientific gates separate from collection |
| Citation and offline release tooling | Include CFF/Zenodo metadata and the existing packaging guide; no publication or remote draft creation |

No new scientific insight is promoted. The existing convention-first,
shared-machinery, source-identity, evidence-preservation and cache-diagnosis
rules cover these observations. Historical performance claims remain bound
to their recorded source and representation.

## Final qualification and source chain

The [source-bound affected run](affected-tests-final.log) passes **1259 tests**,
with190 explicit optional/campaign skips and4 slow tests deselected.
[Before/after identities](affected-source-identity.json) confirm all867
production/test/tool Python and Fortran files plus pytest configuration stayed
unchanged throughout that run. The earlier successful affected replay is
retained as `affected-tests-intermediate.log`; the later run is authoritative.
Final cleanup removes surplus terminal blank lines only from one Fortran
and two Python test files, plus two guides. The
[explicit formatting chain](final-formatting-identity.json) records exact
before/after hashes and unchanged Python ASTs; the hook also normalized
the terminal newline in the CCSDT attribution file without changing its text.
The [post-format replay](post-format-replay.log) passes109 checks with15
optimized-backend skips, exercising the affected snapshot, optimizer-resume
and live original valence interfaces. Scientific gates remain unchanged.

An isolated copy containing only commit candidates and exact small test fixtures
passes **457 essential tests** ([log](isolated-essential.log)) and **82 fixture-
dependent checks**, with5 optional skips ([log](isolated-fixtures.log)).
[Opened scientific inputs](test-document-inputs.json) record their byte sizes
and hashes; the required raw exceptions are tracked explicitly with upstream
license notices. Omitted local research data cannot conceal a missing fixture
in these two isolated panels.

The wheel is checked against every current Python/Fortran source member
([identity](wheel-validation.json)) and imports from its own separate extraction
([smoke](wheel-smoke.log)): the hydrogen solve converges at
-0.5000054122283553Ha. Citation and Zenodo versions/licenses agree with0.2.0.
Full CFF schema validation and remote Zenodo operations were not run for this
cleanup; previous source-bound release results remain historical.
The first wheel-import attempt had an incorrect setup working directory and
fell back to the editable checkout; `wheel-smoke-initial.log` is retained and
is not accepted as an isolated-wheel check. The final check asserts the exact
wheel import location.

Commit hooks temporarily save unstaged changes. Test/build observations that
overlapped the first research commit are not used as final source qualification;
the subsequent source-bound affected run and isolated panels ran with stable
software. Future verification should finish before invoking a hook that may
change the worktree. No new scientific rule is promoted from this operational
observation.


## Offline archive check

The initial packaging policy omitted10 of the22 explicitly tracked fixture and
license exceptions because they lived below historical scratch directories.
The [selection check](release-selection-check.json) binds the old/new script
hashes and verifies preservation of all22 exceptions, four untracked-artifact
exclusions and rejection outside allowed snapshot roots. The policy now honors
tracked inputs before applying local build/scratch filters; no physical code,
fixture bytes or scientific acceptance rule changed in this release increment.
The current release guide documents that ordering.

The complete [offline preparation](release-build.log) succeeds. Its
[archive verification](release-archive-validation.json) checks18237 source-file
hashes, all22 fixture/license exceptions, every wheel package member against
the current source and40 current documentation entry points with zero broken
links. All output-file checksums pass. The source ZIP is201,695,229B and remains
under ignored `dist/grouped-cleanup-20261016/`; no upload or remote draft was
created. The archive records its actual inventory-time dirty source and
precedes this final validation paragraph. No full scientific suite was rerun
from the ZIP; the separately isolated source panels above and exact fixture/
source hashes provide the stated evidence.

Current-root validation after staging the final group: Ruff passes for `src`,
`tests`, `scripts`, `benchmarks` and `tools`; all40 current documentation entry
points have zero broken tracked repository links. Production/current-guide
whitespace checks pass. Archived evidence whitespace is intentionally preserved.
Every grouped commit runs the installed staged-snapshot project-index hook.
