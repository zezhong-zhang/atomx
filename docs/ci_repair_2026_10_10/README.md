# GitHub Actions repair and documentation organization — 2026-10-10

Status: final local essential suite (452 tests), affected numerical panel, documentation tests and lint pass; final essential GitHub matrix pending. Full campaigns were canceled at the user’s request and are not claimed to pass.
Base: `233fec1a9710e95cf42bf64ecb821d287d4336e2`. Repairs are isolated from the
main checkout's unrelated scientific changes in branch `fix/ci-docs`.

## Expectation and distinguishing controls

Keep the physical Hamiltonians, complete spaces, states, normalization and all
existing scientific tolerances. Change packaging and numerical algorithms only
where a fixed-input control identifies the failure. Stop each diagnostic once it
distinguishes the cause; do not scan new physical models to make CI pass.

1. Missing fixture: compare the local native Ne table with its recorded checksum.
   Accept only the identical bytes, published in `tests/data/`.
2. Nitrogen snapshot: check the independent complete signed GRASP Breit matrices
   before refreshing the internal total-energy snapshot. DC/QED and tolerance stay fixed.
3. Ionic closure: replay the identical native CI block with LAPACK `evr`, `evd`
   and `ev`; require the original `2e-14` orthogonality gate and small native
   eigenvalue residuals. Changing energy zero tests numerical covariance, not a
   different interaction or state. No post-solve projection or truncation.
4. Sodium: compare the identical HF solve at 40 and 200 iterations. Accept a
   larger fixture budget only if the original convergence gate is reached.

## Failed evidence and measured results

- [Test run 37744023823](https://github.com/zezhong-zhang/atomx/actions/runs/37744023823):
  both Python versions had two failures and four fixture errors. The missing path
  was ignored `docs/xfel_real_rates_2026_10_03/ne_1000eV.json`; nitrogen differed
  by `2.01408e-5 Ha`; magnetic closure errors reached `3.29e-14` (3.10) and
  `4.03e-14` (3.13). [Original logs](github_test_failure.log.gz).
- [Nightly run 37923003934](https://github.com/zezhong-zhang/atomx/actions/runs/37923003934):
  two sodium convergence failures, 32 passes and three strict expected failures.
  Later sequential gates did not run. [Original logs](github_nightly_failure.log.gz).
- [Local baseline](baseline.log.gz): two failures, four errors and one pass in
  45.35 s. Closure passed locally, demonstrating environment sensitivity rather
  than disproving the GitHub failure.
- [Sodium control](sodium.jsonl): 40 iterations leave residual `0.0039941691`;
  the unchanged solve converges in 46 iterations to residual `4.6380e-7`, energy
  `-162.07808078842032 Ha`, same-kappa overlap `5.39e-17`. The fixture now permits
  100 iterations and shares its read-only result across the two EDM checks.
- [Dense comparison](dense.jsonl), [captured matrices](native_blocks.npz): the
  four-dimensional clustered ionic block has `evr` orthogonality error
  `1.2152e-14`, versus `5.5511e-16` for `evd`. Native residuals are respectively
  `2.8422e-14 Ha` and `2.6645e-14 Ha`. At energy zero offsets +100/-1000 Ha,
  `evr` errors are `2.7266e-14`/`6.0547e-14`, while `evd` stays below `9e-16`.
  Full spectra use `evd`; partial spectra keep subset-capable `evr`.
  [SciPy's driver contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)
  documents these driver and subset options. The new replay gates preserve
  eigenvalues, residuals and complete signed projectors.
- [Affected panel](affected.log): **65 passed**, including every previously failing
  test file, independent signed GRASP Breit matrices, radial snapshots and cluster
  refinement. [Dense regression](dense_regression.log): **6 passed**.
- [Documentation/index tests](documentation_tests.log): **9 passed**;
  [Ruff](ruff.log) passes. All 33 entry points have valid tracked link targets.
  [Source/input identities](source_identity.json) and the compressed repair patch
  preserve the actual numerical source. Original failed logs are losslessly
  compressed, retaining their bytes and whitespace.

## Fixture provenance

`tests/data/ne_native_1000eV.json` is a byte-for-byte copy of the original native
134-configuration Ne table. SHA-256:
`17d9126593cda0a18a7e47f1d62833b125e57d5110e603eea351402765c008ae`.
[Original identity manifest](ne_original_identity.json) and
[original producer](ne_original_build.py) retain the dated source identity.
This is stored native regression data, not a fresh live rebuild or an independent
external-code calculation. The older `ne_1keV_rate_table.json` remains a distinct
model. The XATOM KLL scalar anchor remains unchanged in the consuming test.

`tests/data/ne_clustered_ci_block.json` records the native matrix consumed by the
eigensolver regression. Recreate its capture with [probe_dense.py](probe_dense.py)
on the recorded base; [probe_sodium.py](probe_sodium.py) supplies the iteration control.
Dense CI already stores quadratic-size matrices; `evd` adds quadratic workspace
(roughly 36 MB for two 1500-by-1500 double arrays). No large new physical space is
allocated. These checks make no timing or end-to-end performance claim.

## Documentation disposition and maintenance

| Change | Owning documentation | Disposition |
|---|---|---|
| Fixtures, dense solver and sodium budget | [Testing](../TESTING.md), this record | Update commands, coverage and provenance; retain original numerical gates |
| Routine test scope and serial kernels | [Testing](../TESTING.md), workflow YAML | Replace broad collection with reviewed essential selection; remove nightly and repeated heavy panels |
| Current navigation | [Documentation index](../README.md) | Group guides by task; move dated reports to [research](../RESEARCH.md) |
| Maintenance routing | [Development](../DEVELOPMENT.md) | Move the existing code-to-documentation table intact; retain its old incoming anchor |
| Historical selected-panel counts | [Validation history](../VALIDATION_HISTORY.md) | Move dated results out of current test instructions without changing evidence |
| Ignored local evidence links | Root documentation entry points | Preserve labels/paths as explicitly local evidence; do not imply GitHub availability |
| Link and index freshness | [Knowledge workflow](../AGENT_KNOWLEDGE_WORKFLOW.md) | Add CI mechanical checks; semantic/physics review remains an agent responsibility |

Original scientific archives, failed results and local in-flight additions are
preserved. Current architecture, physical capabilities and migration contracts
are retained: no physical API/default change occurs in this repair.

Distillation: refine existing knowledge-layers and preserve-metric guidance.
When a fixture passes only in a dirty checkout, verify tracked inputs; when
closure fails, repair the eigensystem before changing its threshold. Scope is
this native real dense CI and these workflow contracts. No new root rule needed.

## Essential-suite scope decision and measurement

User decision, 2026-10-10: remove long tests and target about three minutes for
routine CI. This supersedes the initial repair's independent nightly matrix;
it is a coverage decision, not a measurement that the large models got faster.
The test selection is explicit in `pyproject.toml`; affected extended regressions
are still selected by path when their implementation changes.

Fixed conditions: unchanged production source after the dense-driver repair,
original models and tolerances in every retained test, existing Python 3.10
venv and one BLAS/OpenMP/Numba thread. Hypothesis: removing broad campaigns and
redundant repeated heavy solves makes essential coverage fit the three-minute
target. Accept only passing selected gates; stop the experiment at the target
rather than launch another full campaign. Cold hosted-runner timings remain a
separate observation from this warm local checkout; no controlled speedup is claimed.

The initial [selection trial](essential_trial.log) passes **457 tests in 106.80 s**.
A collection check caught that pytest expands `testpaths` as filesystem patterns,
so a node selector did not collect determinism. The final selection uses only
file paths and splits the optional bounds diagnostic into its own module.
The determinism panel retains C, Ne and multi-configuration Be with two fresh
processes and signed wave-byte comparisons. Cf and four/twelve-repeat campaigns
are removed. The separate [determinism check](essential_determinism.log) passes in **46.38 s**.
Final default collection contains **458 tests**, including determinism. Local
selected checks total about **153.18 s** across these two invocations; the final
combined command and cold-runner runtime are verified by GitHub, not inferred
from summing these times.

Three campaign-only modules moved unchanged to the
[source archive](../archive/2026-10/long_running_tests/README.md), with SHA-256
and original commit identities. The Pr/Eu/Nd models and ns=256/320 DHF tests
were not reduced to manufacture a passing large-model result. Optional rendering
extras and Cairo/Pango setup were removed from routine CI because the essential
selection does not import those backends. Runtime target: about three minutes;
job safety timeout: five minutes including setup.

The broad local non-slow run was interrupted at approximately 41% (exit 130;
[partial log](full_non_slow.log.gz)). GitHub Test
[38032113586](https://github.com/zezhong-zhang/atomx/actions/runs/38032113586) and
Nightly [38032155079](https://github.com/zezhong-zhang/atomx/actions/runs/38032155079)
were canceled. Nightly book and bounds jobs had passed before cancellation;
slow/determinism did not complete. These partial results do not establish a
passing full suite. Lint [38032113551](https://github.com/zezhong-zhang/atomx/actions/runs/38032113551)
passed on the first repair commit.

Documentation review: update Testing, README, Slater-scaling gate locations and
historical verification and DF-control current-scope annotations; retain original archived
measurements. Register the user decision under `knowledge-layers` rather than
adding a performance claim or a root rule. No new physical insight.

### Hosted turnaround control and final selection

[Hosted trial 38032963022](https://github.com/zezhong-zhang/atomx/actions/runs/38032963022)
completed all **458 numerical tests** in 292.30 s (3.10) and 283.39 s (3.13).
The 3.10 job then hit its five-minute safety timeout; 3.13 completed successfully.
This fails the turnaround target, despite passing numerical assertions. The
[complete raw trial log](essential_hosted_trial.log.gz) is retained.

Dominant 3.13 calls: determinism 59.95 s; four-electron absolute reservoir
37.57 s; Be analytic/variational comparison 29.00 s; the first He analytic
solve 19.16 s; three-electron absolute reservoir 11.33 s. Those two absolute
reservoir tests move with unchanged function ASTs to
`tests/test_xfel_reservoir_campaign.py`, outside the essential selection.
The unchanged analytic Slater–Condon live panel is also now explicitly selected,
not routine. Signed GRASP/FAC blocks, complete CI traces, the canonical public
energy panel and the live three-electron closure remain essential. Final
expected collection: **452 tests**. This is a declared coverage choice, not a
large-space convergence or same-input performance claim.

The determinism test now reuses compiled kernels while retaining two fresh
interpreters and the identical C/Ne/Be physical inputs. The default suite
exercises ordinary compilation; the explicit bounds diagnostic still uses an
isolated cache. The [shared-cache local check](determinism_shared_cache.log)
passes in 24.01 s; the [bounds diagnostic](bounds_diagnostic.log) passes in
121.29 s. These observations are not controlled timing pairs. No production
source, tolerance, state, Hamiltonian or test-function model was changed.

[Final local default command](essential_final_local.log): **452 passed in 37.83 s**
with existing warm compilation caches. This is one observation, not a controlled
speedup or cold GitHub prediction. [Final selection identity](essential_final_identity.json)
records actual test/config hashes and unchanged ASTs of the moved campaign tests.
