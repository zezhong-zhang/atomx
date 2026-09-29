# Full first-20 native shooting campaign and weak-exchange repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Queue and provenance

`generate_native_shooting_nist.py` now schedules the complete existing H–Ca
configuration queue, rather than only ground states or an excited-state panel:
**7,543 configurations, 15,086 two-grid attempts**. It includes the 17 calcium
configurations previously classified as basis-limited. The source ledger is
read-only; new native results are written to independent SQLite databases.

The default pair is 3,200/6,400 points. For diffuse boxes, the initial grid
increases in blocks of 3,200 to make the estimated outer-cell core log step
`2*Z*R/N` at most 60, below the kernel's rejection threshold of 100. The
finer grid doubles that count. This is an initial resolution policy, not an
error certificate. Actual refinement differences and failures are retained.

The coordinator alone writes results, holds an OS-level output lock, and
commits completed attempts individually. Worker crashes leave uncommitted
attempts pending. Source maps, frozen job-input identity and settings must
match on resume. File-change checks stop execution before committing results
under changed source. The guard was exercised when the numerical repair below
changed the solver during development: it stopped the old campaign and released
the coordinator lock. A tool-session interrupt alone was not sufficient evidence
that the redirected coordinator had stopped; the lock and terminal traceback
were checked subsequently.

`--limit` bounds a single invocation without reducing the stored queue. The
current validation tranche has a 750-attempt limit, covering H, He and the
beginning of Li. Resume the same database without `--limit` once its diagnostics
have been reviewed. An existing failed attempt is retained, not silently retried
or overwritten. New physics/numerical source requires a separate revision.

## Input coverage audit

The archive contains **35,802 standard observed H–Ca line rows** with positive
wavelengths. Their `(Z, charge, row_index)` keys exactly match the first-20
`line_jobs` keys. There are no extra queue keys or absent observed keys, and
every nonempty normalized endpoint configuration has a configuration job.
Evidence: `native_shooting_first20_input_coverage.json`.

This proves queue inclusion, not completed calculation or correct resolved-line
assignment. The source ledger still distinguishes 2,790 missing classifications,
93 unsupported configurations and other unresolved/averaged cases. Its old
comparison statuses are not copied as new shooting comparison results.
Separate isotope tables are outside this particular coverage count; their
calculation coverage and the original NIST plasma-diagnostic gap remain open
parts of the overall task.

## Failures exposed by the full queue

The first campaign (`native_shooting_first20.sqlite`) stopped with 135 successful
and 288 failed committed attempts after a source-change guard. The small panel
had missed a dense group of helium Rydberg failures.

### Retain normalization bounds near poles

For a fixed source, the squared norm of the resolvent solution is

```text
N(e) = sum_j |<j|chi>|^2 / (epsilon_j-e)^2.
```

Between poles, its second derivative is positive. On the increasing branch,
`N<1` gives a lower normalization-root bound and `N>1` gives an upper bound.
The solver previously discarded this information. A small pole overshoot could
restart at `1.5*e`, consuming the Newton limit repeatedly for weak exchange.
The repaired corrector retains those bounds alongside the pole-count bracket.

He I 1s10d then converged. Its corrected energy agrees with the existing
B-spline ledger to approximately 1.5e-10 Ha. No Newton limit or convergence
tolerance was loosened to make it pass.

### Resolve an exchange-driven pole below floating-point resolution

He I 1s13i had source amplitude around 1.6e-11, but the required pole shift
was below a floating-point spacing. An apparently converged energy step left
the squared norm near 68 and introduced an unphysical outer lobe. Removing
the node check would have accepted a damaged wavefunction.

The numerical fallback instead separates the local eigenstate:

```text
psi = a*phi + u,       <phi,u> = 0,
e = epsilon - <phi,chi>/a,
a = sqrt(1-<u,u>),    (H-e)u = Q*chi.
```

An off-pole constrained inverse computes the orthogonal response through a
resolvent iteration. It retains the entire exchange source and finite radial
box. Normalization, overlap and iteration checks must pass; responses outside
the weak-source regime are rejected. The fallback handles both unresolved
normalization near a pole and a singular full-box Green action for an
unconstrained orbital. The ordinary node guard remains active on ordinary
solutions; the separated solution's reference state is selected by the local
bound-state solver and verified by its small orthogonal response.

He I 1s13i and 1s11i now converge. He 1s13i's corrected energy agrees with the
existing B-spline ledger within approximately 1.5e-10 Ha.

## Verification and limits

- Three queue tests passed: diffuse-grid allocation, resume/failed-attempt
  preservation, and source/job identity rejection.
- Three weak-exchange tests passed, including an exactly specified two-state
  source with a sub-ulp pole component. Its nonzero orthogonal response is
  explicitly checked, so dropping exchange would not pass.
- Independent B-spline comparisons cover He 1s10d and 1s13i with a 2e-9 Ha
  energy bound. The diffuse B-spline orbital-change residual stagnated around
  1e-8 on the selected mesh; its control uses a declared 1e-7 orbital tolerance
  and an additional outer-mesh refinement, retaining the 2e-9 Ha energy bound.
  No 1e-10 B-spline orbital convergence is claimed.
- Twenty-four existing open-shell, exponential-quadrature and full-shooting
  checks passed during the repair. This is not a full test-suite run.

Current campaign: `native_shooting_first20_pole_stable.sqlite`. Earlier
`native_shooting_first20.sqlite` and `native_shooting_first20_pole_resolved.sqlite`
are retained diagnostic revisions and do not match the final source. The
current revision completed its 750-attempt validation tranche with 750 successes
and no failures, including all H/He configurations and the start of Li. Its
14,336 remaining attempts were then resumed without an invocation limit. Use
the live snapshot command for subsequent counts:

```sh
python scripts/summarize_native_shooting_campaign.py
python scripts/render_native_shooting_benchmark.py
```

The interactive report includes a separately labelled full-campaign coverage
table with snapshot time, source-match status, failures and pending attempts.
The earlier ground/centroid benchmark panels remain versioned controls; they
are not relabelled as results of the newly expanded campaign.

The full first-20 queue, isotope calculations, NIST matching review and
all-element generation remain unfinished. Large physical-model discrepancies
cannot be inferred to disappear from these convergence repairs.
