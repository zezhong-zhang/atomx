# Retain the exact same-point CI Hamiltonian for checkpoints

This is a source-bound design, **not an implemented optimization or a measured
speedup**. No new H build, CI solve, gradient, optimizer call or scientific
source edit is performed. The live run003 source and all its frozen preparation
files stay unchanged. The [contract](contract.json), [source identity](source_identity.json),
[source audit](source_audit.json), [proposed diff](proposal.diff) and
[verification](verification.json) define the preparation cutoff.

## The measured inputs and source-level redundant work

The original source map is10d481, complete353 Python files. The archived
[CI producer](reference_source/atomx/methods/ci.py) builds H once but returns
only energies, vectors and freeform provenance. The
[functional](reference_source/atomx/methods/csf/variational.py) calls that producer
for an energy trial; the [optimizer](reference_source/atomx/numerics/bspline/orbital_optimization.py)
already forwards an accepted `trial_ci` into the same-point gradient, without
another CI solve. Its accepted checkpoint callback receives this exact CI.

The [frozen run003 callback](../public_mixed_resume_run003/run.py) nevertheless
calls public `build_ci_matrix` again to archive fullH and check residuals.
This design removes that redundant callback build only. Full signed densities,
radial gradients, all531 eigenpairs, the original P/S/x/inverses, physical
tracking and both1e−7/1e−8 gates remain. Uncached independent endpoint replay
still rebuilds H/CI and evaluates a fresh full gradient.

Completed [run002](../public_mixed_resume_run002/result_unverified.json) records
161 accepted checkpoints,202 energy-trial calls and161 optimizer-gradient calls
plus two baseline gradients. Each successful checkpoint's frozen callback
contains one redundant full-H build. Thus161 such callback calls are justified
by source and saved checkpoint records. The202 energy-trial counter is not an
instrumented count of completed H builds; exceptions can occur inside an
evaluation. A hypothetical366 total builder calls assumes all202 energy trials
complete their normal path and includes two baseline builds. It is not an
observed count or a timing result. Future builder counters must distinguish
attempts, completed builds, accepted points and independent controls.

One dense531 float64 H occupies2,255,688 bytes, about2.15MiB. The retained
byte-owned copy adds this much per live CI result; accepted and trial results
can coexist, so account for at least two retained matrices as well as the
producer's live matrix/block copies. Do not retain an unbounded trial history.
No additional full workspace/radial copy is necessary for the proposed owner;
its input binding contains existing public content hashes.

## Smallest reusable public path

Use a new physics-owned `CIHamiltonianSnapshot` and public
`build_ci_hamiltonian_snapshot`, then an opt-in
`ci_solve(..., retain_hamiltonian=False)` field
`CIResult.hamiltonian_snapshot`. A checked `CIResult.hamiltonian_for(...)`
delegates to the typed owner. The getter requires the actual model/point/operator
inputs and returns the existing immutable dense matrix. There is no caller
`matrix_override`, public arbitrary-matrix import, or matrix inside provenance.

The existing public `CIReferenceBinding.capture/validate` already binds signed
P/Q coefficients, full B-spline workspace, Z/ns and sampled nuclear values.
The [full finite-CI producer](reference_source/atomx/methods/ci_effective.py)
already reuses it with an empty core ledger. Reuse that same public path with
all provided radial keys and an empty core ledger. Keep the keys explicitly
beside their ordered spinor hashes. Add the complete ordered CSF signatures,
electron count/J/parity, orbital-basis keys, declared DC operator terms, units,
representation/schema and determinant resource cap. A signature's `.key` lacks
parity, so parity must be explicitly bound. Actual source/objective/P/S/x/inverse
identity remain separate scientific-checkpoint manifests.

The producer captures this binding before the unchanged public matrix build,
verifies it and signed CSF order again after the build, then owns finite
float64 matrix bytes via `np.frombuffer(H.tobytes(), dtype=np.float64)`.
The bytes backing prevents `setflags(write=True)` and writable alias mutation.
It preserves every returned entry, including residual asymmetry; no
symmetrization, truncation, zeroing or energy shift is permitted. Caller mutation
between builds/getters rejects. Parallel mutation during a single mutable-input
build is unsupported; owned `CIRadialSnapshot.solve` is the existing stronger
immutable-input option when required.

`CIRadialSnapshot` is an integral evaluator owner, not a matrix owner. Reusing it
still rebuilds angular projection/H. The new owner can consume that existing
optional snapshot through the public matrix builder; its checked binding does
not inspect private radial snapshot fields. `solve(**options)` already forwards
the proposed retention option, so no radial snapshot implementation change is
needed for this first increment.

First support is explicit: complete real dense B-spline DC only. Retention
rejects grid, Davidson/matrix-free, subset spectra, Breit/frequencies, effective
one-body or scalar shifts before a build. Existing `retain_hamiltonian=False`
behavior for those routes remains available. Coverage extensions need their own
typed operator/binding contracts; a bare DC label must never disguise extra terms.

The functional's opt-in flag is passed to every energy trial, since acceptance
is not known before CI returns. An accepted trial therefore carries its exact
H through the current optimizer without another solve. A supplied CI result
with an owner must validate the actual point before RDM/gradient contraction.
Existing explicitly verified legacy CI remains usable for gradient-only
controls, but a missing owner must reject a reuse checkpoint; no silent full-H
fallback in a supposedly measured reuse path. The baseline saved-CI comparator
is distinct from accepted checkpoint promotion.

The callback retrieves checked H, retains its full eigenpair/Gram/Herm/radial/
no-pair/target checks and writes exactly the same complete archive. Snapshot
fingerprints/source/model/coordinate SHA join the manifest. Equal physics plus
valid result eigenpairs is mandatory: possession of a typed H does not validate
mutable CI energies/vectors or certify stationarity.

## Source-bound qualification and performance decision

A future implementation uses a new archived source epoch and prospective
contract. Never change live10d481/run003 or reuse their byte-source identity.
Bind the saved160 coordinates, full531 model, original target tracker, nucleus,
operators and metric; explicitly review the only-source-field objective epoch
transition. The saved160 endpoint is nonstationary. Its independent988-check
reference anchors the new source, not a new stationarity pass.
Root has since completed320 steps and a1011-check independent endpoint replay.
Prefer that final checkpoint's exported complete g/pg as the future same-point
reference, after binding its exact final hashes in a new runtime contract. This
design keeps its earlier immutable input cutoff and does not claim to have
replayed or bound the newer physical endpoint.

Before timing, compare retained and fresh public H entrywise, all roots, complete
symmetry-block residuals/Gram, signed density contractions, full radial/projected
g and original pg/residual/indicator. H must be bitwise unchanged because the
builder/arithmetic remain the same. Keep all existing scientific tolerances;
phases/complete degenerate spans are compared as projectors/densities rather
than arbitrary saved columns. Immutable-owner mutation and stale orbitals,
workspace, nucleus, signed CSF row order, key labels, terms/cap and missing-owner
controls must reject. These are proposed qualification gates, not tests run here.

Instrument public builder and CI calls in a separately admitted same-point
panel. For each successful CI solve, build once and retain; accepted saved-CI
gradient and checked callback build zero additional H. Keep one separately
fresh independent endpoint build/CI/full gradient. Count caches and retain no
unbounded snapshots. The full531 versus exact target261-block representation
is a separate increment; neither a partial owner nor a target-only spectrum may
be labeled full531. Qualify any later composition against both complete scopes.

The prospective qualification cap is CPU1/3.95GiB/120s, at most three public H
builds, two CI solves and two full gradients, zero optimizer calls. The later
same-point panel cap is CPU1/3.95GiB/240s, five warm old/new pairs plus one cold
pair, at most20 H builds/12 CI solves/12 full gradients and zero optimizer calls.
After all successful normal-path pairs, expected H counts are old12/new6.
These are proposed caps, not actual runs. Resource/identity failure stops;
raw failures remain and the full physical problem is never shrunk.

With one CPU/BLAS thread and admitted host resources, alternate at least five
warm old/new pairs; report cold separately. Same point, model, target, coordinate
steps and gates are fixed. Include full input binding, byte ownership, residual
checks, compression/fsync and callback costs. Record peak RSS and actual completed
builder/CI/gradient counters. Hash scans can outweigh a cheap cached H build.
First require redundant callback builder count1→0, then measure net callback
and full same-point energy/gradient/checkpoint pipeline cost. That is a stage
measurement, even when its complete local pipeline is included. Adopt as a
performance improvement only if those paired medians improve with no
gate/resource regression. Whole-solve speedup requires separately authorized
complete public optimizer trajectories at the same starting point and gates.
Failed load guards mean observational times only. No speedup is inferred from
the removed call count or historical run duration.

The [readonly audit runner](verify_design.py) hashes existing evidence, parses
source/AST and checks links/schema. It performs no Atomx import, H/CI/gradient
or optimizer call and runs no new physics tests. No implemented capability or
global guidance is changed by this design; the existing source-bound reuse and
preserve-metric principles remain applicable.
