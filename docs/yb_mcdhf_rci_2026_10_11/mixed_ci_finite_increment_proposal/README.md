# Native mixed-CI finite increment: reviewed preparation

This is a **proposal, not an implementation or a numerical qualification**.
No shared source, frozen evidence, threshold, target, orbital or metric was
changed. This preparation performs zero H builds, CI solves, radial gradients
or optimizer calls. The [contract](contract.json) binds current354 source
`3304f5895bcfce70708f1aa9a77bca8a7cfc3eca9a9d393380035c42ed67df2d`,
the existing [239-check finite-matrix identity](../mixed_ci_stationarity_probe/mixed_difference_algebra.json),
the [earlier conditional design](../mixed_stable_increment_design/README.md),
and the actual accepted320 and retained-H qualification/performance evidence.
The earlier source353 design remains historical and unchanged.

The next source edit and finite qualification require a root seal of this
concrete contract. Full531 diagnostics and optimizer adoption require separate
authorization after that qualification. No additional unchanged320 optimizer
iterations are proposed here.

## Public-path audit and purpose

The source names the user paths `Atom.solve(...)`, `MCDHF.solve(...)` and
`MCDHFSolver.solve(...)`; there is no `solve_mcdhf` symbol under `src/atomx`.
Both public MCDHF facades construct the shared variational CSF solver. Its
correlated B-spline branch constructs `CSFOrbitalFunctional` without opting
into Hamiltonian retention; it supplies no H-consuming archive callback.
The optimizer already passes **the actual accepted trial CI object** into its
full gradient and then to the accepted-point checkpoint callback. No extra CI
solve is needed there. The retained-H checked-chain median speedup of
[1.242593810757533](../retained_hamiltonian_performance/RESULTS.md) covers the
measured archival consumer. It does not establish a gain for ordinary public
MCDHF without that consumer.

Consequently this proposal adds neither an unused public solver retention flag
nor checkpoint matrix data in freeform provenance. The already qualified
`CIHamiltonianSnapshot` provides the actual H needed by a concrete mixed-CI
increment consumer. Initial reference/trial CIs must explicitly retain their
matrices; the consumer obtains each by the existing strict `hamiltonian_for`
interface. It cannot accept an arbitrary H or synthesize a missing owner.

The accepted320 original residual is **1.9434848235489216e-6**, still above
1e-7. Its indicator is 9.225371955806855e-12 Ha, below 1e-8 Ha. Those original
gates remain unchanged. Neither large core energy nor a small final decrement
demonstrates a rounding obstruction. The completed public runs retained
strictly decreasing accepted EOL and zero line-search fallbacks. The finite
increment is reusable numerical work; it is not a diagnosed cure for the
remaining residual or experimental gap.

## Exact rank-one algebra and semantic domains

First resolve the **same physical rank-one target** at both radial endpoints,
in the same complete real signed ordered CSF representation. Keep the original
eigenvectors c0/c1 and save one whole-column sign sigma in {-1,+1}. Choose it
only after physical target matching; never rephase individual CSFs, rebase the
frozen reference, or select another root to reduce the coefficient difference.

Let n_i = c_i.T c_i > 0, x0 = sigma*c0, d = c1-x0,
A0 = H0-lambda0*I, r_i = (H_i-lambda_i*I)c_i and r0bar = sigma*r0.
Then the exact finite reported-root identity is

```
lambda1-lambda0 = (
    c1.T deltaH c1
    + d.T A0 d
    + 2*d.T r0bar
    + x0.T r0bar
    - c1.T r1
) / n1
```

The trial radial term is contracted with the **complete unfiltered signed**
`contract_bilinear(c1,c1)/n1`. Sum target densities using the unchanged
ensemble weights, before streaming radial source pairs. Keep every term,
including negative weights and off-diagonal interference; remove exact zeros
only. The CI norm n1, each orbital ket norm and ensemble normalization are
three separate quantities.

For the quotient domain, independently form q0 = c0.T H0 c0/n0,
A0q = H0-q0*I and r0qbar = sigma*(A0q*c0). Replace the response terms accordingly
and omit the trial reported-root residual offset. The full identity is

```
q1-q0 = (c1.T deltaH c1 + d.T A0q d
         + 2*d.T r0qbar + x0.T r0qbar) / n1
```

Retain x0.T r0qbar numerically even though it vanishes in exact arithmetic.
Report both domains separately, plus q_i-lambda_i = c_i.T r_i/n_i and the
floating EOL averaging/subtraction offset. The reported-root identity cannot
recover information already lost by stored lambda rounding. The quotient
domain is the quotient of retained vectors in the retained matrices, not a
claim of exact eigenvalues or silently changed production EOL.

Center diagonal entries **after lifting** H and lambda/q into the declared
working dtype. Archive the actual mantissa bits; `longdouble` is not a promise
of IEEE quadruple arithmetic. No finite residual offset may be omitted merely
because a CI residual gate passed.

## Native radial operation

For each ordered one-body h_ab = N/s, with N = a.T h_D b and s = b.T S b,
preserve the full quadratic increments

```
deltaN = da.T h_D b + a.T h_D db + da.T h_D db
deltas = db.T S b + b.T S db + db.T S db
deltah = (deltaN - h_ab*deltas) / (s+deltas)
```

Keep h_ab and h_ba ordered. The angular contraction must reproduce the
public real Hamiltonian's final symmetrization and canonical column permutation.
Both s and s+deltas must be finite and positive.

For each component pair density rho(a,b), retain
delta-rho = rho(da,b)+rho(a,db)+rho(da,db). For the same native cell operator K,

```
deltaR = <delta-rho_ab, K rho_cd>
         + <rho_ab, K delta-rho_cd>
         + <delta-rho_ab, K delta-rho_cd>
```

No cubic/quartic contribution is discarded. The proposed physical helper uses
the **same bank/banded** operator as `matrix._radial_evaluator`, all pp/qq/pq/qp
components, its component orders/extents, sorted within-pair convention and
actual source/target orientation. It does not use the functional's ns>=96
semiseparable `_pair_potential` as an unqualified replacement. Near-roundoff
arithmetic closure and continuum/radial accuracy are separate questions.

The exact pair-band loop may be factored into one private function in
`physics/csf/matrix.py`, with identical order, scalar dtype, allocation shapes
and kernel calls, reused by the ordinary evaluator and new finite operation.
No kernel migration, new fastmath flag, global cache or matrix override is
part of this edit. Its ordinary H/default-CI bitwise parity must pass before
qualifying the increment. Existing fixed-density and constrained APIs, their
guards and their numerical path remain unchanged.

Explicit radial increments must come from the actual coordinate map P*delta-x;
the diagnostic archives them and the endpoint reconstruction difference.
It must not silently replace them with subtraction of two rounded endpoints.
The exact algebra above assumes deltaH = H1-H0. The radial polynomial instead
uses the declared reference plus explicit increment; that mathematical endpoint
can differ from the separately rounded stored trial. Keep its reconstruction
defect and its independently measured energy contribution separate from
radial-contraction error. The diagnostic must not label a nonzero unresolved
endpoint contribution as an exact reported-root identity. The Decimal oracle
evaluates both endpoint domains and the complete finite difference between them.
The native float64 contraction, including existing fastmath bank arithmetic,
has its actual dtype and measured oracle error recorded. Final compensated
sums do not certify the preceding angular or Coulomb products.

## Exact module scope and interface proposal

| File | Smallest responsibility |
|---|---|
| new `numerics/ci_energy_increment.py` | Pure real-array shifted rank-one response, finite norms/residuals/phases and separate reported/quotient decompositions; imports no methods/physics |
| `physics/csf/matrix.py` | Extract only the existing B-spline pair-band loop without changing ordinary evaluation ordering |
| new `physics/csf/radial_increment.py` | Complete ordered ket-normalized DC finite radial change using that shared native cell operation |
| `methods/csf/variational.py` | New diagnostic consumer validating endpoint owners, target identity/weights and signed trial density, then composing the two operations |
| new affected finite tests | Independent arithmetic/physics controls and ordinary-path bitwise regression |

Proposed signatures, **not implemented capabilities**:

```python
# Pure arrays; no arbitrary-H path exposed through the physical method.
real_rank_one_response(H0, H1, c0, c1, lambda0, lambda1, *, phase)

# Physical operation; caller's lifetime is one validated endpoint pair.
bspline_dc_radial_increment(reference, increments, workspace, Z, ns,
                           *, nuclear_potential, cache)
# Returns a radial-key evaluator, not a new CI solve.

# New diagnostic-only method. No float adapter or implicit optimizer wiring.
CSFOrbitalFunctional.mixed_ci_energy_difference(
    reference, trial, reference_ci, trial_ci, *, increments)
```

The diagnostic returns distinct reported and quotient totals, per-target
responses/norms/residuals, full signed radial term, original phases and bindings,
endpoint reconstruction/Gram defects, working dtype and explicit resolution
status. A measured Decimal discrepancy is an oracle measurement, not a rigorous
general interval. No certified decision bound is claimed in the first version.

Reject before radial contraction for charge functional/subclass, complex data,
non-DC/Breit/shift/effective operators, changed nucleus/workspace/keys/signed CSF
order, missing/subset owners, nonfinite/nonpositive norms, ambiguous physical
matching, changed weights/target, or selected rank>1/degenerate-span targets.
Reuse actual full-H eigenpair and Gram validation; owner integrity alone does
not validate mutable CI amplitudes. Unsupported requests explicitly raise and
perform zero additional H/CI work. No result invokes the optimizer.

## Bounded independent finite qualification to seal next

The proposed cap is CPU1, 3.95 GiB and 300 seconds, **18 H builds / 13 complete
CI solves / 3 complete radial gradients / zero optimizer calls**. Declare
started/completed counts, wall/RSS and progress failures; stop at the cap and
preserve every failure. This preparation did not run that experiment.

Use the existing signed Helium fixture: Z=2, point potential -2/r, ns35/ks9,
r_min=.005/r_max=30, all nine NR configurations
`1s2`, `1s1 2s1`, `1s1 3s1`, `2s2`, `2s1 3s1`, `3s2`,
`2p2`, `2p1 3p1`, `3p2`. Enumerate every allowed CSF/J block with the unchanged
default determinant cap; no active-space truncation is permitted. Choose the
same two resolved J=0 even isolated levels as the existing fixture, with
weights 1/4 and 3/4 and a frozen physical reference. Their complete signed
off-diagonal character must be nonzero; nominal selectors alone are not the
matching gate.

The 13 CIs are one baseline plus two distinct deterministic physical tangent
directions, both signs, and h in {1e-4,1e-6,1e-8}: twelve endpoints. A zero step
uses the saved baseline rather than another solve. Five separate ordinary H
builds are the baseline and four largest signed endpoints. The three gradients
are the candidate baseline, its frozen-source baseline comparator using the
same validated saved complete CI, and one largest signed candidate endpoint;
the caller supplies already solved CIs. No hidden warmup/CI, no optimizer and
no timing claim.

Use seeds 413711 and 413712 with NumPy PCG64 standard-normal arrays in canonical
kappa/key order. In each original positive-space coordinate group form
eta = z-x0*sym(x0.T*z), normalize its combined Euclidean norm, and use the
existing polar retraction of x0+h*eta. x0 is the original coordinate selector
for the nuclear positive-space fixture orbitals, rather than a coordinate
recovery from rounded waves. Compute delta-x after the actual retraction and
explicit delta-wave=P*delta-x, preserving P/x/seed/directions in the archive.
Each corresponding full waveform is separately P*x. The actual test source and
its generated input hashes must be sealed before evaluation; no favorable
direction or phase is selected after seeing a decrement.

Use stdlib `decimal` at 80 digits for a separate complete oracle. Convert
**every** retained float64 angular projection/table, one-body/overlap block and
native cell bank coefficient exactly with `Decimal.from_float`. Independently
contract the full signed determinant/CSF matrix, canonical permutation and real
symmetrization at both endpoints, including ket division. Decimal is an
arithmetic test interpreter for the same frozen native discretized operator,
not another electronic-structure backend. Source-pair caching may reuse exact
Decimal contractions; it cannot omit keys or lower precision to fit the cap.

The test matrix must include zero/forward/reverse/tiny increments, both
directions, original full-column sign flips, nonunit CI norms and deliberately
nonzero finite residuals, large common matrix offsets, off-diagonal one-body
norm conventions, all P/Q components, a negative signed contraction and the
response term itself. Synthetic pure-array cases use the same response formula
with independent Decimal dense arithmetic and do not consume additional H/CI.
Their finite reported and quotient domains stay distinct.

Seal the pure-array fixture as H0=[[2,.3],[.3,3]],
H1=[[2+4e-8,.3+8e-8],[.3+8e-8,3-3e-8]], c0=(.6,.8),
c1=(.61,.795), lambda0=2.75 and lambda1=2.75-2e-8, interpreted as the
actual binary64 assets. These deliberately have finite residuals. Include
common offsets 0 and 1e8, independent amplitude scales 1, 2.3 and .71,
and either original whole-column sign. The residual identity applies to
arbitrary finite lambdas; these fixtures must not be mislabeled as solved
atomic eigenstates. Decimal independently calculates the norms, both complete
quadratic forms and all residual terms from the stored assets.

The minimum affected controls are:

- `ordinary_bank_factorization_is_bitwise_frozen354` at the five complete
  independent-H comparator points, including off-diagonal and all J blocks.
- `default_CI_source_and_keywords_unchanged` and
  `default_full_gradient_is_bitwise_frozen354` at the shared baseline, with
  actual same-point CI/signed density and no norm/weight substitution.
- `mixed_rank_one_terms_match_independent_Decimal` for both semantic domains
  on all twelve complete signed endpoint changes and zero.
- `finite_phase_norm_residual_and_common_offset_identity`, retaining the
  changed phase on its residual vector and offset.
- `complete_cell_pair_polynomial_and_reconstruction_domains`, with all
  linear/quadratic pair terms, mixed P/Q and forward/reverse changes.
- `unsupported_scopes_reject_before_work` and
  `legacy_fixed_and_constrained_mixed_guards_unchanged`, with zero H/CI/gradient
  counters for the refusal calls. Existing verified mixed/tracked rejection
  remains a valid result; the new diagnostic does not weaken it.

These are concrete planned tests, not a passing suite. Counters cover all
physical work in the finite qualification; a later broader regression batch
needs its own explicit resource/call scope and cannot be hidden inside 18/13/3.

Separate gates are required: ordinary H/default-gradient bitwise parity with
the frozen source, byte-unchanged CI solve implementation/defaults; full
public-H Rayleigh/eigenpair/Gram gates; exact zero and phase/norm identities;
complete signed Decimal finite-increment error and sign margin. Proposed new
finite-test limits are 1e-12 Ha absolute for either full finite semantic total,
1e-13 Ha absolute for the signed radial polynomial, and error at most 10% of
the independently retained nonzero decrement whenever its magnitude exceeds
1e-13 Ha. Pure-array response error must be <=1e-13 Ha at the sealed small
offset and <=1e-11 Ha in the declared large-offset stress case; this distinction
is retained explicitly. These are finite-test gates, not general error bounds
or permission to accept an Armijo step. Exact test vectors/directions, the
eight operation scopes and source epoch must be root-sealed before execution.
Original H/eigenpair, Gram and g/pg gates are not replaced. A tiny-step
unresolved error or a bank extraction parity failure is retained as a failed
qualification, not hidden by looser original scientific gates. This proposal
does not certify 1e-14 accuracy of the future polynomial.

Any later full531 test binds accepted320 or another separately verified point,
the same five target UIDs/weights, full radial pool and original P/inverse metric.
Original g/pg equality limits 1e-9/1e-10 and stationarity gates 1e-7/1e-8 remain.
It needs a separate resource/call contract. A production Armijo adapter needs a
qualified decision error bound for the same objective, with unchanged 1e-4
coefficient; it is excluded from the finite implementation scope here.

## Disposition

Only this new proposal directory is written. The [read-only verifier](verify_preparation.py)
and its [result](verification.json) check complete source identity, actual
API boundaries, historical input hashes, local links and unchanged protected
scientific files without importing Atomx or NumPy. Numerical tests above are
**planned**. Frozen qualification/performance/checkpoint evidence is not rewritten.
The first metadata invocation checked the generated `verification.json` link
before that output existed; its [failed result](preparation_failure001.json)
is retained. A rerun after generation resolves that link, without any scientific
or contract relaxation. The verifier import formatting was then fixed by Ruff.
Root owns any source edit, finite implementation seal, current navigation,
later full531 qualification and adoption decision.
