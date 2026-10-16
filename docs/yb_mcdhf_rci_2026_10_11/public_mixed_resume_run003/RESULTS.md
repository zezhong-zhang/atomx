# Completed native 531-CSF continuation: 320 accepted steps, still nonstationary

The native public optimizer completed another 160 accepted steps on the same
531-CSF, five-state EOL Dirac–Coulomb objective. The original orbital residual
fell from **4.7170476466e−5 to 1.9434848235e−6**, a factor of 24.27, but still
fails the unchanged **1e−7** gate. The energy indicator is
**9.2253719558e−12 Ha**, passing its unchanged **1e−8 Ha** gate. Both gates are
required, so this endpoint is **not a converged 531-CSF MCDHF result**.

This is the authoritative completion successor for the archived preparation
[README](README.md) and [historical pending result](result_unverified.json).
Those original records retain their preparation/pending flags. The new
[completion proof](completion/completion.json) binds the actual executions,
final checkpoint, independent endpoint replay and saved trajectory audit.
No old source, preparation, result or array was rewritten.

## What was actually run and verified

The operator, nucleus, basis and targets remain fixed: Yb⁺ with 69 electrons,
Fermi A175 nucleus, 24 radial orbitals, complete retained 531 CSFs in 16 signed
J/parity blocks, original005 nuclear no-pair projector, original layout and
inverse blocks, and equal 0.2 weights for 6s₁/₂, 5d₃/₂, 5d₅/₂, 6p₁/₂ and
6p₃/₂. This finite core-single/valence-replacement model is not complete
core-double/CV correlation. The scientific source map remains
10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f.

- [Root solve execution](root_execution.json): one real public
  optimize_orbitals invocation, exit 0; 161 optimizer gradient evaluations,
  two separate full531 same-point baseline gradients and 252 energy trials.
  Termination was maximum_iterations; there were zero line-search fallbacks
  and zero stable-energy comparisons.
- [Independent endpoint replay](independent_endpoint_verification.json):
  **1,011 checks passed**, including a new complete native H/CI evaluation and
  one full gradient at the exact accepted endpoint. It reproduces the complete
  2,856-component projected-coordinate g and preconditioned pg with maximum
  errors **0**, passing the original **1e−9 / 1e−10** same-point gates.
  Passing these replay gates does not make the failed orbital stationarity
  gate pass.
- [Saved-only audit](completion/verification.json): **4,963 checks passed**
  in 4.478 s / 71.78 MiB under actual CPU29, one-thread, 1 GiB address-space
  and 30 s alarm caps. It imported no Atomx module and performed zero new H
  builds, CI solves, gradients or optimizer calls. The audit checks stored
  eigenpair algebra, not a second physical H/gradient replay.

The two 160-step segments use the same objective and exact accepted-point
continuation, but each starts with empty secant history. “320 accepted steps”
therefore means two native segments, not a claim of bitwise equivalence to one
uninterrupted 320-step optimizer history. The solve/replay elapsed times in the
root execution records are uncontrolled observations, not efficiency benchmarks.

## Energy changes and original-metric residual history

All 161 saved points carry complete g/pg, coordinates and immutable geometry.
The saved audit independently applies each original inverse block and the
actual x tangent projection. Its maximum full-vector pg error is **0**, as are
the residual and indicator replay errors. Every saved P/Q is **bitwise equal**
to the direct original-projector-times-x unpack. The original metric byte
digest remains aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe.

The archived diagnostic string “screened Fock spectral gaps” is inherited
metadata. This resume actually consumes the frozen original inverse blocks;
it does not reconstruct Fock gaps, change the no-pair projector, or redefine
the residual yardstick. The alternative no-pair leakage yardstick is likewise
not recomputed on resume.

The whole segment lowers the EOL energy by **6.8542249210e−7 Ha**. All 160
recorded accepted energy differences are negative; none are zero or positive.
At this total energy, one float64 spacing is **1.8189894035e−12 Ha**.

| Accepted interval | Net EOL energy change, Ha | Median recorded decrease, ULP | Decreases at most 128 ULP |
|---|---:|---:|---:|
| All additional 160 steps | −6.8542249210e−7 | 136 | 78 / 160 |
| Last 20 steps | −1.4642864699e−9 | 13.5 | 18 / 20 |
| Last 10 steps | −1.4370016288e−10 | 3.5 | 10 / 10 |
| Final accepted step | −1.8189894035e−12 | 1 | 1 / 1 |

The final residual is the minimum saved residual, although it increases across
69 adjacent steps. **No saved point passes both original stationarity gates.**
The [full trajectory](completion/trajectory_report.json) preserves every
accepted energy, residual, indicator and original-coordinate diagnostic.

The last accepted differences are observed near float64 spacing. This does
**not** prove that rounding caused the stop or that the optimizer failed.
Accepted-only records do not provide rejected trial endpoints, Armijo margins
or an independently validated exact mixed-CI energy-increment oracle. The
observed stop is the declared iteration limit, not a recorded line-search
failure. The gradient indicator and accepted energy changes are not rigorous
bounds on omitted physics or experimental error.

## Responding CI and physical state identity

Selected columns were recovered independently from the complete CSF signed
J/parity support and local-root energy order at every saved point. No global
“first five” shortcut was used. All target blocks remain local root 0; there
are no selected global-column switches. Maximum saved selected-eigenpair
residual is **2.5956412687e−11 Ha**, and selected symmetry leakage is **0**.

| Target | Phase-aligned CI coefficient change from segment entry, norm | Saved frozen-reference overlap² at endpoint |
|---|---:|---:|
| 6s₁/₂ | 7.80004e−5 | 0.99867502 |
| 5d₃/₂ | 9.91824e−5 | 0.99847108 |
| 5d₅/₂ | 9.92047e−5 | 0.99493335 |
| 6p₁/₂ | 1.12707e−4 | 0.99384058 |
| 6p₃/₂ | 1.09426e−4 | 0.99100350 |

The signed coefficient changes show responding CI in the moving orbital basis;
they are not physical overlaps between wavefunctions with different orbitals.
Physical overlap² values are the hash-bound, original-reference tracker
records, not bare-configuration probabilities, and were not recomputed by this
saved-array audit.

The parent002 archive stores x/geometry/CI but does not store g/pg arrays.
The audit compares parent x exactly and binds parent residual/indicator to its
independent 988-check proof; it does not invent a parent whole-vector g/pg
comparison. The initial consumer schema error was retained in
[failure001](completion/failure001.json) with its original script, seal,
execution and log. The corrected [execution seal](completion/execution_contract002.json)
changes only this consumer schema assumption; all scientific inputs and gates
remain fixed.

## Same lower/upper spectrum changes and NIST comparison

These are the already independently rebuilt DC endpoint rows, reordered by
physical label. The common ground energy is **−14067.494148180154 Ha**, a
**−2.7159148885e−7 Ha** change from the parent160 ground. Upper and lower
changes are kept separate; no states from different endpoints are spliced.

| Target | Absolute upper change from parent160, Ha | Same-ground transition change, cm⁻¹ | Endpoint excitation, cm⁻¹ | Signed nominal NIST gap, cm⁻¹ |
|---|---:|---:|---:|---:|
| 5d₃/₂ | +1.8172613636e−7 | +0.09949172 | 25271.65359659 | +2310.85359659 |
| 5d₅/₂ | +1.1902120605e−6 | +0.32082880 | 26069.43659720 | +1736.74659720 |
| 6p₁/₂ | −1.8654372980e−6 | −0.34980872 | 24977.51071576 | −2084.30928424 |
| 6p₃/₂ | −2.6620164135e−6 | −0.52463763 | 28476.05564704 | −1916.17435296 |

All four nominal gap magnitudes increase slightly in this additional segment;
all four nominal 1% gates still fail. Lower EOL energy is not an experimental
accuracy certificate. These nonstationary DC rows cannot establish the final
MCDHF correlation-model gap.

No static Breit spectrum was calculated at the total320 endpoint. The previous
[static RCI result](../public531_endpoint_static_rci/README_result.md) belongs
to accepted160 and retains that point identity; it must not be relabeled or
combined with the new320 DC endpoint as a same-point Breit comparison.

## Evidence and remaining discriminator

The [prospective saved-audit contract](completion/contract.json), full
[759-file input manifest](completion/input_file_sha256.json),
[execution record](completion/execution.json), [completion JSON](completion/completion.json)
and [card evidence](completion/card_evidence.json) retain the source, model,
root calls, endpoint and original gates. Completion is confined to numerical
trajectory verification, not production, highest-level or experimental
certification.

The next useful discriminator is a prospectively bounded check of stable
responding-CI relative-energy evaluation and convergence in the same original
metric before extending space scans. A numerical issue must be demonstrated,
not inferred from the final ULP alone. Missing physical correlation and operator
coverage remain a separate assessment after the declared model is stationary.

Documentation impact: this new completion page and its new proof directory
are the successor account. Frozen README/preparation/pending/array records are
retained with explicit scope; root owns current guide/index/card integration.
No production code, tests, website or existing evidence changed here. No new
general rule is promoted: the existing requirement to check the original
stationarity metric already covers this observation.
