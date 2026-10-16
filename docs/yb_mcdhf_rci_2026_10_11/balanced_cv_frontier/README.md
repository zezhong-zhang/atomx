# Yb⁺ balanced core-single / valence-replacement frontier

This is verified enumeration and signed angular support preflight, **not a computed spectrum**. It uses the stationary [native005 MCDHF baseline](../native_run/nuclear_eol005_projected_increment/result.json), its identical 24 radial-orbital pool, 69 electrons and frozen source map `9dff33cc24acecff9e821c4f9b836e02efca0d615291d6f92ff5a7014a726c3f`. No shared source, old native evidence, orbitals, metric or scientific gates were changed.

The [prospective contract](contract.json) was written before enumeration. From each bare `[Xe]4f14 6s/5d/6p` reference, remove the sole valence electron and one 4f, 5p or 5s core electron, then add every unordered pair in `{6s,5d,6p}`: `6s²`, `6s5d`, `6s6p`, `5d²`, `5d6p`, `6p²`. All three starting references lead to the same 18 candidate occupation configurations; their parent identities remain recorded. The three bare configurations add five CSFs. Every Pauli-allowed jj split and signed coupling tree is retained in every J/parity sector.

## Exact space and resource counts

| Space | CSFs | J/parity blocks | Largest CSF block | Largest raw M=J block | Dense CSF matrices, bytes | Raw-pair work upper count |
|---|---:|---:|---:|---:|---:|---:|
| bare | 5 | 5 | 1 | 1 | 40 | 5 |
| 4f | 289 | 16 | 33 | 135 | 56,472 | 99,791 |
| 5p | 174 | 12 | 27 | 85 | 27,520 | 27,943 |
| 5s | 63 | 10 | 10 | 29 | 3,960 | 3,131 |
| nested_4f | 294 | 16 | 33 | 136 | 58,160 | 100,978 |
| nested_5p | 468 | 16 | 56 | 209 | 156,224 | 230,077 |
| nested_5s | 531 | 16 | 64 | 232 | 205,944 | 284,992 |

The full union has 531 CSFs in 16 complete blocks, with exact full magnetic dimension 3,384. Raw M=J dimensions include higher-J states before CSF projection. They are exact integer counts from the public `pattern_m_counts`, not rounded estimates. The unchanged public per-block determinant cap is 2,000; the union maximum is 232. The full dense raw-determinant matrix storage arithmetic is 2,279,936 bytes over all blocks. These byte totals exclude angular caches, Python objects, sparse-table work arrays and radial workspace; they are not resident-memory bounds. The combinatorial `sum(raw²)` work count is an upper count, not measured timing.

| Hole family | ss | sd | sp | dd | dp | pp | Total CSFs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4f | 2 | 39 | 24 | 81 | 113 | 30 | 289 |
| 5p | 2 | 23 | 18 | 45 | 65 | 21 | 174 |
| 5s | 1 | 8 | 7 | 16 | 23 | 8 | 63 |

The independent occupation-only dimension is `hole_capacity × C(18,2)` because the particle pool has 18 magnetic spinor states. This gives 2,142 / 918 / 306 magnetic states for the 4f / 5p / 5s families, matching the complete CSF degeneracy sums and independent public integer M-scheme counts. Each configuration also passes `n(J)=N(M=J)−N(M=J+1)` and `count_j_resolved_csfs` against the explicit signed generator.

Full configuration strings, parent substitutions, orbital keys, signed coupling trees and per-configuration counts are in [enumeration.json](enumeration.json). Complete block raw dimensions and exact integer resource arithmetic for every family, bare-plus-family and nested union are in [preflight.json](preflight.json).

## What can couple to the lower and upper states

Public `CIRadialDensity.contract_bilinear` evaluates signed angular radial-key weights of the scalar Dirac-Coulomb operator between each bare CSF and every family CSF in its complete matching J/parity block. Public `verify_csf_span` checks the signed J eigenspace and Gram matrix. Public `canonical_static_radial_key` combines reciprocal static Coulomb keys; all coefficients, including tiny and zero values, are archived. The 1e−11 cutoff classifies robust support only and never removes a CSF or coefficient from the physical space. Cancellation-sized nonzero coefficients remain explicitly ambiguous. Static Breit angular coupling is not measured here.

| Bare target | 4f robust CSFs (ambiguous) | 5p robust CSFs (ambiguous) | 5s robust CSFs (ambiguous) |
|---|---:|---:|---:|
| 6s1/2 | 11 (1) | 17 (0) | 7 (1) |
| 5d3/2 | 22 (1) | 24 (0) | 10 (0) |
| 5d5/2 | 30 (0) | 21 (0) | 9 (0) |
| 6p1/2 | 13 (0) | 16 (1) | 7 (0) |
| 6p3/2 | 25 (0) | 27 (0) | 10 (0) |

The omitted `4f13 5d6p` pair family supplies 11 robust ground-6s angular couplings. In contrast the old `4f13 6s6p` family has one same-symmetry ground CSF with only 1.74e−17 canonical angular weight, recorded as ambiguous; the old `4f13 6s5d` family has the wrong ground parity. The full 5p family supplies ground support through `6s6p` and `5d6p`; the full 5s family supplies it through `6s²`, `5d²` and `6p²`. The D/P targets have support in all three hole families through different pair sectors. Thus the old [68-CSF fixed005-orbital experiment](../cv_discriminator/same005_orbitals001/result.json), where the lower 6s energy stayed unchanged while P fell by about 600 cm⁻¹, tested a structurally unbalanced subset. It did not test the complete declared two-particle frontier.

**Angular support does not certify a nonzero physical matrix element or predict an energy decrement.** Real radial integrals can cancel, including stationarity/Brillouin effects for single substitutions. No radial integral, Hamiltonian energy, diagonalization, candidate experimental gap or expanded-space MCDHF stationarity was computed here. See all raw signed coefficients and detailed per-configuration support in [angular_support.json](angular_support.json).

## Next discriminating calculations

1. Fixed005-orbital complete 4f family plus bare refs: 294 CSFs. Compare both 6s and D/P absolute shifts and same-model excitation shifts against native005 and the old 68-CSF subset. The newly present 4f13 5d6p lower-state coupling makes this the first direct balance test.
2. Add the complete 5p-hole family to that common space: 468 CSFs. Preserve cross-family interference and the embedded signed state identities.
3. Add the complete 5s-hole family: 531 CSFs. Measure whether the finite same-pool lower/upper differential trend stabilizes.

At every stage use unchanged native public DC and complete static Breit operators with all roots of all complete J/parity blocks. Replay the native005 bare principal matrix and require original CI residual, Hermiticity, orthogonality and cross-symmetry-zero gates. Track exact J/parity, embedded full-state overlap, bare character, global-ground identity and intruders; energy rank alone is insufficient. Report absolute lower/upper energy shifts, excitation shifts, D/P centroids and splittings separately. Never splice states from different models, add isolated family shifts as errors or fit the experimental gap. The frozen-space angular preflight does not authorize a relaxed stationarity threshold.

Only after this balanced differential evidence should a bounded native multi-reference EOL MCDHF objective be chosen and its response/gradient work estimated. This frontier still omits core-double and other core-core correlation, higher virtual n/l and other core-hole shells; it is not a complete core-valence correlation model.

## Verification and evidence

129 scientific identity/count/signed-span checks pass. The actual single-thread preflight took 1.82 s and peaked at 346.1 MiB RSS under a 4 GiB / 360 s cap. Timing is uncontrolled and contains enumeration plus angular work only; it is not a performance comparison or a bound for a later RCI/MCDHF run. Ruff passes. [verification.json](verification.json) binds the result and scientific artifacts; [source_identity.json](source_identity.json) binds every imported frozen005 Python module and the runner.

- [result.json](result.json), SHA `dd18dda2879edf3bfb715176a37a340d6f71e3ae9a278e36979eff70078a5c4c` — stable card summary, counts, angular support and next-stage protocol.
- [contract.json](contract.json), SHA `154fac6699fdbc0e53e2b8681dfda104be80fe58744ece6074a5d047d2f78757` — prospective fixed scope, resources and stopping rule.
- [run.py](run.py) and [run.log](run.log) — replayable preflight using the saved source. A replay regenerates result timestamps/resource observations, so use a separate copy when preserving these hashes.

Replay command from the repository root:

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/balanced_cv_frontier/run.py
```

## Documentation impact and distillation

Type: current preflight record. Dependencies: the frozen005 source and input hashes in the contract; public configuration, signed angular, density and static radial-key APIs named above. No public source/API/default changed. Existing native005 and 68-CSF result artifacts are retained unchanged and linked; this folder owns the new model enumeration, not those older measured spectra. Root owns topic navigation and current consolidated cards.

Candidate action rule: when a restricted core-hole increment shifts upper states but leaves the lower state structurally uncoupled, enumerate the complete allowed pair family in the same orbital pool and inspect signed lower/upper coupling before diagnosing a general correlation trend. Reason: the present omitted 4f13 5d6p sector restores a lower-state angular mechanism. Scope: this declared one-core-hole two-particle finite pool; a real energy trend remains pending RCI. This is a topic-local candidate, not a new global rule or proof of experimental improvement.
