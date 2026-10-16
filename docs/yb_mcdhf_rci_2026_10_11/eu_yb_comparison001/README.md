# Why the successful Eu calibration does not certify Yb precision

The compared cases are **Eu³⁺ / Eu IV** and **Yb⁺ / Yb II**, not neutral Eu
and Yb. Both use Atomx native physics. Their models and validation targets
are different.

| Contract | Historical Eu³⁺ | Current Yb⁺ |
| --- | --- | --- |
| Model | Frozen statistical Xe core; 4f⁶ multiplet/CF space | Native correlated Dirac–Coulomb MCDHF followed by RCI |
| Parameter use | Pr prior fixes s2; s4/s6 fitted to three ⁵D targets | No experimental-energy fit |
| Reported precision | Calibration RMS8.2 cm⁻¹; cached NIST comparison9.03 cm⁻¹ | Four-transition terminal RMS1283.77 cm⁻¹ |
| Independent scope | Calibration targets are driver-declared Carnall aquo-ion anchors; cached NIST targets are bracketed semiempirical levels | Four nominal NIST comparisons, original1% gates failed |
| Unfitted observables | Six ⁷F levels differ by3.52–6.52% from the current cached NIST rows | Orbital optimization is not stationary |
| Complete space claim | Historical full-M4f⁶ space contains3003 determinants | Five complete retained symmetry sectors contain1595 CSFs; parent3789, other2194 unsolved |

The [Eu audit](../../atomic_native_roadmap_2026_10_11/eu_baseline/README.md),
[driver](../../../scripts/eu3_tierA_anchor.py) and
[saved log](../../review_2026_10_02/codex_fix_validation/eu-tier-a.log)
retain the successful calibration. The log's NIST heading does not override
the reference declared by the driver. Its old7F5 constant3807 cm⁻¹ also differs
from the cached bracketed NIST3910 cm⁻¹: the historical7.7% and current4.88%
comparisons have different references and must remain distinct.

Solving every state in a retained4f⁶ manifold completes that angular/model
space. It does not include unrestricted core excitations, all radial orbitals,
or all correlation orders. CSFs are symmetry-adapted combinations of Slater
determinants, so Eu's3003 determinants and Yb's1595 CSFs are not interchangeable
workload or completeness counts.

For Yb the lower6s and upper5d/6p energies must be balanced. Published
[MCDHF/RCI work by Itano](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=50237),
section D, explicitly treats4f/5s/5p core–valence correlation and additional
core–core contributions. That work concerns quadrupole/hyperfine observables;
it supports these candidate correlation channels, not a quantitative estimate
of Atomx's present transition gap or a precision certificate for this model.

The [JSON-only occupation census](correlation_scope_audit001.json) audits all91
current NR configurations:7 have no core hole and84 have exactly one hole
in4f/5p/5s, all with69 electrons. None has two core holes. Therefore adding
radial orbitals while preserving this occupation restriction cannot insert
explicit double-core-hole configurations. Their differential effect on these
transitions requires the separately bound current31 measurement; this audit
does not prove they dominate the experimental gap. Mean-field/core relaxation already retained must not be
counted again as a missing correction.

That last statement concerns the **current31 point**. The independently closed
[old24 rank-only experiment](../rank_only_discriminator/decision.json), with
[3247 saved-array controls](../rank_only_discriminator/saved_array_audit/verification.json),
already measured the conditional 5p-double response at accepted0160:
the four D3/D5/P1/P3 transitions shifted by +461.25/+596.71/+194.49/+148.67
cm⁻¹. All five absolute energies decreased, but nominal RMS worsened from
2022.83 to2225.97 cm⁻¹. This is a substantial measured channel, not evidence
that it alone repairs the present gap. The [current31 preparation](core_core_discriminator_preparation001/README.md)
retains all91 configurations and adds nine complete5p-double configurations
without new radial orbitals. Exact integer counts give2574 target CSFs;
four raw sectors exceed the old2000 resource cap. A separately declared whole
resource admission is required before numerical execution; this is not a
physics tolerance or permission requirement. No configuration is pruned.
The [source-based resource appendix](core_core_discriminator_preparation001/resource_admission_proposal001.md)
proposes an explicit2659 raw cap and bounded8GiB run, preserving the full
space. Scalar-compatible overlap pairs have a complete integer upper census
of66654; Python angular-table/cache allocation remains the unresolved peak
resource in that prospective census. The [current31 execution](core_core_discriminator_preparation001/execution_preparation002/RESULTS.md)
has now built the full2574 matrix at accepted13 within explicit2659/8GiB
bounds. It closed exit1:157 PASS, one original exact nested1595-byte FAIL,
maximum H difference8.88e−16 Ha. No expanded eigensolve was entered. The
failed gate and complete saved H remain unchanged; separate saved-matrix
analysis can diagnose the arithmetic and conditional physics without
qualifying that source or declaring convergence. Work counts and52.697s
uncontrolled elapsed are not efficiency measurements.

The [separate saved-matrix diagnostic](core_core_discriminator_preparation001/saved_matrix_diagnostic001/RESULTS.md)
has completed all five full eigensystems and original531 signed tracking,
then stopped at its final module-import boundary gate (56 PASS/1 FAIL).
Its retained accepted13 conditional shifts are+468.06/+673.70/+149.14/+160.82
cm⁻¹. All absolute energies decrease, yet four-line nominal RMS worsens
910.72→1113.17 cm⁻¹. D/P centroids move+591.45/+156.93 and splittings
+205.64/+11.68 cm⁻¹. Exact old-block row-sum arithmetic bounds fixed ordered
transition differences below4.973e−10 cm⁻¹; eigensolver/physical errors and
hypothetical re-identification are excluded. Both original failures and
nonstationarity remain. This is a current13 saved diagnostic, not a correction
to terminal16 or a qualified native source/model. The class is significant
but is not by itself the solution to the experimental gap.

The [closed terminal16 observable decomposition](terminal_observable_decomposition001.json)
also separates cross-configuration energy balance from fine splitting:

| Multiplet | Centroid gap from nominal NIST, cm⁻¹ | Splitting gap, cm⁻¹ | Relative splitting gap |
| --- | ---: | ---: | ---: |
| 5d | −1486.70 | +16.34 | +1.19% |
| 6p | −854.06 | +723.73 | +21.73% |

These are the same four terminal transitions, using degeneracy weights4:6
and2:4, not new calculations or independently certified precision. They show
that D splitting and D excitation balance have different errors. P splitting
also needs attention. This motivates separate diagnostics of differential
correlation, radial relaxation and state mixing; it does not identify their
individual contributions before paired same-point measurements.

The [new chronological replay](../screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageA_execution_preparation001/RESULTS.md)
passed200 controls without new physics calls. At point12 the original L-BFGS
direction is64.29 times the original preconditioned steepest direction norm;
new5f columns carry73.48% of its squared norm. The complete source reproduces
the accepted13 endpoint exactly. This is measured optimizer behavior, not yet
proof of a faulty physical gradient or an overshoot.

The [actual-direction physical check](../screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageB_execution_preparation001/RESULTS.md)
has now closed with743 controls passing and two ordinary reported-energy
derivative gates failing. The complete centered finite-CI response agrees
with the public analytic derivative at both h to about2e−10, below the
original3.83e−7 threshold. This supports one local directional derivative;
ordinary production failures, stationarity and missing physics remain open.
The [closed scalar-ledger diagnosis](../screened_f_next_step208/native31_f043_relaxation_preparation001/growth_step12_source_diagnosis001/stageB_execution_preparation001/reporting_difference_diagnosis001/README.md)
quantifies ordinary-minus-centered differences: the largest contribution is
the radial-polynomial versus retained-H quotient closure slope, −1.26e−6 and
−2.05e−6 Ha per unit direction. Reported-root/quotient offsets and signed
actual-Gram terms are retained separately. EO ULP alone is not an adequate
explanation. The [complete saved-H replay](../ai_assisted_iteration001/gradient_producer_source_audit001/saved_contraction_preparation001/RESULTS.md)
now closes96 controls at baseline/h0-plus: matrix radial plus complete finite
CI response agrees with direct quotient change to1.53e−20 Ha. The saved radial
polynomial differs from matrix radial by2.8954e−11 Ha before adding response;
this narrows the producer investigation before CI/export at this pair.
The [complete endpoint discriminator](../ai_assisted_iteration001/gradient_producer_source_audit001/endpoint_radial_contraction_preparation002/RESULTS.md)
now closes191 controls/all19589 keys: density-route versus matrix contraction
accounts for92.12% of this local difference and polynomial versus ordinary
endpoints7.88%. Forward signed-pair increments are within2.38e−13 Ha of
stored-H per target; baseline density and native scalar replay match exactly.
The total approximately6.35e−6 cm⁻¹ energy scale is distinct from the large
spectral gaps. No certified arithmetic bound, callback/gradient repair,
original-failure upgrade or stationarity change follows. The first preparation
order failure is retained separately from the successful metadata successor.
The [complete forward increment discriminator](../ai_assisted_iteration001/gradient_producer_source_audit001/forward_linear_increment_preparation001/RESULTS.md)
closes98 controls/399 guards: direct `T@ΔR` versus transpose agrees within
8.24e−20 Ha per target; weighted difference from the saved separate-endpoint
subtraction is2.6692e−11 Ha. This localizes arithmetic nonadditivity of
endpoint assembly, with no exact-operation bound, gradient repair or
stationarity qualification. Complete trial key coefficients are now retained.
Separately, the same-point
DC/staticBreit/zero-frequency RCI attempt stopped at its original exact DC
whole-H comparison before entering either transverse model. The
[closed saved-array diagnosis](../transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/scalar_request_observation_successor001/saved_DC_failure_diagnosis001/RESULTS.md)
finds matching entire owner/PQ identities, maximum H-entry difference
7.28e−12 Ha, and all complete residual/Gram controls passing. Its conservative
finite-matrix excitation difference bound is about0.0051 cm⁻¹; this particular
source discrepancy cannot explain the current thousand-cm⁻¹ gap. The exact
byte failure remains failed. The new [same-source baseline successor](../transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/scalar_request_observation_successor001/same_epoch_DC_baseline_successor001/RESULTS.md)
completed the exact DC matrix and86 controls, then hit its declared1200s stop
in static construction after261889 scalar returns. Static H and zero-frequency
RCI remain incomplete, so no new Breit shift was measured. This original
recovered31 point also differs from terminal16; no correction may be spliced
between them. The [layout-preserving private Gaunt candidate](../transverse31_samepoint_preparation001/phaseA_owned_execution_preparation001/source_batch_reuse_note001/layout_preserving_successor001/RESULTS.md)
now passes26 controls: all16384 complete component keys agree bitwise with
the original native integrator, including repeated keys and snapshot mutation
checks. Initial Yk producer calls decrease32768→256; all8 component waves
are reused. The original contiguous-copy candidate failure remains failed.
This is a bounded small-model arithmetic/work-count qualification; full31 H/CI,
static retardation, controlled speed and shared-source adoption remain open.
No new Breit spectral contribution was measured.

The [complete current13 gradient comparison](../physical_budget31_after_gradient/physical_next_decision003/expanded_gradient_execution_preparation002/RESULTS.md)
now evaluates all31 orbitals in the saved2574 model. Same original metric
residual increases0.153978→0.221316 after adding the full5p double-hole channel;
the original direction remains aggregate EOL descent. The combined native
filter/actual-norm route delta has residual1.19e−14. This supports a common radial-displacement
measurement of correlation-by-relaxation interaction; no new four-line
response or optimized spectrum is measured. Eu's calibrated historical fit
therefore remains incomparable to a newly stationary Yb spectrum.

No new Eu solver was run for this comparison. No fitting or tolerance changes,
card transmission, or timer restart is part of this step. The complete
core-space arm is the explicitly declared configuration increment above.
The [AI contribution ledger and advice](../ai_assisted_iteration001/README.md)
retain different contexts and failures; current31 predictions abstain for
insufficient comparable independent groups. No AI speedup is certified.
