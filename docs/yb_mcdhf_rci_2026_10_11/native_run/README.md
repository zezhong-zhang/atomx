# Native Yb II MCDHF and RCI

The user selected Atomx native MCDHF → RCI as the primary atomic route. These
calculations use the public `MCDHFSolver.solve` and `ci_solve` implementations,
with a frozen copy of all 353 Atomx Python source files for each retained run.
External software does not supply the orbitals, many-electron states or energies.

The current [005 result](nuclear_eol005_projected_increment/result.json) passes
the unchanged original orbital stationarity gate after a complete solve from
the original initialization: **5.143009363125294e-8 < 1e-7**. This certifies the
declared five-CSF numerical baseline. It does not certify complete core-valence
correlation, experimental accuracy, radial convergence or the highest atomic
accuracy. Every earlier failed source epoch remains archived.

The bounded first model has 69 electrons, `[Xe]4f14{6s,5d,6p}` references, five
CSFs in five different J/parity blocks, and **24 simultaneously optimized radial
orbitals**. The common-orbital EOL objective uses equal fixed weights on the five
resolved levels. Each symmetry block contains one CSF, so this model has radial
relaxation but no dynamical core-valence correlation or within-block CI mixing.
The Fermi nucleus uses the public empirical radius for A = 175; this is not an
isotope-resolved experimental prediction. The selected no-pair projector is the
fixed nuclear positive-energy B-spline space.

Actual runs:

| Evidence | Actual result |
| --- | --- |
| [First public run](nuclear_eol001/result.json) | 57 optimizer steps, line-search failure; 140.94 s, peak 861.99 MiB |
| [Observed replay](nuclear_eol002_trace/result.json) | Same full orbital diagnostics and all five DC/RCI energies as the first run |
| [Original coordinate replay](nuclear_eol002_trace/line_search_diagnosis/result.json) | Original residual and analytic gradient reproduced exactly, with the same saved inverse blocks and projector |
| [Source and artifact checks](nuclear_eol002_trace/verification.json) | 353 source hashes and all recorded scientific artifacts verified; actual script snapshot retained |
| [Complete stable-difference successor](nuclear_eol003_stable_delta/result.json) | Same original initialization and physical model; 82 stable energy comparisons, 56 steps, line-search failure |
| [Stable-difference observed replay](nuclear_eol004_stable_delta_trace/result.json) | Full stationarity diagnostics and five DC/RCI rows bitwise identical to 003 |
| [Stable-difference original metric replay](nuclear_eol004_stable_delta_trace/line_search_diagnosis/result.json) | Original residual 2.670347390185656e-6 and analytic gradient reproduced exactly |
| [Complete explicit-increment successor](nuclear_eol005_projected_increment/result.json) | 68 steps, original residual 5.143009363125294e-8, energy estimate 3.447850250858113e-15 Ha; 132.34 s and peak 860.29 MiB |
| [Independent final stationarity replay](nuclear_eol005_projected_increment/stationarity_replay002/result.json) | Actual radial gradient and objective replay exactly; same original metric hash, residual 5.14300923282462e-8 after coordinate recovery |

The first source epoch's stationarity residual is **5.38729624987869e-6**, above the unchanged
1e-7 threshold. The estimated energy error is 3.2200135033192855e-11 Ha, below the
unchanged 1e-8 energy gate; passing this energy gate does not replace stationarity.
Independent CI replay matches every DC energy exactly, and common radial metric
orthogonality error is 3.33e-15. Thus the failure is in orbital optimization, not
an unresolved CI eigensystem.

The latest same-orbital complete static Breit RCI uses the stationary 005
orbitals. Its signed differences from the retained NIST handbook five-level
reference are:

| Upper level | Diagnostic RCI − handbook (cm^-1) |
| --- | ---: |
| 5d3/2 | +796.49 |
| 5d5/2 | −107.14 |
| 6p1/2 | −2078.22 |
| 6p3/2 | −2593.15 |

The table uses the latest complete 005 explicit-increment source epoch.
These numbers do not certify an experimental gap or the accuracy of MCDHF with
correlation. No experimental energies entered orbital optimization, corrections
or root selection. The handbook does not supply uncertainties for these rows.

The observed replay retained all 56 rejected trials of the final line search in
[failed_line_search.json](nuclear_eol002_trace/failed_line_search.json), and its
accepted coordinates, signed gradient, spectral inverse blocks and projector in
[failed_coordinates.npz](nuclear_eol002_trace/failed_coordinates.npz). There is no
endpoint restart with a different residual metric. At the accepted objective
−14067.391460357456 Ha, one float64 spacing is 1.819e-12 Ha. At fallback step 0.1,
the directly subtracted total energies give +6.548e-11 Ha while the derivative
trapezoid gives −5.143e-12 Ha and the gradient norm falls from 3.125e-5 to 2.078e-5.
The unit-direction central difference at ±1e-4 agrees with the analytic derivative
to about 0.104%; larger ±1e-3 displacements show higher-order curvature. This
supports a cancellation-limited energy comparison near stationarity. A gradient
trapezoid is diagnostic evidence, not the proposed acceptance rule.

The [prospective successor contract](nuclear_eol003_stable_delta_contract.json)
requires exact differences of the same cell-integral finite polynomial, including
normalization, and a complete solve from the original initialization. It preserves
both numerical gates and the physical model. The fixed-RDM specialization is
declared only for the present one-CSF blocks; a future mixing model must account
for CI state response or use its original full CI path. The independent shared
implementation checks passed before 003 executed. This successor still fails
the original stationarity gate: residual **2.670347390185656e-6 > 1e-7**, despite
estimated energy error **3.2140910014161694e-12 Ha < 1e-8**. The original residual
metric hash is unchanged across all four complete runs. No endpoint restart,
tolerance widening or experimental fitting occurred.

The final 004 fallback direction predicts only −4.78937e-20 Ha at step 2^-27,
while the exact same-objective increment is +4.47300e-12 Ha. The
[polar endpoint experiment](stable_polar_endpoint002/result.json) compares the
original eig polar and new wide Gram correction at identical saved coordinates.
Both satisfy orthogonality and reproduce the original metric and gradient
exactly, but at zero step their radial objective increments are +2.7943e-12 and
+4.48555e-12 Ha. The improved polar map therefore does not alone establish an
improved physical objective. The [radial constraint discriminator](radial_constraint_endpoint002/result.json)
measures the actual S Gram defect, up to 2.3838e-15, after projector multiplication.
Its normal energy contribution uses the actual CI one-body ket-norm quotient
derivative. At step 0.1, complete metric defect subtraction, including the
quadratic Gram increment, leaves −4.4903e-13 Ha (old polar) or −4.4974e-13 Ha
(new polar). This is a diagnostic constraint decomposition; it has not certified
a new stationary solve.
The initial endpoint verifier selected the first retained trace residual instead
of the final failed iteration; its [failed evidence](stable_polar_endpoint/failure.json)
is retained, and the corrected independent replay is `stable_polar_endpoint002`.

The full prefilter angular objective also contains 264 nonzero block terms at
or below 1e-15, with maximum absolute coefficient 3.0531e-17. This count does not
bound their energy contribution. The paired unfiltered-versus-retained complete
increment differs by at most 3.26e-25 Ha on these retained candidates. That result
is specific to this model and endpoint. Small coefficients do not authorize
their omission.

A second [linear-map increment experiment](linear_map_increment_endpoint/result.json)
computes radial increments directly as `projector @ delta_coordinates`, preserving
the same linear map without subtracting separately rounded radial spinors. Its
independent full quadratic/quartic expansion reproduces the original complete
difference API within 2.15e-22 Ha for the same rounded increments. With explicit
increments, the zero-step polar centered discrepancy falls from about 1e-15 Ha
to 1e-23 Ha, and the step 2^-27 centered change is −4.81e-20 Ha, agreeing with the
original tangent derivative. The exact identity control is zero. The same stored
CI trial, states and original stationarity metric are retained. Actual increments
are archived separately, and a [prospective numerical contract](nuclear_eol005_projected_increment_contract.json)
requires independent implementation validation before a new complete solve from
the original initialization. The [shared API real-endpoint replay](centered_API_projected_endpoint002/verification.json)
passed first, with original complete-difference API semantics unchanged and
centered-versus-independent formula error at most 1.1671e-22 Ha. Then 005
actually completed from the original initialization, using 37 stable comparisons
and the same source map as that independent endpoint check.

The final [independent stationarity replay](nuclear_eol005_projected_increment/stationarity_replay002/result.json)
evaluates the actual stored radial spinors and reproduces their gradient and
objective exactly. The original inverse blocks and projectors retained in 004
are used only after reproducing their complete metric hash, identical to 005.
Coordinates recovered with that original projector are not bitwise the original
successful optimizer coordinates: forward radial reconstruction error is
2.987e-13, and the residual differs by only −1.303e-15. Both original gates pass.
The first hash verifier used a tuple where the native layout used a list; its
[failed diagnostic evidence](nuclear_eol005_projected_increment/stationarity_replay/failure.json)
is preserved alongside the corrected independent replay. No endpoint optimizer
restart or altered preconditioner was used.

After stationarity passed, the [fixed-orbital RCI discriminator](../cv_discriminator/README.md) added
`[Xe]4f13 6s 5d` (39 CSFs) and `[Xe]4f13 6s 6p` (24 CSFs) without increasing the
24-orbital radial set, for 68 CSFs in total. All DC and complete static-Breit roots
were calculated on the saved 005 orbitals, with prospectively declared signed
target-character tracking. This unbalanced family lowers P transitions by about
600–670 cm⁻¹ while barely changing 6s, worsening the P experimental gaps.
It is an actual correlation-mixing diagnostic, not a 68-CSF MCDHF optimization
or expanded-model orbital-stationarity certificate.

[card_evidence_stationary_baseline.json](card_evidence_stationary_baseline.json)
is the current source-bound card input. It marks the five-CSF numerical baseline
as stationary, while experimental certification and highest accuracy remain
false. The original [002 card evidence](card_evidence.json) and
[004 stable-difference card evidence](card_evidence_stable_delta.json) remain
immutable and explicitly failed.
Neither kernel nor end-to-end parallel speedup is claimed from these runs; all
times are observed resource use under shared-machine load.
