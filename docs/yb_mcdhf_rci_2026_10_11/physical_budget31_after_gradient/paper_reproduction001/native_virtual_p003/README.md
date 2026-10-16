# Native Yb complete6p virtual continuation

2026-10-15. Current owner for the virtual-space continuation of the immutable
[complete4f calculation](../native_tensor_4f002/README.md). The user authorized
continued reproduction, with native physics and performance work. MCDHF→RCI
remains the main route; RCC is its separately authorized comparison. No card,
schedule, external electronic solver or fitted experimental correction is used.

## Contract and physical space

The common68e closed-core reference, empiricalA171 Fermi nucleus,
ns180/order9/R500/RKB bank, Dirac–Coulomb operator, correlated5s²/5p1/2²/4f14
(18 electrons), frozen50 electrons and complete6s/5d3/5d5 target virtuals are
unchanged. Complete6p1/2 and6p3/2 add6 magnetic spinors: **36 total,18 occupied,
18 virtual**. Frozen5p3/2 remains explicitly frozen. This is not a complete-core
or radially/angularly converged model.

[Sahoo2025, Table1 and method](https://arxiv.org/html/2504.01438v1) describes
40 GTOs per symmetry throughi, RCCSDT throughg and separate higher-angular/
Breit contributions. Adding the first p pair is a controlled discriminator
inside that broader reproduction task, not an exact paper-basis reconstruction.
No paper correction is added to our energies.

| Retained coordinates | Count |
|---|---:|
| Core T1/T2/T3 | 324 /23409 /665856 =689589 |
| Exact core M=0/even coordinates | 37533; all original689589 equations checked |
| EA R1/R2/R3 | 18 /2754 /124848 =127620 |
| Conservative complete numerical payload | 6,941,275,920 bytes;8192MiB admission |

Full18e/19e Fock sectors are never constructed; dense orbital tensors and
complete declared rank coordinates remain. There is no amplitude screening,
virtual-energy/count adjustment after observing results, Breit/QED/recoil or
hyperfine Hamiltonian.

## Original radial pencils and off-shell discriminator

[Preparation/probe driver](prepare_virtuals.py) and [predeclared contract](probe_contract.json)
consume the earlier owned p-channel H/S/core columns. The full occupied
complement is formed by QR of S C, followed by its generalized symmetric
pencil. All roots above−c² define the electron branch. The lowest is refined
by exactly two bordered corrections in the **original** H/S, with occupied
orthogonality and physical normalization constraints; no radial/cut/root scan.
The [analytic three-case correction anchor](../yb_dhf_table001/constrained_pair.py)
is rerun. Each new p channel has172 electron columns; selected6p roots:

| State | DHF energy /Ha | Original projected dual /Ha |
|---|---:|---:|
| 6p1/2 | −0.301112939172 | 1.748e-14 |
| 6p3/2 | −0.288306665239 | 1.420e-14 |

Physical norm defects≤2.221e-16 and occupied overlap≤4.523e-18.
[Saved original pencils/selected roots](probe001/virtuals.npz) retain the complete
initial branch as well as selected states. Only the selected complete6p pair
is added to the integral bank. [Operator](operator.json), [native integrals](integrals.npz)
and [fixed embedded inputs](fixed_inputs.npz) carry the new identity.

All old labels form the first30 spinors; one-body/Fock/ERI/frozen constant and
common-origin gates recover the previous operator within1e-10Ha. Embedding
all old amplitudes with exact zero new components recovers all189834 old core
equations and energy; embedding the old6s R3 root recovers all34860 old EA
equations. The newly opened core channels have rank1/2/3 maxima
0.001132503802/0.007156530444/0.001079471711Ha; the old6s state has new-channel
EA residual0.020383981509Ha. These show missing channels at the old solution,
not a line-error bound or a predicted improvement.
[Probe result](probe001/result.json), [execution log](probe_execution.log):
26 controls pass,51.955s total, one core residual35.599s, one EA action7.449s,
peakRSS1531.641MiB. These are cost/admission measurements, not controlled
before/after speedups. Probe conditions: CPU27, six thread variables=1,
AS24GiB,240s wall cap, load1≤8 and available memory≥32GiB.

## Numerical solving and independent anchors

The local [Fock-Jacobi/DIIS implementation](diis_core.py) solves the original
complete SDT equations. For active coordinates x, constant unshifted Fock
excitation diagonal D and residual r(x), form F(x)=x−D⁻¹r(x). DIIS minimizes
the norm of the last six D⁻¹r errors under coefficients summing to one and
combines the corresponding F(x). One common Gram scale preserves that
minimizer. There is no damping, coefficient clipping, level shift, regularized
linear solve or history-rank cutoff. Nonfinite/zero denominators and failed
original equations stop the trial. Only exact M/parity-forbidden coordinates
are omitted in the large solve; all retained689589 equations determine acceptance.
The production `solve_core` implementation/defaults are unchanged.

Before the large solve, the [qualification contract](diis_qualification_contract.json)
and [independent finite/old-Yb replay](qualify_diis.py), [result](diis_qualification.json)
validate the new numerical iteration. The complete3e/6spinor fermion Hamiltonian
gives an independent FCI ground energy: discrepancy9.350e-12Ha, complete
SDT residual7.727e-11Ha,9 evaluations. The old actual16spinor Yb1324-coordinate
solution from zero takes8 evaluations: all-amplitude discrepancy9.983e-12,
energy discrepancy7.284e-14Ha, complete residual2.666e-11Ha. All5 controls pass.
This validates solution identity, not a measured speedup against a differently
seeded or continued historical Krylov solve.

[Large solve driver](solve.py) and [contract](solve_contract.json) preserve the
operator, prescribed S/D3/D5 identities and1e-10Ha original residual gates.
The same signed highest-weight R3 Davidson of the previous source is reused;
its correction projector is ker(J+) within the complete M=J/even sector.
All target magnetic partners follow signed ladders and each full R3 equation
is reapplied. Canonical-R norms are not exp(T) physical-state norms. The
large resources are CPU27/threads1, AS24GiB,8192MiB admission,2400s wall cap,
60 complete core residual calls and48 Davidson basis vectors per target;
independent caps stop at whichever comes first. No automatic failure retry.

## Completed physics and four-model comparison

[Actual result](actual001/result.json), [log](solve_execution.log):
**PASS_COMPLETE_6P_NATIVE_CCSDT_R3**,526 live controls. All689589 SDT equations
pass at max1.93213e-11Ha; scalar-core signed J+ defect8.017e-15. DIIS takes
14 complete residual calls,312.692s; no numerical Jacobian directions are used.
T1/T2/T3 canonical norms are0.001542408/0.058550528/0.003296162. The core
correlation energy is−0.006971177791Ha, a change of−0.000274263168Ha from the
12-virtual complete4f core. A common core energy change cancels from transitions.

| Target | EA /cm⁻¹ | EA change from12virtual /cm⁻¹ | Canonical bareR1 weight |
|---|---:|---:|---:|
| S1/2 | −91904.772317 | −1003.537945 | 0.992406236 |
| D3/2 | −66740.693674 | −422.150190 | 0.996258182 |
| D5/2 | −66332.445906 | −287.126311 | 0.997284093 |

Complete highest-M/even R3 sectors have7524/6899/5786 coordinates. The
three targets take16/18/18 search actions; all2/4/6 signed magnetic partners
pass their complete127620-coordinate original equations (max4.232e-11Ha),
J² (max3.555e-14) and canonical norm gates. Total701.344s includes previous
root replay, the new core solve/readback, angular work, target search and partner
checks; peakRSS1376.188MiB. Prior radial/probe work is separate. These are scoped
costs, not a paired comparison against historical Krylov/other-model timings.

The [predeclared frozen4f companion](corner_contract.json),
[driver](frozen_corner.py), [result](corner001/result.json) closes a two-by-two
core/virtual-space panel. Exact frozen normal ordering of the same36-spinor
operator moves14 complete4f spinors into the frozen core:4 correlated/64 frozen,
18 virtual,22 retained spinors. It recovers the previous16-spinor restriction,
all retained ERIs/Fock and frozen constants within1e-10Ha. No orbital refitting
or integral changes are used. Complete4254 SDT and5526 R3 equations, all12
partners and288 live controls pass. Its core takes9 DIIS calls; total6.199s,
peakRSS309.828MiB, CPU26/threads1,AS8GiB,512MiB admission,180s cap. This
independent physical corner ran concurrently with the large target search;
neither supplies a controlled timing ratio.

| Core correlation / virtual space | S→D3 /cm⁻¹ | S→D5 /cm⁻¹ | D5−D3 /cm⁻¹ |
|---|---:|---:|---:|
| 4core,12 sd | 24372.156946 | 24708.853564 | 336.696618 |
| 18core,12 sd | 24582.690888 | 24855.914777 | 273.223888 |
| 4core,18 sdp | 24512.423775 | 24936.067217 | 423.643442 |
| 18core,18 sdp | **25164.078643** | **25572.326411** | **408.247768** |

[Same-state comparison driver](compare_models.py),
[all four cells and reporting inputs](four_model_comparison.json) keep each
state's EA/core identity. The first cell's stored69e energy is reduced by its
own core energy before calling it an EA. No lower/upper state is spliced
between spaces. The original positive experimental gaps grow; adding6p has
stabilized6s more than either5d state, raising the two transition energies by
**581.387755/716.411634cm⁻¹** in the complete4f model.

| Conditional spectral increment /cm⁻¹ | D3 | D5 |
|---|---:|---:|
| Open4f with12 sd virtuals | +210.533942 | +147.061213 |
| Open4f with18 sdp virtuals | +651.654869 | +636.259194 |
| Add6p with4f frozen | +140.266828 | +227.213653 |
| Add6p with4f active | +581.387755 | +716.411634 |
| Difference of conditional increments | **+441.120927** | **+489.197981** |

This demonstrates nonadditivity of these finite correlation-space increments.
The last row is a model difference, not a separately identified diagram,
universal correction or experimental uncertainty. Earlier4f shifts cannot be
transplanted as fixed corrections to another virtual space. The current
numerical residual is much smaller than the observed model shifts, but is
not a physical truncation bound.

[Angular admission contract](angular_coverage_contract.json),
[census](angular_coverage.json) compare native rank1 reduced-C nonzeros with
independent triangle/parity conditions. S allows p1/2,p3/2; D3 allows p1/2,p3/2,
f5/2; D5 allows p3/2,f5/2,f7/2. The current virtual bank includes only the p
members of this rank1 census. This is an **angular-space imbalance inference**,
not proof of a missing-f energy correction or its sign. It motivates a balanced
angular/radial convergence experiment rather than selecting a channel because
it moves toward experiment. Merely adding the first full f pair would raise
virtual spinors to32: conservative dense payload38,972,589,248 bytes and
4,123,824 core amplitudes. That is metadata admission only; no such tensors
or f-model were allocated. Further expansion first needs a scalable contraction/
angular representation and its own predeclared resource/physics qualification.

## Validation and scope

The unchanged production source has362 Python files, map
`ea402a15130e5041dab452940c288804b300008b8c44dafb6e7a7d1c3f53c373`.
[Source map](source.json) and [archive](source.tar.gz) preserve actual dirty source.
No production or existing test source is changed by this increment.
The current [tensor regression panel](tensor_tests.log), [JUnit](tensor_tests.xml)
reruns all12 cases: **12PASS in14.71s**, no skips. Earlier59-case evidence
remains at its owner; it is not counted again here.

The [saved-data reviewer](verify_saved.py), under its [review contract](review_contract.json), checks every core equation,
three highest-M roots/all12 partners, the frozen4f companion, original p pencils,
new native-Fock/radial-root ledgers, source archive and input/previous-result identities.
The [saved review](saved_data_review.json), [log](saved_review.log) pass137 controls,
including fresh full core/partner equations and the two native-Fock/radial-root
ledgers (maximum discrepancy2.048e-14Ha). Closing receipts bind these results. Its resource call fields describe the main
model (one core/twelve partner actions); the executed review additionally
checks one core/twelve partner actions in the small frozen4f companion, as
clarified by the execution record. Reapplying the same production action is
replay evidence, not an independent full18e FCI oracle. Large full-H omitted-rank
residuals, physical exp(T) state normalization, left/dressed properties and
radial/virtual/core/experimental convergence remain uncomputed. Original MCDHF
failed gates are unaffected.


[Documentation impact](documentation_impact.json), [execution record](execution_record.json),
[validation receipt](validation_receipt.json) and [increment receipt](increment_receipt.json)
close the source/input/process chain. All task sessions are reaped. Link validation
covers present working-tree artifacts, including untracked/ignored local evidence;
it makes no clean-checkout publication claim. Existing archived reports/receipts
are preserved. No new global rule or production API is introduced.
