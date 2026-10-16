# Complete 4f12 fixed-orbital RCI: PASS

All five complete B sectors and their exact A restrictions were calculated once with native numeric streaming and Davidson. All **265/265 live gates** and **554/554 offline saved-result checks** pass. The offline reviewer imports no Atomx, builds no H and performs no eigensolve; direct full-operator residuals are independently required live gates, rather than reconstructed by that reviewer.

A173/B182 NR configurations retain all82 new 4f12 configurations and every declared CSF row. The problem is original accepted13: the same69-electron Yb+, Z70/A175 Fermi,31 waves, P/S/inverse geometry and unshifted Dirac–Coulomb. Breit/QED are off. These results do not include orbital relaxation or transfer a correction to terminal16.

## Individual states and excitation energies

State shifts compare each new state against its own arm's original accepted13 state. Excitation energies use the fresh6s ground in the same arm/model. Their shifts are therefore differential state shifts, not absolute total-energy changes. Units are cm^-1, converted with the unchanged native HARTREE_CM1 = 219474.63136314112.

| State | A state shift | B state shift | A excitation | B excitation | A excitation shift | B excitation shift |
|---|---:|---:|---:|---:|---:|---:|
| 6s1/2 | -7645.065987 | -7444.733588 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5d3/2 | -7349.750613 | -7171.764784 | 24259.924611 | 24705.642959 | 295.315374 | 272.968804 |
| 5d5/2 | -7348.326067 | -7179.886440 | 25082.834734 | 25724.646042 | 296.739920 | 264.847148 |
| 6p1/2 | -7397.568209 | -7258.516437 | 26214.437018 | 26302.297705 | 247.497778 | 186.217151 |
| 6p3/2 | -7446.707230 | -7313.307972 | 29638.940160 | 29732.830061 | 198.358756 | 131.425616 |

The common correlation lowers the states by about7,172–7,645 cm^-1, but the four excitation energies rise by131–297 cm^-1. This finite specified class contributes substantially to total energies while most of that contribution cancels in these transitions. Its shift is not a constant correction and is not a bound on omitted physics or radial/orbital convergence.

| Observable shift / cm^-1 | A | B |
|---|---:|---:|
| D degeneracy-weighted centroid | 296.170101 | 268.095810 |
| D fine-structure splitting | 1.424546 | -8.121656 |
| P degeneracy-weighted centroid | 214.738430 | 149.689461 |
| P fine-structure splitting | -49.139021 | -54.791535 |

Centroids use D weights4:6 and P weights2:4. The A/B interactions of the new class with the nine B-specific configurations are −22.346570 / −31.892772 / −61.280627 / −66.933140 cm^-1 for D3/2, D5/2, P1/2, P3/2. Thus adding independently estimated correlation corrections would not reproduce the complete paired calculation.

## State character and numerical gates

| State | A accepted13 weight | B accepted13 weight | A 4f12 probability | B 4f12 probability |
|---|---:|---:|---:|---:|
| 6s1/2 | 0.973440916 | 0.974606684 | 0.026192311 | 0.025073979 |
| 5d3/2 | 0.973845865 | 0.974929668 | 0.025907951 | 0.024856269 |
| 5d5/2 | 0.974156850 | 0.975147695 | 0.025636003 | 0.024667025 |
| 6p1/2 | 0.971562884 | 0.973607376 | 0.025329280 | 0.024408951 |
| 6p3/2 | 0.970109365 | 0.973066531 | 0.025554779 | 0.024629797 |

All ten transported target states retain more than97% squared overlap with their signed owned004 accepted13 references. The complete 4f-occupation projectors give2.44–2.62% probability to4f12, retaining every saved coefficient in the probability sums. The maximum direct full-operator residual over all twenty returned roots is **9.46485954348e-10 Ha**, below the predeclared1e-8 Ha gate; the native Davidson tolerance remains1e-9 Ha. All selected Gram, original-subspace action, complete census, source and input gates pass.

Numerical isolated-root transport uses the declared >0.5 reference-weight and observed two-root-gap checks. The Bessel omitted-overlap argument is conditional on an exact isolated eigenstate; these finite-residual checks do not rigorously certify an omitted-spectrum gap. Original531 tracking is ancestry only because those earlier orbitals differ. This is no public full-spectrum CIResult/tracker/snapshot.

## Complete resources, without a timing speedup claim

| Sector | Raw determinant rows | CSF rows B/A | Complete candidate pairs | Numeric Hdet entries | Seconds | Peak RSS / MiB |
|---|---:|---:|---:|---:|---:|---:|
| 6s1/2 | 92,715 | 5,717/5,575 | 43,720,175 | 42,396,635 | 427.141 | 2326.023 |
| 5d3/2 | 87,182 | 10,418/10,192 | 40,882,406 | 39,627,831 | 453.890 | 2310.457 |
| 5d5/2 | 76,800 | 13,432/13,196 | 35,054,314 | 33,983,268 | 468.761 | 2220.824 |
| 6p1/2 | 92,377 | 5,704/5,561 | 43,261,341 | 41,970,635 | 405.843 | 2328.547 |
| 6p3/2 | 86,878 | 10,415/10,183 | 40,517,908 | 39,294,707 | 432.338 | 2332.070 |

Each worker used one CPU/thread, at most8 GiB address space and1200 s; no more than three workers ran concurrently. The maximum observed process RSS was2,332.070 MiB (~2.277 GiB). Each sector made one H build, one full numerical determinant CSR, zero angular coefficient tables,63 cold null-space calls and two native Davidson solves. Runtime ranges406–469 s for this one diagnostic per sector. These are source-bound feasibility observations, not paired controlled timing or measured parallel speedup.

## Remaining physics and evidence

The specified missing class is now measured in the fixed orbital space. It does not close the original MCDHF stationarity, three005 line-FD failures, experimental gates or full-spectrum database objective. Next work must first provide a qualified memory-bounded enlarged-state orbital-density/gradient path and resolve the original derivative admission gates before an optimizer continuation. Reoptimizing the enlarged physical model needs new same-point stationarity and differential-observable checks; no accepted13 shift is spliced into terminal16.

- [Aggregate and every individual energy](result.json), [offline reviewer](../review_results.py).
- [Full worker and predeclared caps/gates](../worker.py), [contract](../worker_contract.json), [root admission review](../root_source_review_and_admission.json).
- [Complete A/B source qualification](../../matched001/RESULTS.md) and [increment](../../README.md).
- The five sibling `actual_J*_even|odd` directories retain signed basis signatures, selected vectors, live results and partial checkpoints. Every worker/result/input identity is preserved in the aggregate and closure receipts.
