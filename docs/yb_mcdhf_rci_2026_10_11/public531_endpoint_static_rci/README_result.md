# Complete531 endpoint static RCI: verified result

This is the current result account for the frozen [prospective preparation](README.md). Root executed the contracted calculation successfully; a separate [saved-array replay](saved_array_verification.json) passes **3,710 checks**, with **zero new H builds, CI solves, gradients or optimizer calls**. The native result has72 integrity checks. The original accepted160 DC orbital residual remains **4.717047646600387e-5**, failing the unchanged1e-7 gate; the energy indicator6.8897239137767995e-9 Ha passes1e-8 Ha. These are fixed-orbital RCI diagnostics at a nonstationary mixed531 DC point, not a converged MCDHF result.

The complete69-electron,24-radial-orbital,531-signed-CSF model and all16 J/parity blocks are retained. One shared public Atomx H construction adds Gaunt and zero-frequency retardation including off-diagonal configuration interference. A consumer scipy.linalg.eigh evd solve diagonalizes every symmetry block. The DC comparator is the exact same saved point already independently rebuilt in the [public endpoint verification](../public_mixed_resume_run002/independent_endpoint_verification.json); this stage adds no DC build or gradient. Both models use the original full531 physical tracker and equalfive target weights. The tracked6s state is the actual finite-space ground in both models; no global-column-order assumption is used.

## Same-point transitions and nominal NIST gaps

All entries are cm⁻¹; gap means model minus retained NIST nominal. The four lines use the same lower6s state within each model.

| Target | DC | Static Breit RCI | Static−DC | NIST nominal | DC gap | Static gap |
|---|---:|---:|---:|---:|---:|---:|
| 5d_3/2 | 25271.554105 | 25273.938195 | +2.384090 | 22960.80 | +2310.754105 | +2313.138195 |
| 5d_5/2 | 26069.115768 | 26040.364436 | -28.751332 | 24332.69 | +1736.425768 | +1707.674436 |
| 6p_1/2 | 24977.860524 | 24957.163708 | -20.696817 | 27061.82 | -2083.959476 | -2104.656292 |
| 6p_3/2 | 28476.580285 | 28432.318731 | -44.261554 | 30392.23 | -1915.649715 | -1959.911269 |

Static Breit improves the magnitude of the D5/2 nominal gap by28.751332 cm⁻¹; D3/2 and both P gaps grow. Allfour static nominal1% goals fail. The remaining absolute gaps are1707.674436–2313.138195 cm⁻¹, versus2.384090–44.261554 cm⁻¹ static shifts. **Within this exact frozen model and nonstationary point, static Breit does not resolve the dominant experimental discrepancy.** This is not a rigorous missing-correlation bound, not a quantified dynamic-Breit/QED bound, and not evidence that increasing a cutoff alone suffices. Continued original-metric DC stationarity and a physically balanced correlation-space discriminator remain separate requirements.

## Absolute upper/lower corrections and cancellation

The same lower-state static−DC change is **+14.148346124558884 Ha**. Each upper change is recorded independently; a large common absolute correction is not a transition shift.

| Target | Absolute static−DC (Ha) | Transition static−DC (cm⁻¹) |
|---|---:|---:|
| 6s_1/2 | +14.148346124559 | +0.000000 |
| 5d_3/2 | +14.148356987273 | +2.384090 |
| 5d_5/2 | +14.148215123851 | -28.751332 |
| 6p_1/2 | +14.148251822922 | -20.696817 |
| 6p_3/2 | +14.148144454080 | -44.261554 |

The saved-array replay explicitly checks the identity Δtransition=(Δupper−Δlower)×Ha_to_cm⁻¹, preventing a splice of old and new lower states. Full numerical arrays are in [arrays.npz](arrays.npz); the reported model matrix is not independently reconstructed in this replay.

## Multiplet centroids and fine-structure splitting

Centroids use statistical2J+1 weights4:6 for D and2:4 for P. Splitting means higherJ minus lowerJ, independent of global energy ordering.

| Observable | DC | Static Breit | NIST nominal | Static−DC | DC gap | Static gap |
|---|---:|---:|---:|---:|---:|---:|
| 5d centroid | 25750.091103 | 25733.793940 | 23783.934000 | -16.297163 | +1966.157103 | +1949.859940 |
| 5d splitting | 797.561664 | 766.426241 | 1371.890000 | -31.135422 | -574.328336 | -605.463759 |
| 6p centroid | 27310.340365 | 27273.933723 | 29282.093333 | -36.406642 | -1971.752969 | -2008.159610 |
| 6p splitting | 3498.719760 | 3475.155023 | 3330.410000 | -23.564737 | +168.309760 | +144.745023 |

D centroid gap decreases slightly, but D splitting gap worsens from−574.328336 to−605.463759 cm⁻¹. P centroid gap worsens; P splitting gap improves from+168.309760 to+144.745023 cm⁻¹. Opposite responses show why centroid and splitting must be reported separately from a single aggregate accuracy score.

## Validation, resources and evidence boundary

The [exact root execution contract](execution_contract.json) limits one new full static H and one complete spectrum, CPU29 (one accountingCPU), numericalthreads1,4 GiB address space,180 s. Actual uncontrolled time is31.610722 s, including27.226387 s for H construction and0.002347 s for complete blocked CI; peakRSS1051.828125 MiB. These are resource observations, not paired software or parallel-efficiency benchmarks. The independent replay has separate1 GiB/30 s caps and imports no Atomx. It checks all353 frozen source hashes,16 immutable inputs, all531 rootwise Rayleigh/residual/norm equations, full CI Gram, exact symmetry blocks, all24 saved waves, signed coefficient overlaps, equalweights, target labels and independent transition/centroid/splitting arithmetic.

The source epoch is10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f. [result.json](result.json), [card_evidence.json](card_evidence.json) and preparation files remain byte-identical; their historical pending/prepared fields are retained. [completion.json](completion.json) and [card_evidence_verified.json](card_evidence_verified.json) bind the completed independent proof instead of editing those artifacts. No frequency-dependent Breit, QED, recoil, complete core-double correlation or higher virtual pool is included. Retained [NIST handbook data](../../rare_earth_yb_spectrum_2026_10_08/experimental_reference.json) have unspecified uncertainties, so printed digits are not treated as uncertainty bounds. Numerical integrity passes; orbital stationarity, production, experimental, full-correlation and highest-level certification remain false.

## Documentation impact and distillation

New consumer result → this result account and verified card evidence added. Frozen preparation README/contract/proofs and all previous run archives retained as prospective/historical evidence; no shared source or old science document changed. Root integrates the current result link into the parent topic and website. Existing AGENTS §2–§3 already require same-space lower/upper differences and separate stationarity, so no duplicate global rule is added. Trigger → action → reason → scope: when an absolute correction is large and largely common, report matched lower/upper differences plus centroids/splittings; cancellation and differing responses prevent its magnitude being mistaken for observable improvement. The numerical values above apply only to this recorded531 nonstationary point.
