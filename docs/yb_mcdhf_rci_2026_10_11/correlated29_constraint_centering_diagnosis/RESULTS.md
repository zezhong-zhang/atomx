# Conditional fixed-multiplier result

One foreground saved-only invocation closed exit0:3,432,060 exact scalar products,9.0568s inner /9.1726s outer elapsed,141008KiB peak RSS. All H/CI/gradient/Dirac/eigen/SVD/RDM/cell producer counts were zero. This is an uncontrolled diagnostic timing.

The centered finite Rayleigh central derivative is0.0014943764939743555. Its error relative to the original analytic0.0014943766122014388 is−1.182270833100045e−10. Its error relative to the conditional centered analytic0.0014943766121803242 is−1.182059687508311e−10. Both are below the unchanged1.494376612201439e−8 gate. The raw prior Rayleigh derivative remains0.001494167474901981, and the original002 gate remains failed.

The baseline fixed multiplier uses the archived retained complete radial gradient plus the exact specified raw-h/ket-normalized one-body ambient correction. The resulting actual ambient action is0.001494376612182307; its ideal Gram-normal directional term is1.9826384424433477e−15. This is a near-identity fixed multiplier, not an exact normal projection for arbitrary nonideal Gram matrices.

| Saved sign | Complete Rayleigh increment (Ha) | Full fixed-Lambda normal (Ha) | Centered increment (Ha) | Retained finite CI Rayleigh response (Ha) |
|---|---:|---:|---:|---:|
| plus |1.5874403096811365e−9|−2.016543756470691e−13|1.5876419640567836e−9|2.400813481478005e−13|
| minus |−1.4008946401228255e−9|2.1638376910180221e−13|−1.4011110238919273e−9|2.4008117805245433e−13|

All signed entries of the seven complete same-kappa Gram blocks were retained. Exact dyadic/rational arithmetic verified each trial-minus-baseline Gram against its linear-plus-quadratic form and explicit-plus-reconstruction split. Maximum baseline Gram defect is3.5109192772464932e−15; maximum trial defect is1.809841803417006e−15, within the original1e−10 gate. Quadratic normal contributions individually reach1.0546e−10Ha in kappa2 and cancel corresponding linear curvature terms; they were never dropped. Off-diagonal contributions and every full matrix are archived in [actual/result.json](actual/result.json).

Centering reduces the reconstruction contributions to−4.6573e−16 and−2.3005e−16Ha. The finite CI Rayleigh and reported responses, CI norm/residual/phase history, raw-gradient multiplier control and original direct-stored-H quotient seam remain separately recorded. The direct stored-H quotient central0.001492693371496832 was not substituted for the polynomial-plus-reconstruction-plus-response diagnostic.

This is a conditional finite same-three-point constrained-gradient control. It does not re-evaluate or fully normalize the mixed-CI RDM, bound unfiltered two-body gradient differences, certify the finite centering remainder or provide a universal descent-sign bound. Both original derivative failures, both3451 stationarity failures, experimental gaps and optimizer hold are unchanged. The next smallest work is a separate saved d0 external-direction control and a separately admitted baseline normalized/unfiltered gradient control. No production objective, callback, source or physics model was changed. The [existing preserve-metric experience](../../AGENT_INSIGHTS.md#preserve-metric) already states these remainder limits.
