# Saved one-body and motion result

The single authorized foreground run closed exit0. It consumed only archived001/002 arrays, exact binary64-as-dyadic arithmetic and the original direction20261012/h1e−6. There were2,284,625 exact scalar products, zero H/CI/gradient/Dirac/spectral/RDM/cell calls,6.0686s inner elapsed and149584KiB peak RSS. Outer elapsed was6.1715s. These timings are uncontrolled.

| Measured contribution (Ha per direction unit) | Ideal P times seed | Explicit polar odd increment | Rounded physical odd increment |
|---|---:|---:|---:|
| Full saved radial gradient action |0.001494376612198415|0.001494343818183005|0.0014944307370519225|
| Ket-normalization derivative change |−1.5139e−14|−4.578101151429142e−8|−2.632620185524154e−7|
| Raw minus logical one-body derivative |−9.69e−16|−9.69e−16|−9.69e−16|
| Full minus retained global one-body coefficient slope |5.860384394624036e−16|5.860384443042079e−16|5.860384556409718e−16|
| Conditional corrected full-gradient action |0.001494376612182307|0.001494298037170522|0.001494167475032401|

The archived measured explicit-radial central slope is0.0014942980369557922; the conditional explicit prediction differs by2.1473e−13. The archived Rayleigh central slope is0.001494167474901981; the conditional rounded-physical prediction differs by−1.3042e−13. Its conditional shift from the ideal prediction is−2.0913714990594516e−7, compared with the failed measured derivative error−2.0913729945775707e−7. The measured reconstruction contribution is−1.3056213885886901e−7 and the retained finite CI Rayleigh response is8.50476730769422e−14.

This rules out the **measured global one-body coefficient difference** as the dominant explanation at this point and direction. It supports orbital-reconstruction/constraint-normal sensitivity as the next discriminator. The saved full/retained table discrepancy includes accumulation rounding and the old per-block cutoff, so it is not a pure discarded-term bound. The raw native h representation is strongly asymmetric entrywise, but its measured raw/logical directional seam is about1e−15 here; an entrywise difference alone is not the derivative-error scale.

This remains a conditional linear explanation. The complete two-body coefficient contribution, finite polar curvature, full retained-CI response and cell arithmetic error have not acquired an error bound. The original002 derivative gate stays failed, both3451 stationarity gates stay failed, and optimization remains on hold. No correction was applied to experimental spectra. The next proposed saved-point test must center the full signed same-kappa Gram change with the actual quotient-normal gradient, retain finite quadratic terms and CI response, and state its same-point finite scope. It cannot become a production callback or relax an old gate.

The [result](actual/result.json), [authorization](root_authorization.json), [execution](root_execution.json), [contract](contract.json) and [driver](diagnose_saved.py) are immutable execution evidence. The [existing preserve-metric record](../../AGENT_INSIGHTS.md#preserve-metric) already cautions that finite centering has a remainder and a roundoff guard does not establish a universal sign bound.
