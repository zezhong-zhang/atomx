# J-resolved one-electron shooting — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The configuration-average campaign cannot resolve transitions involving
individual p, d, or higher-l fine-structure levels. A new independent benchmark
uses the NIST configuration and J to select the Dirac kappa explicitly:
`kappa=l` for J=l−1/2 and `kappa=−l−1` for J=l+1/2. Occupation is one in
that subshell, without statistical averaging over its fine-structure partner.
The resolver rejects many-electron inputs and incompatible angular identities.

`scripts/benchmark_shooting_one_electron_lines.py` retains all 454 standard
observed one-electron lines with Z≤20 and queues 218 distinct classified
states, each on both inherited radial grids. Missing configurations and missing
or ambiguous J remain explicit unresolved records. Nuclear mass, isotope,
radial box and starting grid requests come from the original first20 queue.
The same strict correction stack applies: DBSR-style Breit, model self-energy,
vacuum polarization, operator normal/specific recoil and the hydrogenic mass
remainder. The two-electron Breit and specific recoil terms vanish here.

Each state has an initial solve (tolerance 1e-12, maximum 300 iterations) and
a warm implicit-exchange solve (tolerance 1e-13, maximum 100 iterations).
Both stages must converge. The comparison requires both grids and checks the
recorded angular identity and endpoint Hamiltonians. It reports grid changes
separately from experimental residuals. It does not resolve hyperfine structure
or implement complete high-order bound-state QED.

## Initial hydrogen diagnostic

The 3200-point probe for 2p₁/₂ → 2s₁/₂ gives:

| Quantity | cm⁻¹ |
|---|---:|
| J-resolved shooting + corrections | 0.03534811168 |
| Observed NIST interval | 0.03528597696 |
| Difference | 0.00006213472 |

The previous configuration-average interval was −0.2085807 cm⁻¹. Resolving
J restores the physical ordering, but leaves an approximately 0.176% residual
for this small Lamb-shift interval. The initial probe alone has no two-grid
uncertainty assessment; the full queued comparison must supply that check.
No experimental values were inserted into calculated energies.

## Status and provenance

The completed calculation used frozen source under
`/tmp/atomx-one-electron-j-20260929`, with 436 grid attempts. Its output directory
is `output/nist_spectra/shooting_one_electron_j`: the immutable manifest records
input states, full observed rows, parent identity and source hashes; attempts
are checkpointed in JSONL; the final comparison is written after the queue
finishes. This campaign is separate from the configuration-average recovery
databases. Never resume it with changed source under the same output directory.

Eighteen angular-resolution, queue and existing generation/recovery tests
passed. The queue test explicitly retains lines missing J and distinguishes
p₁/₂ from p₃/₂.

## Completed run and observed-feature review

All **436 grid attempts** succeeded; all **218 states** passed the paired
angular-identity comparison, enabling **337** classified line comparisons.
The other **117** records lack sufficient J information. Maximum interval
change between grids is 9.75e-7 cm⁻¹. At the finer grid the hydrogen interval
above is 0.03534811446 cm⁻¹; its grid change is −7.88e-9 cm⁻¹ and its residual
is 6.21375e-5 cm⁻¹. The remaining discrepancy is not removed by this grid
refinement.

Matching calculated J endpoints does not prove that an observed spectral
feature resolves them. Across the retained records, **148** have shared
intensity markers and **216** share an observed wavelength, medium and
reference with another record. NIST defines the intensity asterisk as shared
intensity, not automatically a resolved measurement or a definite blend.
See the [official NIST line help](https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html).

For example, Mg XII 1s → 6p₃/₂ differs from the recorded observation by
14,457.65 cm⁻¹, but from its Ritz reference by about −8.97 cm⁻¹. The observed
wavelength has no reported uncertainty in this record and is also assigned
to the other fine-structure component. Neither agreement with Ritz nor the
large observed discrepancy alone establishes high experimental accuracy.

`review_shooting_one_electron_lines.py` preserves observed residuals, adds
separate Ritz diagnostics and shared-feature metadata, and writes a separate
reviewed report. The native campaign website offers this J-resolved snapshot.
It does not include these shared-feature comparisons in its strict unique-line
RMS. Source and manifest hashes remain available with the records.
