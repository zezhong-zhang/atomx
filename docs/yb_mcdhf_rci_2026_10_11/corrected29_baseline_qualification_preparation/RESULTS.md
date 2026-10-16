# Corrected-source baseline and two constrained controls: actual result

The sole root-authorized foreground run closed with exit 0. All 186 recorded live
controls passed. It used **5 complete H builds, 5 complete CI solves, 3 public
weighted gradients, no Dirac or optimizer**, plus 42 nuclear h/S calls, 13 tracker
resolves, 65 SVDs and 14 overlaps. Consumer time was 70.38 s (outer foreground 70.65 s), uncontrolled.

The private corrected `f0435dd6…` source retained the original full1046 space,
29 radial keys, five equal physical targets, seven original nuclear projectors,
3451 coordinates and inverse metric. Fresh source-specific H, signed eigenpairs,
gradients and owner/UID metadata were saved completely. Full normalized-table
gradient reconstruction passed the original whole-vector and metric gates.

The two fixed directions at `h=1e-6` passed their new constrained Q-Lagrange
controls: d0 error `−1.09315e−10` against `1e−8`, d1 error `−1.18129e−10` against
`1.49438e−8`. Full signed Gram quadratic and reconstruction identities, complete
CI phase/norm/residual/response, and the fixed baseline multipliers were retained.
These results qualify this baseline and these two conditional controls; they do
not introduce a Rayleigh production objective or an energy callback. The original
reported and uncentered tiny-step derivative failures remain failures.

**The point is still nonstationary:** original metric residual
`0.021499084147010395` exceeds `1e−7`; indicator
`1.207034808866711e−4 Ha` exceeds `1e−8 Ha`. A bounded optimization requires its own
contract and exact root execution seal. This result is not a converged MCDHF
spectrum or experimental certification.

The corrected versus old-source H difference is only about `4.16e−17 Ha`, while
reported eigenvalues differ by as much as `3.46e−11 Ha` and resulting lines by
roughly `1e−6 cm⁻¹`. These latter differences include assembly/eigensolver rounding;
they must not be called a physical improvement or conflated with the preceding
fixed-CI expectation correction (maximum `7.68e−12 cm⁻¹`). Neither measurement
explains the much larger experimental gap.

[Actual result](actual/result.json), [root receipt](root_execution.json),
[complete baseline arrays](actual/fresh_baseline.npz),
[all gradient and metric arrays](actual/all_gradients_metric.npz), and
[completion binding](completion.json) preserve the exact source and input chain.
Completion performed seven metadata/hash checks and zero scientific calls. Its
186 numerical checks are the sole live consumer's records, not a second numerical
replay. Frozen preparation prose and all historical failed evidence remain intact.
