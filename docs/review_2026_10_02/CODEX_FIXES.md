# Repairs and validation for the Codex review

This report closes the eleven findings in the Codex review of `5018fab8`,
started on 2026-10-01. Its numbering refers to that review, rather than the
separate, concurrent review in this directory's `README.md`.

The main repairs were incorporated into `9eb22ca` by a concurrent session.
The final He RPA fixture and test-audit repairs, plus documentation below, are
subsequent working tree changes. This session did not create a commit. The original external
FAC reference columns were preserved; only computed atomx columns and explicit
Hamiltonian flag metadata were refreshed.

## Finding closure

| Finding | Repair | Validation |
|---|---|---|
| F1: sparse CI reused stale physical state | Fingerprint orbital contents and grid data, reset radial memoization per assembly, limit cached CSR matrices to four entries and 128 MiB. | In-place orbital mutation changes the spectrum; a cached result agrees with a fresh assembly. In-place quadrature mutation changes the fingerprint. |
| F2: particle-particle integrals substituted for RPA particle-hole tensors | Contract normalized particle-hole tensors with direct and exchange spin-orbital Coulomb elements for both A and B. Reuse fixed-bra radial bilinears. | Independent magnetic contractions reproduce both matrices; canonical He singlet and triplet TDA blocks agree with CI excitation gaps to about 1e-15 Ha. |
| F3: local continuum virtuals mixed with nonlocal DHF occupied orbitals | Build virtuals in the metric-orthogonal complement of the occupied space of the common nonlocal V^N Fock operator; check the native orbital Gram matrix. | Maximum occupied–virtual overlap is about 3.6e-16, compared with 0.666 in the original probe. Deliberately overlapping inputs raise an error. |
| F4: fixed-M crystal-field workflows dropped nonzero-q components | Enumerate the full M space for nonspherical operators in the Ce and Eu drivers. | A field containing only q≠0 reproduces the complete one-electron operator matrix on Python and Numba paths. Ce cubic degeneracy gates pass; Eu runs on all 3003 determinants. |
| F5: external one-body operators dropped cross-κ terms | Allow all spinor partners on the external operator path. | Cross-j one-electron matrices agree with the explicit CF operator to 1e-14 Ha. |
| F6: Sigma2 hole-line exchange buckets broken | Evaluate direct and exchange buckets independently once; also repair bubble row-table selection and enforce M conservation. | Both kernels satisfy off-diagonal Hermiticity and bra/ket fermionic antisymmetry across the complete small valence space, with nonzero gates. |
| F7: Slater scaling used direct radial slots in exchange | Swap the ket radial slots in the exchange contribution. | Scaling through the hook agrees with direct radial-kernel scaling to about 1e-17 Ha. |
| F8: orbital MBPT had wrong signs/slots, discarded terms, and swallowed errors | Implement diagonal closed-shell on-shell GF2 with both 2p1h and 2h1p terms, all allowed ranks and antisymmetrized exchange; retain bound virtuals and same-subshell partners; propagate basis errors. | A nonzero explicit spin-orbital sum matches production to relative 1e-11; bound-virtual and injected-failure tests pass. |
| F9: MBPT rebuilt the same eigensystems inside its sum | Build each κ virtual basis once per calculation and reuse radial bilinears. | The reviewed four-channel Be workload requires four basis builds, compared with 64 before repair. Current warm time is 13.7 ms on this host. |
| F10: tests missed defects and the ordinary suite was red | Replace duplicated arguments and shape-only checks with operator comparisons, off-diagonal identities, nonzero corrections and analytic limits. Diagnose failing fixtures and numerical branches. | Full-suite and focused results are recorded below. |
| F11: scripts, exports and lint were broken | Repair fit coordinates and undefined names, use repository-relative imports, move the reusable rare-earth workspace into production code, remove the nonexistent export and clear configured lint violations. | Ce CF, Eu Tier-A, Eu CF and dense-Ne RPA drivers complete. Configured Ruff scan passes. |

## Additional numerical and validation repairs

The inhomogeneous orbital-source solver now finds the minimum of its convex
normalization residual between adjacent poles before bracketing the physical
root. This repairs the excitation-autoionization fixture errors. A manufactured
two-pole problem gates the normalized equation directly. The private occupied
rotation helper again accepts list-backed buffers.

The one-electron CI test now uses a basis demonstrated to converge to the
analytic Dirac energies. The continuum channel identity compares the same
finite-frequency photon operator on both sides. The Na hyperfine fixture now
requires an actually converged SCF result; the hydrogen hyperfine check uses
the independent Dirac 1s expression. The Be excited-orbital check no longer
imposes an unjustified hydrogenic bound on a correlated orbital multiplier.
Obsolete benchmark API requests and the obsolete RPA scaffold test were
repaired. Implementation snapshots were refreshed after the independent
numerical gates; the FAC accuracy thresholds were retained.

The FAC structure driver now explicitly matches static Breit and QED flags
on both programs. Local FAC defaults enable static Breit; the previous
documentation incorrectly stated otherwise. For the stored panel, the largest
absolute difference is 11.13 eV and the largest relative difference is 0.02395%.
At Z=82, the He-, Ne- and Ar-like differences are 0.81, 6.08 and 11.13 eV.
Stored external FAC energy and PI/RR cross-section values were not altered.

Deprecated spherical-harmonic calls were replaced with a version-compatible
adapter, eliminating the warning flood. The Planck-weight helper avoids
evaluating overflowing exponentials. The thin compiled CG adapter no longer
uses a disk cache that can retain a stale imported Wigner implementation.
Closure bindings and explicit numerical-module imports replace fragile implicit
dependencies uncovered by lint.

## Independent He RPA anchor and reference consistency

The concurrent commit `887a452` introduced two strict expected failures using
a tiny-occupation, separately optimized 2s orbital. Its reported multiplier
was -0.0929086335 Ha, whereas the common V^N Fock expectation was +0.0240783367
Ha. The resulting energy-gap mismatch was 0.1169869705 Ha in both angular
channels. Substituting that multiplier into canonical TDHF did not test the
stated common-reference equations.

The repaired fixture starts from a closed-shell He reference and diagonalizes
its occupied-orthogonal V^N virtual complement. Both tests are ordinary passing
tests with a 1e-9 Ha tolerance, tightened from the previous 1e-3 Ha:

| Block | RPA A, Ha | Independent CI gap, Ha |
|---|---:|---:|
| J=0 | 0.8671164737683195 | 0.8671164737683184 |
| J=1 | 0.8669789222353463 | 0.8669789222353448 |

The identification of A with the CIS excitation matrix assumes the common
Hartree–Fock reference, as described in the
[Q-Chem CIS documentation](https://manual.q-chem.com/6.2/subsec_CIS.html).
The numerical diagnosis is archived with the validation artifacts.

Two additional backward-kernel tests independently match the ground-to-double
excitation CI matrix element: B=+6.877576648656228e-5 Ha for ph J=0 and
B=-6.877576648656225e-5 Ha for ph J=1. The latter is allowed: particle-hole
J=1 recouples an allowed particle-particle J=0 pair, and does not require the
Pauli-forbidden particle-particle J=1 matrix element.

## Measured workflows and scope

The ns=40 dense-Ne calculation retains 51 virtuals and 91 particle-hole pairs.
It gives TRK=10.26, independent-particle α=1.969, TDHF α=2.374 and the lowest
pole at 17.99 eV. These are finite-box, finite-energy and finite-partial-wave
results, rather than converged experimental predictions. The sign of the
response correction is determined by the equations rather than prescribed by
a screening-direction assertion.

The Eu fit optimizes two coordinates with the remaining scale fixed at its
prior. The resulting scales are (-0.2726, -0.1424, -0.1669), residual RMS
8.2 cm^-1 and Jacobian condition number 24.5. The three calibration levels
are 17275.5, 19015.4 and 21508.9 cm^-1. The unfitted 7F ladder retains
3.4–7.7% discrepancies and is reported as a holdout. The full-M CF trace
residual is 3.52e-4 cm^-1; the direct operator trace is gated separately,
and the assembled subtraction uses a roundoff budget based on its large
absolute diagonal energies.

The orbital GF2 helper estimates orbital self-energies. Its difference is not
a complete neutral-excitation or transition-operator correction. Empirical
Slater scaling remains a calibration, and correlation, radial, box and
partial-wave truncations still require convergence studies. Other pre-existing
expected failures and findings from the separate concurrent review are outside
the eleven-finding closure reported here.

## Reproduction and final checks

Use the repository virtual environment with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`.
The broad run uses `NUMBA_CACHE_DIR=/tmp/atomx-fix-numba-clean` to avoid
inherited compiled artifacts. A separate Wigner regression run also passes
with the default cache (21 tests).

Commands:

```text
.venv/bin/pytest -q -m 'not slow' --tb=short
.venv/bin/pytest -q tests/test_rpa_tdhf.py tests/test_review_regressions.py --tb=short
.venv/bin/ruff check src tests scripts benchmarks
git diff --check
.venv/bin/python scripts/ce3_cubic_cf_validation.py
.venv/bin/python scripts/eu3_tierA_anchor.py
.venv/bin/python scripts/eu3_cf_anchor_pipeline.py
.venv/bin/python scripts/rpa_ne_dense_continuum.py
```

The complete non-slow run completed with **3931 passed, 11 skipped,
28 deselected, 21 xfailed, zero unexpected failures and one warning** in
986.63 seconds. Its collection preceded the final He fixture repair: two of
those expected failures have since become ordinary passing tests, verified by
the focused final run. The current tree retains the other 19 existing expected
failures. The slow-marked tests were excluded. The sole warning concerns a
Matplotlib `tight_layout` plot, rather than the former spherical-harmonic flood.

The broad run is preserved in
[non-slow-suite.log](codex_fix_validation/non-slow-suite.log).

Focused final checks: **20 passed** for RPA and the new review regressions;
**2 passed** for the repaired legacy Sigma2 comparison and full-M CF test;
**21 passed** for the default-cache Wigner cancellation regression. Lint and
`git diff --check` pass. The two legacy test repairs now compare the actual
same-J and full m-scheme values, and enumerate the M-changing CF destinations.

Evidence: [metadata](codex_fix_validation/metadata.json),
[operator and RPA tests](codex_fix_validation/final-anchors.log),
[He reference diagnosis](codex_fix_validation/he-anchor-diagnosis.log),
[Ce CF](codex_fix_validation/ce-cf.log),
[Eu Tier-A](codex_fix_validation/eu-tier-a.log),
[Eu CF](codex_fix_validation/eu-cf.log),
[Ne RPA](codex_fix_validation/ne-rpa.log),
[FAC panel](codex_fix_validation/fac-structure.log), and
[MBPT measurement](codex_fix_validation/mbpt-performance.json).
