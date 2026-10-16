# Native joint generalized eigenblock refinement

The new shared
[`refine_generalized_eigenblock`](../../../../../src/atomx/numerics/bspline/dirac_assembly.py)
repairs leading eigenstates in the original full real H/S coordinate pencil,
jointly orthogonalizes all supplied columns, then applies shifted Ritz only
within the leading band. It retains the original lower-defined H, S, columns
and roots. The numerical approximation to the initial electron span may change;
the routine does not define a physical no-pair branch or extra constraints.
This is a native numerical engine, not an external electronic-structure backend.

The [saved-matrix diagnosis](../block_diagnosis001/README.md) explains why a
fixed-span repair was insufficient. Full119-dimensional Ritz restored Gram to
about1e−14 but left a full S-dual residual about1.46e−7 Ha, mainly outside
that approximate span. Repeating wide contractions did not eliminate it.
The original overlap metric has condition number3.34e8 and unselected energies
reach6.43e8 Ha. These observations motivated full239-coordinate residual
correction and a separate leading-band Ritz, rather than an enlarged cutoff.

```python
energies, coefficients, info = refine_generalized_eigenblock(
    H_active, S_active, original_energies, original_coefficients,
    leading_count=number_of_requested_leading_roots,
    dimension_cap=256, column_cap=128, memory_cap_bytes=128*1024**2)
```

Inputs must contain finite real matrices, an exactly symmetric positive metric,
strictly ordered isolated roots and an S-orthonormal supplied block. H's lower
triangle defines its operator. The original scalar Newton helper keeps its
default Euclidean objective; the explicit block consumer uses the full S-dual
objective. Two corrections per selected root act in all original coordinates.
Ordered Cholesky QR preserves the leading span while restoring the entire
block's Gram. Wide shifted contractions feed a leading-only double-precision
Ritz solve. Root order, overlap, gap-relative energy movements and joint Gram
remain checked. Unselected columns are retained but not newly residual-certified.
Extra occupied constraints must already be represented in the supplied pencil.
Neither factory defaults nor an ongoing MCDHF frozen projector is changed.

Resource admission precedes factorizations: at most256 coordinates,128 columns
and an estimated128 MiB NumPy payload. Python/BLAS/SciPy RSS overhead is extra.
Per leading count k, explicit kernel work is at most2k bordered solves,
3k+2 triangular solves, two Cholesky calls and one leading Ritz solve.
Unsupported extended precision and invalid/degenerate inputs reject.

The [real frozen-Yb68 application005](../yb_high_l005/RESULTS.md) passed all
802 original controls. Each g partner retains119 electron columns and every
one of67 original roots below50 Ha, including six bound roots. The original
H/S, core bank, isotope/geometry, branch, band and thresholds remain fixed.

| Full-band or full-block measurement | κ4 | κ−5 |
|---|---:|---:|
| Original band S-dual max / Ha | 1.81922e−7 | 5.50694e−7 |
| Corrected band S-dual max / Ha | 1.85548e−11 | 3.55956e−11 |
| All119-column S-Gram error | 2.22045e−15 | 1.99840e−15 |
| Complete metric-projector error | 4.31191e−15 | 3.73188e−15 |

The band gate remains1e−8 Ha; Gram/projector/Gauss gates remain1e−10.
The original numerical span changes only by measured maximum S-norm
3.88e−12/1.26e−11 outside components. Source/ownership, complete-band membership,
ordered-root identity and Gauss checks pass. High roots at/above50 Ha remain
present, with residual maxima2.37e−6/2.77e−6 Ha that are not given the band gate's
certificate. Core stationarity is not newly measured. Strict paper inputs,
high-l correlation and experimental-gap improvement remain unverified.

Work accounting clarification:005 records four instrumented **kernel** Cholesky
calls, plus two additional driver projector Cholesky calls visible in source;
there are also internal solver factors outside that explicit census. Its
sealed RESULTS shorthand is clarified by the independent
[result review](../block_diagnosis001/result_review005.json). The actual one
fresh Fock build and frozen κ4 reuse are different scopes; no speedup is claimed.
All earlier failures and the unexecuted fixed-span004 proposal are preserved.

[Six new tests](../../../../../tests/test_generalized_eigenblock.py) use an
independently constructed pencil and exact spectral states. They check recovery
of omitted-span components, signed physical projectors, all retained columns,
joint metric identities, unchanged input ownership/lower-H convention and
pre-factorization caps. The selected seven-file panel passed **86 tests, no
skips,32.49 s**: generalized blocks, Rydberg refinement, orbital accuracy,
hydrogenic Dirac, complex-scaled Dirac and both architecture panels. Existing
`.venv`, CPU26, BLAS/OMP/Numba threads1 and a task-local Numba cache were used.
This duration is validation scope, not a performance comparison.

Documentation disposition: update DF controls, capabilities, testing, migration
and paper navigation. Earlier source-only contracts and failed numerical results
remain their own epochs. Existing native-metric/residual principles cover this
finding; no new general rule is added. The [physical-priority audit](../physical_priority_audit001/README.md)
separately directs the next spectral test to common31-orbital correlation×radial
relaxation, without identifying a virtual residual as a spectral error bound.
[Receipt](receipt.json) binds current source and completed evidence.
