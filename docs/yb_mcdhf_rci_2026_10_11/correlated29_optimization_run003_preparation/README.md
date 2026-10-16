# Corrected-source bounded native29 MCDHF preparation

This consumer is prepared, not executed. It reuses the public native orbital
optimizer and ordinary reported five-state EOL objective. It supplies no energy
difference callback, Rayleigh objective adapter or replacement physical kernel.
Only the separate root's exact execution seal can permit the numerical run.

The initial point is the corrected private `f0435dd6…` source's own
[qualified complete baseline](../corrected29_baseline_qualification_preparation/RESULTS.md):
1046 signed CI eigenpairs, full29 production gradient, original seven nuclear
projectors, 3451 coordinates and inverse metric. That baseline and its two fixed
constrained derivative controls passed; its residual `0.021499084147` and
`1.20703e-4 Ha` indicator remain above the original `1e-7` and `1e-8 Ha` gates.
The earlier reported and uncentered tiny-step derivative failures remain failures.

Point zero must independently reproduce **this corrected baseline's** H, reported
roots, signed C and physical waves byte for byte. The entire raw coordinate and
radial gradient use its own `1e-9` gate; its own original-metric preconditioned
vector uses `1e-10`. The owner input and retained-H matrix SHAs are bound
separately from the complete implementation source identity. Corrected values
are never required to equal the old500ac outputs.

The single public call allows at most 16 new accepted steps, 65 H builds,
65 complete CI solves and 17 full production gradients. `from_arrays` validates
the original seven nuclear positive spans and restores immutable P, coordinates
and inverse blocks without replacing them or inheriting secant history. All29
orbitals may relax. Original 69 electrons, Fermi A175 nucleus, five physical UIDs,
equal weights, complete target sectors and frozen original tracker are retained.
The 2163-CSF NR parent is not computed; its declared five target sectors contain
the complete 1046-CSF calculation here.

Every accepted point archives H, all signed eigenpairs, full radial and coordinate
gradients, original metric, physical waves and source/owner/UID identities before
qualification. It validates complete sector residuals and Gram matrices, radial
constraints, no-pair leakage and unchanged metric. The existing exact dyadic
stored-H consumer then checks normalized signed `cᵀHc/(cᵀc)` against the actual
reported EOL and the optimizer's actual Armijo step/slope/RHS metadata. This guard
checks finite stored-H descent; it is not a uniform radial derivative bound.
The first unresolved candidate is retained unqualified, the previous qualified
checkpoint remains authoritative, and no further optimizer update is made.

Each qualified checkpoint separately records the physical lower and upper state
energies, four transition shifts, centroids, splittings, nominal NIST gaps/RMS and
original stationarity gates. Label/UID matching determines the ground and four
upper states; insertion order is forbidden. EOL descent does not guarantee gap
reduction. Missing NIST uncertainty data prevents uncertainty-based certification.
Neither early accepted steps nor a small total-energy change proves physical
completeness, converged radial correlation or highest atomic accuracy.

Resources are CPU31, all numerical threads one, 3.95 GiB address space, 600 s
inner and 615 s outer caps, with optimizer soft wall 570 s. The budget explicitly
includes seven validation Dirac calls (at most 21 spectral attempts), 36 geometry
SPD validations, at most 462 nuclear h/S calls, 346 SciPy spectral calls,
81 tracker resolves/405 SVDs/82 overlaps, 448 polar calls and 85 full signed target
density recontractions. Exact stored-H arithmetic is bounded to 17 points,
85 quotients and 3 million examined pairs. These caps do not promise 16 steps.
A new empty source-isolated NumBa cache is required; timing is uncontrolled.

[Preparation verification](preparation_verification.json) passed 14 stdlib,
source, JSON, ZIP-field, AST, UID-order and refusal checks with zero numerical
imports or scientific calls. Ruff passed. The public optimizer invocation's AST
is unchanged from frozen run002; only source-specific prerequisite/point bindings
and owner metadata changed. The old preparations, source archives and failed
results were left intact.
