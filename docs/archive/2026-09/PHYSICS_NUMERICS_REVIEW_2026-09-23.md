# Physics and numerical-performance review — 2026-09-23

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Reviewed commit: `c14b2bb66d59b2621fe8b08038ec07cb8e0e3ba9`.

This is the historical baseline review. Subsequent fixes and their validation
are tracked in [the remediation record](../../review_2026_09_23/REMEDIATION.md).

**Assessment: the reviewed implementation returns materially incorrect
physics through several public paths.** The exact CI angular Hamiltonian is a
substantial improvement over the older implementation, but that alone does not
make the orbitals, transition operators, continuum matching, or plasma kinetics
consistent. Eight high-priority correctness findings and four additional
correctness/performance findings are reproduced below. No library code was
changed during this review.

The review combined source inspection, analytic limits, independent operator
constructions, and execution. Earlier review documents were used for orientation;
their findings and performance numbers were not assumed to describe this commit.
This is a targeted deep review, not an exhaustive certification of every module.

## Findings requiring correction

### 1. [P1] MCDHF feeds nonorthogonal orbitals into orthonormal CI and hides the inner failure

Locations: [inner solve](../../../src/atomx/methods/mcdhf/solver.py#L997),
[outer convergence](../../../src/atomx/methods/mcdhf/solver.py#L1070),
[CI construction](../../../src/atomx/methods/mcdhf/matrix.py#L468).
Reproduction: `probes.py orthogonality`.

For `Atom(2, config=['1s2', '1s1 2s1'], engine='bspline', nuclear='point',
corrections='none').solve(tol=1e-10)`:

| Quantity | Observed |
|---|---:|
| Same-kappa overlap `<1s|2s>` | 0.05303017123 |
| Last inner `DFSolver` convergence | `False` |
| Outer MCDHF convergence | `True` |
| Reported CI ground energy | −2.86799111168 Ha |
| Energy after removing the overlap, in the same CI span | −2.86181232632 Ha |
| Spurious lowering | 0.00617878536 Ha, approximately 0.1681 eV |

The Slater–Condon rules used by `exact_ci` assume orthonormal spin orbitals.
The code verifies the abstract angular CSF vectors, but not the radial overlap
matrix on which those rules depend. `DFSolver` already detects this overlap and
rejects convergence; `MCDHFSolver` discards that verdict and accepts a small
change in the CI energy instead.

The independent check leaves 1s fixed and replaces 2s by
`(2s − <1s|2s> 1s)/sqrt(1 − |<1s|2s>|²)`. For this particular configuration
space, that transformation preserves the span of `1s²` and the singlet `1s2s`
CSF. Thus the energy difference is not a larger-basis correlation effect.

**Required change:** propagate inner failure and orbital-overlap diagnostics.
Enforce the radial orthonormality constraints before using ordinary
Slater–Condon algebra, or implement the appropriate nonorthogonal matrix
elements and generalized eigenproblem. Small energy changes cannot replace
these checks. The existing SCF orthogonality limitation was documented earlier;
the MCDHF suppression of its failure makes it substantially more dangerous.

### 2. [P1] Default MCDHF transition amplitudes do not represent the CI wavefunctions

Location: [CSF transition matrix elements](../../../src/atomx/properties/transitions/calculator.py#L661).
Reproduction: `probes.py lines`.

For point-nucleus He, configurations `['1s2', '1s1 2p1']`, with corrections off,
I constructed the electric-dipole operator explicitly as
`D_q = sum_ab <a|r C^1_q|b> a†_a a_b` over determinants and projected it onto
the **same** CSF vectors and CI eigenvectors as the production calculation.
The comparison shares the radial integrals and CSF basis; the operator
construction is independent of `one_body_csf_coupling` and the transition
calculator's population/recoupling factors.

| Decay to ground | Default `A` (s⁻¹) | Explicit operator, same wavefunctions (s⁻¹) |
|---|---:|---:|
| Lower J=1, predominantly triplet | 4.3662781 × 10⁹ | 10.79169 |
| Upper J=1, predominantly singlet | 5.8934991 × 10⁸ | 2.6512242 × 10⁹ |

The intercombination rate is wrong by approximately **4.05 × 10⁸**. In the
nonrelativistic limit an electric dipole cannot change total spin; the
singlet/triplet distinction requires coherent amplitudes with the correct
CSF phases. Configuration-average population factors and a special `sqrt(2)`
multiplier cannot supply the general many-electron reduced matrix element.

The optional `line_resolution='ci'` uses a separate implementation for a
restricted configuration-pair case and can re-solve different orbitals. Its FAC
tests do not validate the default transition operator on this MCDHF result.
Likewise, agreement between two radial gauges can survive the same erroneous
angular prefactor in both gauges.

**Required change:** evaluate one-body operators in the same CSF convention as
the Hamiltonian, then contract with the actual mixing coefficients. Gate the
low-Z spin-forbidden limit, coherent cancellation, and a determinant trace sum
rule in addition to individual strong lines.

### 3. [P1] Multichannel R-matrix matching computes the inverse reaction matrix

Location: [boundary matching](../../../src/atomx/scattering/r_matrix/cc_outer.py#L595).
Reproduction: `probes.py matching`.

The stated boundary condition is `u = a R u'`, with `u = F + G K`.
Substitution gives

```
(G − a R G') K = a R F' − F.
```

The implementation instead solves

```
(F − a R F') K = a R G' − G.
```

For one channel this is the reciprocal of the required K. It is subsequently
passed to `arctan(K)` and `(I+iK)/(I−iK)` as if it were the usual reaction matrix.

Take zero residual charge, l=0, k=1, boundary a=30, and the analytic solution
`u=sin(kr+delta)`, so `R=tan(ka+delta)/(ka)`. With exact sine/cosine reference
functions injected solely to isolate the matching algebra:

| Input delta | Required K=tan(delta) | Returned K | Returned phase |
|---:|---:|---:|---:|
| 0.1 | 0.1003346721 | 9.9666444233 | 1.4707963268 |
| 0.3 | 0.3093362496 | 3.2327281438 | 1.2707963268 |

Without that injection, the production Coulomb-function approximation also
introduces numerical error, but the inversion is still obvious (K=9.0948 and
3.1383). In the free-particle limit the desired K tends to zero; this formula
tends toward a pole instead. S remains unitary for a real scalar K, so a
unitarity-only test will not catch this.

**Required change:** rederive and fix matching, including channel flux
normalization for unequal momenta. Do not use `0.5*(K+K.T)` to conceal a
normalization error. Add an analytic free/known-phase boundary test, then
revalidate cross sections and resonances.

### 4. [P1] R-matrix collision strengths violate reciprocity

Locations: [cross-section weights](../../../src/atomx/scattering/r_matrix/cc_outer.py#L123),
[collision strengths](../../../src/atomx/scattering/r_matrix/cc_outer.py#L167).
Reproduction: `probes.py reciprocity`.

Both formulas divide by the incoming projectile's `(2 l_i + 1)`. That makes
the reported collision strength direction-dependent whenever incoming and
outgoing projectile angular momenta differ.

For channels `(target 1s, projectile p)` and `(target 2p, projectile s)` in
the same `L=1, S=0, odd-parity` block, a symmetric input
`K=[[0.1,0.2],[0.2,0.3]]` produces:

```
Omega[0,1] = 0.13558173036
Omega[1,0] = 0.40674519109
Omega[0,1] / Omega[1,0] = 1/3
```

S symmetry and unitarity errors are only 2.8e-17 and 2.3e-16. The failure is
therefore in the observable conversion, independently of finding 3.
Time reversal requires equality of the collision strengths at the same total
energy; see the explicit symmetry relation in
[Fritzsche et al., Collision Strengths of Astrophysical Interest](https://www.mdpi.com/2218-2004/11/5/80).

**Required change:** derive the target/projectile spin averages and LS weights
once for both sigma and Omega. In the standard unpolarized LS convention, a
channel contribution to Omega carries `(2L+1)(2S+1)/2`, not division by
`2l_i+1`. Verify both reciprocity and `Omega = g_i k_i² sigma / pi`.

### 5. [P1] `CRM.from_atom` turns spontaneous decay into excitation

Location: [Atom-to-CRM adapter](../../../src/atomx/plasma/crm.py#L250).
Reproduction: `probes.py crm`.

Atom transition rows use `initial_label` for the lower state and `final_label`
for the upper state; their Einstein A describes upper-to-lower emission.
`from_atom` assigns these to `upper` and `lower` in the opposite order.

For `Atom(2, config=['1s2','1s1 2p1'], corrections='none')`, the imported
CRM contains `upper=0, lower=2` and `upper=0, lower=4`, although State_0 is
the ground state. At zero electron density, the ground population consequently
has an outgoing spontaneous rate of **4.9556e9 s⁻¹**, with no incident photons
or collisions. The rate matrix also skips the collisional terms because the
reversed energy difference is negative.

Additionally, all imported levels receive `g=1`, including J=1 and J=2
states. Fixing the direction alone will still leave excitation and Boltzmann
ratios wrong. The single-configuration branch's use of occupied orbital
energies as many-electron levels is a further modeling restriction.

**Required change:** map emission direction using explicit lower/upper state
identities (and validate energy ordering); obtain `2J+1` from CI symmetry
labels. Test the imported rate matrix itself: spontaneous evolution from a
ground-state initial condition must not excite it.

### 6. [P1] CRM spontaneous RR includes an implicit thermal radiation bath

Location: [recombination coefficient](../../../src/atomx/plasma/crm.py#L654).
Reproduction: `probes.py crm`.

The implementation sets `alpha_RR = PI_Planck(T_e)/Saha(T_e)` and applies
that rate even when there is no radiation field. Detailed balance against
a Planck bath requires **spontaneous plus stimulated** recombination. It
does not determine the spontaneous coefficient alone.

Milne followed by Maxwell averaging gives a spontaneous kernel proportional
to `sigma_PI(omega) omega² exp(-E/T)`, with `omega=I+E`. Substituting a full
Planck occupation instead introduces the extra factor
`1/[1−exp(-(I+E)/T)]`. The effect becomes large for T comparable to or above I.
The Milne/Maxwell construction is also described in
[Kaastra et al., section 3.3.1](https://ned.ipac.caltech.edu/level5/Sept08/Kaastra/Kaastra3.html).

Independent integration of the **same Kramers PI cross section**, I=13.6 eV,
g_bound=g_continuum=1:

| T (eV) | Code / spontaneous Milne integral |
|---:|---:|
| 1 | 1.0048 |
| 13.6 | 1.3077 |
| 136 | 4.5059 |
| 1360 | 24.0568 |

At T=0.01 eV the code returns zero because Saha underflows, while the direct
spontaneous integral is **7.6332e-13 cm³/s**. There is no activation barrier
for electron capture that would justify that zero.

**Required change:** integrate spontaneous RR directly in outgoing electron
energy with a stable Maxwell kernel. Add stimulated RR only for an explicitly
provided photon field. An LTE cancellation test that defines RR by dividing
the same PI rate by Saha is circular and cannot establish correctness in the
zero-radiation coronal limit.

### 7. [P1] MCDHF silently ignores the requested radial box and grid refinement

Locations: [Atom construction](../../../src/atomx/atom.py#L879),
[grid reconstruction](../../../src/atomx/methods/mcdhf/solver.py#L978).
Reproduction: `probes.py grid`.

The MCDHF calls do not forward `basis.r_max` or the grid overrides; the
automatic grid is reconstructed using only ks, Z, and nuclear model. The
constructor explicitly accepts grid overrides for this path, so the user
receives no rejection.

For point-nucleus Z=92, corrections off, one electron:

| Config | Basis | ns | Energy (Ha) |
|---|---|---:|---:|
| `'1s1'` | default | 88 | −4857.606126071 |
| `'1s1'` | hi=.005, he=.1, r_max=30 | 157 | −4861.189023899 |
| `['1s1']` | default | 88 | −4857.606126071 |
| `['1s1']` | hi=.005, he=.1, r_max=30 | 88 | −4857.606126071 |

The list-form result retains the old last knot, 50.2873 a0; the refined
scalar-form grid ends at 30.8006 a0. Merely wrapping the configuration in a
list disables a **97.49 eV** energy improvement. Every run says converged.
The exact point-Coulomb 1s energy, `c²[sqrt(1−(Z alpha)²)−1]`, is
−4861.197903210 Ha: the default and refined errors are 97.7372 and 0.2416 eV.

**Required change:** pass the complete basis settings through every solver
route and construct the grid from them. Gate the actual grid and a physical
observable for scalar/list/configuration-average variants; constructor
acceptance is insufficient. SCF convergence and discretization convergence
must remain separate diagnostics.

### 8. [P1] Shooting PI/RR loses a factor of two in the initial-state average

Location: [shooting PI helper](../../../src/atomx/scattering/radiative_recombination/__init__.py#L238).
Reproduction: `probes.py rr`.

The initial subshell has `2j+1 = 2|kappa|` magnetic states. The shooting path
divides the reduced line strength by `|kappa|`; the B-spline PI and multipole
paths correctly divide by `2|kappa|`.

For He²⁺ capture into He⁺ 1s, using the shooting Coulomb continuum with 4000
grid points, the RR/Stobbe–Milne ratios at electron energies 1, 10, 100 eV are
**2.00232, 2.00043, 2.00212**. This isolates the factor-of-two error against an
analytic nonrelativistic reference, rather than against another approximate
continuum solver. The B-spline ratios on that grid are 0.8664, 0.9705, 0.9737;
its remaining pointwise pseudostate error is a separate convergence issue.

**Required change:** share the angular normalization across implementations
and add an absolute hydrogenic test for the shooting path. RR, RR rates, and
recombination-continuum spectra that use this helper inherit the error.

### 9. [P2] Solver occupation floors leak into reported physical occupations

Locations: [occupation floor](../../../src/atomx/methods/mcdhf/solver.py#L725),
[result occupations](../../../src/atomx/methods/mcdhf/solver.py#L1100).
Reproduction: `probes.py occupations`.

He `['1s2','1s1 2p1']` has an unmixed ground configuration with exactly two
electrons. The result instead reports occupations `{1s:2, 2p1/2:.001,
2p3/2:.001}`, totaling **2.002**. For Be `['1s2 2s2','1s2 2p2']` the sum is
4.0000960032. These are not physical expectation values of the particle-number
operator. The fields are consumed by density/property code.

**Required change:** calculate result occupations from raw integer CSF
occupations and normalized CI weights; retain artificial floors only inside
the solver where required. Check `sum occupations = Z−charge` (or the
configuration's validated electron count). For general densities, also include
off-diagonal one-particle density terms when configurations mix radial
orbitals of the same symmetry.

## Numerical performance

### 10. [P2] Continuum energy scans repeatedly diagonalize the same matrix

Locations: [continuum solve](../../../src/atomx/numerics/bspline/continuum.py#L152),
[PI loop](../../../src/atomx/numerics/bspline/continuum.py#L307).
Reproduction: `probes.py perf_rr`.

For fixed kappa, potential, and grid, H and S do not depend on the requested
continuum energy. Nevertheless each energy rebuilds them and solves the full
generalized eigenproblem. The requested energy is used only afterward to
select a positive-energy pseudostate.

He⁺ 1s E1 RR, 80 energies from 1 to 500 eV, ns=278, one BLAS thread:

| Implementation in diagnostic process | Diagonalizations | Median wall time, 3 runs |
|---|---:|---:|
| Current | 160 | 3.825 s |
| Cache identical `(H,S)` eigensystems | 2 | 0.630 s |

Every output array is **bit-identical**. Measured speedup: **6.08×**. The
diagnostic cache still hashes and reconstructs the matrices, so this does not
require a speculative rewrite or an accuracy tradeoff.

**Required change:** create a continuum workspace per grid/potential/kappa,
diagonalize once, reuse spectral normalization and bound-continuum radial
matrix elements, and bound cache lifetime. Invalidate on potential changes.
Cache only the energy-independent eigensystem for this pseudostate method;
an exact-energy shooting solution has a different dependency structure.

### 11. [P2] The exact-CI size guard executes after combinatorial allocation

Locations: [determinant generation](../../../src/atomx/methods/mcdhf/exact_ci.py#L205),
[late cap](../../../src/atomx/methods/mcdhf/exact_ci.py#L801),
[quadratic table](../../../src/atomx/methods/mcdhf/exact_ci.py#L835).
Reproduction: `probes.py cap`.

`PatternLayout` enumerates and retains **every M** determinant before
`_block_structure` checks the requested M=J block against `max_determinants`.
Requesting a cap of just 1 for
`((4,-4,4),(4,3,3),(5,-5,5))` first constructs **352800 determinants**, uses
approximately **67.4 MiB additional peak RSS**, and spends 0.43 s, then raises
because the selected block has 21718 determinants. The layout remains cached.
Adding the half-filled `(5,4,4)` shell multiplies the all-M count by 70 to
**24,696,000** before the same guard. That larger allocation was not executed.

Within the cap, construction still scans all determinant pairs; the sparse
table has `D²` rows, and evaluation materializes a dense `D×D` matrix. The
module's major caches use `lru_cache(maxsize=None)`, retaining angular spaces
and tables across unrelated runs. Calling the coefficient table sparse does
not make the full algorithm sparse or memory-bounded.

**Required change:** count dimension before allocation, generate only the
required M sector, reject over-budget layouts early, and use bounded caches.
For larger spaces generate connected single/double replacements and apply H
as an operator, instead of testing all determinant pairs and building dense
matrices. Preserve the existing angular-oracle gates during that work.

### 12. [P2] Symmetry-blocked CI ignores the requested Davidson solver

Location: [solver dispatch](../../../src/atomx/methods/ci.py#L164).
Reproduction: `probes.py davidson`.

When there is more than one J/parity block, `ci_solve(..., solver='davidson',
n_states=1)` always executes `_solve_blocked`, which computes every dense
eigenpair of every block, assembles the full mixing matrix, and then slices
the result. The requested iterative solver and residual controls have no
effect. The probe isolates dispatch using a four-dimensional diagonal
Hamiltonian and observes two dense eigensolves and `solver='dense-blocked'`.

**Required change:** select the algorithm within each block, request only the
needed low states, then merge candidate eigenvalues globally. Small dense
blocks can use `eigh(subset_by_index=...)`; see
[SciPy's documented subset support](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
A matrix-free build is also needed if Davidson is intended to remove the
Hamiltonian storage bottleneck, rather than only reduce diagonalization cost.

## Model limits and validation priorities

These are distinct from the reproduced defects above:

- The MCDHF radial iteration optimizes an averaged monoconfiguration Fock
  problem using squared CI weights and occupancy products. A fully variational
  MCDHF functional `E=sum_rs c_r c_s H_rs` also has off-diagonal density/operator
  contributions to its orbital derivative. It needs constrained orbital
  gradients, not just an unchanged energy and weights. The formal variational
  requirements are described in the authors' [GRASP manual](https://www.mdpi.com/2218-2004/11/4/68).
  Label the present orbital optimization approximation explicitly.
- The default B-spline RR helpers pass no residual-core SCF potential to their
  continuum solves: they use the bare nuclear field even with many-electron
  bound orbitals. Hydrogenic benchmarks cannot establish accuracy for that
  approximation. The pseudostate method also returns the nearest box energy,
  which explains why pointwise threshold cross sections need separate box/grid
  convergence tests even after energy normalization is fixed.
- The default high-Z grid limitation remains real even on paths where the
  refinement settings are honored. Test basis/box convergence separately from
  the SCF fixed point.
- The CI Breit operator is Gaunt-only; frequency-independent retardation is
  explicitly absent. QED uses model corrections. These are stated modeling
  choices, not demonstrated bugs in this review; “Breit” or “frontier QED”
  alone does not communicate an accuracy budget.
- A nonrelativistic exact energy is not a rigorous lower bound on a
  relativistic Dirac–Coulomb calculation. Some invariant-test prose makes that
  claim. Variational gates must compare the same Hamiltonian and use an
  appropriate positive-energy/no-pair convention; numerical proximity to an
  NR limit is a useful sanity check but not that theorem.

The most valuable new gates are the same-wavefunction operator comparison,
radial Gram-matrix constraints, analytic scattering boundary conditions,
reciprocity, imported CRM ground-state stability, and independent spontaneous
Milne integrals. Existing positive checks for angular algebra, radial
quadrature, Uehling integration, and eigensolver residuals remain useful.
They do not establish that adjacent layers share their conventions.

Recommended order: stop false convergence and incorrect transition rates
(1–2); repair the two independent R-matrix errors (3–4); fix CRM direction
and RR physics (5–6); restore numerical control forwarding and normalization
(7–9). The eigensystem cache (10) is a low-risk performance improvement.
Dimension preflight and block-aware iterative solving (11–12) should precede
claims of scalable large-CI calculations.

## Reproduction and scope

The executable diagnostics are in [review_2026_09_23/probes.py](../../review_2026_09_23/probes.py).
Each mode must run in a separate process because some deliberately replace
functions in memory to isolate dispatch/algebra or measure caching. None edits
the library. They print observations; they are not a new passing test suite.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/atomx-mpl \
  .venv/bin/python docs/review_2026_09_23/probes.py MODE
```

Modes: `orthogonality`, `lines`, `matching`, `reciprocity`, `crm`, `grid`,
`rr`, `occupations`, `perf_rr`, `cap`, `davidson`.

Environment: Linux x86-64, AMD Ryzen 9 9950X3D, Python 3.10.21, NumPy 2.2.6,
SciPy 1.15.3, Numba 0.65.0. pytest 9.1.1 and SymPy 1.14.0 were installed in
the existing virtual environment for the review; manifests were not changed.
Performance numbers are local measurements, not projections from the older
macOS audit. Peak-memory differences use Linux `ru_maxrss`.

Initial selected regression run: **134 passed in 95.20 s** across
`test_first_principles_invariants`, `test_ci_analytic_slater_condon`,
`test_ci_grid_kernel`, `test_rr`, `test_plasma_physics`, `test_davidson`,
`test_gauge_invariance`, and `test_scf_nan_sentinel`.

Second selected run: **132 passed, 1 xfailed in 168.73 s** across
`test_r_matrix`, `test_r_matrix_phase2_vs_burke1971`,
`test_ci_matrix_vs_mscheme_reference`, `test_rkb_bank_sharing`, `test_scf`,
`test_breit_operator`, `test_qed_uehling`, and `test_slater_integrator`.
Total: **266 passed, 1 expected failure**. In particular, the existing
R-matrix resonance gates pass despite findings 3 and 4: locating a resonance
does not validate the absolute background phase or cross-section weights.

Recorded diagnostic output is in
[review_2026_09_23/probe_output.txt](../../review_2026_09_23/probe_output.txt).
The diagnostic script passes `ruff check` and was executed mode by mode.
The full repository suite and fresh external FAC/GRASP calculations were not run.
Molecular Auger, the complete high-Z QED model family, all distorted-wave
channels, and every property implementation were not exhaustively audited.
