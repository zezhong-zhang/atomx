# B-spline performance review — 27 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Scope and measurement

This review covers the bound-state B-spline DHF path, its shared Coulomb
integrals, and the B-spline-dependent MCDHF/CI and continuum consumers.
Timing controls fixed OpenBLAS, OMP, and MKL to one thread before importing
NumPy. The measured solve was Ne I `1s2 2s2 2p6` on the production Fermi
physics grid, with three DHF sweeps; `hmax=1.0` has 113 splines and
`hmax=0.18` has 333. These three-sweep timings isolate numerical cost;
they are **not** claims of converged spectral accuracy. Cold runs include
bank construction; warm runs reuse the same-grid process cache. Timed
Python wrappers add overhead and overlap their callers, so their durations
must not be summed as disjoint wall time.

| 333-spline measurement | Full-bank direct | Compact direct |
| --- | ---: | ---: |
| Warm uninstrumented three-sweep solve, two interleaved repeats | 2.23–2.25 s | 1.36 s |
| Final energy, warm scoped profile | 0.75 s | 0.02 s |
| Three-sweep energy | −128.68455490495919 Ha | −128.6845549049612 Ha |
| Energy difference | | 2.0e-12 Ha |

Before direct cell assembly, the compact route's warm scoped 333-spline
run spent approximately
0.71 s in 15 Dirac eigensolves and 0.61 s in 120 exchange convolutions.
Direct and final-energy compact contractions together took about 0.06 s.
That implementation's first run spent about 2.45 s acquiring full Coulomb
banks. It retained 12 transposed full banks totalling **733.5 MiB**,
plus 35.4 MiB of compact overlap strips and 0.26 MiB of pair moments;
process peak RSS was about **1.16 GiB**.

The subsequent direct cell builder now assembles the exact overlap strip
without a full bank. A compact mixed P/Q exchange contraction also removes
the last full-bank requirement from batch DHF at 320 splines or more. In a
333-spline, three-sweep Ne control with one BLAS thread, the intermediate
direct-cell version retained three `qpqp` banks (182.7 MiB), ran in 2.06 s
cold and 1.34 s warm, and reached 632.5 MiB peak RSS. The final version
retained **zero full banks**, 35.4 MiB of overlap strips, and reached
425.0 MiB peak RSS in a fresh warmed-JIT process; its cold solve took
1.67 s, and a repeated same-process solve took 1.45 s. The final and
intermediate three-sweep energies differed by 1.6e-12 Ha. These are
distinct measurements, not a controlled interleaved speed comparison.

## Physics behind the implemented speedup

For a radial B-spline pair (u_{i,p}(r)=B_i(r)B_{i+p}(r)), define moments

\[
M^+_{i,p}=\int u_{i,p}(r)r^k\,dr,\qquad
M^-_{i,p}=\int u_{i,p}(r)r^{-k-1}\,dr.
\]

If the supports of pairs (u) and (v) do not overlap, the Coulomb kernel
\(r_<^k/r_>^{k+1}\) separates exactly: the integral is
\(M_u^+M_v^-\) or \(M_u^-M_v^+\), according to radial order. Only the
strip where their supports overlap needs the original two-electron cell
integral. This follows from the same radial kernel and cell construction
used by [DBSR_HF](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=919310);
the compact contraction and its crossover are Atomx measurements.

The prior change used these factors for **symmetric exchange** at 320 or
more splines. This review adds a prefix/suffix contraction for the **direct
Fock term** and all four P/Q blocks of the **final Slater energy** at 256
or more splines. A suffix is accumulated directly, avoiding cancellation
from subtracting two nearly equal prefix sums in a diffuse tail. The
underlying cell integral, angular coefficient, kinetic-balance basis,
Breit/QED settings, and isotope mass are unchanged. On a physical
668-spline bank, a direct contraction fell from 10.83 to 0.42 ms;
symmetric exchange was previously 15.89 versus 11.19 ms. Tests compare
the compact route with the full cell bank for all four P/Q blocks under
kinetic balance, multipoles through k=6, and a converged Ne DHF solve.

## Ranked remaining work

### 1. Migrate remaining full-bank consumers and budget cache bytes

The DHF direct, symmetric exchange, mixed P/Q exchange, and scalar Slater
paths now use the directly assembled overlap strip at large grid sizes.
The off-diagonal contribution uses exact one-dimensional moments. For fixed
spline order, the strip grows linearly with the number of radial splines,
whereas the full bank grows quadratically. Smaller bases and other
bilinear Slater, MBPT, CI, and scattering consumers still request full
banks. Keep full-bank materialization as a compatibility path while moving
these callers to the compact representation with independent numerical
checks.

The cache still bounds the *number of grids* at four rather than their
bytes. Enforce a byte budget and report bytes by grid and multipole after
the remaining consumers migrate. Eviction must respect live solver
references, or it will cause repeated rebuilds rather than reduce peak
memory. Size multiworker jobs from measured per-worker peaks, especially
for the 668-spline Rydberg cases.

### 2. Treat exchange and the dense Dirac eigensolve together

After the direct change, these are the largest **warm** costs. The Fock
matrix is dense because nonlocal exchange couples distant radial basis
functions; a banded generalized eigensolver cannot simply replace the
current dense solver, even though the one-body Hamiltonian and overlap are
banded. LAPACK's [DSBGVD](https://www.netlib.org/lapack/lapack-3.1.1/html/dsbgvd.f.html)
requires *both* matrices to be banded. A matrix-free eigensolver would need
an exchange matvec, a preconditioner based on the banded one-body/direct
part, and reliable selection of the physical Dirac branch. It must preserve
the kinetic-balance/no-pair guard and same-κ orthogonality constraints;
the [B-spline Dirac Galerkin analysis](https://arxiv.org/abs/0806.3067)
explains why avoiding spurious Dirac states is a design constraint.

A lower-risk experiment is exact reuse of Fock fields for independent
angular-coefficient rows while the orbital density is frozen. The present
SCF updates orbitals sequentially, so sharing across targets changes the
iteration map unless introduced as a controlled block update. Compare
iteration count, occupied radial nodes, stationary energy, and transition
intervals—not only per-iteration speed. Profile per κ and grid size before
changing LAPACK drivers. An exploratory in-process OpenBLAS thread-count
switch crashed after the two-thread probe on this host; fixed one-thread
workers remain the reliable baseline until separate-process tests establish
an optimum for large matrices.

### 3. Control radial degrees of freedom by error, not a blanket box step

For diffuse levels the outer turning radius scales roughly as
\(2n^2/Z_{\mathrm{eff}}\), so a constant outer knot step makes basis size
grow with the entire box. Yet blindly coarsening that step already shifted
the Li I `27p` control by about 1.3 cm⁻¹. High-Z K shells have the opposite
need: the inner grid must resolve the nuclear/Coulomb scale; the default
grid's U hydrogen-like 1s error is documented in the repository at about
97.7 eV. A robust adaptive grid should insert knots *locally* from a
posteriori orbital residuals and target transition-energy errors, preserving
kinetic balance and radial-node labels. Nested knot insertion lets each
refinement inherit the previous solution and supports meaningful
variational convergence checks. Keep distinct inner-core, oscillatory
Rydberg, turning-point, and asymptotic-tail criteria. The validation target
is an interval or ionization threshold on two successive grids, not only
an absolute total energy.

### 4. Reuse invariant one-body and CI structure across solves

The DHF solve already caches one-body matrices per κ *within* a solve, and
Coulomb banks by exact grid content *across* solves. In MCDHF CI,
`build_ci_matrix` still clears `_SLATER_MEMO` on entry, including the
orbital-independent `("h0", ns, κ, Z)` blocks; `_check_orbital_metric`
also rebuilds a Gram audit for every CI construction. The existing
[HPC research note](HPC_NUMERICS_RESEARCH_2026-09-27.md) reports large
gains from retaining invariant h0 and reducing redundant metric audits on
an earlier worktree. Reproduce those A/B timings on the present branch
before editing the CI path. Any metric-check cadence change must retain a
final exact check and one after any orbital/grid change. The B-spline
continuum eigensystem is **already** cached by grid, potential and κ in
`continuum.py`; a prior review suggesting that cache as missing is stale.

### 5. Keep validation and performance gates representative

Use three workload tiers: compact closed shell (113 splines), open-shell
core/valence with a large radial box (roughly 200–350), and high-n or
heavy-ion box (500–700). Measure cold build, warm SCF, final energy,
peak RSS, and multiworker peak separately. Pin BLAS/Numba threads before
imports. Compare full and compact Coulomb blocks at the matrix level for
all four component types and several multipoles, then converged energies,
orbital labels/nodes, and representative NIST transition intervals. Keep
independent analytic Dirac and DBSR_HF anchors. Do not use a timing gain
that changes the physical branch or hides a failed convergence check.

## Verification completed in this review

- Compact direct contraction: four P/Q block types, kinetic balance,
  k=0, 1 and 6 controls; maximum relative differences were about 1e-15
  against the full-cell direct kernel in the exploratory matrices.
- Compact final Slater energy: k=0, 1 and 6 four-block comparisons.
- Interleaved 333-spline Ne three-sweep A/B: 2.23–2.25 s versus 1.36 s;
  energy difference 2.0e-12 Ha.
- Relevant SCF, convolution, cache, and NIST job suite: 121 passed with
  `PYTHONPATH` pinned to this checkout; Ruff and diff checks passed.
- Direct overlap-strip assembly matches every near-cell value from an
  independently built full bank for all four P/Q blocks and k=0, 1, 6.
  Mixed exchange matches the full-bank kernel on asymmetric densities;
  the 333-spline three-sweep Ne energy moved by 1.6e-12 Ha while retained
  full banks fell from three to zero.
