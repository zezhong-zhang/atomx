# DHF convergence and numerical performance review (27 September 2026)

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The NIST production rung remains B-spline nonlocal DHF with perturbative
Breit, QED, and the selected isotope's leading normal mass shift. Broad
workers were stopped while convergence and performance changes were checked.

## Physical and numerical choices

DBSR_HF's `hf_matrix.f90` constructs direct and nonlocal exchange from
angular coefficients and radial Coulomb integrals. Its `hf_solve_HF.f90`
updates occupied orbitals in sequence and constrains same-symmetry roots;
`hf_rotate.f90` uses a local pair-rotation step. Atomx retains the full
quartic pair-rotation search. GRASP's `rmcdhf90/scf.f90` uses orbital
self-consistency ordering and damping. FAC's `faclib/radial.c` optimises a
local model potential, so its radial speed is not a substitute for nonlocal
DHF exchange. The local copies reviewed are `~/code/dbsr_hf`, `~/code/grasp`,
and `~/code/fac`. The [DBSR_HF paper](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=919310)
is the external description of the B-spline reference method.

Atomx also has a nonlocal shooting DHF solver. It solves a Dirac boundary
value problem with a nonlocal exchange source via a Green function. The
same angular average enters the two engines, but shooting's Schmidt handling
of off-diagonal same-kappa open-shell constraints is approximate. A
converged shooting energy is therefore an independent diagnostic, not an
automatic replacement for a B-spline result on every open-shell spectrum.

## Implemented changes

1. **Pair rotations.** The Fock field is linear in the density. Rotating a
   pair makes that density a constant plus cosine and sine of twice the
   angle. Three Fock samples for each target reconstruct all four derivatives
   needed by the existing exact quartic search. The fields are projected
   onto the two-orbital subspace before interpolation. A perturbed Li I
   occupied pair returns to its stationary DHF energy using six Fock builds
   instead of eight.
2. **Exchange symmetry.** Equal-component Coulomb exchange matrices are
   symmetric. Their convolution computes the lower triangle and mirrors it;
   the mixed large/small block still uses its full matrix. On a real
   113-spline Coulomb bank the half contraction took 0.381 ms versus
   0.695 ms, with maximum matrix difference 1.1e-16 in the check.
3. **Large-basis bank memory.** At 256 splines or more, the cache retains
   the SCF's contiguous transposed Coulomb bank and provides a zero-copy
   view for final-energy Slater contractions. A single 668-spline pppp
   bank of order 8 is about 218 MiB, so this avoids another 218 MiB copy
   per multipole and integral type. A 180-spline forced-policy trial gave
   identical Slater values and a view contraction time of 8.25 ms versus
   9.28 ms from the raw bank (30-call means).
4. **Semiseparable large-grid exchange.** For a B-spline pair
   `B_i B_(i+p)`, define its `r^k` and `r^(-k-1)` moments. If two pairs
   have disjoint radial supports, their Coulomb integral is exactly the
   product of one rising and one falling moment; only the overlapping
   strip needs the original two-electron cell integral. The new symmetric
   exchange path stores that strip and the moments and streams the three
   radial ranges separately. It activates at 320 splines, where measured
   kernel timings become favorable. Subsequent work extended the same
   exact factorization to mixed P/Q exchange and assembled the strip
   directly from cell moments, so large-grid batch DHF no longer needs
   these full banks.
5. **Semiseparable direct and final-energy contractions.** Prefix and suffix
   sums over the same radial pair moments replace the distant-pair loops
   in the direct Fock term and all four P/Q blocks of the final Slater
   energy. Direct scalar contraction at 668 splines fell from 10.83 ms to
   0.42 ms. On a 333-spline physical Ne grid, interleaved warm three-sweep
   solves fell from about 2.24 s to 1.36 s; the total energy changed by
   2.0e-12 Ha. The later direct-cell and mixed-exchange change also reduced
   cold-start time and peak memory; see the B-spline performance review.
6. **Shooting seed for the validated Cu I `13p` retry.** A converged shooting spinor is sampled
   at the target B-spline quadrature nodes and projected by solving
   `S c = <B|psi>`, with the target overlap metric. The projection reports
   its weighted relative spinor L2 error. The retry path rejects an
   unconverged shooting run, incomplete grid coverage, projection error
   above 2%, a failed B-spline SCF, orbital energy disorder, or a guarded
   radial-node mismatch. Only the final B-spline DHF+B+QED+mass energy is
   written to the NIST ledger. The production selector permits this route
   only for Cu I `13p` until other configurations pass the same branch and
   grid checks. Ordinary pending jobs do not use the new seed path yet.

### Measured controls

All runs below fixed OPENBLAS, OMP, and MKL to one thread. Timings are
wall-clock observations on this host, not portable speed guarantees.

| Control | Before | After | Numerical check |
| --- | ---: | ---: | --- |
| Mo VI `9p`, bare DHF, same basis, warm median of two repeats | 5.84 s | 4.18 s | 7 sweeps both; total energies differ by <2e-12 Ha |
| Mo VI `10p`, five-sweep cProfile including cold QED table | 6.21 s | 5.04 s | exchange convolution 2.74→1.71 s; pair rotations 2.05→1.18 s |
| Cu I `13p`, isotope-aware DHF+B+QED+mass | TFDA continuation 274 s, 400-sweep allowance | shooting seed plus B-spline 11.8 s, 6 B-spline sweeps | 2.1e-7 Ha from long TFDA result; projection L2 1.62e-4; ordered orbitals |

On a physical 668-spline `pppp`, `k=0` bank with a symmetric rank-three
density, symmetric exchange took 15.89 ms with the full bank and 11.19 ms
with the semiseparable representation (11-call medians, one thread). The
factor construction took 8 ms; maximum matrix difference was 3.5e-18 in
that control. A 320-spline Ne DHF solve converged in 14 iterations to the
same energy with either path. Its warm full-solve times, 8.54 and 8.52 s,
show that eigensolves and other work dominate that particular system; the
kernel gain must not be represented as a whole-solver speedup.

Cu I `13p` on an independently refined outer grid (`hmax=6.0` versus
production `6.5`) gave `-1652.0141640029929` Ha, only 4.9e-9 Ha from
the shooting-seeded production-grid result. Both `13p` spinors had eleven
significant radial nodes, as required by `n-l-1`.

The first targeted ledger retry exposed a separate validation bug: the
orbital-order check used strict `zip` on a sequence and its one-element-
shorter tail. It raised after a successful seed solve and sent the job into
the expensive fallback chain. The check now uses `itertools.pairwise`.
The next targeted retry completed in 12.82 s with
`-1652.0141639980734` Ha and provenance `shooting_hf_galerkin_seed`.
The comparison website was rebuilt from that ledger. At this checkpoint
the full ledger has 14,382 done, 2 failed, and 2,203 pending configuration
jobs; it is not yet a complete periodic-table calculation.

## Would shooting replace B-splines?

Shooting was faster on the tested configurations, and comparison of
transition energies is more informative than absolute total energies:

| Control, no Breit/QED/mass | Shooting | B-spline | Difference |
| --- | ---: | ---: | ---: |
| Ne I total | 0.28 s | 1.02 s | 7.95e-6 Ha |
| Fe XVII total | 0.16 s | 0.38 s | 6.58e-5 Ha |
| Mo VI `9p` total | 0.72 s | 4.40 s | 3.45e-4 Ha |
| Mo VI `8p→9p` interval | — | — | 0.18 cm^-1 |
| Mo VI `9p→Mo VII` limit | — | — | 7.86 cm^-1 |
| O I `2p4→2p3 3s` excitation | 0.28 s for excited solve | 2.91 s | 36.6 cm^-1 |

The Mo VI shooting total is stable within 4.7e-7 Ha when its radial grid
is refined from 1000 to 2000 and 4000 points. The cross-engine offset
therefore should not be dismissed as merely a coarse shooting grid. The
O I open-shell excitation shows why one cannot switch the NIST final rung
on speed alone. The coherent policy is to use shooting for a fast independent
estimate and, on difficult retry jobs, a Galerkin initial guess; retain the
validated B-spline result and explicit method provenance in comparisons.

## Remaining hot spots and acceptance gates

The earlier Mo VI profile recorded 1.71 s in exchange convolution, 1.12 s
building the Uehling spline once per process, 0.59 s in final Slater-energy
contractions, and 0.46 s in closed-shell canonicalization. The new direct
path removes most of the final-energy contraction cost on large grids.
The current 333-spline Ne three-sweep profile places about 0.71 s in
Dirac eigensolves, 0.61 s in exchange convolution, and only 0.02 s in the
final energy; the cold full-bank build still costs about 2.4 s in that
profile. See the dedicated B-spline performance review for the remaining
memory and solver work. Wider shooting-seed use still needs sampled
transition and radial-node checks.

The focused exchange, cache, shooting, and NIST job tests passed (71 tests).
Ruff and `git diff --check` passed. A broader 93-test run had one existing
nitrogen open-shell CI snapshot failure: the expected and computed energies
also disagree on a clean `4aba25c` checkout, so that CI mismatch is not a
consequence of this numerical change.
After the semiseparable exchange addition, 44 SCF, convolution, cache, and
exchange tests passed, including a forced 256-spline Ne comparison against
the original full-bank route. Its 14-iteration converged energy was identical
to the shown precision. The physical-bank tests also covered `pppp` and
`qqqq` under kinetic balance and multipoles `k=0` and `k=2`.
