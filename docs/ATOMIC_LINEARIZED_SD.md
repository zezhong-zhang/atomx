# Native atomic linearized SD

The [native roadmap](atomic_native_roadmap_2026_10_11/README.md) distinguishes
native finite equations and partial J-block diagrams from the current large-core
pCI-derived shared-library consumer. Native production equations must be
implemented and independently validated; a backend integration is not that goal.

`atomx.methods.linearized_sd` provides an exact scalar SD resource planner
and a finite native closed-core linearized CC equation anchor.
`atomx.methods.core_sd` supplies direct reduced DC contractions through
source-bound original pCI core diagrams. Both are separate from
[nonlinear CCSD/FS-CCSDT](ATOMIC_FSCC.md), the finite determinant nonlinear
CCSD oracle, and NR-CISD configuration diagonalization.
`atomx.methods.valence_sd` now adds a separate fixed-energy CI+all-order
valence numerator solver and signed Sigma1 rows on the same native DC core.
`atomx.methods.monovalent_sd` supplies a separate bounded standalone SD
feedback/state anchor on that actual core. `atomx.methods.reduced_monovalent_sd`
now supplies an opt-in native
J-reduced equation/feedback consumer, validated against complete finite magnetic
spaces. Its production normalized-state/property and full-Yb gates remain open.
The fixed-energy valence interface retains its original contract.

## Equation and scope

The equation is `Q (H + [H,T1+T2]) |reference> = 0`, with linked energy
`E = H00 + sum_mu H0mu t_mu`. A complete declared scalar J=0 singles/doubles
space is retained. For noncanonical orbitals, `T1` acting on the singly
excited part of `H|reference>` must be subtracted. Solving only
`(H_QQ-H00)t = -H_Q0` would omit that term. The implementation contracts
signed excitation operations without allocating full fixed-N generators.

This is linearized SD, not the nonlinear exponential equations, a normalized
CI state, a variational upper bound, a complete many-electron eigenstate or
an accuracy guarantee. It omits nonlinear cluster products and connected
triples. This closed-core interface does not itself supply monovalent feedback,
CI+all-order screening, dressed observables or physical resonances. The separate
standalone and fixed-energy consumers are documented below. The pCI
CI+all-order valence equations have a different
energy-feedback contract from standalone monovalent SD.

## Finite standalone monovalent SD

`FiniteMonovalentSD(core_problem, core_result, valence_index, **caps)` consumes
the **actual native CoreSD owner and its converged real DC result**. It keeps
core T1/T2 fixed and retains every independent valence single and core–valence
double at the reference total M/parity, including doubles with a particle in
the reference valence orbital. Core amplitudes retain their closed-core
denominators; valence correlation energy never shifts those denominators.
Core rho expansion is recertified against the full core linked residual and
all three core energies. A normalized CI vector is not a cluster amplitude.

The connected source is the attached `H+[H,T_core]` action minus its core
spectator. The valence map uses **core normal ordering**. An ordinary
attached-reference `[H,S_v]` incorrectly subtracts `S_v1` times the bare
core-single source induced by valence occupation. The implementation restores
precisely this linear singles diagram; it adds no `T_core*S_v` cluster product.
Literal independent tensor contractions of Eqs. 8, 9 and 13 of the
[primary method paper](https://arxiv.org/html/0905.2578) anchor canonical
terms; separate full Fock-space actions anchor noncanonical Fock terms.

For source `b,c`, homogeneous map `K,e` and valence amplitudes `s`:

```text
b + K s = deltaE_v s
deltaE_v = c + e s
attachment energy = epsilon_v + deltaE_v
total energy = E_core_SD + attachment energy
```

The solver follows the reference-connected eigenvector `(1,s)` of the real
directed pencil `[[c,e],[b,K]]` along an explicit coupling continuation. It
rejects ambiguous/complex roots, lost reference projection, close roots and
small energy denominators; it applies no shifts or matrix averaging.
The separation guard detects failure of this bound-state scalar-reference
prescription; it is **not a continuum/resonance-pole detector**. The complete
signed direct action recertifies the final pencil root.

```python
from atomx.methods.monovalent_sd import FiniteMonovalentSD, one_body_matrix_element

v = core_problem.integrals.labels.index((2, -1, -1))
problem = FiniteMonovalentSD(core_problem, core_result, v, work_cap=200_000_000)
state = problem.solve()
print(state.attachment_energy_Ha, state.projected_residual_Ha)
# O_spinor is explicitly chosen in the same complete spinor order.
expectation = one_body_matrix_element(state, state, O_spinor)
```

Results own amplitudes and normalized signed determinant coefficients of
`(1+T_core+S_v)|Phi_v>`. The physical SD bilinear follows Eq. 14 and is distinct
from exponential CC left/response properties. The one-body consumer requires
actual certified endpoints and the same native integral/spinor phase owner,
accepts real/complex complete operators and preserves weak nonzero values.
It returns a component of the supplied operator; it does not infer photon
frequency, reduced J elements, units or oscillator strength from arbitrary O.

Projected/energy residuals, physical state norm, Hamiltonian expectation and
**full eigenstate residual** are separate outputs. Linked SD energy is not a
variational expectation. A normalized SD state need not satisfy the complete
Schrodinger equation because cluster products/triples and higher diagrams are
omitted. Do not add its expectation–linked-energy difference as a correction.

The entire core/attached fixed-N intermediate bounds are counted before
integrals, cluster decoding or equation work. Defaults: 64 spinors, 256
amplitudes, 2000 determinants per sector, 64 MiB additional payload and 50
million conservative elementary actions. The twelve-spinor example requests
200 million actions explicitly; this changes allowed work, not physical space
or residual gates. Memory estimates include term/dictionary/eigensolver
allowances, excluding already-owned core/integral caches and interpreter RSS.
Oversized requests reject whole. The public method builds no full determinant
H or ordered quartic amplitude tensor. Its finite dense amplitude pencil is a
correctness anchor, **not a production J-reduced CPU/GPU implementation**.

Local evidence `atomic_pci_sd_native_2026_10_09/monovalent_sd006/` retains 195
complete arrays: all ten magnetic states in a common native Be+ twelve-spinor
pool and eight actual line-energy E1 controls. Primary equation/action/root
differences are at most `2.81e-15 / 2.50e-16 / 9.17e-17 Ha`; independent amplitude
difference is `1.37e-11`, within the original `1e-10` gate. Multiplet energies
agree at represented precision; J-squared state defects are at most `4.09e-11`,
within the declared `1e-9` gate. Normalized line bilinears agree with elementary
fermion action to `1.78e-15 a.u.`. The **full** eigenstate residual reaches
`2.552e-3 Ha`; length/velocity reduced E1 amplitudes differ by about 2.5%.
Neither is a physical precision certification.

Failed commutator convention (001), singles-only diagnosis (002), and an
overestimated minimum-feedback-size fixture (003) are preserved. The latter
was replaced by the original raw/amplitude gates in 004 without changing
physical comparison tolerances. Ten public native checks pass. The affected
15-file regression passed 159 checks initially; three external FSCC launches
failed because their child processes lacked the pinned libgfortran runtime.
With that runtime supplied, all three plus the ten new checks passed without
skips. This validates 162 distinct affected checks across the two runs, not
a single uninterrupted passing run. The final formatted-source replay (006)
is bitwise equal to the pre-format scientific arrays (005). Large-Yb standalone SD, nonlinear/triples,
Breit/QED/recoil, analytic SD response, continuum poles and basis/experimental
convergence remain outside this increment. Multi-valence consumption is separate.

Maintenance: this section owns `methods/monovalent_sd.py`, `numerics/fermion.py`
and `test_monovalent_sd.py`; failed and source-bound replay cases are local
owner evidence. These new lower-level files are in the working tree; a clean
published checkout/release has not been certified.

## J-reduced standalone monovalent SD

`ReducedMonovalentSD(core_problem, core_result, valence_keys=...)` is a separate
opt-in real native DC interface. Build its backend with
`tools/build_pci_monovalent_backend.py` / `build_monovalent(...)`, using the same
complete core dimensions and the `nxv/nvh` from `plan_reduced_monovalent_sd`.
The external pinned pCI source/license remains in the explicit reference/build
directory. This extension retains the existing fixed-energy `ValenceSD` ABI and
equations; a source-bound standalone extension is required, not inferred from a
fixed-energy library. Core results must pass their actual raw/source/energy gates.

The coordinate vector contains physical valence singles for every declared
virtual radial orbital of the reference kappa except the reference itself,
followed by **all original redundant direct rho_k core–valence doubles** in
channel/Fortran radial order. Doubles with a particle in the reference valence
orbital remain present. No amplitude, radial-window or rank pruning is performed.
The redundancy is inherited from the pinned diagrams; it is not a determinant
state metric, nor is the Euclidean norm of this vector a wavefunction norm.

At every native action the original diagrams run at bare `ev`. The wrapper
passes `D0_double*rho_double` to the original numerator ABI, which divides by
unshifted D0 and recovers physical amplitudes. Core numerator conversions cancel
against their fixed closed-core denominators. The wrapper restores the original
`term31` with **physical** singles and adds scalar offdiagonal Fock contractions.
Correlation energy enters only the separate consuming equations:

```text
D0_single = eps_v - eps_m
D0_double = eps_v + eps_b - eps_m - eps_n
(D0 + deltaE_v) rho = R(rho)
deltaE_v = R_reference(rho)
attachment energy = eps_v + deltaE_v
total energy = E_core_SD + attachment energy
```

For noncanonical occupied/virtual Fock blocks, scalar S1 spans every M of
the valence radial multiplet. Its other-M action is needed on the **bare core
Fock-single source**. Acting instead on the full attached H source introduces
Coulomb contractions excluded by core normal ordering. The direct rank-zero
coefficient `rho(v,m,v,b) = -[j_v]*S_mv*F_vb` has valence in its first particle;
particle antisymmetrization supplies the reversed order. The finite anchor now
includes the same missing bare-Fock contribution, independently checked against
literal primary Coulomb equations and full core-sector Fock matrices. Previous
`monovalent_sd006` data remain historical; the corrected source is revalidated
in `monovalent_sd008`, rather than relabeled as a formatting-only change.

The coupled solve uses source-separated native actions, matrix-free GMRES
inverses, analytic scalar feedback derivatives and eight explicit coupling
stages. The core COMMON image is restored and all actions run under its owner
lock. Small bare/shifted denominators, failed inverse/raw/energy gates and
singular feedback slopes raise; no level shift, Hermitian averaging, coordinate
truncation or backend fallback is applied. This numerical continuation is not
an intruder-free, unique-root, forward-error or continuum-resonance certificate.

```python
from atomx.methods.reduced_monovalent_sd import (
    ReducedMonovalentSD, plan_reduced_monovalent_sd,
)

# Plan the complete extension before compiling/loading the native image.
plan = plan_reduced_monovalent_sd(core_plan, requested_valence_keys,
    valence_term2_blas=True, valence_term2_threads=8)
# Include solve's Arnoldi/Hessenberg scratch in the caller's resource gate.
sd = ReducedMonovalentSD(core_problem, core_result,
    valence_keys=requested_valence_keys)
result = sd.solve((6, -1), residual_tolerance_Ha=1e-10)
energy_map, full_rhs_Ha = sd.evaluate((6, -1), result.amplitudes)
```

`plan` includes the complete retained core/valence bound, two additional full
native double planes and four single vectors, packed int32 coordinates and
sixteen scalar scratch vectors. `solve` adds the explicit Krylov-vector budget
and Hessenberg scratch before initialization; constructor preflight precedes
an uncached parent's re-certification action. This is a payload bound excluding
interpreter/JIT/runtime RSS. Denominator assembly avoids millions of boxed
Python floats. Existing opt-in streamed BLAS double kernels are reused with
their source/runtime/inner-one worker checks. CPU execution is implemented;
this interface has no GPU or end-to-end performance certification.

The current affected panel validates 150 distinct checks across a 147-pass,
three-missing-serial-runtime-skip run and a three-pass runtime revalidation.
Its 84 new reduced cases cover four matched complete twelve-spinor native
fixtures (original, streamed BLAS, .37rad virtual rotation, .08rad occupied–virtual
rotation), four valence columns and every magnetic state; the changed core
projector is not called a fixed-projector invariance experiment. Current
fixed-energy columns remain bitwise identical after standalone interleaving.
The source-bound `reduced_monovalent008` replay retains independent dense
core-sector diagrams, complete exact-J angular projectors and all raw defects.
The stronger unrestricted-M normalized-state comparisons failed in 006/007
and remain recorded separately; exact total-J validation does not erase them.

Complete original 68-core/414-radial-virtual Yb planning retains roughly
2.02–3.77 million amplitudes per requested column. With forty Krylov vectors
and restart twenty the maximum payload bound is 9,750,842,316 bytes, within
the earlier explicit 32GiB host cap. This is arithmetic preflight only, not a
new large-space action, root, spectrum or timing result. Production normalized
SD states/properties, full-Yb roots, Breit/QED/recoil, nonlinear triples,
experimental convergence and actual continuum poles remain separate gates.

The [complete-space diagnostic driver](atomic_pci_sd_native_2026_10_09/probe_yb_full_monovalent.py)
now exercises the original Yb input without shrinking its internal space.
Whole-run planning also includes historical-guess re-certification of the
28,218,500-coordinate closed core: its maximum bound is 12,844,848,516 bytes,
distinct from the standalone-column bound above. The first actual attempt
[002](atomic_pci_sd_native_2026_10_09/yb_full_monovalent002/failure.json) rejected
a changed shared angular source before solving. An explicit audited-source
transition and complete stored signed core-action comparison precede the
[003 run](atomic_pci_sd_native_2026_10_09/yb_full_monovalent003/contract.json).
Its actual current core raw defect is 4.438e-13 Ha; the complete stored signed
core action differs by 2.776e-17 Ha, and source/energy/denominators match.
The first complete 6s signed-map gate passes at 3.711e-17 Ha. Its
[explicit stop](atomic_pci_sd_native_2026_10_09/yb_full_monovalent003/stopped.json)
preserves eleven complete actions, without a solved feedback root.
Historical amplitudes are iterative guesses; only an actual current public
core residual/energy check supplies the new transient certificate.

Two private owner controls retain the same equations: terminal term31 radial
FP64 GEMM, with a callable original signed component; and directed augmented
Davidson using complete bare-ev action images. Cached images are independent
of coupling and feedback energy; each of eight coupling stages repeats the
actual complete public raw/energy gate, followed by a final affine check.
The source-bound [finite control](atomic_pci_sd_native_2026_10_09/monovalent_davidson002/result.json)
passes 84 independent checks and retains 144 arrays across four columns,
all ten M and four complete orbital variants. Paired original/private errors
reach 6.662e-16 Ha. At the same 1e-14 Ha goal, native action counts are
262–698 versus 22–25; these are counts, not end-to-end timing speedups.
The [004 full-Yb run](atomic_pci_sd_native_2026_10_09/yb_full_monovalent004/contract.json)
passes all five original/GEMM component and signed-map gates, with maximum
affine/component defects 6.182e-17/3.897e-20 Ha and exact double-only regression.
Its [explicit stop](atomic_pci_sd_native_2026_10_09/yb_full_monovalent004/stopped.json)
retains complete current-source map arrays and avoids duplicate CPU/GPU root
contention; no CPU feedback root is claimed. Its additional private scratch is
6,506,240 bytes; whole-run bound is 12,851,354,756 bytes under the unchanged
32GiB cap. These controls do not change public solvers, defaults or GPU policy.

The private standalone CPU/CUDA control subsequently passes 164 independent
finite checks in each of two source-bound executions, including exact term41
and terma4 GEMMs in the second. Small complete GPU roots are slower in single
timing observations; no automatic selection is adopted. An explicit
[full-GPU execution wrapper](atomic_pci_sd_native_2026_10_09/probe_yb_monovalent_cuda.py)
reuses the complete input/core driver and counts the combined host bound of
12,929,879,172 bytes under the original 32GiB host and 2GiB device caps.
All five complete original CPU map arrays must pass before any GPU root;
every resulting root repeats an actual complete CPU raw/energy check.
Full005 passes five paired signed-map columns/50 arrays and two feedback roots
with fresh CPU raw checks, then explicitly stops to avoid duplicate numerical16
contention; no five-root result or comparison is claimed. A separate source-bound
numerical64 wrapper starts Full006 after the finite164 gate and an independent
complete42-matrix stress test that actually uses wider dimensions. Additional
complete vectors are guarded before native load, with conservative whole-host
bound15,824,548,740 bytes under32GiB; all original physical spaces/gates remain.
Full006 now completes all five feedback roots, all50 paired CPU/CUDA arrays and
fresh actual full CPU raw/energy checks. Maximum raw/energy defects are
2.320e-11/2.838e-15 Ha; native action counts56/78/93/71/70 retain all physical
coordinates. The frozen nominal excitation comparison ranges from+0.925% to
+4.141%; only one of four excitations meets the unchanged1% gate. Both fine
splittings fail, with the5d discrepancy−9.362%. These are distinct numerical
and experimental outcomes, detailed in the [owner](atomic_pci_sd_native_2026_10_09/README.md).

A private radial physical-norm control retains core spectators, Pauli blocking
and signed core-S1/valence-S2 interference in `(1+T_core+S_v)|Phi_v>`.
Independent complete fermion expansions pass840 finite component/M gates after
preserving an equal-radial-hole antisymmetry failure. Small angular Grams and
zero-copy radial views avoid magnetic quartic tensors; six boundary/allocation
checks pass. The complete Yb consumer requires fresh current core amplitudes
measured bitwise equal to the seed and all18 magnetic projections. Its complete
execution passes all18 projections, with zero measured within-multiplet norm
spread and381 source/33 payload hashes unchanged; see the owner for terms,
norm values and actual timing scope. This private scalar norm does not supply
production state assembly, dressed properties, full-H stationarity or physical
correlation convergence; public defaults and capabilities remain unchanged.

Maintenance: this section owns `reduced_monovalent_sd.py`, `_pci_monovalent.f90`,
the standalone builder and `test_reduced_monovalent_sd.py`. Current-source
snapshots and failures 001–008 belong to the [active owner](atomic_pci_sd_native_2026_10_09/README.md).

## Native entry point

```python
from atomx.methods.linearized_sd import compute

sd = compute(
    "1s2", ("2s", "2p"), native_orbitals, native_workspace, Z=4, ns=30,
    reference_identity="owned native Be2+ snapshot",
    nuclear_potential=native_nuclear_operator,
    include_breit=False,
)
print(sd.total_energy_Ha, sd.equation_residual_Ha)
```

All reference NR shells must be closed. Every target NR shell must be
disjoint from the reference and supply all its spin-orbit radial components
in `native_orbitals`; every component retains its magnetic multiplet. The
native CI metric and nuclear/signed radial binding are checked and recorded.
`frozen_orbitals=("1s",)` excludes that NR shell from excitation amplitudes
while retaining it in H and the reference energy. `include_breit=True`
uses the full existing Gaunt plus static-retardation CI operator throughout
the equations, including their noncanonical singles subtraction. No QED,
recoil, dynamic Breit or orbital reoptimization is silently included.

The dense reduced-CSF solve is bounded (defaults: 512 CSFs, 2000 magnetic
determinants, 128 MiB for estimated equation-matrix work). Missing orbitals,
invalid/oversized spaces and singular equations reject the entire request.
Matrix working estimates exclude Python/vector, native radial and compiler-
cache overhead, and are not total-RSS bounds. The planner checks scalar
dimensions and matrix memory before CSF/vector/H materialization.

`solve` is the lower-level signed projection consumer. It checks a supplied
real symmetric H, orthonormal determinant expansions of pure SD ranks and
the unique reference. Its producer must establish a genuine one-/two-body H
and physical completeness of the supplied projection. Native `compute`
independently checks completeness by exact scalar counting. Results own
amplitudes and equation matrices; provenance includes basis and H hashes,
reference mask, native binding and Hamiltonian coverage.

## Exact large-pool preflight

```python
from atomx.methods.linearized_sd import plan_scalar_sd

plan = plan_scalar_sd(correlated_closed_core_keys, complete_virtual_keys)
print(plan.resources())
```

Counts use antisymmetric normalized radial pairs with the same total J and
parity on occupied and virtual sides. Identical radial pairs obey Pauli
selection. Singles require equal kappa. Grouping keys by kappa avoids
enumerating each radial pair during planning; only exact symmetry zeros are
removed. Frozen keys must be excluded explicitly from the occupied argument.

These are independent scalar excitation coordinates, not redundant
particle-hole `rho_k` storage in pCI. The count alone does not provide tensor
equations or a solve. A real amplitude-vector size excludes all
iteration/intermediate/integral storage. The native full Yb l<=3,<5 Ha band
has 13,165,103 independent scalar amplitudes: one real vector is about
100.44 MiB, but a dense equation matrix is still prohibitive. Direct reduced
linked contractions are required for that workload.

## Direct reduced DC core equations

`methods.core_sd.CoreSD` uses the original pinned pCI core diagrams with an
exact sparse factorization of the native B-spline cell Coulomb quadrature.
P/Q have separate auxiliary-node tags because kinetic-balance component
orders and same-cell triangular quadratures differ. Its weighted density
action must not be integrated a second time with Gauss weights. Full scalar
off-diagonal Fock terms, occupied/virtual couplings and singles energy are
included; no orbital canonicalization or Hamiltonian replacement is hidden.

The auxiliary action follows the shared rank-aware triangular quadrature:
nonnegative rank `k` uses the component order plus `floor(k/2)` inner nodes.
Fixed polynomial density coordinates exactly resample the additional native
subnodes; outer quadrature and negative-power moments are unchanged. Rank
workspace is admitted before powers or temporary arrays are evaluated.
The auxiliary node/edge counts have changed: plan and rebuild case-specific
backends from the current feature identity. Old source-bound measurements
remain historical. The [cleanup qualification](repository_cleanup_2026_10_16/README.md)
records signed native-cell controls and a fresh complete finite pCI/DC solve;
it makes no full-Yb, convergence or performance claim.

First construct `CellCoulombFeatures(workspace)` and call
`plan_core_sd(occupied_keys, virtual_keys, features.identity)`. Compile the
exact planned dimensions with `tools/build_pci_sd_backend.py`, specifying
the pinned source, fresh build directory, compiler and optional runtime
directory. The build retains GPL-3.0 source/license, unchanged diagram
routines, the native radial adapter and artifact hashes. Then prepare:

```python
from atomx.methods.core_sd import CoreSD

problem = CoreSD(
    native_orbitals, native_workspace, Z, ns,
    occupied_keys=closed_core_keys, virtual_keys=complete_virtual_keys,
    nuclear_potential=native_nuclear_operator,
    backend_directory=isolated_case_build,
    memory_bytes=2 * 1024**3,
)
result = problem.solve(residual_tolerance_Ha=1e-10, restart=30, maxiter=100)
```

`solve(initial_amplitudes=complete_signed_vector)` can warm-start a fresh
problem. It records the full initial vector hash and recomputes the current
operator, source, energy and final raw residual. The seed is not a transferred
result or an inherited source identity. Compare complete physical inputs
explicitly when replaying a historical calculation; malformed, complex or
nonfinite seeds reject before the native action.

`methods.atomic_space.build_atomic_space` builds every retained VN bound
and pseudo-continuum root in a declared partial-wave/energy band. FSCC and
direct SD use this shared policy. `tools/run_native_core_sd.py` supplies a
complete-band workload with native inputs, plan, residuals and stage times.

The direct representation stores original redundant particle-hole `rho_k`,
so its vector count differs from the independent SU(2) count. Physical core
doubles obey simultaneous particle/hole exchange; same-hole, same-angular
radial blocks must be symmetric under particle exchange. The low-level
`action` is the raw redundant numerical map, not an independent physical
commutator for every unconstrained coordinate vector. Caller-provided physical
amplitudes and seeds must preserve the representation's exchange constraints;
shape/finite checks do not certify them. The [current four-core domain audit](atomic_physics_correctness_2026_10_11/README.md)
checks every projected coordinate, the complete physical image and allowed
null directions against an independent fixed-N commutator. It retains the
unconstrained-random negative rather than changing the Hamiltonian or gate.
This finite proof does not certify every production shell topology. The new
[occupied-p/mixed-j control](atomic_physics_correctness_2026_10_11/sd_pcore/README.md)
keeps all14 spinors/1001 fixed-N determinants for the declared Ne DC model.
Its real zero/source and first-Born inputs pass all310 signed SD components,
and its independent/native MBPT2 coefficient agrees within3.723e-24Ha.
Nonzero p3/2-p3/2 and s-p3/2 hole branches are measured above roundoff.
The separated interaction-only action has signal1.106e-5Ha and error1.374e-18Ha.
This is a weak-coupling model with diffuse VN virtuals, without a complete
native mixed-j rho-domain derivation, arbitrary-input/all-direction action,
converged all-order root or strong-correlation/Ne spectroscopy certificate.
Shared radial/angular integrals also leave radial-operator validation distinct.

The [matched-H method ladder](atomic_physics_correctness_2026_10_11/method_rank/README.md)
solves actual public dense finite `linearized_sd.solve`, nonlinear CCSD and
CCSDT against complete FCI at unchanged Z4 DC plus finite-frequency transverse H.
The core method errors are -0.0131872, +2.27875e-5 and +3.17000e-6Ha while
all retained equation gates pass. Its LCCSD core anchor is not native HPC
CoreSD, and no matched-H LCCSD attachment was executed. That transverse-H
model differs from the Z6 DC native-domain control; their energies/amplitudes
must not be spliced or their conditional method responses transferred to Yb.
Nonlinear/rank coverage and complete physical operators remain separate from
orbital cutoff convergence. Matrix-free
GMRES retains every active virtual slot, with a diagonal preconditioner and
an explicit complete storage cap; it does not allocate a dense equation
matrix or magnetic quartic ERI array. Original pCI arrays are still padded
and some intermediates are large. Preflight includes scratch and requested
Krylov vectors; estimates exclude interpreter/JIT/runtime RSS. Each live
problem requires a separate library path because pCI uses mutable COMMON.

`radial_pair_cache_size=16384` is explicit at both planning and preparation;
its array upper bound is included. Reusing the smaller generic integral cache
caused repeated full-Yb radial work; a cache change does not change the space.
The default `term21_threads=0` retains the original diagram. Optional
`term21_threads=N, term21_mode="compact"` combines disjoint P/Q node tags and
distributes complete core-pair outputs. `term21_mode="streamed"` generates
each core-independent virtual-pair potential once per partial/rank block and
uses LP64 BLAS for full projections. The streamed build requires an explicit
`blas_library`; `blas_library` also selects the exact X1/X2 initializer.
`core_singles_blas=True` selects an exact BLAS contraction of the complete
direct-plus-exchange `terma4` singles diagram. It requires positive workers
and explicit BLAS, and its shared/per-worker arrays are included in preflight.
`core_single_to_doubles_blas=True` separately selects exact native weighted
BLAS contractions for both complete core singles-to-doubles `term31/32`.
It shares the full trial density and core/virtual potentials and writes
disjoint output channels. Positive workers, explicit BLAS and the full
shared/per-worker scratch bound are required. All three builders expose
`--core-single-to-doubles-blas`; pass the matching option to planning and
`CoreSD` too. Both optional BLAS switches default to false.
For more than one streamed or either BLAS worker, the build requires an OpenMP
OpenBLAS runtime. On load, its actual configuration must match the bound
runtime and its inner thread count must be one (`OMP_NUM_THREADS=1`). A
serial BLAS with one internal thread is not by itself safe for concurrent
callers; the full-Yb run exposed a memory-pool warning in that combination.
The original and compact `term21` paths do not call BLAS inside their workers.
The optional singles and singles-to-doubles paths do, including with compact `term21`.
Pass identical thread/mode options to the planner, builder and `CoreSD`.
Workers are bounded by the number of complete core pairs; the actual selected
count, mode, scratch and generated source hashes are carried in provenance.
The cap includes all retained original arrays plus shared and per-worker heap
scratch. Compile only the heap-backed worker with OpenMP, preserving the
original static scratch and a reentrant native density action. No rank,
virtual-orbital or amplitude cutoff is introduced by either path.

Signed full-action/root checks pass for the recorded d/f, kinetic-balance and
complete Ne g workloads. The compact four-thread full action measures about
0.65 s versus 2.1 s. Earlier streamed four-thread timings used an unsupported
concurrent serial-BLAS configuration and remain withdrawn. The bound OpenMP
replay validates full Ne actions at 3.34e-16 Ha and measures paired four-worker
actions of 0.342–0.348 s versus 2.30–2.33 s; requested 16 workers are clamped
to the 10 complete Ne core pairs. Historical arrays/logs remain unchanged.
The later singles-BLAS correctness replay also passes, but its load guard
fails, so that batch supplies no controlled timing claim.
`problem.set_diagram_progress(True)` enables per-diagram timings on stderr;
it is disabled for controlled timing. No full-Yb or paired end-to-end gain
is implied by a kernel/stage measurement.

An optional `PolynomialCellCoulombFeatures(workspace)` retains the original
quadrature through the exact coordinate transform `A.T @ K_native @ A`.
Native spline pair densities have degree at most `2*(max_order-1)` in each
cell. Values at `2*max_order-1` tagged interpolation nodes therefore determine
them without dropping density coordinates. The new nodes do not replace
integration nodes or weights. Plan from this feature identity and pass
`radial_representation="polynomial"` to `CoreSD`; native remains the default.
The shared engine checks memory before interpolation and records both native
and transformed geometry identities. Original convolution, isolated P/Q terms,
complex c-products, an origin-cell integral and the finite noncanonical SD
anchor pass. The complete Ne g band also passes signed actions and solved
amplitudes against the original native representation (maximum action difference
3.34e-16 Ha, raw residual 8.44e-13 Ha). A separate Yb control retains all 68
core electrons and two virtual roots per partial wave: the original pCI action
at the new solution has residual 1.31e-12 Ha. This finite virtual control does
does not establish correlation convergence for the production 414-virtual band.
The later complete all-68-core/414-virtual run008 passed the full original
28,218,500-coordinate action gate (1.39e-16 Ha), and solved the retained raw
equations in 36 actions with maximum residual 4.44e-13 Ha and correlation
energy -0.6363421005996561 Ha. Its source is archived separately; it predates
the optional term31/32 kernel and warm-start API. This is a numerical equation
result, without a Yb optical spectrum or radial/partial-wave convergence claim.
These later timing batches fail the load guard or lack paired timing, so they
support correctness only. See the continuation for complete vectors and logs.

The term31/32 prototype full-band control retains the same 414 virtual roots:
the complete initial action differs by 1.39e-16 Ha, and independently stimulated
signed singles give complete term31/32 differences 8.95e-19/1.52e-18 Ha.
Public integration passed 32 affected live tests without skips at the archived
pre-warm-start source; the subsequent warm-start core panel passed 26 tests
without skips. The public full-band kernel replay003 also passed all original
signed gates. The complete Yb valence consumer001 passed source, denominator,
energy and raw core replay gates, then failed the native valence channel-order
check. Complete owned dimensions repair that layout; the next full entry exposed
an OpenMP-induced large-array stack fault. Separate serial compilation repairs
that fault, and the identical full-space source map returns with all channels
and finite outputs. Complete valence roots/screening remain under verification. Single observed kernel times
are not a controlled performance or end-to-end speedup claim.

Native problem provenance captures the Python source identities at import,
including shared angular and radial modules. It must identify the code actually
running even if files are subsequently edited. Nonfinite source, native action
or correlation energy is rejected; successful solver status does not override
these checks or the raw equation residual gate.

The valence backend binds complete NXV/NVH in its own include, independent of
the precompiled optimized core's placeholders. Serial valence/screening units
are compiled separately without OpenMP so their large original local arrays
remain static; calls share the owning core lock. Only explicitly heap-backed
core worker units retain OpenMP. The full-Yb channel-layout failure and
OpenMP-induced stack fault, their sources and full-entry diagnosis are retained
in the [native continuation](atomic_pci_sd_native_2026_10_09/README.md).

Coverage here is **real DC closed-core linearized SD**. Frozen-core,
Breit/QED/recoil, monovalent Bloch equations, CI+all-order screening,
nonlinear products, triples, dressed observables and resonance certification
are not provided by this path. The finite anchor retains its separate
DC/DCB/frozen-shell capabilities. Numerical residuals, correlation/radial
convergence and experiment remain distinct gates.

The [direct-equation continuation](atomic_pci_sd_native_2026_10_09/README.md)
records independent signed commutators, complete occupied/virtual rotations,
solved physical amplitudes and full-band Ne workloads. It supersedes the
earlier statement that only a pCI angular function had been compared; the
sealed earlier evidence itself is unchanged. Full Yb work is tracked in
that continuation, and no new Yb optical precision is asserted here.

## Validation and reference boundary

The [implementation evidence](atomic_reduced_sd_2026_10_09/README.md) includes
independent full-Fock commutators with signed rotations and noncanonical
singles, separated-fragment extensivity, weak-coupling limits, exact scalar
counts against full jj-CSF enumeration, native DC/DCB/frozen-core probes,
and an executed pCI angular-function comparison. That angular comparison
does not certify equality with pCI radial integrals or complete SD equations.
No new Yb optical accuracy result follows from this closed-core increment.


## Fixed-energy native CI+all-order valence equations

`methods.valence_sd.plan_valence_sd(core_plan, valence_keys)` keeps every
declared core virtual and all angular channels for each requested column.
`tools/build_pci_valence_backend.py` compiles the pinned original pCI core and
valence diagrams plus the native full scalar Fock commutator. Use a fresh
source-bound build with the planned NXV/NVH dimensions. Its eight explicit
COMMON-layout reconciliations use the owning core radial dimension, including
when NX>NXX. Original sources, GPL-3.0 license and artifact hashes are retained.
Core planning, builder and preparation must use identical performance options.

An optional exact term2 specialization reuses complete virtual-pair Coulomb
potentials and batches projections before signed angular recoupling. Set
`valence_term2_blas=True, valence_term2_threads=8` in both
`plan_valence_sd`/`plan_sigma2_sd` and the combined backend builder (CLI
`--valence-term2-blas --valence-term2-threads 8`). The default keeps original
term2 and one worker. Its heap workers have disjoint core-channel outputs;
the shared complete potential is never screened. Legacy large-array wrappers
and diagrams still compile serially. The source-bound manifest determines
the consuming `ValenceSD` plan. Complete per-worker scratch is included;
parallel use requires bound OpenMP BLAS with one inner thread. Core BLAS
binding must also be enabled in the build/preparation. Full signed term2 and
analytic-response arrays match the original in the complete 68-core/414-virtual
Yb space; [18 public checks](atomic_pci_sd_native_2026_10_09/valence_term2_public_parallel007/pytest.log)
and [independent 16-magnetic-state equations](atomic_pci_sd_native_2026_10_09/public_valence008/result.json)
pass without skips or looser gates. This does not establish an end-to-end
speedup, complete valence solve or experimental precision; the
[current owner](atomic_pci_sd_native_2026_10_09/README.md#价响应源分离与完整term2继续验证)
retains the full inputs, original map, failures and performance contract.

The homogeneous Krylov action returns values only. The optional streamed
term2 uses one shared kernel with an explicit response policy and skips its
unconsumed derivative arithmetic for those calls. Full affine root checks,
the implicit-response source, public `evaluate`, and joint Sigma2 response
retain the complete analytic derivative. Both modes retain the full scratch
bound; requesting values does not supply a derivative or change the space.
A fresh combined build is required for the new value entry point and source
manifest. [Public adoption controls](atomic_pci_sd_native_2026_10_09/valence_response_policy_public016/contract.json)
pass22 affected checks without skips, the complete original full-band value/response
and mode restoration, six independent radial columns/16 magnetic states, and
three alternating warm value-request pairs (median1.36). The timing scope excludes
full solves; current production and experimental status belongs to the owner.

```python
from atomx.methods.valence_sd import ValenceSD, plan_valence_sd

valence_plan = plan_valence_sd(problem.plan, requested_valence_keys)
# Prepare problem with the fresh combined backend, then solve its core.
core_result = problem.solve(residual_tolerance_Ha=1e-10)
valence = ValenceSD(problem, core_result, valence_keys=requested_valence_keys)
column = valence.solve(
    requested_valence_keys[0], energy_Ha=column_energy_parameter,
    solve_response=True, residual_tolerance_Ha=1e-10,
)
print(column.residual_max_Ha, column.sigma1_Ha, column.sigma1_derivative)
```

The column energy is an explicit CI+all-order parameter in the same native
core-Fock energy zero, not a solved physical excitation energy. Core numerators
remain fixed when differentiating it. Full implicit valence numerator response
and explicit denominator derivatives are included. Bare F_mv belongs to H1;
Sigma1 contains correlation only. This path follows the original fixed-energy
CI+all-order loop: it excludes term31, valence-singles feedback, standalone
monovalent Bloch energy feedback, nonlinear products and triples. It does not
construct a Hermitian multi-valence CI effective Hamiltonian, a dressed
transition operator, Breit/QED/recoil corrections or complex resonances.

The public matrix-free solver separates the exact homogeneous action from
its source, refines with the actual operator and checks complete raw root and
response residuals. Its complete root map supplies the explicit derivative;
the exact homogeneous action on the implicit numerator response completes
Sigma1's derivative without a third evaluation of all fixed-core source diagrams.
Original full-map and re-solved finite-difference controls remain independent
checks. It restores core COMMON after interleaved public core
actions, binds the actual solved core amplitudes and physical/source identities,
and rejects nonfinite inputs, wrong sources and over-budget requests. Packed
int32 indices and reusable per-column buffers avoid millions of Python tuples
and repeated full padded allocations; all radial columns remain complete.
The cap includes full core/valence storage and requested Krylov vectors, not
interpreter/JIT/runtime RSS. Results own immutable output vectors and explicitly
leave space convergence, forward accuracy certification and experiment false.

`solve(..., initial_numerators=..., initial_numerator_derivative=...)` accepts
complete finite real initial guesses, counts their owned buffers in the cap and
records their hashes. A response guess requires `solve_response=True`. Both
guesses enter Krylov only; the current complete source/affine/root/response
operators still certify the returned result. Saved-state consumers separately
bind physical input, energy, channel order and denominator identity before
reusing an archive. The [checkpoint panel](atomic_pci_sd_native_2026_10_09/valence_checkpoint_public017/pytest.log)
passes26 checks without skips, including perturbed complete guesses and four
original/optimized backend modes.

`solve(..., iteration_callback=consumer)` optionally emits a read-only complete
numerical guess after each GMRES restart, including root/response stage,
refinement, ordinal and current core/amplitude/source/packed-coordinate/
denominator hashes. The callback adds no operator action, preserves restart
budget semantics, propagates consumer exceptions and reserves two extra full
vectors in the payload cap. Caller-retained snapshots need their own storage
budget. Every snapshot is explicitly uncertified; recovery still executes the
current complete raw/affine/response gates. The [restart callback panel](atomic_pci_sd_native_2026_10_09/valence_iteration_public018/pytest.log)
passes26 affected checks in all four backends. A [private persistence panel](atomic_pci_sd_native_2026_10_09/valence_restart_storage019/pytest.log)
passes five tests: real interruption/replay in four backends and atomic integrity/
resource rejection. The full-workload driver commits immutable NPZ generations
with atomic stage heads and exact actual-source/physical/coordinate binding;
it does not modify the older loaded CPU004 process.

Binding `ValenceSD` can reuse the full raw residual just measured by the same
`CoreSD` instance for its most recent immutable result. The private in-memory
certificate binds the complete source, denominators, amplitudes, energy/residual
scalars, provenance and action callable. It is never saved or transferred.
Reconstructed results, changed inputs/operators/provenance and results that
miss the consumer's residual bound still execute the original complete raw
action. This removes one duplicate core action; it changes no equation or
acceptance threshold. The [current panel](atomic_pci_sd_native_2026_10_09/valence_core_certificate_public021/pytest.log)
passes56 affected core/valence/Sigma2 checks with no skips. A separate
[current-source GPU replay](atomic_pci_sd_native_2026_10_09/public_valence011/result.json)
passes all16 magnetic states against independent long-double equations; the
largest complete signed numerator-response difference is6.092e-14Ha against
the unchanged1e-11 gate. GPU dispatch remains a private candidate.

The current inverse remains restarted GMRES. A proposed same-column GCROT
subspace reuse passed60 finite live checks but diverged in the independent
rotated4s case, failing the unchanged raw gate. It was withdrawn from the
public API and production; current source was restored exactly to the
certificate021 version. The [failed candidate and withdrawal](atomic_pci_sd_native_2026_10_09/valence_recycling_public022/withdrawal.json)
retain the actual source, SciPy dependency and complete failed GPU012 trace.
No automatic fallback, tolerance change or physical-space reduction was made.
The failure's numerical cause remains unassigned.

Three live public tests pass with no skips, including NX>NXX, full response and
COMMON restoration. The [current public replay](atomic_pci_sd_native_2026_10_09/public_valence002/result.json)
uses a rotated four-electron native DC core, six s/p radial columns and all 16
magnetic states. Independent extended-precision contractions on identical
float64 physical inputs match complete signed actions/roots/responses to
2.09e-16/2.82e-14/1.12e-12 respectively. Buffer/index optimization preserves
all five public output arrays bitwise against the preceding source snapshot.
This finite anchor does not establish a large-valence performance gain or
Yb spectroscopic precision. The current source-separated response passes
[10 live checks](atomic_pci_sd_native_2026_10_09/valence_response_split001/pytest.log)
and a [full independent 16-magnetic-state replay](atomic_pci_sd_native_2026_10_09/public_valence005/result.json).
The stronger .61rad changed-reference response fails its unchanged raw gate;
[complete conditioning diagnosis](atomic_pci_sd_native_2026_10_09/valence_condition006/result.json)
retains that negative result and does not establish high precision. Failed controls004–008 remain archived; their
shallow-state conditioning amplified representation/arithmetic differences.
The later [actual-native arithmetic controls](atomic_pci_sd_native_2026_10_09/README.md#强参考响应的实际算符精度控制)
retain this public-double failure. On identical complete physical inputs, the
private113-significant-bit six-column control passes raw gates; it identifies
an amplified error in the preceding80-bit basis-coefficient reference. The
retained matrix-free80-bit strong-column response agrees with that fresh
control within3.48e-13. This is a finite mixed-width arithmetic diagnosis, with
double core fields/angular helpers retained; it supplies neither a scalable
public extended backend nor a physical uncertainty bound.
The unchanged full-response gate passes in control009 after improving the
independent reference arithmetic, with no space or threshold change.

## Complete fixed-energy reduced Sigma2

`methods.sigma2_sd.plan_sigma2_sd(core_plan, valence_keys)` counts complete
screening and lookup arrays before allocation. Compile its exact `sizes` with
`tools/build_pci_sigma2_backend.py`; this includes the public valence backend,
original pinned `sdvw-ci` Coulomb diagrams and the independently anchored
irreducible Fov term. Every declared core/virtual intermediate remains, while
external valence columns are explicitly selected from that complete pool.
Kappa angular templates are reused across radial columns. The cap covers core,
valence, native screening/response scratch, owned outputs and packed column
indices; simultaneous column-solve Krylov storage must be added separately.
Interpreter/JIT/runtime RSS is outside this payload bound.

```python
from atomx.methods.sigma2_sd import Sigma2SD, plan_sigma2_sd

screening_plan = plan_sigma2_sd(core_plan, requested_valence_keys)
# Compile a fresh combined image with dict(screening_plan["sizes"]).
# Construct its CoreSD and ValenceSD, solve the core, then each explicit energy.
columns = tuple(valence.solve(key, energy_parameters[key])
                for key in requested_valence_keys)
screening = Sigma2SD(valence).evaluate(columns, joint_response=True)
print(screening.radial_Ha.shape, screening.provenance["ci_metric_constructed"])
```

The result owns directed reduced values in original parity-free S1
normalization. `direct_element(m_label, n_label, v_label, w_label)` restores
`(-1)^k S1(m,v)S1(n,w)` and the signed magnetic rho coefficient. Right labels
must belong to solved external columns; left labels may span the complete
retained virtual pool. No approximate index swaps, Hermitian averaging or
energy extrapolation are applied. Joint response varies every column energy
with slope one, including full implicit numerator response; screening
explicit denominators have slope two. Core numerators remain fixed.

Every supplied root and response is checked with the actual bound valence
operator. Source, nucleus, orbital, core-amplitude and complete channel
identities must match. COMMON is restored under the shared core lock, and
checked packed indices are reused without a second native X1/X2 initialization.
Three public screening tests pass; the combined screening/valence panel passes
six tests with no skips. The [current public replay](atomic_pci_sd_native_2026_10_09/sigma2_solved005/result.json)
checks all 16 magnetic states/65,536 signed elements over 25 solved cases.
Independent values/analytic derivatives agree to 4.09e-17/3.73e-16 Ha; the
public magnetic query passes its own complete signed comparison. All three
predeclared steps pass high-order re-solved derivative gates. Optimized and
preceding complete cases are [bitwise equal](atomic_pci_sd_native_2026_10_09/sigma2_public_reuse_equality.json).
Earlier build, method-role binding and second-order difference failures are
retained. No large-workload or controlled timing gain follows from this panel.

A passed directed screening equation does not supply a Hermitian CI operator.
The [same-case consumer audit](atomic_pci_sd_native_2026_10_09/directed_effective_audit001/comparison.json)
retains all 14,400 antisymmetric pair elements. Simultaneous particle-swap
covariance passes at 2.09e-17 Ha, but the chosen fixed-column-energy probe has
Sigma1/Sigma2 pair adjoint defects of 0.01674/0.03687 Ha; the combined frozen
valence matrix defect is 0.03933 Ha. These are finite convention diagnostics,
not experimental error estimates. Ordinary Hermitian CI injection is therefore
unsupported: energy conventions, left/right states and the effective metric
must be established explicitly. No physical excitation root, dressed
observable, monovalent Bloch feedback, nonlinear/triples contribution,
Breit/QED/recoil correction or complex pole is certified by this interface.

The [partial-wave common-energy replay](atomic_pci_sd_native_2026_10_09/sigma2_solved006/result.json)
passes the same 25 signed/response cases. Its complete pair audit reduces the
Sigma1 adjoint defect to 0.002295 Ha, while Sigma2 remains 0.03687 Ha. This
shows sensitivity to the explicit energy protocol; it does not identify the
remaining cause. Non-Hermitian Bloch/FSCC effective operators are legitimate
when their wave operator and left/right physical metric are established.
Forcing symmetry is therefore not an equation repair.

The [strong native occupied–virtual control](atomic_pci_sd_native_2026_10_09/public_valence003/result.json)
rotates occupied 2s and virtual 3s by 0.07 rad before constructing all integral
and Fock identities. Actual Fov is 0.07740 Ha. Full independent magnetic
root/action/response checks pass, including signed response error 1.53e-12 Ha.
The [same-reference screening replay](atomic_pci_sd_native_2026_10_09/sigma2_solved007/result.json)
also passes all 25 complete signed/re-solved cases, with measured nonzero Fov
contributions. This deliberately changes the occupied projector; it is not
a claim of orbital-rotation invariance or stationary orbital optimization.

## Exact finite Bloch wave operator and physical metric

`methods.ci_effective.FiniteCIModel.bloch_reduction(root_indices)` now provides
an independently anchored metric for its own full native finite CI Hamiltonian.
Select exactly one full root per P dimension; no overlap-based replacement is
made. Split degenerate subspaces, singular/ill-conditioned P projections,
resource-cap violations and full residual failures reject the request.

For selected full orthonormal states Psi and C=P Psi, it constructs
`Omega=Psi C^-1`, `G=Omega.T Omega` and `H_B=P H Omega`. It checks full
`H Omega=Omega H_B`, `H_B.T G=G H_B`, biorthogonality and physical state norms.
The orthonormal Hamiltonian uses the isometry `Omega G^-1/2`; no entries are
averaged. `effective_operator(O, representation="bloch"|"orthonormal")`
retains the complete signed full-space property, including Q contributions.
Results are immutable and retain the parent Hamiltonian/basis/metric identity.
The dense anchor has explicit dimension and storage caps; it is not the large
atomic SD/FSCC engine and does not infer a metric for arbitrary directed Sigma.

Nine finite Q-space/Bloch tests pass with no skips. The
[native Be+ 17-CSF/two-P-state control](atomic_pci_sd_native_2026_10_09/native_bloch_metric001/comparison.json)
has a directed adjoint defect 4.79e-6 Ha, while its physical metric-adjoint
residual is 1.94e-17 Ha. All signed reconstructed states, eigenenergies and
core-occupation matrix elements pass original gates. This is a finite physical
metric anchor, without radial/correlation convergence or experimental accuracy.

The finite consumer now also provides parent-bound rectangular transitions,
with both endpoint metrics and normalized left/right state consumption.
`properties.transitions.ci_effective.finite_ci_electric_line` supplies native
electric photon operators across J/parity, at each actual many-electron line
frequency, and retains signed amplitudes and upper/lower statistical weights.
Twenty-one finite CI tests and the broader 240-check affected angular/CSF/
transition panel pass without skips. Four shared angular amplitude magnitude
cutoffs are removed, with a normalized degenerate-state weak-component anchor.
The complete three-sector Be+
16-control determinant/full-state comparison has maximum signed amplitude
difference 5.78e-15 a.u. Local evidence is
`atomic_pci_sd_native_2026_10_09/finite_bloch_transitions.md`.
This does not supply the missing large-space SD wave operator, monovalent
feedback or metric; fixed-energy SD columns retain their original scope.

## Isolated two-valence screening anchors

The active [native owner](atomic_pci_sd_native_2026_10_09/README.md) retains
finite source-bound screening controls. Original pCI and paper unantisymmetrized
numerators use equivalent symmetric particle gauges; complete same-gauge
responses and the physical antisymmetrized difference are both checked.

The two-valence control retains every s/p valence column, all 16 magnetic states
and 65,536 signed Sigma2 matrix elements. Original sdvw-ci values match independent
magnetic Eq.38 contractions to 7.05e-19 Ha. Its parity-free S1 radial normalization
is explicitly restored. None of these finite canonical anchors establishes
noncanonical native valence coverage, production-space convergence or new Yb
spectral precision. The isolated joint-energy derivative now passes a separate
variable/response audit: both valence energies shift together, solved numerator
response is retained, and every denominator has slope two. The wrapper preserves
original values bitwise. Full signed analytic derivatives match the independent
oracle to 3.18e-19; six re-solved controls match three-step centered differences
to 2.77e-13 at the smallest step. The original single derivative field has a
different explicit denominator slope and must not be relabeled as this joint
variable. See the [re-solved gate](atomic_pci_sd_native_2026_10_09/sigma2_joint_resolved_comparison.json).

The separate noncanonical Sigma2 Fock primitive retains all 65,536 signed
elements and their joint derivative, with errors 3.40e-16/1.53e-13 Ha.
Its independent fermion check covers all 14,400 antisymmetric pair elements
and passes at 4.79e-16 Ha. A raw two-valence projection includes embedded
Sigma1: that independently evaluated one-body contribution must be subtracted
with the same energy denominators before calling the result irreducible Sigma2.
The failed unsplit control001 is retained. These are declared complete input
directions, not a solved noncanonical Sigma2 spectrum or a public Sigma2 API.

## Private GPU arithmetic controls

The [native owner](atomic_pci_sd_native_2026_10_09/README.md#gpu实际径向批量控制)
retains actual float64 CUDA controls using shared tagged cell geometry and the
original complete signed term2. The complete Yb density bridge passes values,
analytic responses and full/value/full restoration at its archived source.
Three transfer-inclusive pairs against eight CPU workers fail their declared
load guard; they establish no controlled speedup or production adoption.
A separate whole-GPU candidate derives every angular row from the retained
native kernel and keeps all intermediate projections on-device. Its full414virtual values/analytic responses and mode recovery pass the
retained original kernel. Transfer-inclusive eight-worker pairs fail their
predeclared load guard, so the observed ratio is not a controlled gain.
A private additive native dispatcher retains every other SD diagram, scalar
Fock block and complete source. Five finite Na+ columns pass full affine/
homogeneous/value/response restoration and fresh CPU/GPU solved equations.
Caches are invalidated at every owned COMMON initializer and source/field
snapshots retained per context. The [full-Yb additive map control002](atomic_pci_sd_native_2026_10_09/cuda_valence002/result.json)
passes all five s/p/d columns in the unchanged68core/414virtual space:
complete affine, homogeneous, value, analytic-response and mode-restoration gates
pass, maximum signed affine error1.388e-17 Ha. This is a complete-map proof,
without fresh production roots or controlled timings. The [fresh005 production contract](atomic_pci_sd_native_2026_10_09/yb_full_valence005/contract.json)
has completed certified GMRES for all five fixed-energy columns and directed
screening/joint response, retaining complete root/response gates and source-bound
numerical checkpoints. See the [complete result](atomic_pci_sd_native_2026_10_09/yb_full_valence005/result.json).
The private two-pass recycled inverse now passes the
[independent six-column gate014](atomic_pci_sd_native_2026_10_09/public_valence014/result.json)
and [fresh public certification015](atomic_pci_sd_native_2026_10_09/public_valence015/result.json).
Private results are numerical guesses only; the actual public GMRES result
retains all source/root/response gates before screening. Full production006
has completed this sequence within the original32GiB host cap. The [complete
five-column/screening comparison](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_006.json)
passes unchanged signed gates: maximum column difference1.348e-12Ha,
Sigma2 value/joint-response differences1.123e-15/3.635e-14Ha.
A separate [resident-vector matrix gate002](atomic_pci_sd_native_2026_10_09/cuda_krylov002/result.json)
passes the complete retained64-coordinate physical operator and response rhs.
The private numerical engine owns an8GiB device pool/stream, retains all FP64
vectors and refreshes actual images; combined with the original2GiB diagram
pool its explicit cap is10GiB. Complete physical actions still use the native
operator bridge. [Full independent resident-Krylov certification017](atomic_pci_sd_native_2026_10_09/public_valence017/result.json)
passes all six radial columns/16 magnetic states and actual public
recertification after the allocator/failure/resource gates. Source-bound
production008 has completed all five public-certified columns and directed
screening within its6.708GB resident numerical bound. The [complete matched
comparison](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_008.json) passes
unchanged gates; controlled end-to-end performance remains unverified.
These private controls do not add CUDA parameters or a GPU backend to the
public CoreSD, ValenceSD or Sigma2 APIs. CPU acceptance and actual-source
contracts remain as documented above; no extended/quad GPU precision is claimed.


The private continuation now checks complete homogeneous stage decomposition,
exactly reachable batched GPU projections, disjoint complete angular CUDA
accumulation, and original terminal term41/P/Q singles GEMMs. Frozen full414virtual
five-column angular value/response replay and all six radial/16 magnetic
independent root/response controls pass. Full five-column original homogeneous
component/map comparison passes. [Fresh complete map005](atomic_pci_sd_native_2026_10_09/cuda_valence005/result.json)
and [fused map006](atomic_pci_sd_native_2026_10_09/cuda_valence006/result.json)
pass all five original affine/value/analytic-response arrays, maximum8.674e-17Ha.
006 explicitly reuses certified original CPU arrays with bitwise same inputs;
it freshly executes the GPU path and core certification, without a fresh CPU
timing claim. [Candidate production012](atomic_pci_sd_native_2026_10_09/yb_full_valence012/contract.json)
has completed all five public-GMRES roots/responses and directed screening.
The [complete005/012 comparison](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_012.json)
has maximum signed column difference1.735e-17Ha and bitwise identical directed
Sigma2/joint-response arrays. This does not certify physical convergence or
controlled performance. These numerical controls
retain the existing physical coverage and do not change the public backend.
See the [active evidence](atomic_pci_sd_native_2026_10_09/README.md#完整图项分解与精确可达gpu投影).

The private [Numba CPU/CUDA continuation](atomic_pci_sd_native_2026_10_09/README.md#numba-cpu与cuda共享数值核)
now compiles the same ordered scalar density algorithms for CPU `prange` and
CUDA launches, with explicit real-FP64 layout, memory and backend contracts.
All eight complete s-density ranks and fixed-geometry density directions agree
bitwise with the original public action. A zero-copy same-stream adapter also
passes all five complete frozen-field term2 values, analytic energy responses
and mode-restoration gates, maximum5.421e-20Ha against the certified original
GPU path. This retains the transitive original CPU/map anchor, rather than
claiming a new whole SD solve. The CPU1/CPU8 density-stage ratio is4.6131;
GPU exclusive-device timing rejects and no controlled GPU or end-to-end ratio
is reported. These are private controls; no public Numba GPU option is added.


## Complete transverse J-pair bridge and finite core anchor

`atomx.methods.sd_transverse.build_j_coupled_sd_interaction` consumes the public
owned signed spinor integral ABI, including static/dynamic transverse exchange,
and returns immutable normalized antisymmetric J/parity pair blocks with the
unchanged effective one-body, Fock and reference-energy ledger. All magnetic
coordinates are reconstructed and checked; complete multiplets and spherical
core identity are mandatory. Whole-space work/memory limits reject before
expensive ERI contraction. This is a bounded correctness bridge, without a
quartic magnetic ERI tensor.

`solve_two_electron_core_sd` reuses the existing finite linked scalar SD solver
with that complete Hamiltonian. [Independent evidence](atomic_pci_sd_native_2026_10_09/parallel_sd_transverse001/README.md)
checks signed full three-electron matrices, jj-CSF phases, frozen-core constant,
complex orbital phases and DC/static/dynamic finite SD residuals. This new
finite anchor does not extend the existing large-core pCI Coulomb RINT diagrams.
`methods.sd_pair_diagrams.CorePairLadders` now consumes complete scalar doubles
source, particle-particle and normal-ordered hole-hole ladders directly from
those signed J/parity blocks, plus the linked doubles energy contraction.
It retains every reachable scalar pair coordinate and the noncanonical
one-body/Fock/reference identity. Independent 495-row four-electron actions
check all ten scalar doubles directions for DC/static/dynamic exchange and
complex phases. This is a complete diagram subset; singles, Fock/mixed-rank,
ring and remaining core terms, valence and Sigma2 are still required for a
complete transverse SD solver. See the [six-direction evidence](atomx_hourly_progress_2026_10_10/parallel_six002/README.md).

[Complete Yb tail preflight](atomic_pci_sd_native_2026_10_09/parallel_tail007/README.md)
reuses public band and rho planners for all five states and lmax3/4 × Emax5/10.
Its host bounds include norm and CUDA staging. The original lmax4 norm-scratch
failure is retained. A subsequent [bounded norm adoption](atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_integration001/README.md)
keeps complete signed Gram matrices in an owned 256-entry/4MiB cache, evicting
and recomputing whole matrices without pruning. Fresh 672 finite and 168
multicore component/M gates pass against independent fermion expansion.
Explicit planner injection preserves all five states and original numeric64
counts, with four-band norm scratch bounds of14.15–18.54MB below the unchanged
64MiB cap. The subsequent six-direction control actually consumes the unchanged
complete saved Yb five-state wavefunctions with the adopted helper: all18M,
28,218,500 core coordinates and the original signed component/M gates.
This certifies current metric consumption of those historical source-bound
wavefunctions; it does not solve new roots or measure new lmax4/Emax10 tails. Live GPU
admission, constructor angular-plan count and a source-bound production tail
executor remain mandatory; no new tail roots or experimental accuracy are certified.
