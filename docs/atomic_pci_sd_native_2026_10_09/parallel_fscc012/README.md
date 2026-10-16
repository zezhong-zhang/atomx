# Noncanonical 0h1p Fov–T2 equation audit and isolated candidate

The unchanged ten-spinor dynamic-transverse case now has a specific equation
diagnosis and a passing isolated candidate. The validated baseline executable
and checkout remain unchanged. The candidate only removes the duplicated
outgoing-particle antisymmetrization in EXP-T `const_terms_0h1p`, `dgd2b_2`.
This closes the finite strict target, without establishing production
large-space FS-CCSDT, experimental accuracy or highest-level certification.

## Contract and complete amplitude evidence

[The frozen original-backend audit](final_audit004/contract.json) fixes the
saved `transverse_precision007/tight` Hamiltonian, native orthonormal metric,
two occupied and eight active spinors. It retains all 120 three-electron
determinants and eight directed model columns. The strict effective-matrix
gate is `1e-10 Ha`; the complete Bloch-oracle gate is `1e-11 Ha`. No Fov
deletion, fitted shift, orbital removal or precision-budget change is used.

The bounded [reader and probe](probe.py) decodes every actual signed
`t1c`, `t2c`, `t3c`, `s1c`, `s2c`, `s3c` block, including permutation-related
blocks, from the pinned uncompressed little-endian `ATXDG002` format. It
rejects unknown formats, truncation, trailing bytes, inconsistent spinor axes,
unresolved permutation parents and external block files. The pinned reference
`diagram.c`, `block.c`, `sort_1e.c`, `sort_pyscf.c` and integral reader are
included in each source archive. This is a finite determinant diagnosis,
not a tensor-production amplitude solver.

EXP-T stores bra-first amplitudes. In particular, its `hp`/`hhpp` inputs use
`F[i,a]`/`g[i,j,a,b]`; ket excitation coefficients require conjugation.
The exponential uses `1/(k! k!)` for core rank `k` and attachment uses
`1/(k! (k-1)!)` for incoming valence plus `k-1` core holes. Full complex
rotations independently check this convention.

[Actual original amplitudes](final_audit004/result.json) give:

| Quantity | Maximum |
| --- | ---: |
| Fock decode minus original signed Fock | 0 Ha |
| Actual core cluster minus exact core cluster | 9.324e-18 |
| Actual core projected equation residual | 2.949e-17 Ha |
| Actual attachment Bloch residual, P / Q1 / Q2 | 1.776e-15 / 4.164e-7 / 1.791e-16 Ha |
| Actual HEFF minus projection of actual wave operator | 2.109e-15 Ha |
| Original HEFF minus complete exact HEFF | 8.729e-10 Ha |

Define ket amplitudes `tau = conjugate(t2c)`. The complete residual equals
the open rank-two operator with coefficients

```text
C[v,i,a,b] = sum_j F[j,v] * tau[j,i,a,b]
C_operator = (1/2) sum_{viab} C[v,i,a,b] a†a a†b ai av
```

Subtracting this *complete signed operator* from the original 120×8 residual
leaves at most `1.776e-15 Ha`. This is stronger than an energy/Fov correlation:
it locates the operator term and preserves all off-diagonal phases.
The first test in [the independent panel](../../../tests/test_fscc_equation_audit.py)
uses complex six-spinor coefficients and an independent occupation-list
action. The exact open projection of `[Fdeexcitation,T2]` requires one copy;
doubling it fails. A whole unprojected commutator is **not** the open term:
its spectator and additional normal-ordered contributions are different.
The unsuccessful whole-commutator attribution remains in
[audit002](audit002/result.json).

## Isolated candidate and same-H gates

The reference source already contracts `vh` with antisymmetric `t2c`, then
applies `perm("r2", "(34)")` once more. This doubles the open Fov–T2 term.
The [exact candidate patch](candidate001/equation.patch) removes that extra
permutation. [The build contract](candidate001/contract.json) binds every
reused object and archive, original source, compiler, compile/link flags and
original executable before compiling only the changed object into a fresh
binary. The original files are rechecked after compilation.

The candidate executable SHA is
`36576b763bde756ef272770f0967de6282aaa3448748a5238439bba06886cfcc`.
Its [manifest](candidate001/atomx_backend.json) retains the original upstream
revision and interface identities and adds the explicit equation patch.
The validated original executable remains
`1605f4d9daf0adb5ecc543fc785da4cd5e32d96121260dc00e9951bdbdd0d33f`.
No deployed backend is replaced by this task.

[The final controls](final_panel.json) retain each case's own complete FCI,
core-cluster and intermediate-normalized Bloch oracle:

| Candidate case | Complete Bloch max, Ha | Full HEFF error, Ha | Eight-root FCI error, Ha |
| --- | ---: | ---: | ---: |
| Original dynamic H, C1 | 1.776e-15 | 4.219e-15 | 1.599e-14 |
| Original dynamic H, axial | 1.776e-15 | 4.219e-15 | 1.599e-14 |
| Declared zero-Fov H | 8.119e-14 | 7.327e-15 | 1.776e-14 |
| Declared twice-Fov H | 3.553e-15 | 1.055e-14 | 1.954e-14 |
| Dirac–Coulomb H | 3.553e-15 | 1.887e-15 | 4.263e-14 |
| Static-transverse H, ns=42 | 3.553e-15 | 4.552e-15 | 1.599e-14 |
| Complex occupied/virtual rotation, seed 83 | 4.615e-14 | 1.034e-14 | 1.776e-14 |
| Complex phase rotation, axial | 5.597e-14 | 6.471e-15 | 2.665e-14 |
| Dynamic-transverse H, ns=100 | 3.553e-15 | 5.662e-15 | 1.066e-14 |

For every candidate solve, `ATOMX_ONE` and `ATOMX_TWO` bytes match its source
input. Zero/doubled Fov cases are explicitly changed Hamiltonians, each with
its own complete oracle; zero Fov is not adopted as a repair. The complex
rotation preserves the original operator under a complete determinant-space
unitary transformation to `1.066e-14 Ha`.

`final_static` was initially given a misleading directory label: its actual
saved input was the duplicate dynamic case in the axial test. Its original
records are preserved with an explicit `label_correction` in `final_panel.json`;
the actual static result is [final_static002](final_static002/result.json).

The initial [complex amplitude audit](complex_candidate001/result.json)
failed because the diagnostic lift omitted bra-first conjugation; its
core-reconstruction gate rejected interpretation even though the candidate
HEFF already passed. The convention was corrected and
[complex_candidate002](complex_candidate002/result.json) passed. Exact earlier
probe/preparation/validation sources were recovered and checked against their
recorded hashes in [historical_sources](historical_sources/identity_proof.json).
Negative results and their source versions remain available.

## Replay and resource scope

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_fscc012/probe.py \
  --output /path/to/fresh/audit

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_fscc012/build_candidate.py \
  --output /path/to/fresh/candidate

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_fscc012/validate_candidate.py \
  --executable /path/to/candidate/expt.x --output /path/to/fresh/validation
```

Candidate execution additionally needs the existing isolated toolchain library
environment recorded by the owning topic. All science processes use one BLAS
thread, a `4 GiB` address-space cap and a `120 s` solver timeout. Native EXP-T
uses its separate `512 MiB` tensor cap, tile size 4, full CCSDT and raw threshold
`1e-14`. Available memory and load are recorded before each experiment. No
physical model is silently reduced.

The original audit phase took about 0.5 s and process peak RSS was about
392 MB in this environment. Archive/import time is outside that phase timing;
this is an observed small finite workload, without a controlled speedup or
production CPU/GPU scaling claim. Source archives include actual dirty files,
signed inputs, actual backend amplitudes and model matrix. The
[delivery identity](delivery_proof.json) binds the frozen tools and test evidence.

The [four-test panel](checks.xml) passes without skips. Ruff and formatting
checks pass on the five owned Python files. Shared guide/navigation changes
and optional public-builder adoption are handed to the parent integration
task; this directory does not modify shared production physics.

## Remaining boundary

This establishes a corrected finite noncanonical attachment contraction and
complete signed validation across real/complex C1/axial representations.
Larger occupied/virtual spaces, more sectors, production tensor contractions,
properties/left states, J-reduced native transverse SD, physical tails and
experimental spectra remain separate gates. A small oracle does not certify
the entire scientific software or highest attainable atomic accuracy.
