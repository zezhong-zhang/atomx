# FAC independent evidence review

This owner is read-only with respect to production source, reference builds,
scientific runs, and other benchmark owners. It audits actual source and runtime
identity, complete numerical inventories, and paired statistics independently.

## Source-level findings before measurement

- Local FAC is at `35f9846c1d93d5b6eb3b47ce66038388aeb6b70c`; the actual
  worktree also contains changes to `setup.py`, `python/consts.py`, and package
  metadata. HEAD alone is not the executable identity.
- `sysdef.h` defines `USE_MPI 2`; configure selected `--with-mpi=omp`.
  `InitializeMPI(p)` initializes FAC's worker-local rank, scheduling locks,
  potential copies, and workspace in addition to setting the OpenMP team.
  `InitFac()` does not call it. Setting `OMP_NUM_THREADS` alone does not initialize
  these facilities. A requested team can be clamped by `OMP_THREAD_LIMIT`;
  process thread observation remains the resource gate.
  The retained serial `actual002` run completed without this call; its omission
  alone does not invalidate that observation. Final `actual005` explicitly
  initializes one worker.
- The local static links use `libfac.a`, `libmlapack.a`, and `libmblas.a`;
  dynamic requirements include `libgomp`, `libgfortran`, `libm`, `libmvec`, and
  libc. `libgfortran` does not resolve under the default loader path on this
  machine. The supplied library directory and actual mapped runtime must be
  recorded; `ldd` without that environment is not a runtime measurement.
- The private diagnostic frontend adds seven dump commands to stock SFAC.
  It links the same FAC numerical archives without editing those libraries.
  `DumpPhoton` explicitly clears a multipole cache and sets a fixed-frequency
  grid; that diagnostic changes interpolation state. The present DC common
  integral campaign does not call it. CI integral/spinor exports must retain
  their complete inventories and must not be counted as stock workflow output.
- `LimitArray(0,0)` disables the Yk cache. The cache uses float storage plus a
  fitted asymptotic tail, whereas its disabled path computes Yk in double
  precision. This is a declared numerical option, not a change in CI space.
  Default-cache controls and cache-disabled references need distinct labels;
  any failing default control stays in the record.

The prospectively fixed current campaign disables the Yk cache. No fresh
default-cache negative control was registered or measured in this source epoch;
historical default-cache differences therefore cannot quantify this option's
current accuracy or speed effect.

The [official FAC README](https://github.com/flexible-atomic-code/fac) identifies
OpenMP as its primary parallel mechanism, documents the independent SFAC/PFAC
frontends, and states that PFAC is not thread safe. Independent concurrent jobs
therefore require separate processes. This audit does not infer speed or
accuracy from those documented capabilities.

## Comparison boundaries

Common complete FAC radial integrals test independent angular CI assembly;
common FAC spinors additionally test Atomx's radial reconstruction. Neither
establishes independently optimized experimental accuracy. Frozen-config DC
workflows with separately optimized orbitals are model comparisons, so their
absolute timing does not rank identical Hamiltonians. The declared complete
spaces, two photon modes, all levels and lines, actual source epoch, and
accepted load/thread gates must survive every paired observation.

The timed stock FAC child writes text tables and binary Hamiltonians; conversion
to the review NPZ/JSON happens afterward. The timed Atomx child also captures
signed matrices, eigenvectors, and radial data and writes NPZ/JSON. Own-code
scaling keeps that export work fixed. Absolute cross-code costs include those
different instrumentation and export scopes and cannot rank identical
production or equal-accuracy workflows. Failed or incomplete children are
retained but cannot contribute an accepted complete four-ion workflow cost.

No new general insight is introduced: these findings apply existing
convention-first, actual-runtime, complete-inventory, and measured-scope rules.

## Final independently verified evidence

[final_audit.json](final_audit.json) is authoritative. Its `passed` field covers
evidence identity, completeness, arithmetic, and admission; it does not claim
that the scientific targets passed. The audit rehashed both complete owner
manifests: 3,042 files and 221,132,894 bytes, including retained failures.
It verified actual source archives containing 868 precision files, 1,084 initial
performance files, and 1,086 fresh-cache files. All 1,084 common performance
source files stayed unchanged; the two additions are private cache helpers.
The mapped runtimes and the before/after source identities remained unchanged.

The independent reader decoded 670 precision arrays, 1,002 arrays from the
initial performance epoch, and 5,112 fresh-cache arrays. It checked complete
inventories, level/line identities, and all declared resource/load gates. The
final precision inventory has 635,443 coordinates, 35 symmetry blocks, 192
levels, and 2,003 integrals. All 64 excitation and 40 E1 model-spread rows were
recomputed from their underlying outputs. That matching uses configuration,
J/parity, and energy rank, without state-overlap or NIST validation.

Three within-round paired observations reproduce the accepted rows below.
Efficiency divides each paired speedup by the requested worker count; actual
process thread peaks are a separate resource measurement.

| Backend | Requested workers | Median full four-ion seconds | Paired speedup | Logical efficiency | Actual thread peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| FAC | 1 | 4.769212 | 1.000000 | 1.000000 | 1 |
| FAC | 2 | 4.718081 | 1.010534 | 0.505267 | 2 |
| FAC | 4 | 4.730715 | 1.009978 | 0.252494 | 4 |
| FAC | 8 | 4.751842 | 1.005121 | 0.125640 | 8 |
| Atomx, fresh cache | 1 | 8.928899 | 1.000000 | 1.000000 | 1 |
| Atomx, fresh cache | 2 | 8.981222 | 1.001278 | 0.500639 | 3 |
| Atomx, fresh cache | 4 | 8.964119 | 0.998994 | 0.249749 | 7 |

The requested Atomx eight-worker warmup reached 15 threads against the fixed
eight-thread process cap. Its accepted time, speedup, and efficiency stay null.
Original cache-failure children remain archived and excluded from complete
workflow cost. Same-backend admitted full arrays are bitwise unchanged.

The original precision gates remain absolute 1e-11 Ha and relative 1e-10.
Common-integral eigenvalues pass both gates for Z=10, 6, and 26; Z=82 fails
the absolute gate at 3.092282e-11 Ha despite its 2.286268e-15 relative error.
The common-spinor 4,000-point aligned eigenvalues fail the strict gate in all
four cases, with maximum absolute errors from 1.244688e-8 to 1.261316e-6 Ha.
These failures, unmeasured fresh default-Yk controls, unequal diagnostic costs,
and the unmeasured experimental accuracy prevent a highest-level claim.

The independent audit introduces no new scientific runs or production edits.
[audit.py](audit.py) contains the reproducible arithmetic and manifest checks;
[ears.json](ears.json) and [documentation_impact.json](documentation_impact.json)
record the handoff and documentation disposition.
