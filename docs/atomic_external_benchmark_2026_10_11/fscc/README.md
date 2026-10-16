# Finite matched-Hamiltonian CI / EXP-T benchmark

This owner measures a **local, finite** FS-CCSDT backend and its Atomx wrapper.
Atomx FSCC calls EXP-T, so the wrapper and bare executable are the **same
amplitude algorithm**. A complete Fermi determinant CI supplies a separate
finite accuracy oracle; its Python matrix builder is not a production CI
software performance ranking. Neither fixture establishes experimental
precision or large-scale atomic throughput.

The [parent contract](../contract.json) fixes signed inputs, residual gates,
thread/memory limits and a common timing lock. [benchmark.py](benchmark.py)
records actual dirty Python source, actual binary and source manifest, complete
input files, signed output tensors, all selected roots, native metric and
timing scopes. No production or reference source is changed by this benchmark.

## Accuracy and independent oracle

Both fixtures retain their archived complete signed finite-frequency
transverse Hamiltonian, all magnetic multiplets and all eight attachment
model columns. The two-core case has 10 spinors, 45 complete core and 120
attachment determinants. The four-core case has 12 spinors, 495 complete core
and 792 attachment determinants. Its orbitals remain those of the declared
two-core radial producer; this is a finite method/truncation control, not a
new optimized experimental atomic calculation.

The local default EXP-T build includes Atomx's integral ABI, exact compact
tensor index and axial symmetry patches, while retaining the original
noncanonical attachment equation. The explicit corrected build additionally
contains the Fov–T2 repair. Both use upstream revision
`aa8d809ad5818d4aa618dcdadbd7f12257ecbc28`; their source manifests differ only
at `src/rcc/models/sector01.c`. [Runtime provenance](runtime_provenance.json)
retains full hashes and this source distinction. This is not a comparison
against an untouched upstream executable.

The old noncanonical two-core backend passes a root-only error at a loose
scale while failing the full signed Bloch equation. A software agreement
claim therefore requires the independent operator gate. Four-core CCSDT
also has genuine omitted T4/S4 residuals and FCI method errors even after
the retained equations pass; those method errors are reported separately.

[run003](run003/result.json) is the final run. Every baseline retains the
complete signed T1/T2/T3/S1/S2/S3 tensors, complete determinant Hamiltonians,
all eight wave-operator model columns and the signed 8×8 HEFF. The independent
[validation](validation.json) decodes all 20 repeated tensor/HEFF outputs
again: their maximum difference from the corresponding corrected baseline
is exactly zero. The finite-CI validation retains every eigenvector and root,
with independent residual and orthogonality checks.

| Fixed finite input / backend | Retained attachment equation, Ha | Full Bloch residual, Ha | Maximum error of all 8 roots vs FCI, Ha |
|---|---:|---:|---:|
| 2 core / 10 spinors, local default | 4.163911e−7 | 4.163911e−7 | 8.728964e−10 |
| 2 core / 10 spinors, opt-in corrected | 1.776357e−15 | 1.776357e−15 | 1.243450e−14 |
| 4 core / 12 spinors, local default | 1.031257e−2 | 1.031257e−2 | 3.515326e−3 |
| 4 core / 12 spinors, opt-in corrected | 6.483789e−14 | 2.283871e−3 | 9.093015e−6 |

The corrected core residual is 2.948777e−17 Ha for two core and
5.620071e−14 Ha for four core. Four-core core CCSDT–FCI energy error is
3.170001e−6 Ha. The corrected signed HEFF projection is within 5.4e−15 Ha
and intermediate normalization is exact in both fixtures. These are
retained-equation checks, not evidence that the four-core truncated wave
operator equals FCI or that either finite model matches experiment.

## Resource admission and performance

The actual link line statically links bundled OpenBLAS and dynamically links
libgomp. The BLAS discovery value in CMakeCache is not the executable link.
The source implements external OpenMP tensor-block contractions; each block
GEMM requests one BLAS thread. The executable can still retain additional
runtime threads.

[run001](run001/stopped_thread_budget.json) is the preserved requested-only
attempt: requested teams 1/2/4/8 were observed as 1/4/11/25 total process
threads. The benchmark stopped its own verified process group once the
8-thread parent budget violation was recognized. Its p4/p8 timing and
efficiency claims are invalid.

[thread_preflight001](thread_preflight001/result.json) prospectively set
`OMP_THREAD_LIMIT=p` and `OMP_MAX_ACTIVE_LEVELS=1`. A stricter total-count=p
observer stopped p2 when a third runtime thread appeared; the additional
threads had zero sampled CPU ticks at that instant. This does not prove they
would always be idle.

[thread_preflight002](thread_preflight002/result.json) retains the original
parent maximum of 8 total process threads, separately records requested team
size, and first rejects p4 at 9 observed threads. The observer's final sampled
peak reached 10 before the owned child exited; the frozen run003 summary
retains its original first-over-limit field of 9. Its p1/p2 complete signed tensors
and HEFF passed unchanged-input gates. No further low-information runtime
configuration scan is attempted; p4/p8 remain resource-rejected rows.

The original [run002](run002/result.json) used `subprocess.run(timeout=120)`.
Its wall times remain observations of the deployed public wrapper path,
including Python's exponential wait polling (up to 50 ms). The explicit
[timing review](run002/timing_review_withdrawal.json) withdraws inference about
backend algorithm cost, extremely narrow timing dispersion and lower-level
efficiency from those records; it does not discard their scientific arrays
or claim that the measured deployment latency was fictitious.

The prospective [run003 contract](run003/contract.json) declares a private
blocking `Popen.wait()` launcher and a separate 120-second owned-child
watchdog. The bare executable and **private low-overhead wait wrapper** use
that same launcher. The wrapper temporarily replaces its subprocess object
only inside this private benchmark process and restores it in `finally`.
The actual production file, defaults, integral producer and binary remain
unchanged. Accordingly, this is not untouched public-wrapper latency.

Run003 measures five alternating rounds at admissible p1/p2 settings, with
every tensor coordinate rechecked. Bare timing includes private launcher
startup/watchdog, executable startup, solve, scratch I/O and exit; pre-export
input setup and parsing are excluded. Private wrapper timing additionally
includes integral export, executable validation and result readback. Python
imports, source build and orbital/integral production are prebuilt and
unmeasured. All measurement panels share the parent lock; all source,
correctness, load and resource gates pass.

The authoritative [paired statistics](paired_statistics.json) report the
median of five within-round ratios `T1/T2`. Frozen run003 also retains its
original ratio-of-time-medians definition, which happens to give the same
median here. Logical requested-team efficiency is `S/p`; the separate
process-thread resource ratio is `S/n_actual_peak` using the aggregate peak.

| Requested OpenMP team | Wall median [min, max], seconds | Actual total threads, sampled peak | Paired speedup | Logical team efficiency | Process-thread resource ratio |
|---|---|---:|---:|---:|---:|
| 1 | 0.627102 [0.623729, 0.638829] | 1 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 1.121025 [1.050059, 1.375366] | 5 | 0.559400 | 0.279700 | 0.111880 |
| 4 | resource rejected | 10 (first rejection at 9) | null | null | null |
| 8 | resource rejected in requested-only run001 | 25 | null | null | null |

The five paired p2 speedups are 0.591725, 0.516595, 0.593994, 0.559400 and
0.460496. On this tiny four-core fixture p2 is slower. Sampled CPU/wall is
0.963–0.978 at p1 and 2.695–2.785 at p2; the maximum number of threads with
positive CPU-tick increments between 10 ms reads is 1 / 3. These are sampled
lower bounds and distinct from instantaneous utilization. Additional runtime
pool threads cannot be assumed inactive throughout the solve. Peak sampled
backend VmHWM is 274,235,392 bytes across the timed repeats; benchmark parent
peak RSS is separately 552,722,432 bytes. Neither is an exact whole-process-tree
peak. This single-node CPU fixture supplies no GPU, MPI or large-atom scaling
claim.

## Costs with explicit work scopes

| Fixed finite input / work | Five-round wall median, seconds |
|---|---:|
| 2 core, corrected naked EXP-T, launcher/solve/I/O/exit | 0.312197 |
| 2 core, private low-overhead wait wrapper | 0.315232 |
| 2 core, complete CI matrix construction | 0.113111 |
| 2 core, complete CI eigensystems | 0.001160 |
| 2 core, CI construction plus complete eigensystems | 0.114263 |
| 4 core, corrected naked EXP-T, p1 | 0.627102 |
| 4 core, complete CI matrix construction | 2.365901 |
| 4 core, complete CI eigensystems | 0.200859 |
| 4 core, CI construction plus complete eigensystems | 2.568621 |

The median paired private-wrapper extra cost is **5.759497 ms**; paired
deltas range from −2.459557 to 11.792380 ms. The difference of the separate
time medians is **3.034781 ms**, a different statistic. Five short rounds do
not establish a precise asymptotic adapter overhead. Atomx and naked EXP-T
here use the same solver; this is an adapter-cost measurement, not an
independent algorithm speedup. The complete CI Python oracle performs
different work (all roots/eigenvectors) and the four-core CCSDT omits genuine
T4/S4 correlations, so these scoped costs do not establish a
precision-equivalent software ranking.

## Reproduction and evidence

Fresh output directories are mandatory. Source epochs and registered inputs
are snapshotted before execution; complete old/new source archives are
independently verified as 802 / 813 / 814 registered files. Runtime-mapped
dependencies, actual binary/link/static-BLAS identity, every full-array output
and all failure records remain hashed. All 960 registered run003 outputs
verify. [Owner result](result.json) joins current evidence and explicit
historical review overrides without rewriting any frozen run summary.

```sh
.venv/bin/python docs/atomic_external_benchmark_2026_10_11/fscc/benchmark.py \
  --workers 1 2 --output docs/atomic_external_benchmark_2026_10_11/fscc/new_run
.venv/bin/python docs/atomic_external_benchmark_2026_10_11/fscc/validate.py \
  --run docs/atomic_external_benchmark_2026_10_11/fscc/run003
```

Validation/test CPU work also used the parent lock. [Test records](test_results.json)
retain 16 scientific/ABI/projection passes (2 explicitly deselected live tests)
and 2 owned-child launcher/watchdog passes; the benchmark itself supplies the
fresh external full-array scientific gates. [EARS](ears.json) and
[documentation impact](documentation_impact.json) record the integration scope.
No production/default promotion is implied, and highest atomic accuracy and
large-scale throughput remain unproven.
