# Matched AMBiT measurement, 2026-10-11

This bounded panel compares the existing Atomx matrix-self-energy interface with independent, unmodified pinned AMBiT MBPT diagram code. It does not compare their independent radial solvers or rank physical atomic-spectrum accuracy.

The common Ne DC inputs contain 19 radial orbitals, all 37,350 declared radial integrals, and three complete signed 3×3 same-κ blocks. Each κ in (−1, +1, −2, +2, −3) retains its declared first three Vᴺ virtual orbitals; this is a finite panel rather than a complete physical continuum. All 27 matrix entries retain separate 2p1h, 2h1p, and total outputs, including nonzero signed off-diagonal entries. The original 10⁻¹¹ Ha / 10⁻¹⁰ relative gates remain fixed.

Five alternating paired warm operator runs all passed the signed-output gates. Median full fixed-energy output work took 41.16 ms in Atomx and 18.63 ms in AMBiT; the median paired workflow ratio was 2.259. Atomx additionally constructs reusable energy-dependent poles/residues and captures the full radial binding; AMBiT directly sums each fixed-energy pp/hh/total request. The timed regions exclude common radial generation, parsing/imports, full warmup, and output formatting. This ratio is scoped to these interfaces and cannot establish an end-to-end software ranking.

AMBiT's full independent BeII smoke calculation uses its unchanged 4000-point input, HF, basis, MBPT, and CI. Each invocation starts in a fresh scientific workspace. Shared-library, OS, and complete angular-data caches are warm. There is no matched Atomx BeII radial/Hamiltonian/space consumer for this calculation, so its parallel result is AMBiT's own fixed-work scaling, not a cross-software precision result. All complete basis, one-/two-electron integrals, six signed roots, and their CI vectors are retained.

The first whole-CLI observations in [result.json](result.json) include Python's timeout-wait polling latency. They are retained in full. A predeclared [timer amendment](timer_amendment.json) replaces only the stopwatch with a C++ monotonic fork/exec/wait4 wrapper; the original executable, inputs, original tolerances, 600-second timeout, 8-GiB address limit, and all scientific work stay fixed. The corrected child records include wall time, CPU time, maximum RSS, and PID. The corrected [final aggregate](result.corrected.json) contains every paired observation. All original resource/load guards passed, and all native basis, one-/two-electron integrals, six roots and signed vectors remained byte-identical:

| OpenMP threads | Paired serial median (s) | Parallel child median (s) | Median paired Sₚ | Eₚ=Sₚ/p |
|---:|---:|---:|---:|---:|
| 2 | 1.01831 | 0.82443 | 1.2391 | 61.95% |
| 4 | 1.02835 | 0.75139 | 1.3788 | 34.47% |
| 8 | 1.05617 | 0.76207 | 1.3898 | 17.37% |

Each row uses its own five alternating 1-thread/parallel-thread pairs, so the serial median differs between rows. The ratios are medians of all paired ratios, rather than ratios of unmatched median times. The small fixed BeII calculation saturates near 4 threads in this panel; this does not predict production CI/HPC scaling.

All timed pairs hold the shared absolute timing lock. An entire panel must retain every predeclared pair and pass every scientific and CPU-load/memory/disk/load-change guard before reporting a controlled ratio; otherwise speedup and efficiency are null. No slow observation is filtered. Threads are configured through OpenMP and confirmed by the program's own log; they are not a claim of exclusive physical-core occupancy.

Evidence: [fresh complete signed inputs and precision result](fresh_inputs001/matched_sigma_results.json), [timed AMBiT adapter](timed_sigma.cpp), [Atomx complete consumer](atomx_action.py), [full benchmark driver](benchmark.py), [corrected timer driver](accurate_scaling.py), [actual native/Python source/object archive identity](execution_identity.json), [mapped runtime dependencies](execution_identity.json), [source precheck](pre_source_guard.json), and [native angular inputs](angular_inputs_identity.json).

The first adapter attempt supplied four-coordinate saved states to a three-coordinate virtual hook. Its failure, source, stdout, and nonzero exit remain in [failure_adapter001](failure_adapter001/benchmark.stderr) and [matched_round00](matched_round00/atomx.stderr). Mapping P,Q,κ,E to the hook's P,Q,E fixed the adapter without changing orbitals, integrals, diagrams, or gates.

Four meaningful owner tests passed: complete signed independent replay, preservation of slow/failed observations, exact original resource/load boundaries, and child-timer failure/RSS propagation. See [final JUnit evidence](final_test_checks.xml). These private report scripts changed no production physics implementation. Syntax/name checks passed; full style lint reported import-order and safe synchronous closure/zip style findings, retained as review observations rather than a physics failure.

Final checks rehashed 621 actual source/object/binary files, 137 mapped runtime dependencies, and 8 preserved native angular-cache files with no mismatch. [Complete artifact manifest](artifact_manifest.json) and [postchecks](post_identity_guard.json) bind all raw arrays, failures, cold observations, both timing epochs, archives and tests. Shared-library runtime files are hashed rather than copied; the actual source/object archive and timing-refinement source/binary/complete angular archive are retained.
