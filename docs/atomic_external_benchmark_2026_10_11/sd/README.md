# Complete SD operator agreement and CPU strong scaling

This owner measures the current public native SD consumers and exact streamed
kernel against pinned original pCI-derived diagrams. The pCI checkout is
`4d685d82bfdf27876098a87808dadc5fcf3038d1`. The common native radial service and
upstream equations are shared. Agreement establishes full-array compatibility;
it does not establish independent software accuracy, a production pCI benchmark,
or improved agreement with NIST.

The [parent contract](../contract.json) fixes all signed coordinates, Hamiltonian,
nucleus, core, complete virtual pool, energy, units and response. The accuracy
threshold is 1e-11 Ha and relative 1e-10. The shared timing lock serializes measured
scopes across benchmark owners; each observation records load before/after.
`max(load) + workers <= affinity CPUs` and load change at most 4 are required.
Failed timing admission preserves observations and sets controlled speedup to null.

The [pCI availability audit](pci_availability.json) found no compiled all-order
executable; the MPI compiler and launcher are also absent. The upstream all-order
Makefile targets are serial and do not require MPI. Complete standalone main-program
builds and matched input exchange were not established. The fresh matched control
uses the original pinned diagram ABI and explicitly declares its inherited lineage. No pCI production
or MPI efficiency ranking can be inferred.

## Measurements

- [Finite full-action probe](finite_action_probe.py) uses all ten closed-core
  electrons of Na+, ns38, point nucleus, all declared 3s4s3p4p3d virtual radial
  orbitals and all magnetic partners. Every map consumes all five s/p/d valence
  columns, full signed numerators and supplied analytic derivatives. It calls
  public `ValenceSD.evaluate`, retaining source, Fock and every diagram present
  in that interface. Five alternating rounds compare original diagrams and
  current 1/2/4/8-worker implementations. It does not time an SD solve.
- [Complete Yb round probe](yb_round_probe.py) retains 68 core electrons,
  all 414 virtual radial orbitals, ns123 Fermi nucleus, lmax3/Emax5 and the original
  6s numerator plus response. Every invocation covers all 1076 stored channels,
  including exactly forbidden zero coordinates. Three independent processes,
  each one unchanged complete 1/2/4/8 round with its own 600-second wall cap and
  8 GiB address cap, separate preparation, flock wait and warm complete term2
  action. Compilation and full SD solves are excluded from the action speedup.

[Input semantics audit](input_semantics/audit.json) retains the original signed
stress-vector generator and seed 314521. Yb uses the complete physical radial
space with signed Gaussian numerator and supplied response of scale 1e-4,
zero forbidden coordinates and a zero core seed. This is a complete operator
control, not a converged physical Yb SD wave. The finite Na+ control certifies
its solved core wave but also applies supplied stress vectors to valence columns.

The original Yb pCI diagram stage previously took 602.4896 seconds per call,
exceeding this task's 600-second limit before setup. Its original full signed and
response arrays are retained and compared; there is no fresh original Yb timing.

## Retained failed pilots and scope

[First preparation](yb_stage001/failure.json) was stopped after an incidental CPU
accounting defect: `RLIMIT_CPU` sums worker CPU time and cannot replace a wall-time
cap. [Second pilot](yb_stage002/stopped_for_round_partition.json) passed a complete
8-worker warmup against all original arrays, then was stopped proactively when
measured cold setup showed the proposed three rounds would exceed the unchanged
600-second wall budget. The incomplete serial call was discarded. All old inputs,
source and completed outputs remain. The actual round contracts retain every
physical coordinate and all original gates.

[Reservation accounting](resource_partition.json) records address space separately
from observed resident memory. At most one full Yb timing-round process is resident
per lane, plus the small finite control. Three full rounds run sequentially.

## Completed finite whole-action result

[Corrected current result](finite_action001/result.corrected.json) retains the raw
scientific results and observations. [Scope correction](finite_action001/measurement_scope_correction.json)
withdraws an unmeasured SCF/virtual clock placeholder to null and labels the overall
workflow clock as including setup, waits and timing. No scientific arrays or
measured action times changed.

| Backend / workers | Median time per complete five-column map | Paired speedup vs current one | Parallel efficiency |
|---|---:|---:|---:|
| Original pCI-derived diagram ABI, 1 | 52.125 ms | — | — |
| Current, 1 | 26.355 ms | 1.000 | 100% |
| Current, 2 | 22.120 ms | 1.195 | 59.77% |
| Current, 4 | 20.375 ms | 1.292 | 32.30% |
| Current, 8 | 22.412 ms | 1.159 | 14.49% |

Five alternating rounds passed every numerical and load guard. The maximum complete
signed/total-response difference is 8.6736e-19 Ha, relative 1.6905e-15.
The four-worker/current versus original-diagram paired median ratio is 2.536.
This is a whole-interface action comparison with shared pCI lineage, not an
independent production-software accuracy ranking. The small workload is faster
with four workers than eight. Its measured efficiency does not predict the larger
Yb stage.

[Independent full-output verification](finite_action001/full_output_verification.json)
re-read all 25 measured NPZ files, 500 complete arrays and 2034 output coordinates
per five-column map. The same 490-coordinate solved core wave passes the original
whole-core residual gate in all five backends; all source/denominator identities
are equal. Peak RSS for all five resident backends is 333,312,000 bytes against a
432,614,844-byte complete physical payload estimate. Backend construction,
complete integral setup and core-wave certificate take approximately 4.14–4.39
seconds per instance. These cold stages are recorded separately and not compared
as a production solve speedup.

## Completed full Yb stage result

All three independently capped rounds passed every signed/response, source,
input, backend/runtime identity and measured load guard. The maximum difference
from the preserved original pCI diagram is 3.117081e-19 Ha, relative 4.898035e-15.
All 1/2/4/8-worker outputs are bitwise equal to the previous serial streamed
output. Each round's independent replay verifies all ten arrays of shape
64×64×1076: 4,407,296 coordinates in each signed array and its same-size response,
including forbidden zeros. All 365 captured source files and six original
scientific input files match the actual repository. There are 30 complete Yb
output arrays, including three full warmups; 24 belong to the 12 measured calls.

| Workers | Round 0 seconds | Round 1 seconds | Round 2 seconds | Median seconds | Paired speedup | Parallel efficiency |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 81.2861 | 83.7682 | 82.1194 | 82.1194 | 1.0000 | 100% |
| 2 | 43.8341 | 44.0492 | 43.9537 | 43.9537 | 1.8683 | 93.42% |
| 4 | 23.8504 | 24.5635 | 25.2658 | 24.5635 | 3.4082 | 85.20% |
| 8 | 16.9710 | 17.0865 | 17.4689 | 17.0865 | 4.7897 | 59.87% |

Speedup is the median of the three within-round T1/Tp ratios, and efficiency
is that value divided by the requested worker count. It is not a ratio of
unpaired medians. Round order is 1/2/4/8, then 8/4/2/1, then 2/8/1/4.
The shared absolute flock serializes complete measured calls across owners.
Before/after load and each admission result remain in every observations file.

Cold construction and full integral preparation take 109.867/115.884/108.009
seconds; total lock waits, including warmups, are 57.776/1.223/0.000095 seconds.
Peak RSS is 3,213,312,000/3,210,616,832/3,223,339,008 bytes. These stages and
resident memory are separate from the warm operator times. Retained preflight
to-result wall markers are 362.284/315.974/305.409 seconds, safely within each
original 600-second process cap; [wall identity](process_wall_identity.json)
labels its exact scope and does not infer an unmeasured import clock.

[Actual thread verification](thread_verification.json) confirms exact current
public kernel bodies compiled with outer 1/2/4/8 and actual loaded inner BLAS=1.
Read-only process thread CPU deltas observed 1/2/4/8 active compute threads in
the last two complete rounds, and 2/4/8 in round 0. Its serial call finished
before sampling started, so that one OS observation stays null. Transition
intervals, cold setup and warmup are excluded; low activity during lock wait
or output compression is not treated as an invented worker observation.

[Final aggregate](result.json), [independent verification](verification.json),
[round 0](yb_round000/full_output_verification.json),
[round 1](yb_round001/full_output_verification.json) and
[round 2](yb_round002/full_output_verification.json) preserve full arrays,
F-order coordinate hashes, compiler/source/input/binary/runtime identities.
[CPU scaling figure](cpu_scaling.png) and [vector export](cpu_scaling.svg)
normalize each workload against its own serial time.

The larger Yb stage benefits from eight workers; the finite full-action workload
does not. Worker choice must follow workload size and measured complete scopes.
The 4.79-fold stage speedup cannot be assigned to a full action, converged SD
solve or production spectrum. No experimental accuracy improvement, independent
pCI software ranking or MPI efficiency measurement is claimed. Public physics,
backends and defaults remain unchanged.

## Replay and retained evidence

The private source-bound launchers are [finite_action_probe.py](finite_action_probe.py)
and [run_yb_rounds.py](run_yb_rounds.py), with [yb_round_probe.py](yb_round_probe.py)
retained in each actual launch source tree. Re-running requires a new owner
directory because existing run directories deliberately refuse overwrite.
Run ` .venv/bin/python docs/atomic_external_benchmark_2026_10_11/sd/verify_and_aggregate.py `
to independently replay every saved complete coordinate gate, identity and
paired statistic; [verify_and_aggregate.py](verify_and_aggregate.py) is itself
hashed by the verification result.
[Input coordinate verification](input_coordinates_verification.json) records all
938 complete arrays across the four actual physical/numerator NPZ inputs,
including individual F-order hashes of the signed numerator and its response;
all numeric members pass a finite-coordinate gate. Its separate
[verifier](verify_input_coordinates.py) can be rerun read-only.
[EARS](ears.json) and
[documentation impact](documentation_impact.json) are the parent's trace handoff.
