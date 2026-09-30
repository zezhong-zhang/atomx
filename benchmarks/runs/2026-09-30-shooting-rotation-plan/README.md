# Shooting rotation energy: fixed Hamiltonian, explicit cache lifetime

Baseline: `af6128b`; numerical change: `b128df2`. This pass optimizes scalar occupied-rotation contractions.
The physical Hamiltonian, coherent amplitudes, grids, root selection and SCF
acceptance criteria are unchanged. Compact results are in
[summary.json](summary.json), source/output hashes in [manifest.json](manifest.json)
and warm Cf profiles in [hotspots.json](hotspots.json).

## Physics and numerical design

For fixed Hamiltonian weights the scalar energy is

\[
E=\sum_a q_a h_{aa}+\sum_t C_t R^{k_t}(a_t b_t,c_t d_t).
\]

Every signed coefficient supplied by the physics owner is retained. For a
coherent CSF state these are the full angular contraction of `c†Hc`, including
interference and odd rotation harmonics. They are not inferred from orbital
occupations. The radial numerical layer selects no averaging prescription.

During a rotation of orbitals `a,b`, densities involving either orbital must
be recomputed. All other densities are fixed only for that pair's callback.
The discrete Coulomb map uses the existing forward/backward SLAG6 integrals;
its two density arguments cannot in general be exchanged exactly on a finite
grid. The implementation sorts labels within each product density but never
interchanges the left and right densities. Coefficient order, grouped
fixed/variable summation, cumulative quadrature and floating-point arithmetic
are preserved, without fastmath or coefficient pruning.

The implementation has three explicit lifetimes:

| Owner | Contents | Lifetime |
|---|---|---|
| Physics functional | One-body weights and signed angular Coulomb terms | Fixed Hamiltonian/state |
| Numerical rotation graph | Read-only density, potential and term indices, copied coefficients | One occupied-rotation sweep; rebuilt after a state update |
| Pair callback | Independent density/potential tables and fixed energy | One pair; changing rows refreshed on every energy evaluation |

The graph borrows live orbital buffers and fixed radial geometry. Callbacks
are not reentrant and only their declared orbitals may change while in use.
No radial scratch is shared between pairs or cached across Gauss–Seidel
updates. Scratch storage is `8*N*(density_pairs + potential_requests)` bytes,
plus existing invocation-owned quadrature temporaries and small index arrays.
Closed shells with no allowed rotation pairs skip graph construction.

The ordinary HF driver now owns dense orbital buffers, so contractions can
operate directly on rows and avoid per-energy list-to-array copies. Seed and
result buffers retain independent ownership. The old per-pair construction
interface was replaced at its callers; there is no production alias module or
fallback implementation.

## Isolated contraction measurements

Fresh before/after processes use one BLAS/OMP thread. Each measurement has ten
batches of 50 operations after warmup. Atom solving and JIT compilation are
excluded. Complete sweep preparation includes graph construction and both
pair contexts; the baseline's metadata factory is only a lightweight closure,
so comparing factory creation alone would be misleading.

Both probes use 1,000 radial points. Na has five orbitals and 25 ordered
Coulomb terms; Cf has 27 orbitals and 903 coherent Coulomb terms. Each has two
allowed rotation pairs.

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| Na rotation energy | 90.19 µs | 73.99 µs | 1.22× |
| Na complete sweep preparation | 185.68 µs | 182.27 µs | 1.02× |
| Cf rotation energy | 1.03342 ms | 0.82493 ms | 1.25× |
| Cf complete sweep preparation | 10.1180 ms | 8.33297 ms | 1.21× |

All **36 sampled energies are bit-identical** between implementations
(nine angles for each of four pairs).
These are kernel improvements on one machine, not portable latency guarantees.

## Complete solve measurements

Four fresh processes ran C/Ne/Xe/Cf panels in before/after/after/before order.
Each element had four solves per process; its first solve was excluded from
the warm median. Imports and constructors are excluded. Default shooting HF,
Hamiltonian averaging and Breit + QED are used in both implementations.

| Element | Before median | After median | Change in time | Iterations |
|---|---:|---:|---:|---:|
| C | 193.20 ms | 194.70 ms | +0.8% | 7 |
| Ne | 142.98 ms | 147.39 ms | +3.1% | 8 |
| Xe | 766.03 ms | 775.58 ms | +1.2% | 10 |
| Cf | 3.35108 s | 3.32446 s | −0.8% | 7 |

**No reliable whole-solver speedup is established by this panel.** Changes are
smaller than, or comparable to, variation between fresh-process medians.
The contraction improvements affect only part of a solve. All 64 solves
converge with bit-identical total/orbital energies, iterations and result
types. C/Cf use dedicated Hamiltonian DF; MCDHF is not selected.

The first solves and source fingerprints are retained in the raw reports.
Initial angular construction and JIT/cache loading remain significant costs;
this pass does not establish a cold-start improvement.

## Verification

- **295 focused tests pass**, covering direct/coherent DF, energy derivatives,
  independent radial engines, native correlated gradients/adjoints, HF warm
  starts, core-hole stationarity, heavy roots, source solvers, existing
  external-reference controls and architectural boundaries.
- **2 fresh-process reproducibility/bounds tests pass**: three repetitions
  of the full C/Ne/Be/Cf panel and an independent bounds-enabled compilation.
- New controls include an analytic hydrogenic Coulomb integral `5Z/8`, its
  fourth-power amplitude scaling, ordered signed/odd interference terms,
  reversed density arguments, strided orbital buffers, empty graphs and
  fresh pair/state contexts. Invalid Coulomb ranks/indices are rejected before
  unchecked compiled reads.
- **26 observable arrays and all seven transition records are bit-identical**:
  native Ne bound waves, full-box response, screened continuum waves,
  continuum phases, E1/M1/E2 transitions in both gauges, and sampled CE/EII,
  photoionization and recombination cross sections.

Collision controls use B-spline bound states and shooting continua because
those coefficient-based public APIs do not accept plain native shooting bound
states. Their small channel spaces are regression controls, not converged
cross-section predictions. The local environment lacks `pytest-timeout` and
reports its two configuration warnings. This is focused verification, not a
full repository audit. Unchanged numerical outputs do not imply better NIST
agreement or additional correlation physics.

## Remaining work

The warm Cf profile drops from 4,754,722 to 4,486,823 calls. The baseline's
64,078 Python potential lookups disappear from the rotation path. Profiling
adds overhead and charges compiled work to its Python wrappers; the following
self times identify remaining work rather than predict solve latency:

| After-profile work | Calls | Self time |
|---|---:|---:|
| Local-field initialization | 1 | 0.691 s |
| Inhomogeneous orbital orchestration | 486 | 0.416 s |
| Coherent two-body source | 405 | 0.309 s |
| Rotation energy callbacks | 236 | 0.198 s |
| Scaled Green preparation | 1,529 | 0.157 s |
| Pair preparation | 36 | 0.132 s |

The repeated coherent Coulomb **source** contractions remain separate from
this scalar energy plan. A source plan must retain every differentiated
occurrence, multiplicity, sign and trial-orbital dependency. Reusing scalar
energy symmetry for the source would require the actual discrete adjoint.
Local-field initialization, source/constraint solves and Breit/QED angular
dispatch still occupy most of a heavy-atom calculation. Further improvements
need independent energy-gradient and root/observable controls, with explicit
invalidation after orbital or Hamiltonian-state updates.

## Reproduction

Select source with `PYTHONPATH` and use separate output and Numba cache
directories for the archived baseline and candidate. For example:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_df_solver.py --z 6 10 54 98 --repeats 4 --out /tmp/df-panel
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_shooting_rotation_energy.py --out /tmp/rotation-energy
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_shooting_observables.py --out /tmp/shooting-observables
```

Invoke the new rotation driver by absolute path when using baseline source;
its historical-interface dispatch is confined to the benchmark. Generated
outputs are retained locally under ignored `raw/`; `python summarize.py`
rebuilds the compact evidence when those outputs are present. Raw outputs are
not needed by ordinary package tests and are not committed.
