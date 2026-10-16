# FAC complete CI numerical precision, 2026-10-11

This panel reruns FAC locally and compares every retained Dirac–Coulomb CI
eigenvalue within its complete J/parity block. It uses FAC's exported radial
integrals first, then independently integrates its unchanged spinors through
Atomx's production radial kernels and public CI builder. It does not compare
independently optimized physical spectra or establish experimental accuracy.
The authoritative fresh run is `actual005/`; the compact consuming record is
[`result.json`](result.json).

All four cases use point nuclei, infinite mass, SE/VP/Breit off, configuration
relaxation shifts off, FAC 8000 radial points with density factor 4, and
Atomx grids of 2000 and 4000 points. `InitializeMPI(1)` explicitly establishes
the FAC serial worker identity. FAC's existing `LimitArray(0,0)` disables
compressed float Yk potentials and fitted tails; the double-precision Slater
cache remains. A fresh default-cache negative control was **not measured** in
this campaign, so no measured current-epoch improvement from that switch is
claimed. Historical cache diagnostics remain historical.

The original parent gates are absolute `1e-11 Ha` **and** relative `1e-10`.
Relative agreement does not replace a failed absolute gate. Raw/native-c and
analytic FAC-c alignment are reported separately; no fitted correction is used.

| Complete finite configuration space | Levels | Radial keys | Common-integral max Δ (Ha) | Original joint gate |
|---|---:|---:|---:|---|
| He-like Ne, all two-electron configurations through n=3 | 98 | 1419 | 1.3500312e-13 | pass |
| C: `1s2 {2s2 2p2, 2s1 2p3, 2p4}` | 20 | 34 | 5.6843419e-14 | pass |
| Ne-like Fe: ground + complete 2s/2p → 3s/3p/3d singles | 37 | 275 | 3.4106051e-12 | pass |
| Same Ne-like space for Pb | 37 | 275 | 3.0922820e-11 | **fail absolute**, relative passes |

There are 192 levels, 2003 radial integrals, and 35 complete symmetry blocks.
The Pb common-integral relative maximum is `2.2862682e-15`; its full-energy
absolute gate remains failed. Different CSF phase and coupling conventions
prevent an unqualified elementwise FAC/Atomx matrix comparison. Every signed
Hamiltonian is nevertheless archived, and the comparison uses complete sorted
eigenvalue multisets per symmetry rather than selecting favorable states.

| Case | Native-c 4000-point spinor max Δ (Ha) | Analytic FAC-c aligned max Δ (Ha) | Original joint gate |
|---|---:|---:|---|
| He-like Ne | 1.2486616e-8 | 1.2446883e-8 | fail |
| C | 1.6730908e-8 | 1.6721096e-8 | fail |
| Ne-like Fe | 1.2734813e-7 | 1.2214741e-7 | fail |
| Ne-like Pb | 2.1293272e-6 | 1.2613164e-6 | fail absolute; aligned relative passes |

Both 2000- and 4000-point results, every radial integral, raw and aligned
matrices, original supported FAC spinors, interpolated P/Q, radial grids,
Jacobians and nuclear potentials are retained. FAC has
`c=137.03599913815452`; analytic alignment uses the fixed-spinor identity
`H(c)=c K - 2 c² Q_overlap + V`. Atomx's production constant is unchanged.
The public complete CI matrix equals its independently evaluated exact blocks
entrywise in this panel and is exactly symmetric. These internal identities
do not remove the independent FAC radial gate failures. This fixed finite-grid
panel is not a continuum/radial convergence certification.

The final run read back all 670 arrays and their 635443 signed coordinates.
Its 868 frozen source/build inputs and 121 mapped runtime files remained
unchanged before/after, and were rechecked against the actual current files at
handoff. Lock waiting is included in the recorded 153.28-second lane wall time;
that value is not a solver performance measurement.

Each case's `complete_signed_arrays.npz` has an `array_inventory.json` with
shape, dtype, bytes and SHA of every decoded coordinate array;
`atomx_complete_csf_signatures.json` retains the complete coupling identity.
Every saved array was read back and checked coordinatewise. The final frozen
[`artifact_manifest.json`](artifact_manifest.json) covers successful, failed
and superseded attempts rather than deleting unfavorable observations.

FAC's actual local sources, dirty Python packaging changes, headers, configure
record, Makefiles, static libraries, private dump frontend and supplied
libgfortran are captured in `actual_sources_and_build_inputs.tar.gz` and
`source_before.json`/`source_after.json`. Actual loaded runtime files are
sampled from `/proc/PID/maps`, hashed on first observation and rehashed after
the campaign. The supplied runtime is
`/data/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu/libgfortran.so.5.0.0`.
No FAC library, Atomx production source or historical fixture was modified.
The frontend contains diagnostic photon methods too; these DC scripts never
invoke them. Its `DumpPhoton` state changes are therefore outside this panel.

The actual address-space cap is 8 GiB, worker/thread budget 8, and each FAC
child has a 600-second watchdog. FAC was observed at one thread, with inner
BLAS/Numba worker limits one. Scientific work and selected CPU regressions take
the campaign's shared `timing.lock`. FAC invocation wall observations include
orbital optimization, CI and large text exports; Atomx component observations
start after angular inventory preparation. These scopes are unequal: this
precision lane makes **no whole-solver speed or parallel-efficiency claim**.
The separate performance lane provides those declared workflow measurements.

Retained attempts:

- `actual001.log`: Python 3.10 lacked `hashlib.file_digest`; adapter failed before science.
- `actual002/`: complete successful implicit-serial numerical panel, retained.
- `actual003/`: complete successful explicit `InitializeMPI(1)` panel, same numerical values.
- `actual004/`: complete first FAC execution followed by a private CSF-signature JSON
  serialization failure; no incomplete campaign promoted as final evidence.
- `actual005/`: final explicit-serial panel with complete serializable CSF identities,
  nuclear arrays, and additional configure/package metadata.

The selected existing regressions use their unchanged archived-fixture
tolerances; see [`selected_tests.json`](selected_tests.json) and
[`selected_tests.log`](selected_tests.log). Their pass does not certify the
tighter fresh campaign gates, whose failures are separately retained above.
All 30 selected tests passed, with no skips, in 34.94 seconds. The first attempt
had 25 passes and five setup errors from Numba cache serialization
(`ReferenceError: underlying object has vanished`). That attempt is retained as
`selected_tests.failed001.*`. A fresh private `NUMBA_CACHE_DIR` resolved the
setup errors without deleting shared caches, changing sources, or altering any
fixture or tolerance.

Replay from the Atomx repository root with its existing `.venv`:

```sh
sh docs/atomic_external_benchmark_2026_10_11/fac/precision/build_frontend.sh
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE \
  NUMBA_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/data/code/atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu \
  .venv/bin/python docs/atomic_external_benchmark_2026_10_11/fac/precision/replay.py \
  --run-dir docs/atomic_external_benchmark_2026_10_11/fac/precision/new-run
```

Use a new directory; frozen historical or campaign owners must not be rewritten.
No further grid/cache/case scan is needed to report this fixed benchmark.
