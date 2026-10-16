# Full531 same-point retained-H checked-chain performance

The actual controlled five-pair warm checked-chain comparison has median
**1.242593810757533 ordinary/retained**, with all predeclared gates passing.
[RESULTS.md](RESULTS.md) reports the exact timing scopes, cold observations,
call counts and remaining stationarity failure;
[verification.json](verification.json) independently passes 10041 saved-array
and statistics checks without new physics or timing. Both sides use Atomx's
native complete531 Dirac–Coulomb model at the exact accepted320 point, candidate
354-file source map `3304f5895bcfce70708f1aa9a77bca8a7cfc3eca9a9d393380035c42ed67df2d`,
the original tracker, five equal weights, Fermi A175 nucleus, 24 radial orbitals,
and original projector/preconditioner metric. No optimizer, spectrum change,
model reduction, or changed numerical gate is included.

The [independent qualification002](../hamiltonian_snapshot_qualification002/README.md)
now has 1491 live controls and 3217 independent saved-array checks passing.
Its [result](../hamiltonian_snapshot_qualification002/result.json) and
[verification](../hamiltonian_snapshot_qualification002/verification.json) are
exact inputs to this preparation. The first
[qualification failure](../hamiltonian_snapshot_qualification/failure.json)
is retained: NumPy int64 RDM metadata could not be serialized after the numerical
path, so that predecessor was never certified. The successor preserves its
science, source, gates and 3H/2CI/2-gradient budget. The original accepted320
residual **1.9434848235489216e−6 fails 1e−7**, while its energy indicator
9.225371955806855e−12 passes 1e−8. Correctness qualification and a chain speedup
would leave that stationarity failure intact.

## Frozen prospective design

[contract.json](contract.json) binds 393 immutable inputs, the exact qualification
result/proof/driver/contract, full354 source, original accepted320 archive, signed
basis/model and metric. [run.py](run.py) implements only an explicitly sealed
runtime path. Root reviewed and sealed the [actual execution contract](execution_contract.json)
after competing physical/resource diagnostics finished. Authorization was checked
before any scientific import. The original preparation files remain unchanged.

The hypothesis is that retaining the physical byte-owned full H removes one
callback H build. Complete owner/input binding can consume that saving. The
measured chain therefore includes:

1. Frozen-input identity validation and an actual public energy-only trial.
2. Full531 trial eigenpair/Gram validation against the independently qualified,
   byte-bound same-point accepted320 H, before supplying that CI to the gradient.
   This matrix is a numerical anchor and is never injected into a solve or owner.
3. The same trial CI's public full signed gradient and original state tracking.
4. Callback H from the public builder for ordinary mode, or the checked physical
   owner getter for retained mode. Actual H must equal the anchor bitwise.
5. Complete H/CI/signed density/target/full g/pg/original metric/radial Gram and
   no-pair gates, followed by symmetric complete compressed NPZ, fsync, readback,
   archive hash and JSON manifest.

Both modes archive the same numeric keys and run the same checks. Each archived
mixing matrix has its full own-H eigenpair/Gram gate. Pair comparisons require
exact physical arrays, weights, densities and gradients, without imposing
arbitrary column identity inside a degenerate eigenspace. Complete signed RDM
terms/order/values must also match the qualified live contraction; this does not
introduce a new unscreened density implementation.

## Timing and resource limits

Two separate fresh processes run cold ordinary and cold retained chains, each
with a new initially empty owned Numba cache. Their full observed process time
includes imports, identities, helper construction and archive. OS filesystem
caches are not flushed, and cold results are separate observations.

A third worker explicitly warms ordinary once and retained once, then measures
five pairs in alternating orders O,R / R,O / O,R / R,O / O,R. No other warmup or
rerun is permitted. All 14 chains count: **21 full H builds, 14 CI solves,
14 full gradients and zero optimizer calls**. Budgets count started and completed
public calls; instrumentation only forwards to the unchanged public operations.
Each worker constructs the qualified context once and reports four actual public
functional constructions and their elapsed time, separately from warm chain
ratios. Complete input-owner/geometry checks within every physical chain stay
inside its stage time. Binding capture/validate timings are inclusive subscopes
and must not be added a second time.

The prospective cap is CPU1/threads1, 3.95GiB address space and 300 seconds for
the entire sequential panel, with a 315-second external backup timeout. Host
admission requires at least6GiB available memory and load1 per original available
CPU ≤0.8 before/after each worker and chain. The denominator is the original root
available affinity, never the one pinned CPU. An exclusive resource window is a
required root seal field. A load/resource/scientific/archive/call-budget failure
stops the panel and preserves partial arrays, logs, progress and actual counters;
partial observations cannot certify controlled speedup. No failed pair is omitted
or replaced.

Warm whole-chain ratios include all stage validation, input hashes, numeric
archive/manifest operations and the phase journals between stages. Initial helper
construction is outside the warm chain but separately measured. Per-case progress
rows and final complete journals/pair summaries are outside the chain and included
in observed worker/panel wall time. Both modes use this identical accounting.
The prospective decision requires all five ordinary/retained ratios >1 and median
≥1.05, all physical/resource controls and all14 chains complete. The result reports
all ratios and their range. A negative result remains useful. No result from this
panel can establish whole-optimizer, GPU, parallel-scaling or experimental accuracy.

## Preparation evidence and commands

[preparation_verification.json](preparation_verification.json) has 397 stdlib
input/source/helper controls passing.
[readonly_verification.json](readonly_verification.json) has 836 checks passing,
including all independent qualification proof dependencies, strict authorization
refusals and the actual NumPy int64/float64/bool nested JSON roundtrip. Arrays,
including zero-dimensional ndarrays, cannot silently become JSON metadata.
Preparation imported no Atomx and executed zero H/CI/gradient/optimizer calls.
[verify_preparation.py](verify_preparation.py) is the bounded metadata-only verifier.

The [initial preparation failure](preparation_failure001.json) records a system
Python3.14.6 versus project .venv Python3.10.21 AST schema difference. The draft helper
AST digest was regenerated with the required project interpreter before freezing.
No helper source, scientific point, threshold or source epoch changed.

From the repository root, these commands only verify preparation:

```bash
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/run.py prepare
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/verify_preparation.py
.venv/bin/ruff check docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/run.py docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/verify_preparation.py
```

**Root-only runtime, after exact review/seal:**

```bash
timeout 315 .venv/bin/python docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/run.py panel --execution-contract docs/yb_mcdhf_rci_2026_10_11/retained_hamiltonian_performance/execution_contract.json
```

No execution contract is manufactured by the preparation scripts. Its required
fields are the five explicit true authorization/exclusivity/completed-diagnostic
flags in `execution_admission`, exact driver/contract/source354 hashes, four exact
qualification dependency hashes, root's chosen CPU and original available CPU
list, and the exact21H/14CI/14-gradient/0-optimizer limits. Existing `actual/`
artifacts refuse replacement; a future retry needs a newly reviewed successor.

Documentation impact: only this new directory changes. Frozen science source,
tests, implementation proofs, original checkpoints/qualifications and existing
public guides remain owned by their prior increments. Root owns current-guide
integration and actual performance execution. [handoff.json](handoff.json) binds
all prepared files; [documentation_verification.json](documentation_verification.json)
records local links, protected-file identity and preparation status.
