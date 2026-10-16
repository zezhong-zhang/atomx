# Private native MCDHF density-band executor preparation

This is source preparation for one private numerical primitive. The engine has
not imported NumPy/Numba/CUDA, compiled, allocated numerical arrays, opened NPZ,
created a device context, or run a physical calculation. No public source,
defaults, tests, or dependencies were changed. The frozen parent
[primitive selection](../PROPOSAL.md) remains historical; this directory is its
new implementation preparation, **not a CPU/GPU qualification**.

The actual native consumer is `CSFOrbitalFunctional.gradient_from_weights`
→ `_pair_potential` → `_convol_case1_semiseparable_t`. The public factors are
shared with the ordinary H consumer. This first adapter will batch **g's source
potentials only**; ordinary H, CI, angular algebra, state normalization, dense
P/Q expansion/actions and signed accumulation remain on their existing public
CPU paths. No adapter to the public functional is installed yet.

## Exact narrow interfaces

[batch_executor.py](batch_executor.py) contains four boundaries:

1. `BatchPlan` fixes the complete source job insertion order, native radial keys,
   source/workspace identities, independent P/Q orders and component extents.
   Each declared `(rank,key_a,key_b)` generates `pppp,pqpq,qpqp,qqqq` in that
   order. A later physical consumer must independently show that this declaration
   is exactly the complete normalized signed RDM source union; the engine cannot
   infer it from a label or SHA.
2. `DensitySnapshot.from_public_bands` accepts the already public `_density_sym`
   P/Q bands, checks native contiguous finite real binary64 and exact structural
   tails, then creates a byte-backed readonly host owner. The caller independently
   binds the trial manifest; the owner also hashes the actual complete band bytes.
3. `PrivateBatchExecutor` accepts the complete public five-factor tuples, owns
   their actual bytes, and retains this geometry only within the same numerical
   source epoch/workspace. Selecting CUDA is explicit. It requires root's
   separately activated existing isolated runtime; it installs nothing and has
   no CPU fallback. Internal/device concurrent mutation is unsupported.
4. `chunks(snapshot)` returns immutable four-band packets in source insertion
   order. Existing `_banded_to_full_add`, actions and signed weighted g additions
   must be applied in their original order by a future narrowly scoped adapter.
   The caller must release packets after consumption; retaining every packet is
   outside the resident streaming bound.

The scalar factory is shared between CPU and CUDA. Prefix is accumulated in
ascending j, suffix independently in descending j, with each jp's `end=ns-jp`.
Each row preserves the exact public `first_near`, before/after clamps, jp order,
falling×prefix then rising×suffix then ascending local near-strip contributions.
Each scan/row is owned by one job. CPU `prange` changes only which independent
job executes first; thread masks are restored. CUDA uses explicit `dadd_rn` and
`dmul_rn`, owned outputs, and its own stream. Inactive scratch/output padding is
reset every chunk. No total-minus-prefix, atomics, tree reduction or fastmath is
introduced. Generated-code IEEE/no-FMA evidence is still required at numerical
qualification; a source review does not supply it.

The engine accepts ns96–123, P/Q orders ≤9, real DC Coulomb ranks0–6 and at most
31 radial keys. ns<96, negative ranks, changed dtype/endian/strides, missing or
duplicate source/component declarations, nonzero RKB density tails, stale
identities, concurrent execution and over-budget models are refused. There is
no increment/change, Breit, QED, recoil, effective-operator, arbitrary angular
or complex adapter. No rank or signed coefficient is removed to fit resources.

## Preparation performed

[preparation_verification.json](preparation_verification.json) records 63
source/AST/hash/syntax controls. It binds current public source map
`bef2f5d6b66d81efb0ce54ec8b2bd37c8f40a1855b07ed6ed3057912b79a50d3`
(356 files), the private module AST and exact f043 numerical consumers.
[descriptor_verification.json](descriptor_verification.json) records 36
standard-library mock admission/refusal controls, under an import blocker.
These checks cover descriptors, complete four-component declarations, job order,
invalid domains and the resource arithmetic. They do not evaluate any floating
point scan, row, derivative, compile or runtime behavior. Ruff passes.

Reproduce **source preparation only**, without numerical imports:

```sh
python3 -I docs/yb_mcdhf_rci_2026_10_11/gpu_mcdhf_next_step208/private_batch_preparation001/prepare.py
```

The authoritative fixed boundary is [contract.json](contract.json); exact inputs
and source epoch are preserved in [public_source_identity.json](public_source_identity.json).

## Resource and transfer upper bounds

These are pure integer bounds for the declared full31 worst case, not measured
actual31 RDM counts. With 496 unique unordered source pairs, seven ranks, ns123
and both orders9, there are at most 3,472 source jobs and 13,888 component jobs.
Chunks contain at most 128 sources / 512 component jobs. Each trial has at most
28 scan launches and 28 row launches; inactive padding is still written.

|Owned/transport item|Bytes upper bound|
|---|---:|
|Resident packed complete factor geometry|38,930,976|
|Unique P/Q source density bands|8,785,152|
|Chunk prefix/suffix|9,142,272|
|Chunk four-band output|4,534,272|
|Chunk job descriptor indices|16,384|
|Device array payload total|61,409,056|
|Host payload incl caller inputs/copies/scratch/CPU dense postprocessing|196,894,048|
|All component-band D2H per trial|122,992,128|

The first implementation deliberately performs independent component scans.
Its 9,142,272-byte scan workspace doubles the frozen parent's 4,571,136-byte
estimate, which assumed shared right-component scans. No silent claim of that
deduplication is retained. Geometry is uploaded once per resident owner; the
complete unique density pool is uploaded per trial; packet D2H/synchronization
is included. Current CPU g is not device-resident, so this is not a zero-copy
whole-gradient route. Allocation/context/JIT overhead, the public H/CI/angular
workspace and actual host RSS require a separate runtime admission. The 512 MiB
owned array cap is not a bound on total process/device runtime memory.

## Pending qualification and decision

The 31-orbital fixed-point calculation in the parent contract had actual
gradient_calls=0 and did not optimize MCDHF. Native31 complete signed fullg,
constraint layout/original metric and derivative reference must be separately
bound when available. The literal208 29-orbital derivative proof cannot be used
as a 31-orbital certificate.

Root must review a later executor adapter and exact anchor/compile/transfer
call contract before any numerical factory or science run. The prospective
same-point comparison keeps all original H/root/complete eigenpair/Gram/signed
degenerate-span/RDM/g/tangent/native-pg/nuclear-P/UID gates. CPU1 is the primary
ordered numerical anchor; CPU parallel and CUDA follow independently. Public
H and CI remain CPU, and first H/CI/g qualification is not a performance test.
Any failure stops; old failures remain unchanged.

Later timing must measure the actual fresh H→g chain, including public density
construction, host packing, transfers, synchronization, CPU postprocessing and
validation/archive costs, with cold/setup costs separate. CPU parallel scaling
and GPU device-exclusive/host-load admission need their own paired measurements.
The prior warm native29 load failure still gives no controlled speed certificate;
old SD GPU evidence gives no native MCDHF certificate. No numerical or speed
claim is made by this preparation.
