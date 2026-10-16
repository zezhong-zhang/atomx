# Source-bound physical attachment matrix

The [direct complex tensor comparison](../reference_tensor_compare004/RESULTS.md)
establishes the physical 0h1p ket matrix as the raw formatted HEFF adjoint.
All four candidate transformations have nearly identical roots; only the
adjoint agrees with complete signed tensor waves, directed entries, retained
Bloch equations and four physical projectors. The independent
[producer/decoder review](../source_convention_review001/README.md) explains
the bra-first storage and LAPACK column-major indexing. No reference source
text is copied into the native physics implementation.

`methods.fscc.compute` now checks the exact five qualified producer signatures
before every 0h1p export. Missing or changed construction/writer/Slater/
eigensolver/model-vector signatures reject. The existing noncanonical Fov×T2
capability guard remains separate. `read_heff` and `effective_matrices` preserve
raw blocks; only the 0h1p `model_space_matrix_Ha` and its saved NPZ use the
immutable physical ket adjoint. Provenance binds both conventions and source
signatures. The matrix is relative to the same correlated core energy, not
total H. No Hermitian averaging, wave metric, automatic property construction
or omitted-rank certification is introduced. Closed-sector output and the
raw 1h0p matrix remain separately scoped; hole ket orientation is unqualified.

[Twelve tests](../../../../../tests/test_fscc_heff_convention.py) check the
complete saved complex wave projection, all four physical projectors and
retained/omitted Bloch rows; ten missing/changed producer-signature requests
reject before export. A live current-public replay uses all seven original
spinors, three core electrons, four active model columns and full nonzero
T3 on the same frozen complex Hamiltonian. Its physical matrix and serialized
NPZ agree with the independent expected matrix; raw blocks remain unchanged.
This toy has incomplete atomic multiplets and is not a Yb or NIST result.

Actual affected panel: **53 passed, no skips, 32.74 s** (eight files: matrix
convention, backend physics capability, signed equation audit, scope projection,
effective properties, spinor-integral FSCC, streaming integrals and the
noncanonical builder). The eligible source-bound public binary runs the live
C1/axial complex, closed and hole checks; it was not substituted by a backend
without the capability. The later new current-public complex replay makes
**54 distinct checks**. The expanded convention file itself passed 12/12 in
0.44 s; the final source epoch's durable single public replay passed in 0.43 s.
Test timing is not a performance claim. Numerical code is unchanged across
the intervening `_model_matrix` docstring clarification.

All actual runs used the existing `.venv`, CPU 26, single BLAS/OMP threads,
the isolated runtime and eligible executable
`cf230c3c8e1c70fb29cf019d794f228b4773d8ea79a498e4ee4aaf095438d7df`.
The final public replay requests one backend thread, 512 MiB tensor cap,
60 s backend/90 s outer time caps, tolerance1e−13 and raw residual bound1e−11 Ha.
Its [actual result](actual/result.json), [request](actual/request.json),
[physical NPZ](actual/model_space_matrix.npz), [raw HEFF](actual/raw_HEFF),
[log](actual/cc.log), [pytest output](public_replay_tests.txt) and
[replay receipt](replay_receipt.json) are retained. Post-test copying initially
counted pytest's numeric/current symlink aliases twice; the
[copy failure](artifact_copy_failure.json) is preserved and corrected by
deduplicating resolved paths, without another solve or changed science.

The direct comparison's matrix error is6.83e−15 Ha, retained Bloch residual
3.22e−14 Ha and four-projector defect1.99e−13. The omitted4p3h Bloch row remains
6.12e−4 Ha and full-H residual5.29e−4 Ha. Strict `FiniteCorrelatedWave` gates
therefore still reject a full-wave completeness claim. Actual MVCOEF01 bytes
were not decoded: sorted backend-row semantics are source-reviewed, while
the coefficient-conjugation check independently diagonalizes raw.T. The
interface returns a matrix, not an inferred physical wave.

Documentation disposition: update [FSCC guide](../../../../ATOMIC_FSCC.md),
[capabilities](../../../../ARCHITECTURE_CAPABILITIES.md),
[testing](../../../../TESTING.md), [migration](../../../../MIGRATION.md) and
parent navigation. Preserve all001–003 failures, source snapshots and older
raw-matrix artifacts; no archived result is upgraded. Existing principles
already require signed wave/convention checks rather than eigenvalues alone;
no new general rule is promoted. Native MCDHF→RCI remains the production main
route and EXP-T remains a comparator. [Receipt](receipt.json) binds current
source, evidence and tests.
