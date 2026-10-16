# Matched angular construction benchmark

The once-only driver compares the complete existing A1595/B2574 spaces in four
fresh workers: old, new, new, old. Every worker retains all five symmetry blocks
of each arm. The old arm injects the two literal archived functions
`_connected_pairs` and `_coefficient_table` from [the before source](../angular_before.py)
into the same current Atomx module. All other imported dependencies are shared
and bound to their actual dirty file hashes in [the contract](contract.json).

The experiment checks bitwise hashes of every candidate connection, full signed
angular table, radial-key order, retained pair coordinates, sparse CSF projection
and signed synthetic integral action. It also checks the independent determinant
bilinear and rejects an absolute-amplitude negative control. It does not assemble
a radial Hamiltonian, solve CI, load the saved radial arrays, or calculate the
larger 4f¹² model.

Each worker uses CPU 27, one BLAS thread, a 6 GiB address-space cap and a 300 s
alarm. Four fresh workers give cold in-process angular caches. Table construction
and projection times are separate. Connection pull time is measured in fixed
1024-coordinate batches, excluding downstream angular work and hashing; its
subtraction from table time is an approximate stage split. Peak RSS is the whole
worker high-water mark. Failed load guards invalidate controlled timing claims.
Any correctness, identity or hard-resource failure stops subsequent workers;
there is no automatic retry or tolerance/model change.

Root must seal the exact driver, contract and resources before execution. The
default invocation and `--prepare` use only the standard library and perform no
scientific work. The root-authorized run completed successfully; see
[results and limits](RESULTS.md), [raw receipt](actual/result.json) and
[standard-library review](review.json). Source preparation and review passed Ruff.

Documentation impact: this is an owning benchmark record, not a new public API
or source default. Root maintains the parent topic and current technical guides.
No new general insight is promoted from preparation alone.
