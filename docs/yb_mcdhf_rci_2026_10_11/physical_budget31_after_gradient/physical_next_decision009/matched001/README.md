# Accepted13 cached/streamed native CI comparison

This preparation binds both algorithms to the **same current dirty Atomx source**,
69-electron Yb⁺ Dirac–Coulomb Hamiltonian, A175 Fermi nucleus, original 31 radial
waves/3689 coordinates, positive-energy projector and S metric, and literal
accepted13. Complete A1595/B2574 signed CSFs and all five symmetry sectors remain.
The frozen 004 full H/E/C arrays provide an external historical control; they are
read only. No larger 4f¹² correlation class is calculated here.

`run.py` reuses the reviewed 006 preparation, NPZ/header and original-geometry
checks. Current source inventory and each imported source hash are additionally
verified. The cached operator is held only for exact comparisons against the
streamed operator. Every old signed eigenvector and every unit coordinate is
applied in batches of 32; diagonals, Hermiticity, cross-sector zeros, residuals
and full reference Gram matrices are checked. Native Davidson solves the lowest
two roots separately in every complete sector, with direct residual, energy,
signed-wave and projector gates. No partial full-spectrum tracker/snapshot is
fabricated. Timings are descriptive; this fixed-order correctness run does not
establish a speedup.

`contract.json` fixes all scientific gates and resource limits: CPU27, six thread
controls at 1, AS8GiB, inner1200s/outer1215s, four H builds and twenty Davidson
solves. Streamed builds must never call `block_terms` or `_coefficient_table`.
The frozen current-source fingerprint is
`4b39195f35c7b2affa5a4841180b2fc2eaa82dbb833694905909e33e13ff77f1`
(358 Python source files).
`freeze_source.py` performs only source hashing and contract binding; it refuses
to rebind after a root seal or an actual evidence directory exists.

Root has frozen the source; the stdlib SOURCE check passes with zero science
calls and zero NPZ loads. Numerical execution still requires a fresh exact
`root_execution_seal.json` from root.
A failure creates preserved once-only evidence; no automatic retry, tolerance,
model, solver, root, or weight change is permitted. Existing 005 derivative
failures and every earlier production/experimental limitation remain unchanged.
