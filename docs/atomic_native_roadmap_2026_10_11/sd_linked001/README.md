# Complete finite native linked SD discriminator

The [premeasurement contract](contract.json) and [actual probe](probe.py)
retain the original 4-electron, 12-spinor Hamiltonian payload in all three
available DC, static-Breit and dynamic-transverse modes. The experiment
executed the existing public native `linearized_sd.solve`, with no pCI or
EXP-T library/executable solve. No production source was changed.

This is the existing fixed four-electron reference on Be2+ DC orbitals,
not a newly optimized four-electron atom. Occupied radial keys are 1s and 2s;
the complete retained virtual multiplets are 3s, 2p1/2 and 2p3/2. The full
fixed-N representation has 495 determinants. Independent SU(2) planning
gives 2 scalar T1 plus 10 scalar T2 coordinates, and the actual signed
vectors have an identity metric to 8.882e-16. The complete model was retained.

The public solver receives the reference-first 13-dimensional projection
of the same Hamiltonian. An independent elementary Fermi action on the
complete 495-dimensional space constructs every column of
`Q(H T_mu - T_mu H)|Phi>` and the affine source. The linked energy contraction
is checked independently. The full noncanonical one-body matrix, occupied
Fock contraction and reference/frozen-core ledger remain present.

The [actual results](result.json) passed the original 1e-11 Ha signed matrix,
source and energy gates and the 1e-12 Ha retained equation root gate:

| Quantity / Ha | DC | Static Breit | Dynamic transverse |
|---|---:|---:|---:|
| Complete source error | 1.323e-23 | 1.388e-17 | 1.388e-17 |
| Complete K, all 12 columns, maximum error | 1.965e-14 | 1.432e-14 | 1.432e-14 |
| Linked correlation-energy error | 0 | 0 | 1.388e-17 |
| Independent complete retained root residual | 5.218e-15 | 3.969e-15 | 3.705e-15 |
| Actual Fov*T2 to scalar T1, maximum | 0.07733393 | 0.07735417 | 0.07735417 |
| Native linked SD total energy | -14.62902448775457 | -14.628321522255671 | -14.628321522433465 |

The actual noncanonical T1-product subtraction is also nonzero:
0.07733393 / 0.07735417 / 0.07735417 Ha. Neither control eliminates core–virtual
Fock coupling. Producer Hermitian roundoff was explicitly bounded before
the public solver's exact real-symmetric input boundary; its measured
projection is recorded separately and does not change the scientific gate.

## Distinguishing missing diagrams from space size

The original `CorePairLadders` doubles source agrees with the independent full
source to 1.388e-17 Ha. Its source, PP and normal-ordered HH subgraph is an
implemented partial operator. Even after including the **complete normal-ordered
Fock action**, it differs from the full linked equations at this unchanged space:

| Coverage discriminator / Ha | DC | Static Breit | Dynamic transverse |
|---|---:|---:|---:|
| Missing off-diagonal T2–T2 action after Fock + PP + HH | 0.07935305 | 0.07937383 | 0.07937383 |
| Maximum missing complete action after Fock + PP + HH | 2.09618602 | 2.09621215 | 2.09621214 |
| Full accepted root inserted into Fock + PP + HH residual | 0.42633733 | 0.42636493 | 0.42636495 |

These values demonstrate incomplete native subgraph coverage independently of
enlarging the orbital space. They are operator/action defects of an explicitly
partial implementation, **not** experimental energy errors, estimates of Yb
shifts, or evidence that the existing pCI full DC solver omits those diagrams.
Individual missing entries have not been uniquely assigned to named diagram
topologies. The next implementation is to complete Fock/singles/mixed-rank/ring
native contractions in this shared J-pair representation and reproduce the full
signed action before introducing scalable blocked CPU/GPU execution.

## Replay and resource identity

[Saved-array verification](verification.json) independently reconstructs each
cluster generator from its signed determinant vector, rather than the pair
stencils used in the original probe, and freshly executes the public native
solver. All three saved roots reproduce exactly; maximum K error is
1.432e-14 Ha and maximum independently replayed root residual is 3.705e-15 Ha.

The unchanged original and [archived input](inputs.npz), 354 current/source-snapshot
hashes, actual arrays and original contract are verified. Original input SHA:
`bb83f71005f99d4662554822c648c1c0658fd101cd019e1119ceae4fccc48667`.
The [archived dirty source](source/) includes the exact executed probe and all
Atomx Python source. Import-style lint findings are preserved in
[lint_import_style_failure.json](lint_import_style_failure.json); the frozen
scientific probe was not rewritten after measurement to erase them.

The actual runtime checked one BLAS, OpenMP and Numba thread. Predeclared
address-space cap was 3 GiB; numerical scratch bound was 16,738,240 B plus
64 MiB additional Python payload allowance. Actual peak process RSS was
408,006,656 B. Complete science execution took 0.5514 seconds, under the
120-second per-case and 300-second total stopping conditions. This is an
uncontrolled resource observation, not a performance or speedup claim.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python docs/atomic_native_roadmap_2026_10_11/sd_linked001/verify.py
```

This increment completes a bounded native physics discriminator. It does not
implement production heavy-atom SD, connected triples/FSCC, new NIST roots,
orbital relaxation under dynamic transverse exchange, or dressed properties.
