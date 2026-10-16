# Independent native normal-ordered PH review

This review reproduces **every signed doubles–doubles element** left after
subtracting the projected Fock matrix and the existing PP/HH pair ladders
from the complete native linked SD equations. All three original fixed
Hamiltonians pass the unchanged `1e-11 Ha` matrix gate. The large diagonal
difference is a real normal-ordered particle–hole contribution in this finite
model, rather than Fock/HH double counting.

The inputs are the immutable [complete native SD discriminator](../sd_linked001/contract.json),
with 4 electrons, 12 spinors, all 495 fixed-N determinants and all 12 scalar
SD coordinates: 2 singles and 10 doubles. Its original Z=4, `1s²2s²` payload
and orbitals are retained; this review does not optimize orbitals, reconstruct
integrals or generate a new heavy-atom spectrum. The original `dc`, `static`
and `dynamic` Hamiltonian identities remain separate.

## Operator and independent check

With virtual indices `a,b` and occupied indices `i,j`, the independently
applied core-normal-ordered interaction is

```text
V_PH = sum_(a,b virtual; i,j occupied)
       g(a i,b j) [a†_a a†_i a_j a_b − delta_ij a†_a a_b].
```

`g` is the original antisymmetrized integral tensor. The four equivalent
mixed occupied/virtual index placements cancel the `1/4` prefactor of the
unrestricted two-body sum, giving the coefficient shown here. The subtraction
is the occupied contraction; dropping it would mix the reference contraction
into the normal-ordered two-body PH diagram.

[review.py](review.py) uses independent rightmost-first Fermi signs on the
complete fixed-N basis. It imports the native packed integral owner to read
the original payload, without importing either the linked probe's generator
or the pair-ladder implementation. The saved signed scalar vectors define
the same physical coordinates; this check independently verifies the PH
action and normal-ordering decomposition, rather than claiming a second
independent angular basis or radial integral calculation.

For each of the three unchanged Hamiltonians, it checks the full `10 × 10`
signed matrix against both the saved difference and its decomposition:

```text
PH_DD = K_DD − projected_Fock_DD − PP_DD − HH_DD.
```

| Original mode | Maximum signed difference / Ha | PH diagonal maximum / Ha | PH off-diagonal maximum / Ha |
|---|---:|---:|---:|
| DC | 1.2434498e−14 | 2.096186016 | 0.0793530494 |
| Static transverse | 5.7731597e−15 | 2.096212153 | 0.0793738341 |
| Original finite-frequency transverse | 7.7715612e−15 | 2.096212140 | 0.0793738288 |

The saved-difference decomposition itself agrees within `1.12e−15 Ha` and
the scalar metric defect is `8.88e−16`. All 300 signed entries pass. No
amplitude or integral magnitude pruning, rescaling, fitted correction, model
shrink or gate change occurs.

## Scope and interpretation

This identifies the PH work required to extend the **native PP/HH-only
subgraph** into a complete native linked SD implementation. It does not
identify a missing PH contribution in the existing complete pCI-based
production map. It supplies a concrete finite oracle for native PH and
mixed-rank implementation; it does not supply their scalable production
contractions, nonlinear CC terms, triples, valence sectors or a new solution.

The `2.096 Ha` matrix element and the original partial-action residual of
about `0.426 Ha` are equation-action diagnostics. They are not missing energy
corrections, Yb spectral errors or experimental uncertainty estimates.

The original saved `Fock_K` is `C_Q† F_N C_Q`. In the DD block it equals the
linked `[F_N,T2]` action, so the PH statement above is exact. In the full
singles/doubles matrix it does not include the noncanonical
`T1 F_N|reference>` subtraction in the DS block. Accordingly this review
labels the original all-block partial residual as **projected Fock + PP/HH
action**, rather than as a complete-Fock-commutator residual. This distinction
does not affect the already accepted complete native K/root checks.

## Evidence and resources

- [contract.json](contract.json): measurement contract, fixed original space,
  unchanged gates and complete original/source/reviewer bindings.
- [result.json](result.json): actual three-mode results, runtime thread count,
  source/input identity checks and resource measurements.
- [matrices.npz](matrices.npz): all signed independent PH and saved-difference
  matrices, with SHA-256 bound in the result.

The actual review completed in `0.04285 s` after imports and source checks;
peak process RSS was `408,006,656 bytes`. These are execution bookkeeping,
not controlled timing or a speedup claim. The declared cap was 2 GiB address
space and 120 seconds CPU/wall time; BLAS/OMP/Numba environment counts were
one and the loaded BLAS runtime also reported one thread. No external
solver was invoked. Original evidence and current production source were
unchanged; all original source snapshots also matched their bindings.

Ruff passes for the final script. The initial import-order lint failure,
premeasurement script and first contract are retained in
[failed_preflight_import_order/failure.json](failed_preflight_import_order/failure.json).
No scientific calculation ran under that first contract. Only import order
was repaired before Ruff passed and a fresh contract was declared.

To reproduce the arithmetic without overwriting the original result, use a
new evidence directory with the same parent owner layout. The original run
used:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -B docs/atomic_native_roadmap_2026_10_11/sd_ph_review001/review.py --prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python -B docs/atomic_native_roadmap_2026_10_11/sd_ph_review001/review.py
```

Both entry points reject overwriting their original contract/result.
