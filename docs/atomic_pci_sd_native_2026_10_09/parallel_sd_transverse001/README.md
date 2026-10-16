# Complete transverse-to-SD pair bridge, parallel increment 001

The new [sd_transverse.py](../../../src/atomx/methods/sd_transverse.py) supplies
a complete, immutable J/parity-coupled pair representation of the actual public
signed spinor Hamiltonian, including static Breit or real orbital-difference
transverse exchange. It carries effective one-body, Fock, frozen-core constant,
reference energy, complete orbital order, frequency/source identity and an
input hash together. Every magnetic ERI, scalar one-body element, occupied
Fock contraction and reference energy is recertified before acceptance.

This is a bounded integral/compiler and finite equation anchor. The existing
large native pCI `CoreSD`, fixed-energy `ValenceSD`, standalone
`ReducedMonovalentSD` and `Sigma2SD` diagrams remain **Dirac–Coulomb only**.
No transverse scalar has been inserted into Coulomb RINT and no frozen-core
Breit shift has been attached to archived SD roots. No highest-level or
experimental-accuracy certification follows from this increment.

The independent scalar count is reused in resource planning: grouped SU(2)
counts precede radial-pair channel materialization, angular stencils and ERI
evaluation. The complete signed magnetic tensor is never retained. Rejection
at the resource gate preserves the entire requested space. The finite bridge
has quadratic J-block storage and a complete magnetic verification cost;
it does not establish large-space contraction performance.

## Convention and consuming interface

For a radial pair in the input radial order,

```text
|ab;JM> = sum_(ma,mb) CG(ja ma, jb mb | JM)
           a†_(a,ma) a†_(b,mb) |0> / sqrt(1 + delta_ab).
V^J_(ab,cd) = <ab;JM| V |cd;JM>.
g_(pqrs) = sum_J U_(pq,ab;JM) V^J_(ab,cd) U_(rs,cd;JM).
```

The phase is Condon–Shortley CG followed by the actual input spinor creation
order. Equal radial orbitals retain only Pauli-allowed even integer J.
Arbitrary signed offdiagonals and complex common-multiplet phases are retained.
No division by electric reduced `C^k` occurs. Pair J is distinct from the
existing redundant particle-hole `rho_k` coordinate convention.

```python
from atomx.methods.sd_transverse import (
    build_j_coupled_sd_interaction, solve_two_electron_core_sd,
)

# owned_integrals is the existing complete public SpinorIntegrals or on-demand owner.
interaction = build_j_coupled_sd_interaction(owned_integrals)
state = solve_two_electron_core_sd(
    interaction, reference_identity="declared complete two-electron core",
)
```

The separate finite solver requires one complete occupied `j=1/2` multiplet.
It retains every scalar single/double in the given virtual pool and reuses
the established linked linearized equation `Q(H+[H,T1+T2])|reference>=0`.
Its full two-electron Hamiltonian uses `h_eff`, not Fock, plus the frozen
constant exactly once. Noncanonical Fock-induced singles-product subtraction
is preserved. A bounded, recorded Hermitian roundoff projection precedes the
existing real-symmetric finite solver; complex equations reject explicitly.
The J-pair integral bridge itself supports complex Hermitian integrals.

## Actual source-bound evidence

The [contract](contract.json) fixes Be²⁺, `Z=4`, `ns=30`, `ks=7`, a complete
1s core and 2s/2p₁/₂/2p₃/₂ VN virtual multiplets: 10 spinors, 45 magnetic
pairs, 17 J-pair coordinates, 57 J-block entries. All three Hamiltonians use
the same native DC orbitals. Dynamic frequencies are the actual common DC
Fock diagonal, fixed during all iterations.

[Final replay004](replay004/result.json) archives the actual dirty Python
source, input contract, complete Hamiltonian and amplitude arrays. Its
complete 120×120 three-electron matrices use an independent elementary
fermion action. The J=0 check uses the independent native jj-CSF angular
convention and shared physical CI operator. Original gates remain
`1e-11 Ha` for complete signed matrices and `1e-12 Ha` for the finite linked
SD raw residual.

| Quantity | DC | Static Breit | Dynamic transverse |
|---|---:|---:|---:|
| Full three-electron matrix error / Ha | 3.553e-15 | 1.776e-15 | 3.553e-15 |
| Independent jj-CSF matrix error / Ha | 1.243e-14 | 1.243e-14 | 1.243e-14 |
| Finite linked SD raw residual / Ha | 1.735e-18 | 3.469e-18 | 1.735e-18 |
| Core–virtual Fock maximum / Ha | 1.080e-9 | 1.272e-4 | 1.272e-4 |
| Singles-product subtraction maximum / Ha | 1.528e-9 | 1.799e-4 | 1.799e-4 |
| Finite SD total energy / Ha | −13.61411105438789 | −13.613445206352665 | −13.613445206321279 |

These linked linearized SD energies need not equal FCI or nonlinear CCSD.
Their static-to-dynamic change is a same-space finite-core result, not a Yb
excitation shift and not a correction certified for another method or state.

The [22 passing checks](checks-final.xml) combine 11 new bridge checks and
11 existing linked-SD checks. The new panel includes complex phase
interference, all magnetic M metrics, frozen-core projection, noncanonical
offdiagonals, immutable arrays, incomplete multiplet and nonspherical
rejection, and preflight-before-channel/ERI evaluation. These selected checks
require no external backend and have no skips. Initial 7/10-check panels and replay001–003 are preserved;
the first test referenced an absent field name and was corrected to the
owning `singles_product_subtraction_Ha` field without changing gates.
Replay002/003 corrected controller JSON metadata, and replay004 follows the
grouped-count preflight source change. Source changes are explicit; old
archives are not rewritten.

Reproduce with:

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_sd_transverse001/probe.py --output <fresh-directory>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q tests/test_native_sd_transverse_contract.py tests/test_linearized_sd.py
```

Each probe enforces one BLAS/Numba thread, a 4 GiB address cap and 120 CPU
seconds. Original replay003 observed 1.249 s and 391,503,872 B peak process
RSS. This includes imports, fixture formation, three integral/bridge builds,
complete magnetic/CSF checks and finite SD; source archival is excluded.
It is an uncontrolled observation, not a measured speedup. The planned
216,080 B bridge payload excludes the already-owned integral caches and
interpreter; it must not be substituted for peak process RSS.

## Next implementation boundary

The current native pCI radial service evaluates electric multipole RINT with
its angular `S/C^k` factors; its prepared angular tables and diagram topology
are DC-specific. Replacing its radial scalar by a Breit scalar cannot supply
all current-current/retardation angular channels or their noncanonical Fock
contractions.

The next reusable capability is a complete generic two-body tensor service
for the native equation contractions, with an explicit recoupling from the
normalized J-pair convention into each retained particle-hole `rho_k` channel.
It must first reproduce DC core, valence and Sigma2 maps on complete signed
vectors, then replace **every** retained interaction occurrence consistently
with the full transverse service. Core source, energy, denominators/Fock,
all linear singles/doubles diagrams, valence source/feedback and directed
screening must use the same declared Hamiltonian. Independent finite
magnetic commutator/attachment maps anchor each contraction before a
large-space CPU/GPU path. The solver must retain the source-bound original
core metric, no Coulomb-parity pruning of valid transverse channels, complete
offdiagonals and one-time normal ordering. Large Yb, production transverse
SD, transverse Sigma2 and dressed properties remain implementation frontiers.
