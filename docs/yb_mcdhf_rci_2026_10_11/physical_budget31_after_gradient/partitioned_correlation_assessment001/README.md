# Native partitioned-correlation source assessment

SOURCE only: no implementation, NPZ access, numerical imports or tests.
[Exact source and line bindings](source_assessment.json) distinguish current
working-tree components from frozen f043; no Yb qualification transfers.
Keep common-orbital MCDHF→RCI as the main route. The parent-reviewed
[2026 PCFI paper](https://arxiv.org/abs/2609.14362) validates Li/Be/Al, not Yb;
this assessment establishes no speed or precision advantage.

Reusable components are real, but narrower than full PCFI:

- [Signed CSF vectors](../../../../src/atomx/angular/csf_hamiltonian.py#L625)
  and [CI overlaps](../../../../src/atomx/physics/csf/overlap.py) use complete
  determinant/minor contractions, including singular cross-orbital overlaps.
- [One-body nonorthogonal tensors](../../../../src/atomx/angular/csf_hamiltonian.py#L764)
  use cofactors. The transition consumer calls them directly; its comments
  about a biorthogonal transform do not correspond to an implemented transform.
- [Per-block orbital solves](../../../../src/atomx/methods/csf/solver.py#L1709)
  actually perform statistical DF and block RCI despite stale no-op comments.
  These separate J/parity blocks are not same-sector correlation partitions;
  `state_average` explicitly rejects `per_block_orbital_sets`.

Missing pieces are a partition/frozen-reference policy, complete cross-bank
scalar one- and two-body H, owned joint many-electron overlap S, generalized
CI and result/transition consumers for several banks in one physical state.
The existing optimizer requires complete same-kappa groups; `gradient_keys`
alone does not certify frozen-reference columns. Current RCI uses ordinary H,
and the effective transition operator explicitly rejects nonorthogonal banks.
If joint orbitals are optimized, qualify cᵀ(dH−E dS)c and CF response;
the existing fixed-amplitude overlap derivative is only part of this derivative.

Preserve every strong-reference CSF once. Remove a CF's reference component
in the full many-electron metric or prove exact joint dependence; different
orbital labels are insufficient. Solve one joint Hc=ESc with all signed H_ab
and S_ab, never sum partition correlation-energy corrections. Singular orbital
overlap is valid; only proved exact null directions of joint S may be quotiented,
with H-null compatibility checked. Retain near dependencies or reject by fixed
gates. Missing disconnected product excitations are a separate coverage issue;
include them or declare contraction loss and test a separable-fragment limit.

The smallest native qualification is N=2 in three s1/2 radial functions:
15 magnetic determinants, six complete J=0+ CSFs, two explicit reference CSFs
and two two-CSF correlation partitions with distinct orthonormal orbital banks.
An independent exterior-power oracle supplies S=FᵀF and H=FᵀH_common F.
Check the common-bank limit, full six-dimensional span, signs/rotations,
singular cross-overlap, duplicate columns, generalized residual and cᵀSc.
Then contract each correlation partition to one CF: compare that same
four-dimensional span and report contraction loss separately. Test small native
Dirac–Coulomb radial integrals before Yb; this proposal has not been executed.

Methods should orchestrate partitions; physics/angular should supply cross-bank
operators and identity; numerics should solve the proved metric-range problem.
Reuse existing MCDHF, overlap and radial kernels; do not add another SCF engine
or fabricate common-bank snapshot ownership. All banks need the same nucleus,
N/J/parity, radial geometry, no-pair subspace and operator coverage.

Documentation impact: only this assessment is added. Shared source, current
guides, root's full-gradient work, EARS, cards and publication remain untouched.
