# Frozen-point full531 / complete target-sector261 preparation

This directory prepares an exact symmetry discriminator. No Hamiltonian,
CI, gradient, or optimizer evaluation has been executed here. The
[418 passing preparation checks](preparation_verification.json) cover
JSON identities, the 353 frozen source files, complete sector selection,
prospective budgets, and fail-closed authorization controls. Both Python
drivers pass Ruff. Numerical equivalence and performance remain unmeasured.

The input is the independently verified
[run002 accepted0160 checkpoint](../public_mixed_resume_run002/checkpoints/accepted_0160.json).
Its original-metric residual is 4.717047646600387e-5, which fails the unchanged
1e-7 stationarity threshold. The energy indicator is
6.8897239137767995e-9 Ha, below its unchanged 1e-8 threshold. The two gates
remain separate; this discriminator cannot turn the failed stationarity
claim into a pass or certify experimental accuracy.

The scalar DC functional is compared with all 531 CSFs versus every CSF in
the five complete target J/parity sectors. The retained dimensions are:

| 2J | Parity | Complete CSFs |
| --- | --- | --- |
| 1 | odd | 39 |
| 1 | even | 38 |
| 3 | odd | 63 |
| 3 | even | 59 |
| 5 | even | 62 |

The 261 retained rows preserve their signed original order. Both variants
still discover all 24 radial keys and use the identical 2856-coordinate
point, original nuclear no-pair projector, and original inverse metric.
No configuration or amplitude pruning is proposed. Both use the same
original `CIBasisTracking` reference: balanced531 DC eigenpairs and target
binding, with native005 radial functions. Tracking is not rebased to the
new endpoint.

The [contract](contract.json) and [source/mathematical audit](mathematical_source_audit.json)
require whole Hamiltonian principal-block equivalence, complete retained
sector spectra, five signed physical roots or complete degenerate spans,
unchanged equal 0.2 weights, EOL energy, coherent RDM, full 24-key radial
gradient, tangent gradient, and original preconditioned residual/energy
indicator. Global CI column equality is not used for target identity.
Existing public angular-density filtering remains unchanged; shared-source
equivalence does not independently bound its physical approximation.

The prospective budget is exactly two attempted public complete CI/gradient
evaluations, plus two explicit verification Hamiltonian builds. Together
with the matrices built internally by the two CI solves, that is four
CI Hamiltonian builds. Functional construction also builds the ordinary
one-electron Dirac matrices. No optimizer or perturbed-orbital solve is
allowed. Caps remain CPU1, 4 GiB address space, and 120 seconds. Original host
affinity/load/memory are recorded before pinning to sorted affinity[-2];
fewer than three available CPUs fails admission. The continuation and static
RCI tasks use separate cores. These caps are prospective, not latency or
memory measurements and not a parallel-efficiency certificate.

`run.py` defaults to a JSON/hash-only preparation report. Its `--execute`
path requires a separate, exact root-reviewed execution binding and checks
all input/source bytes before importing NumPy or Atomx. The
[input proposal](input_binding_proposal.json) deliberately has root
authorization and root tracker attestation set to false, so it cannot
execute. An existing admission/execution record refuses repeat execution
or evidence overwrite. Missing inputs, changed source, changed point,
failed gates, and resource stops retain an unqualified outcome.

The metadata verifier's positive identity control uses an in-memory test
binding while scientific imports are blocked. It never produces a
persistent authorized contract or calls the execution function. The
original pending planning is preserved unchanged as
[contract_planning001.json](contract_planning001.json) and
[mathematical_source_audit_planning001.json](mathematical_source_audit_planning001.json).
Preparation revision 002 binds the available endpoint and uses the recorded
`ks=9`; unsupported separate `ksp/ksq` assumptions from the initial planning
are removed. The full radial geometry remains archive-hash bound.

Preparation identities are frozen as follows:

| Artifact | SHA-256 |
| --- | --- |
| contract.json | f203fb2907d6ec3abd12fded067d4f25c0c51a47d4038a0f59393a170821f222 |
| run.py | 686fd3695557aec7845a6acb4437b02eee22b7a7bb5b772886e9667380bfaab7 |
| mathematical_source_audit.json | 470e1f5011eac717ad39f30f1878cd31a416553fb94e7746d69eac08384ed31d |
| input_binding_proposal.json | ee112b14645b640cb7136756a8e6002eb6037dc688ca33502988eec328e167a7 |
| verify_preparation.py | c03988725ab56ad447e1ece8de516bc8d35f2607aad4897cc1ca55aaa754ff21 |
| preparation_verification.json | 66ccef405dae77a8ff0e9cc35ef5d85e417b82eb804aaf4f5d06d6e0530f47de |

This experiment changes representation only. It supplies no new core-double,
virtual-radial, screened-projector, Breit, QED, or recoil response, and cannot
reduce the NIST gap. The actual evidence and smallest next physics
discriminator remain in [physical_frontier004](../physical_frontier004/decision_and_prospective_experiment.json).
No old003 corrections are transplanted. A passing result would support an
exact target-sector EOL performance path at this declared point and target;
full-spectrum, global-ground, transition-operator, and expanded-space MCDHF
certification are outside its scope.
