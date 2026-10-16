# Complete target-sector equivalence: serialization successor 002

The [predecessor actual invocation](../target_sector_equivalence/failure.json)
used its two complete CI/gradient evaluations and two explicit Hamiltonian
builds, then failed the target-assignment comparison. Its 19 preceding gates
passed; saved full H/spectrum/EOL replay, principal sector H, cross-sector
zero coupling, and full/sector EOL comparison all had zero error. The failure
remains unqualified and all predecessor source, preparation, execution,
resource, and failure records are retained unchanged.

[Read-only diagnosis](failure_diagnosis.json) proves a serialization error in
the driver: frozen `CIBasisTracking.resolve` emits nested tuples, while the
checkpoint JSON contains nested lists. Python's direct comparison rejects
equal contents across these types. The original invocation did not save
actual full/sector target diagnostic values or gradients, so their equality
cannot be certified from that failure. No additional scientific call was
made for this diagnosis.

This successor changes only sequence normalization before exact target
value/order equality, and immediate evidence persistence. Changed root,
parity, or target-group order still rejects. Each completed model now saves
its eigenpairs, radial gradient, state weights, target diagnostics, coherent
RDM, and exact point before subsequent comparisons. Each explicit H is also
saved before checks. These immutable per-model files are listed in progress,
success, or failure evidence. They do not assert a gate passed before it is
checked.

The [contract](contract.json) has identical scientific conditions,
acceptance gates, and resource caps to the predecessor. It retains the exact
run002 accepted0160 point, original balanced531/005 tracker, source map
10d481, complete 531 versus 261 sectors, all 24 radial keys, original2856
metric, and equal five target weights. CPU1/4GiB/120s, deterministic original
affinity[-2] with at least three cores, two attempted CI/gradient calls,
and two explicit H builds remain unchanged. Any new actual invocation needs
a separate root-reviewed execution binding; it is an additional declared
two-call budget and never reuses or hides the first invocation's consumed
budget.

[449 preparation checks](preparation_verification.json) passed with zero H,
CI, gradient, optimizer, or NumPy/Atomx import calls. They bind all 353 frozen
source files and previous failed evidence, check unchanged conditions/gates,
and exercise the source-built tuple/list diagnosis, normalized equality,
and changed-root/order/parity rejection. Both drivers pass Ruff. The
[input proposal](input_binding_proposal.json) retains false root authorization
and false root tracker attestation, so it cannot run. The in-memory metadata
control produces no persistent execution authorization.

| Artifact | SHA-256 |
| --- | --- |
| contract.json | 2ba3c405204958a5c9e56ac4b353b79cf08c80c710b730933027640b56e85c1f |
| run.py | eeca1a408462cf01a3bb09fa9a2189546f4bc39126bdab6d8eb4f3e4437f5963 |
| verify_preparation.py | 95cc0d2350b699fb6fcdfa60289c96109bcfeb722e4a587f47c3ced3ded27b5b |
| mathematical_source_audit.json | d20810903145031e8af4eb2fa70b6a93471fea6fd39e4a540fdd17335348766b |
| input_binding_proposal.json | a7528a9163db429f4f7aaa74862979df15540a35278763918ab3027341f84b09 |
| preparation_verification.json | 46b30370b21215e4caf2f5999d74ef65c1df7a56d23a52ebbd1c5335b3ce626b |
| failure_diagnosis.json | 158733b94ea350b25c2a6e5e37303986d73ee36a24620ddcbf213102d21e4270 |

Numerical target/gradient equivalence remains pending. The original point
still fails the 1e-7 stationarity gate; no full-spectrum, expanded MCDHF,
experimental-gap, or performance certificate is implied. This experiment
adds no core-double, virtual-radial, Breit, QED, or recoil physics, and cannot
make the physical Hamiltonian complete.
