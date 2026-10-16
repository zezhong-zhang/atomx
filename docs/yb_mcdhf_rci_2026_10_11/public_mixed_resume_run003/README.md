# Second bounded public531 MCDHF continuation, preparation only

This prepares a new genuine `optimize_orbitals` call on the complete native
531-CSF Dirac–Coulomb EOL objective. **No new optimizer call, CI solve, gradient
or physical H build has been executed here.** Root will inspect the concrete
driver, verifier, exact source/input bindings and readonly proofs before
sealing an execution epoch and authorizing a single run.

The previous [160-step result](../public_mixed_resume_run002/result_unverified.json)
stopped at its iteration limit. Its
[independent988-check replay](../public_mixed_resume_run002/independent_endpoint_verification.json)
passed identity controls but retained residual4.717047646600387e−5 >1e−7;
indicator6.8897239137767995e−9 Ha <1e−8 Ha alone does not certify stationarity.
All accepted EOL values decreased, minimum81 floating spacings and final1282;
the [stable increment design](../mixed_stable_increment_design/README.md) does
not diagnose numerical cancellation as the obstacle.

The [contract](contract.json), [driver](run.py),
[independent endpoint verifier](verify_endpoint.py) and
[preparation proof](preparation_verification.json) define the new boundary.
The readonly [preflight](preflight.json) enumerates all531 CSFs, max raw M=J232,
retains the original metric SHA and verifies complete saved160 P/Q are exactly
P₊x. It imports only the frozen source's configuration/angular enumeration,
not a functional or solver. Host admission is a feasibility observation;
it never establishes a speedup or controlled timing.

## Same physical problem and exact new starting point

- Consume [accepted160 JSON](../public_mixed_resume_run002/checkpoints/accepted_0160.json)
  and [complete arrays](../public_mixed_resume_run002/checkpoints/accepted_0160.npz)
  by their frozen hashes. Coordinates are authoritative; no PᵀSC recovery,
  physical-endpoint polar repair or new initializer is allowed.
- Preserve all69 electrons,24 radial spinors, ns123/ks9, Fermi A175 nucleus,
 21 NR configurations and531 signed CSFs; retain every full J/parity block.
- Use the exact original004 positive nuclear projector, radial S, inverse
  diagonal and inverse blocks. Metric remains
  `aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe`.
- Preserve the frozen balanced531 CI/native005 radial physical tracking reference,
  complete same source map10d481 and canonical EOL objective
  `ae850ddc1fa246c062ef3c130173eb1c07de0106c363c4ce87120ca1d45fa199`.
  The five6s/5d/6p targets each have weight0.2. Re-resolve this original reference
  on every trial; do not rebase it to the 160-step endpoint.
- The public accepted geometry state contains no secant pairs. The new segment
  explicitly starts fresh history10 at the identical point/objective. It is
  not a bitwise resumption of the earlier optimizer's internal history.

## Actual-entry and endpoint gates

The starting-point guard validates complete accepted flag, exact source,
preparation/execution/input manifests, actual arrays, canonical objective,
original geometry/metric and independently frozen160 replay before stepping.
The [synthetic start guards](starting_checkpoint_guard_verification.json) alter
each provenance component and content on constructed JSON/bytes only.
Runtime tuple UIDs and JSON list UIDs are compared in one complete canonical
form; group/span ordering and measured overlap diagnostics are not identities.
Every J/parity/root UID and weight remains significant. Pure metadata controls
retain complete spans and detect changed/duplicate/invalid targets. Radial keys
and layout shapes likewise use integer tuples; geometry is an exact SHA string.

At actual entry, check saved complete160 eigenpairs against saved fullH and Gram.
Enumerate and independently bind the actual signed CSF order/target payload;
construct the same native full531 functional and unchanged frozen physical
tracker. Two baseline full gradients both evaluate **exactly this same160 point**:

1. Fresh full531 CI and gradient, plus full public H reconstruction.
2. Full531 gradient using the independently validated saved160 complete CI.

Compare full H/all root energies, original metric, actual residual/indicator and
EOL against the frozen independent160 replay. Compare both complete same-point
projected gradients at1e−9 and preconditioned gradients at1e−10. The nonstationary
starting residual is retained and permitted. There is no bare005 functional,
gradient or stationarity assertion at this changed point.

One public call has explicit unpacked orbitals, `resume_state`, same
`objective_identity`, `allow_objective_change=False`, accepted-point callback,
wall budget and returned state. Original1e−7 residual,1e−8 Ha indicator and
Armijo1e−4 remain. `energy_difference=None` and explicit-increment mode remain
off. No consumer line-search or orbital-step loop is introduced.

Initial and every accepted point archive x/P/Q/fullH/allCI/original P/inverses,
actual accepted projected g/pg reused from the full gradient, and complete
source/contract/target/geometry/metric identities. Writes are atomic; latest
updates only after complete accepted archive pairs. No incomplete trial is
promoted into a checkpoint.

The separate endpoint verifier first checks checkpoint/source/input/canonical
objective and saved full eigenpair residual/Gram. It then independently rebuilds
fullH/CI, evaluates one full gradient and the original metric, verifies all
S-Gram/no-pair/target/weights, saved actual g/pg and both original stationarity
gates. No saved eigenvector-column equality is required inside degeneracies.
Upper/lower changes and transitions are reported separately relative to both
fixed531 and parent160, with the same ground state and unchanged operators.
No static-Breit spectrum is changed by this DC continuation.

## Resource and authorization boundary

Use one CPU/BLAS thread, a3.95GiB address-space cap below4GiB,160 iterations,
830s soft accepted/evaluation-boundary stop and890s hard stop. Exactly two
baseline full gradients are outside the optimizer162-call cap: maximum total
run full gradients is164. The independent endpoint process has its own one
full-gradient allowance and admission. Reaching a limit preserves the last
accepted point and nonconvergence; no automatic second solve or model shrink.

Only preparation has run. The command to inspect it is:

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/run.py preflight
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run003/verify_preparation.py
```

After root's exact review, root may seal source and create immutable
`execution_contract.json` binding driver/verifier/contract/source/inputs and
both authorization flags. Only then may root invoke `run.py solve
--execution-contract ...` and separately `verify_endpoint.py
--execution-contract ...`. Missing explicit execution contract fails before
physics. This preparer will not invoke either scientific command.

Original001/002 evidence, first different-point failure, matched old-source
reference, completed static spectra and website stay unchanged. This is a
bounded numerical continuation plan, not full-CV correlation, experimental
agreement, highest accuracy, controlled performance or already achieved MCDHF
stationarity. The current public API and `preserve-metric` guidance remain valid;
no new global rule follows from this preparation.
