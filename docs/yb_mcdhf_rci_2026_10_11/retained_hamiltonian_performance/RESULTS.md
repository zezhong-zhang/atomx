# Actual complete531 retained-H checked-chain comparison

The controlled **warm checked-chain** median ordinary/retained ratio is
**1.242593810757533** (about **19.52% lower elapsed time** implied by that median
ratio). All five pairs pass the prospective >1 individual / ≥1.05 median gate.
This result covers the declared energy trial → accepted full gradient → checked
archive chain at one native531 accepted320 point. Whole-optimizer, GPU and parallel
scaling were not measured.

| Pair | Execution order | Ordinary chain (s) | Retained chain (s) | Ordinary / retained |
|---:|:---|---:|---:|---:|
| 1 | O,R | 2.225588 | 1.784053 | 1.247490 |
| 2 | R,O | 2.209007 | 1.700746 | 1.298845 |
| 3 | O,R | 2.117229 | 1.734081 | 1.220951 |
| 4 | R,O | 2.217375 | 1.784473 | 1.242594 |
| 5 | O,R | 2.184542 | 1.825441 | 1.196720 |

Range: **1.1967198705054731–1.2988454921429593**. Ratios are computed per pair;
the reported median is the median of those five ratios. No pair was deleted,
reordered, repeated or replaced. Two explicit warmups are excluded from the
five-pair statistics and included in all call/resource budgets.

## Physical and numerical identity

Candidate source is the exact354-file frozen map
`3304f5895bcfce70708f1aa9a77bca8a7cfc3eca9a9d393380035c42ed67df2d`.
All 14 chains use the same complete531/16-sector Dirac–Coulomb Hamiltonian,
69 electrons, Fermi A175 nucleus, 24 radial orbitals, accepted320 coordinates,
five equal original tracked targets, projector and preconditioner. The actual
H is bitwise equal to the independently qualified and original full H. Saved
2856 g/pg match the original accepted320 g/pg **exactly** in every chain.

[verification.json](verification.json) independently passes **10041 checks**
without importing Atomx or evaluating new physics. It checks all14 complete
archives, full eigenpair algebra, signed cluster spans, saved original coordinates,
radial gradients, tangent/preconditioner action, all **2173 signed
public density terms**, including **1219 negative terms**, all five tracked
weights, original metric bytes, exact calls and all34 host endpoint records.
The independent audit preserves 7434 signed root-cluster span comparisons;
cluster grouping only allows numerical subspace freedom and is not an extra
physical degeneracy criterion. It imposes no fixed-column phase inside a cluster.

Maximum independent saved-array errors:

| Check | Maximum error | Original limit |
|:---|---:|---:|
| Rootwise Rayleigh energy (Ha) | 2.3647e−11 | 1e−8 |
| Centered eigenpair residual (Ha) | 2.1722e−11 | 1e−8 |
| Complete CI Gram | 2.4425e−15 | 1e−10 |
| Signed root-cluster span | 2.4417e−15 | 1e−9 |
| Independent radial → projected g | 1.5551e−12 | 1e−9 |
| Independent original inverse/tangent pg | 9.6352e−16 | 1e−10 |
| Saved complete g/pg → frozen320 | 0 / 0 | 1e−9 / 1e−10 |

The RDM audit verifies the saved complete signed contraction's coverage,
order/sign/values against the actual live qualification. It does **not** regenerate
an independent physical RDM. Full531 live correctness was separately established
by [qualification002](../hamiltonian_snapshot_qualification002/verification.json):
1491 live controls plus 3217 saved-array checks. Owner byte integrity is not
used as a substitute for the independently checked eigenpair algebra.

The original residual **1.9434848235489216e−6 still fails 1e−7**; its energy
indicator 9.225371955806855e−12 passes 1e−8. This performance experiment does not
improve that point, experimental gaps, or correlation coverage, and certifies no
stationary531 MCDHF result.

## Actual work, timing scopes and resources

All declared work completed exactly: **2 cold chains + 2 explicit warmups +
10 measured chains =14**, **21 H builds,14 CI solves,14 full gradients,0 optimizer
calls**. Ordinary chains perform2H/1CI/1-gradient; retained chains perform
1H/1CI/1-gradient. Every callback still checks the same full actual H. Retention
removes a redundant physical H construction; it removes no physical state,
angular term or gradient term.

The actual panel exited0 in **55.92856850591488 seconds**, within the300s cap,
on one CPU (29) with all numerical threads1 and a3.95GiB address-space cap.
Peak worker RSS was **898.74MiB**. All host guards passed; the original32-CPU
load denominator was retained after pinning. Both sides include identical
pre-gradient/full eigenpair validation, input hashing, full signed gradient,
tracking, original metric gates, compressed NPZ, fsync, full readback and JSON
manifest cost. Final progress bookkeeping lies outside the chain and remains
in full observed worker/panel time, as predeclared.

The following are descriptive stage medians across the same five measured pairs;
**the medians are not additive** and binding is an inclusive subscope:

| Stage median (s) | Ordinary | Retained |
|:---|---:|---:|
| Frozen-input SHA validation | 0.029160 | 0.029015 |
| Energy trial | 0.624607 | 0.681092 |
| Accepted full gradient | 0.246357 | 0.268833 |
| Trial eigenpair/Gram gate | 0.005691 | 0.006827 |
| Callback tracking + actual H | 0.628979 | 0.125454 |
| Remaining complete validation | 0.022106 | 0.023527 |
| Complete NPZ + fsync + readback + manifest | 0.383514 | 0.325590 |
| Owner binding inclusive subscope | 0 | 0.016513 |

The largest measured stage reduction is the callback tracking/H stage.
Retained trials incur extra owner work; no claim equates call-count reduction
with the full chain ratio. Phase journals and timing/bookkeeping between stages
are included in the chain (unallocated whole-chain residual medians0.2614s and
0.2579s). This is the specific checked archive workload defined before execution.

Cold observations are **11.236817s ordinary** and **10.837858s retained** for one
fresh process per mode, including imports, helper setup and full chain. Each has
an initially empty owned Numba cache; OS filesystem caches were not flushed.
These two observations do not supply a statistically established cold speedup.
The warm process took33.665604s including its own4.617767s helper setup and both
explicit warmups. Three fresh helpers each constructed four public functionals;
setup and actual constructor elapsed are recorded separately and included in
process elapsed. Setup is outside the warm chain ratios.

## Frozen evidence and documentation impact

- [Prospective contract](contract.json), [reviewed driver](run.py) and preparation
  proofs remain byte-for-byte as reviewed before execution.
- [Root execution contract](execution_contract.json), [root receipt](root_execution.json)
  and [actual result](actual/result.json) bind the execution and timing records.
- [Independent saved-array contract](verification_contract.json),
  [verifier](verify_actual.py) and [verification](verification.json) add zero
  H/CI/gradient/optimizer calls and zero performance measurements.
- [actual_handoff.json](actual_handoff.json) and [documentation_verification.json](documentation_verification.json)
  bind final prepared/actual/proof/document files. Root owns guide/registry updates.

This increment changes only the owned performance documentation/evidence.
Shared science source, tests, original point and metric, qualification and all
scientific thresholds remain unchanged. The persistent rule is to qualify the
same complete model independently before counting and timing symmetric checked
consumers, and retain each timing scope and negative scientific gate alongside
a performance result. No new GPU or parallel-scaling insight is established.
