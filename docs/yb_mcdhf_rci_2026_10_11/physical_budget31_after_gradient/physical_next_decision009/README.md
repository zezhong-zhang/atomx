# Explicit numeric CI assembly and complete 4f12 contribution

This increment adds an explicit native `angular_storage='streamed'` mode to
`build_ci_matrix` and `exact_block_matrix`. It evaluates the same signed
Slater–Condon terms directly into determinant numeric CSR, retaining each
distinct radial value only for that immutable operator build. It avoids the
large retained angular coefficient table. The default stays `'cached'`;
existing orbital-density/gradient construction still needs angular tables.

Both modes share canonical determinant construction, normal ordering against
the common occupied core, signed pair algebra, CSF projection and span checks.
For the projected diagonal, each CSF column occupies a single occupation
pattern. Its diagonal is therefore exactly determined by that pattern's
principal determinant block. Full matrix actions still include every
cross-pattern coupling. This removes a potentially large `Hdet @ projection`
temporary without excluding terms, small amplitudes, configurations or rows.

## Qualification before physical execution

All **262 affected tests pass**: [30 new numeric-streaming controls](test_receipt.json)
and the [232-test affected regression panel](regression_receipt.json). The new
panel independently sums full Fock creation/annihilation actions, with signed
core contractions, repeated parentage, reordered CSFs, real and complex DC /
static Breit, reciprocal and nonreciprocal radial keys, c-products, adjoints,
fresh per-build caches and explicit resource rejection. Native RKB public
builder checks include operator/dense output and orbital rephasing.

The [complete current A/B qualification](matched001/RESULTS.md) passes
**523/523 live gates and 222/222 independent saved-result checks**. At original
accepted13, complete A1595/B2574 operators in all five sectors agree between
cached and streamed modes to 7.28e−12 Ha. Every owned full reference-C action
agrees with its old dense oracle to 1.82e−11 Ha. Forty selected native Davidson
roots pass the unchanged energy, signed-wave, projector, Gram and residual
gates. This is correctness evidence; the fixed execution order and warmed
caches do not establish a controlled speedup.

The complete current dirty source map contains358 Python files, fingerprint
`4b39195f35c7b2affa5a4841180b2fc2eaa82dbb833694905909e33e13ff77f1`.
[Before-source identity](source_before.json) and the before/after copies of
both modified implementation files preserve this increment's source chain.
The separately frozen004 dense oracle remains a historical source epoch.

## Complete missing physical class

The [full-class plan](full_f12_preparation001/README.md) retains all82 new NR
configurations with two4f holes and three electrons in the existing valence
pool. Together with existing A91/B100, this gives complete A173/B182 NR
models over the same31 radial waves. It adds43,112 target CSFs to each arm;
the complete target totals are44,707 /45,686. This is a complete specified
class in a finite orbital space, not all atomic correlation or a radial
convergence result.

One full B numeric operator is assembled per symmetry sector. Exact signed
CSF signatures restrict it to A, retaining all82 new configurations in both
arms. Signed owned004 accepted13 target vectors are embedded in the same
orbital basis to identify the target roots. Earlier531 tracking is retained
as ancestry; direct old-orbital coefficient overlap is not substituted.
Strict reference weight greater than one half, Gram, direct full-operator
residual and observed two-root-gap gates are declared before execution.
They qualify numerical root transport, without a rigorous omitted-spectrum
gap certificate. No public full-spectrum result/tracker/snapshot is fabricated.

The bound physical problem remains69-electron Yb⁺, Z70/A175 Fermi nucleus,
original positive-energy P, S and inverse geometry, literal accepted13 waves,
unshifted native Dirac–Coulomb, and no Breit/QED or empirical shifts. The
question is the magnitude and sign of the missing-class contribution at this
point. Absolute state shifts, excitation shifts relative to each arm's fresh
6s ground, D/P centroids and fine-structure shifts must remain separate.
No accepted13 correction is transferred to production terminal16.

At preparation, execution is bounded to three workers, CPUs28–30, one thread,
8 GiB address space and1200 s each. These are trial caps, not a proven fit or
runtime forecast. All18 subshell cases /63 cold null-space calls and full
structural pair budgets are retained. There is no amplitude screening, model
reduction, alternate backend, automatic retry or altered tolerance on failure.
The first failure stops siblings and prevents remaining dispatch.

The [complete physical result](full_f12_preparation001/aggregate001/RESULTS.md)
now passes **265/265 live gates and554/554 offline saved-result checks**. All
five sectors were calculated exactly once; all dispatched workers terminated
successfully. The signed vectors and basis signatures remain replayable.
Both arms retain all82 added NR configurations.

| Excitation shift from complete4f12 / cm^-1 | A | B |
|---|---:|---:|
| 5d3/2 − 6s1/2 | +295.315374 | +272.968804 |
| 5d5/2 − 6s1/2 | +296.739920 | +264.847148 |
| 6p1/2 − 6s1/2 | +247.497778 | +186.217151 |
| 6p3/2 − 6s1/2 | +198.358756 | +131.425616 |

Individual state energies fall by7,172–7,645 cm^-1, but most of that correlation
cancels in the transitions. Signed reference weights exceed97%; the complete
4f12 projector has2.44–2.62% probability. All twenty selected roots pass direct
full-operator residuals, maximum9.465e-10 Ha. Per-sector trials take406–469 s,
with maximum RSS2,332.070 MiB (~2.277 GiB), under the declared8 GiB/1200 s caps.
Each sector uses one complete B H, exact A restriction, zero angular tables
and two native solves. These are feasibility measurements, not controlled
speedups. [Actual dispatch record](physical_run_execution_record.json).

This quantifies the whole specified missing class at fixed accepted13. It
does not close earlier MCDHF stationarity, three005 line-FD failures or
experimental gates. Before optimizer continuation, the enlarged-state
orbital-density/gradient representation and original derivative admission
gates need qualification. No fixed accepted13 correction is transferred to
terminal16. Full-spectrum state ownership and the full-element database
remain separate unfinished targets. [Documentation disposition](documentation_impact.json)
records the guide updates; no new general insight is promoted beyond existing
exact-representation, paired-observable and source-bound measurement rules.

[Increment closure and source bindings](increment_receipt.json) retain the completed validation chain.
