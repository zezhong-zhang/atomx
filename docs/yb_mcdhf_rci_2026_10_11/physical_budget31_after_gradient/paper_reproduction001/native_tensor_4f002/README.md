# Native Yb complete-4f CCSDT/EA and matched contraction performance

2026-10-15. Current owner for this continuation; the [previous16-spinor
qualification](../native_tensor_ccsdt001/README.md) remains immutable.
The user requested continued physics and performance work. Native MCDHF→RCI
remains the principal route; this is the separately authorized RCC comparison.
No cards, schedules, experimental fitting or external electronic solver ran.

Complete4f14 is now correlated with5s2/5p1/2²: **18 core electrons,50 frozen,
30 magnetic spinors**. The core and both D lines pass the declared projected
equations. Opening4f raises the two lines by **210.533942/147.061213cm⁻¹**;
it does not close the original positive experimental gaps. The matched
complete-residual kernel is **1.73938× faster**, with all measured output
coordinates bitwise identical. These are finite-space results, not full-paper,
full-core, physical full-H or experimental convergence certificates.

## Physics and identity

The same owned native ns180/order9/R500/RKB radial bank, empirical A171 Fermi
nucleus, Dirac–Coulomb integrals and original68-electron HF origin are retained.
Virtuals are the **complete6s1/2,5d3/2,5d5/2 multiplets**,12 magnetic spinors;
there are no extra radial, p/f/g/h/i virtuals, Breit, QED or recoil terms.
The frozen50-electron ledger has15 complete radial multiplets.
Recontracting newly active4f recovers the old effective h, frozen constant,
Fock and every restricted ERI within1e-10Ha; the ERIs match exactly.
See [probe contract](probe_contract.json), [operator](operator.json),
[integrals](integrals.npz) and [probe measurements](probe_result.json).
The old1324 amplitudes embed exactly; new4f rank2/rank3 off-shell maxima are
0.019525843657/0.000480099508Ha, demonstrating unsatisfied newly opened equations.
This off-shell residual is not a spectral-error bound.

| Coordinates | Full retained count | Numerical solve |
|---|---:|---|
| Core T1/T2/T3 | 216 /10098 /179520 | 6854 exact M=0/even coordinates; every189834 equation checked |
| EA R1/R2/R3 | 12 /1188 /33660 | Highest-M even sectors S3011, D3 2790, D5 2388; signed highest-weight projection |

Complete18e/19e Fock sectors would contain86,493,225/54,627,300 determinants;
they were not constructed. Dense orbital contractions remain; canonical output
accumulation avoids large free-index permutation copies, not all dense storage.

The core converged in57 residual evaluations,293.184s, with maximum complete
SDT residual **1.00018e-17Ha** and signed J+ defect7.927e-15. Correlation energy
relative to the common HF origin is−0.006696914623Ha, a change of
−0.004998305898Ha from the frozen4f core; T3 canonical norm is0.002392398313.
This core change is common to the target energies and cancels from transitions.

| State | EA /Ha | EA change from frozen4f /cm⁻¹ | Total69e energy change /Ha |
|---|---:|---:|---:|
| 6s S1/2 | −0.414176498701 | −76.980028 | −0.005349052713 |
| 5d D3/2 | −0.302169517596 | +133.553914 | −0.004389789491 |
| 5d D5/2 | −0.300924617961 | +70.081185 | −0.004678992525 |

| Transition /cm⁻¹ | Previous frozen4f | Complete4f | Change |
|---|---:|---:|---:|
| S→D3 | 24372.156946 | 24582.690888 | +210.533942 |
| S→D5 | 24708.853564 | 24855.914777 | +147.061213 |
| D5−D3 | 336.696618 | 273.223888 | −63.472729 |

[Same-state arithmetic](state_shift_review.json) uses old Mhalf target total
energies minus its own core energy to recover old EAs; no old total energy is
mislabelled as an EA. [Reporting provenance](reporting_reference_binding.json)
binds the old result to its pre-existing sealed owner receipt. It supplements
the first solve contract's omitted reporting-only reference, without rewriting
that original contract; the new Davidson contracts bind it in advance.
The shifts demonstrate state-dependent core/valence correlation in this declared
space. They do not identify a single missing interaction or make a portable
“4f correction.” The12-spinor virtual space is still very incomplete.

## Solver experiments, including failures

1. [First contract](solve_contract.json), [driver](solve_and_attach.py),
   [result](actual001/result.json), [log](solve_execution.log): all old directed
   sector matrices and independent signed-J² qualification pass; the18e core
   converges. Six smallest-real roots in the complete Mhalf/even sector hit the
   predeclared300-action cap after621.040s. No large eigenpair or line is accepted.
   This is a search-resource failure, not a failed core equation.
2. [Second contract](davidson_contract.json), [driver](targeted_davidson.py),
   [result](actual002/result.json), [log](davidson_execution.log): target-seeded
   nonsymmetric Davidson with Fock-diagonal preconditioning. Small old D3 stops
   when the normalized iteration7 correction basis has J² defect6.774e-7,
   above1e-7. No large target solve is reached. The near-converged residual
   normalization amplifies other-J roundoff; the failure is retained.
3. [New contract](davidson_J_contract.json), [driver](targeted_davidson_J.py),
   [result](actual003/result.json), [log](davidson_J_execution.log): same physical
   operator/core/ranks/acceptance thresholds. For each prescribed J at M=J,
   construct signed sparse A=J+ from [occupation-list ladders](angular.py) and
   apply P=I−Aᵀ(AAᵀ)⁻¹A to corrections. On the complete M+1 space AAᵀ=J+J−
   is strictly positive, with eigenvalue(J′+M+1)(J′−M)>0; kerA contains precisely
   the target-J highest weights. No coefficient cutoff/SVD threshold is used.
   Sparse CG has rtol1e-14,atol0,maxiter100 and explicit convergence/J+ gates.
   Idempotence and direct signed-ladder agreement are checked; the original
   unprojected Hbar equation still determines acceptance.

The J-projected trial qualifies all three old targets against independently
anchored stored dense roots within6.241e-13Ha, then converges the new S/D3/D5
in9/9/10 matrix actions. **28 search actions plus15 direct equation actions**
are distinct from the prior six-root request; this is not a paired solver-speedup
claim. Complete2/4/6 partners follow signed lowering ladders and each full34860
R3 equation is reapplied. All514 live controls pass; maximum partner equation
residual9.571e-11Ha, J² defect2.394e-14 and canonical norm defect4.441e-16.
Elapsed55.815s includes saved-core readback, small qualification, projection,
target searches and partner checks; peakRSS490.828MiB. Core solving is excluded.
The local experiment has not been promoted to a production Davidson API.

All trials used CPU27, six numerical thread variables=1, AS12GiB,
4096MiB operator admission, load1≤8 and available memory≥32GiB. First wall cap
1500s; each Davidson cap600s, unrestarted subspace48 per target. Caps and
failure statuses were not edited after execution. The signed angular operator
is numerical total J; hyperfine coupling is not included.

## Measured performance and production changes

Before→after source maps (362 production Python files each):
`a5d63536ef61f08f2a2bbfae676f9e3b037aef18abba2cf38421616930bf596b`
→`ea402a15130e5041dab452940c288804b300008b8c44dafb6e7a7d1c3f53c373`.
[Before map](before_source.json), [archive](before_source.tar.gz),
[after map](after_source.json), [archive](after_source.tar.gz) retain actual dirty
source. Only `numerics/tensor_recipes.py` and `methods/tensor_coupled_cluster.py`
change in production; all other360 Python files match the prior source.

`projection_storage='canonical'` gathers every canonical output before
accumulation/permutation, while retaining every complete einsum and nested
antisymmetrizer subtraction. Default `'dense'` remains. `solve_core` explicitly
adds exact M/parity scalar coordinates, optional Fock inverse preconditioner
and same-operator warm result. Denominators must be finite/nonzero, with no
clipping or shifts; forbidden amplitudes must be exactly zero. It always checks
all retained equations. Warm already-converged points skip an unnecessary
nonlinear invocation. [Current API guide](../../../../ATOMIC_FSCC.md).

[Profile](before_profile.txt) identifies large permutation/output copying.
[Matched contract](performance_contract.json), [measurement driver](measure.py)
and [ABBA summary](performance_summary.json) bind the identical full4f operator,
fixed embedded old T and deterministic signed EA vector, CPU27/threads1,
workspace and caches. Each cell has one warmup and two timed repetitions;
A1,B1,B2,A2 gives four warm samples per version. Old source runs from its frozen
extracted tree, current source uses canonical mode. Maximum observed load1=1.08936.

| Fixed workload | Before mean /s | After mean /s | Speed ratio |
|---|---:|---:|---:|
| All189834 core residuals | 8.871862 | 5.100589 | 1.73938× |
| Complete34860 EA action | 1.092303 | 1.055304 | 1.03506× |
| Combined | 9.964165 | 6.155893 | 1.61864× |

PeakRSS572.867–574.027→495.941–496.160MiB. All four cells' entire residual
and EA vectors match the frozen probe **bitwise**, maximum difference0.
This measures the fixed off-shell kernel, not solver/end-to-end/experimental
performance. The EA change is small. Conservative workspace admission is not
reduced to the measured RSS. Integral preparation/profiling RSS is a different
scope and is not used for the paired memory claim.

The original in-memory probe operator SHA is
`9f87af78f072430d76276719341ff21d7ba85899f8f8bc036534e70c00fed7b5`;
JSON-roundtrip producer representation used identically in A/B/solves yields
`1a553270fb97b6a9e2be32c8f8ac11f8a207330bed56ee4db9359fa699d43182`.
The current hash includes identity dict repr; tuples become lists on JSON
roundtrip. These are distinct producer identities. Full numerical equality and
explicit source/input chains, not relabelling hashes, qualify the comparison.

## Verification and remaining work

The initial canonical optimization flattened nested permutation subtraction;
[the strict tiny-complex test](initial_tests.log) caught a spurious residual
around9e-218 where cancellation requires exact zero. The implementation now
preserves nested subtraction; atol0/rtol3e-15 is unchanged. This original
1FAIL/11PASS and the initial warm-start warning remain in the record.

The current [affected panel](final_tests.log), [JUnit](final_tests.xml) has
**59PASS/2 external-only SKIP** in47.95s. It includes12 tensor cases covering
both output modes, independent arbitrary off-shell fermion/expm and directed
batched EA anchors, overlapping permutations/tiny complex amplitudes, exact
scalar-space and preconditioner/warm-result controls, plus finite CC/EA,
native integrals and architecture panels. No native scientific trial was skipped.
[Saved-data review](saved_data_review.json), [driver](verify_saved.py),
[log](saved_review.log) reapplies every core equation, all3 roots and12 partners,
checks both source archives/current362 files and original contracts/failed
results, and compares all ABBA output coordinates: **96 controls pass**.
Reusing the production action is a replay check, not a new independent full18e FCI
oracle; independent exponential coverage remains the finite tests/old sectors.

The norms above are canonical projected-R norms, not normalized exp(T) physical
states. Full-H omitted-rank residuals, left states/dressed properties, radial/
virtual/core convergence and experimental accuracy have not been computed for
this large model. The old full-H defects are historical, not transplanted here.
The paper's much broader GTO/angular spaces and additional terms have not been
reproduced. Next physics discriminator: expand complete virtual multiplets and
radial layers while preserving this source/operator ledger, first admitting and
profiling the larger contractions; then compare each state's EA and both lines.
Do not treat only opening4f as the missing experimental correction.

[Documentation impact](documentation_impact.json), [execution record](execution_record.json),
[validation receipt](validation_receipt.json) and [increment receipt](increment_receipt.json)
close this increment. Link validation covers present working-tree artifacts,
including untracked/ignored evidence; it makes no clean-checkout publication claim.
