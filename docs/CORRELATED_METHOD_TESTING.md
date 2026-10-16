# Correlated-method testing

## External accuracy and fixed-work performance

The [2026-10-11 benchmark owner](atomic_external_benchmark_2026_10_11/README.md)
contains prospective contracts, complete saved arrays, original failed runs and
paired CPU scaling for matched AMBiT, pCI-derived SD and finite EXP-T paths.
Numerical gates, experimental accuracy, timing admission and execution-thread
counts are checked separately. A pinned reference executable is rerun for
accuracy; an inherited backend does not establish an independent comparison.
Short executable timings also require checking timeout-wait polling latency;
the original observations and corrected measurement epochs are both retained.
Its verification records belong to this campaign, not the earlier production
method test counts.

The [FAC extension](atomic_external_benchmark_2026_10_11/fac/README.md)
keeps complete signed symmetry blocks and raw radial integrals, including
failed strict gates. Regressions on older FAC fixtures have their original
thresholds; they do not certify the fresh stricter comparison. Complete
four-ion strong scaling requires all process exits, all scientific arrays,
actual total-thread caps, load admission and terminal source/runtime guards.
Requested BLAS teams can create more total process threads. Private cache
failures and incomplete child lifetimes are retained and excluded from
controlled speed claims.

## Standalone monovalent SD feedback and physical-state gate

`test_monovalent_sd.py` uses the live pinned core backend and a complete common
native Be+ pool. Literal primary Eq.8/9/13 tensor contractions independently
check valence singles, energy and doubles; full Fock-space matrices with
elementary signs check noncanonical terms. An independent nonlinear root
checks the directed pencil, and explicit `(1+T_core+S_v)|Phi_v>` reconstructs
the normalized physical state. Tests retain complex signed one-body
bilinears, weak nonzero operators, actual parent/source/energy rejection and
preflight rejection before integral action. Ten tests pass with the pCI
source/compiler/runtime variables described below, without skips.

The local `atomic_pci_sd_native_2026_10_09/monovalent_sd006/` replay additionally
retains all ten magnetic states, J-squared/energy multiplet controls and eight
actual-frequency E1 cases. The ordinary attached-reference commutator failure
and its correction are preserved; full-state residual/gauge differences remain
visible. This anchor supplies finite standalone SD feedback and a physical SD
bilinear, not production J reduction, FS-CCSDT, experiment or continuum poles.
See the [authoritative contract](ATOMIC_LINEARIZED_SD.md#finite-standalone-monovalent-sd).

The 15-file affected panel initially passed 159 checks and failed three
external FSCC launches because their subprocesses could not find the pinned
`libgfortran.so.5`. Supplying that runtime through `LD_LIBRARY_PATH` passed
the three launches and repeated all ten new tests: 13 passed without skips.
Together the runs validate 162 distinct checks; both logs are retained in
005/006. Only source formatting separates the production snapshots; AST
equivalence and bitwise scientific-array equality are recorded separately.

## J-reduced standalone monovalent SD

The separate `test_reduced_monovalent_sd.py` now validates 84 native checks:
all ten M of 2s/3s/2p1/2/2p3/2 in four complete matched Be+ pools, including
streamed BLAS, virtual rotation and changed-core-projector controls. It checks
all signed reduced/direct-rho actions, independent finite feedback roots,
raw and energy residuals, pure J, unchanged fixed-energy columns after mode
interleaving, source/energy rejection and full constructor/solve preflight.
The five-file panel passes 147 checks and skips three serial-BLAS rejection
checks when that runtime variable is absent; their three-pass revalidation
with `ATOMX_PCI_SERIAL_BLAS` completes 150 distinct checks. No scientific
gate is skipped after this complement, but it is not a single all-pass run.

The stronger `reduced_monovalent006/007` normalized-state comparisons are
retained failures at 1.3624e-11 despite stricter feedback solves. The separate
008 exact-total-J oracle decomposes the **complete** fixed-M determinant
space with an independent J² matrix, retains every vector in the target irrep,
and records both unrestricted discrepancies and full unprojected raw residuals.
It does not silently drop weak coefficients or symmetrize the directed pencil.
The current finite state/E1 panel is also revalidated in `monovalent_sd008`;
006/005 remain earlier-source evidence. See the
[current reduced contract](ATOMIC_LINEARIZED_SD.md#j-reduced-standalone-monovalent-sd).

All 40 exact-J replay cases pass: maximum signed random-map difference
3.96e-16 Ha, independent unprojected raw defect 1.73e-15 Ha, exact-J root
difference 4.16e-15 Ha and normalized-state difference 1.78e-15. The unrestricted
comparison still fails its stronger state gate in two cases, reaching
2.353e-11. Its outside-J amplitude reaches the same value; within-J differences
are at most 2.17e-15. These controls locate a finite unrestricted floating
near-degeneracy effect, without erasing the negative comparison or establishing
experimental precision. The owner retains 824 complete arrays and all sources.

Private owner controls are verified separately from the public panel. The
terminal term31 GEMM and complete-image directed Davidson control pass 84
independent finite checks across the same four orbital variants and ten M,
with 144 stored arrays. Seven numerical rejection/restart checks additionally
verify a full nonsymmetric pencil, stale-image rejection, complex-branch
rejection and cap/action-budget enforcement. A distinct CPU/CUDA standalone
term2 control passes 164 independent finite checks and 16 paired feedback
roots in each of two source-bound executions; the second includes exact
term41/terma4 GEMMs. Small complete GPU roots are slower in single observations.
Complete CPU004 passes all five signed maps and is explicitly stopped before
roots to avoid contention. GPU005 passes all five paired maps and two roots,
then stops duplicate numerical16 iterations; no complete five-root claim follows.
The numerical64 finite164 panel retains all96 arrays bitwise, and a separate
complete42-matrix stress test passes3checks using dimensions beyond16. Full006
now passes all five feedback roots, all50 paired arrays and each actual complete
CPU raw/energy audit, under the original32GiB cap and physical gates. Its frozen
nominal experiment passes only one of four unchanged1% excitation gates and
fails both splitting gates. A separate complete physical-norm primitive passes
840 finite fermion-expansion/component/M gates; six cap/type/complete-tail/cache
checks join the7 numerical and3 wide-subspace checks in a16-pass private panel.
Complete-Yb norm consumption now passes all18M, measured bitwise current-core
seed identity and381 source/33 payload postchecks, tracked in the
[current owner](atomic_pci_sd_native_2026_10_09/README.md). These private controls
do not change defaults or certify properties, full-H or experimental accuracy.

## Integral-driven CCSD/FS-CCSDT electronic gate

`test_spinor_integrals_fscc.py` checks signed packing, phase covariance, native
frozen-core projection, complete virtual bands, hash-checked replay and resource
rejection. Set `ATOMX_EXPT_EXECUTABLE` and the built backend library environment
to run real attachment/detachment CCSDT against complete CI, multiple tiles and
complex occupied/virtual rotations. Otherwise this external gate is explicitly
skipped. See [backend setup](ATOMIC_FSCC.md) and [measured runs](atomic_allorder_fscc_2026_10_09/README.md).

`test_streaming_spinor_integrals.py` additionally compares complete signed
packed/exported ERI bytes, spherical/frozen-core contractions, owned snapshots,
cache capacities 1/16384, complete energy bands and pre-export resource rejection.
Its live gate compares full signed FS-CCSDT effective matrices after streamed
input. Include `test_ci_radial_snapshot.py` when changing the bounded shared
radial caches. [Streaming/tail measurements](atomic_streaming_tail_2026_10_09/README.md)
keep native construction/export performance separate from tensor-solver timing.

## Native scalar linearized SD

`test_linearized_sd.py` checks the linked equation matrix against independently
assembled full determinant commutators at noncanonical orbitals, arbitrary
signed rotations and reference phases. It checks separated-fragment
extensivity, independent weak-coupling MP2 limits, exact scalar counts
against full jj-CSF enumeration and brute radial-pair counting, native Be
DC/DCB operator coverage, singular/invalid inputs and whole-space preallocation
rejections. [Executed reference and native probes](atomic_reduced_sd_2026_10_09/README.md)
retain pCI angular phase checks and Be/Ne/frozen-shell equations. A small-space
or count gate does not establish full-Yb SD or experimental accuracy.

`test_cell_coulomb_features.py` independently compares signed PP/QQ/mixed
quadrature, unequal kinetic-balance component orders, complex C-products and
an analytic radial integral against the public cell kernel. The features are
already weighted bilinear actions; multiplying Gauss weights again is wrong.
`test_core_sd.py` checks reduced angular phases, complete raw/resource counts,
the shared immutable virtual band and live case-specific original, compact and
streamed pCI builds. It also checks unchanged signed results when stage logging
is enabled and rejects concurrent streamed use of an incompatible serial BLAS
before building or allocating the native problem.
Set `ATOMX_PCI_SOURCE`, `ATOMX_PCI_COMPILER` and `ATOMX_PCI_RUNTIME` to the pinned
reference/toolchain paths to execute that backend gate. It compares native
matrix-free amplitudes, energies and raw residuals against the independent
finite J=0 commutator solve, including all noncanonical Fock terms.
Set `ATOMX_PCI_BLAS` to a source-bound LP64 OpenMP OpenBLAS and pin
`OMP_NUM_THREADS=1` for threaded streamed calls; `ATOMX_PCI_SERIAL_BLAS`
executes the explicit rejection control. Build/load identity checks guard
against an older serial library being selected by `LD_LIBRARY_PATH`.

The [direct-core continuation](atomic_pci_sd_native_2026_10_09/README.md)
records 97 affected tests passing with live pCI and EXP-T backends before the
subsequent runtime guard, plus 10 live core tests with the OpenMP BLAS identity.
The later singles BLAS/polynomial-coordinate increment initially passed 42
core/radial checks with four invalid option combinations skipped. The current
valid-combination panel passes **43 checks with no skips**, including live pCI
builds, memory rejection, finite-input native-output overflow rejection and a
nonzero action counter. The broader dependency panel passes **120 checks** with
live pCI/EXP-T at its archived source; it predates the final finite/source-identity
guards. Logs and exact source dispositions are retained in the continuation.
Native convolution, all four isolated component terms and the origin-cell
analytic integral anchor the exact polynomial density transform. Full Ne signed
actions/roots, 256 Yb native radial values and an all-68-core Yb finite-virtual
root checked with the original full action have separate workload evidence.
Independent rotated Be/Ne, d/f and unequal-order kinetic-balance amplitudes
also pass. Complete Ne s/p/d/f/g virtual bands establish equation/workload
coverage; their remaining partial-wave changes fail physical convergence.
The full retained Yb run and valence spectra require their own gates.


The native owner additionally contains isolated live valence controls:
`probe_valence_control007.py` compares the original pCI fixed-energy map with
complete magnetic actions, an independent root and re-solved state derivatives;
`probe_sigma2_control002.py` covers all six finite s/p valence columns and compares
65,536 signed screening elements with an independent Eq.38 contraction.
Both completed successfully at their archived control-source snapshots. These
checks use an explicitly canonical validation Hamiltonian; they do not test a
public production valence API. Original/paper redundant numerator gauges and
S1 radial normalization are audited before any physical comparison.
`probe_sigma2_control003.py` also tests the joint two-energy derivative against
an independent analytic contraction; `check_sigma2_joint_derivative.py` verifies
all 65,536 signed derivatives against six fully re-solved controls at three
predeclared steps, with bitwise fixed-input and compiled-source identity checks.
Original values remain bitwise unchanged by the joint-derivative wrapper.

`test_valence_sd.py` exercises the public fixed-energy interface with live
source-bound builds: complete angular/resource planning, NX>NXX core COMMON
reconciliation, full implicit response and re-solved differences, interleaved
core action restoration, immutable outputs, wrong physical/source/core-amplitude
rejection and preallocation guards. The final source-bound panel passes three
tests with no skips; two earlier planning-fixture failures remain archived.
`probe_public_valence002.py` independently compares the optimized public solver
in a rotated four-electron native DC space, all six s/p columns and 16 magnetic
states. Full actions/roots/responses pass; five public output arrays are bitwise
identical to the preceding public source snapshot. This is a finite equation
anchor, not a large-valence performance or physical convergence result.

The same owner retains noncanonical Fock primitive controls001/003 with complete
two-/four-electron cores and all s/p magnetic columns. Independent signed and
actual-fermion checks pass; failed002 records an oracle triangular-core indexing
error. Complete solved-response controls004–008 retain their failed gates.
Control009 preserves identical float64 physics and original gates while using
extended-precision independent contractions and actual-operator LU refinement;
all six columns/16 magnetic states and three predeclared re-solved differences
pass. Conditioning, forward error and residual remain separately reported.
`sigma2_fock002` independently checks all 65,536 signed noncanonical screening
values/joint derivatives and 14,400 physical pair elements. Its irreducible
fermion projection subtracts independently computed embedded Sigma1 at the
same denominators. The failed unsplit001 is retained. These arbitrary complete
input-direction gates do not yet establish solved noncanonical Sigma2.
Full retained Yb remains an active workload; no new optical precision is claimed.

`test_sigma2_sd.py` checks complete reduced planning and source-bound live
screening with NX>NXX, raw supplied-root/response rejection, COMMON restoration
under interleaved core calls, immutable results, signed magnetic label/query
contracts, fully re-solved joint differences, missing-column/response identity
and complete preallocation caps. The combined public screening/valence panel
passes **six tests with no skips** at its archived source; the initial screening
method-role binding failure is retained. `sigma2_solved005` independently checks
all 65,536 directed values/derivatives across 25 full solved cases, including
all 16 magnetic states and a full baseline public magnetic-query comparison.
High-order three-step re-solves pass without changing the original gates.
Complete arrays remain bitwise equal after checked packed-index reuse; no
controlled timing or physical-convergence claim is made. The same-case directed
operator audit separately reports particle covariance and nonzero adjoint
defects; it supplies no Hermitian CI or physical-energy validation.

The current native owner also records [32 affected public core/valence/Sigma2 checks](atomic_pci_sd_native_2026_10_09/public_single_to_doubles_pytest002.log) with no skips, including all three concurrent serial-BLAS rejection modes and exact term31/32 finite signed CSF gates. The subsequent [complete iterative-seed core panel](atomic_pci_sd_native_2026_10_09/core_warm_start_pytest001.log) passes 26 checks without skips. Both retain actual launch sources. Public full-414-virtual term31/32 replay passes every original-action and stimulated-single coordinate; full run008 raw core residual is 4.44e-13 Ha. These are equation gates, without radial convergence or experimental spectra.

The native owner retains [10 no-skip original/optimized valence and screening checks](atomic_pci_sd_native_2026_10_09/valence_static_pytest002.log) after separate complete valence dimension binding and serial-static compilation. The full-Yb native entry fault is diagnosed independently with actual gdb and full unchanged-space inputs. Public optional exact axial FSCC uses `ATOMX_EXPT_AXIAL_EXECUTABLE`; its [19 no-skip checks](atomic_pci_sd_native_2026_10_09/axial_public002/pytest.log) include complete signed native/complex matrices, FCI, all supported sectors and exact-zero/sidecar export guards. Actual cases and launch source are retained; these do not establish production-space accuracy or speedup.

The [source-separated valence response panel](atomic_pci_sd_native_2026_10_09/valence_response_split001/pytest.log) passes10checks with no skips, including original full derivative and re-solved joint screening gates. The [independent 16-magnetic-state replay005](atomic_pci_sd_native_2026_10_09/public_valence005/result.json) passes unchanged signed gates. A stronger .61rad changed occupied reference fails the raw response gate; [same-input conditioning/70-digit coefficient-matrix diagnosis](atomic_pci_sd_native_2026_10_09/valence_condition006/result.json) retains the failure without substituting a high precision native operator. Exact term2 batching passes the complete original68core/414virtual signed/analytic-response gate003. The public optional serial/8-worker paths pass [18 affected checks](atomic_pci_sd_native_2026_10_09/valence_term2_public_parallel007/pytest.log) without skips and [independent six-column/16-magnetic-state equations](atomic_pci_sd_native_2026_10_09/public_valence008/result.json). The [complete eight-worker replay](atomic_pci_sd_native_2026_10_09/valence_term2_parallel_full006/post_binding_check.json) retains bitwise equality with the serial input/output arrays. [Three alternating warm pairs](atomic_pci_sd_native_2026_10_09/valence_term2_paired008/result.json) pass the predeclared load and full signed gates: term2 one/eight-worker median speedup4.84, range4.82–4.91. The production003 archive retains its certified6s column; fresh004 continues p/d/screening. No end-to-end or experimental claim follows.

The [explicit response-policy public panel](atomic_pci_sd_native_2026_10_09/valence_response_policy_public016/adoption_gates.json) passes22 checks without skips across four backend modes; homogeneous values and full responses share the physical action. [Independent replay009](atomic_pci_sd_native_2026_10_09/public_valence009/result.json) retains all6 radial columns/16 magnetic states and unchanged signed gates. [Complete Yb mode recovery014](atomic_pci_sd_native_2026_10_09/valence_term2_policy_full014/post_binding_check.json) retains all866 radial arrays and original signed values/responses bitwise. [Three alternating policy pairs015](atomic_pci_sd_native_2026_10_09/valence_term2_policy_paired015/result.json) pass numerical and load guards (1.34–1.60, median1.36) for warm value requests at fixed8 workers, including C copies. Full response, full solves, radial/space convergence and experiment are separate gates; the production003 archive retains its original full-response source; fresh004 uses the current kernel.

The [complete checkpoint panel](atomic_pci_sd_native_2026_10_09/valence_checkpoint_public017/pytest.log) passes26 checks without skips across all four original/optimized backends. Perturbed complete real numerator/response guesses are re-solved and retain current raw/signed gates; wrong shape, complex, nonfinite or unrequested-response guesses reject. Same-input private mixed-width native [arithmetic controls](atomic_pci_sd_native_2026_10_09/README.md#强参考响应的实际算符精度控制) retain public double/80-bit reference failures and compare actual113-bit native contractions, without changing physical coverage or claiming scalable/public extended precision. Production full003 stopped after a certified6s column; [fresh004](atomic_pci_sd_native_2026_10_09/yb_full_valence004/contract.json) binds the unchanged archive and re-certifies complete checkpoints before continuing.

Private [GPU arithmetic controls](atomic_pci_sd_native_2026_10_09/README.md#gpu实际径向批量控制) now retain complete59s/3481-pair/ranks0..7 bitwise density outputs, full68core/414virtual term2 values/analytic responses and mode recovery, and full414virtual whole-GPU projection outputs. Both complete eight-worker CPU/CUDA pairings fail the predeclared load guard, so no controlled full-map gain or public/production GPU adoption is claimed. All physical inputs, full signed outputs, actual CUDA runtime and failed builds remain replayable in the owner.


The [restart callback panel](atomic_pci_sd_native_2026_10_09/valence_iteration_public018/pytest.log)
passes26 affected checks with no skips across original/optimized backends:
complete immutable restart guesses, fresh signed/root/response certification,
callback consumer failure and malformed input rejection. The separate
[atomic persistence panel](atomic_pci_sd_native_2026_10_09/valence_restart_storage019/pytest.log)
passes five private checks, including actual interrupted native solves and
corrupt/changed-identity/uncommitted archive rejection. These establish finite
restart integrity; older loaded production004 cannot acquire the new callback.

The [current core certificate panel](atomic_pci_sd_native_2026_10_09/valence_core_certificate_public021/pytest.log)
passes56 core/valence/Sigma2 checks with no skips, including four live backend
modes, unchanged producer reuse, reconstructed-result recheck and changed
source/denominator/amplitude/scalar/provenance/action rejection. The initial
[020 run](atomic_pci_sd_native_2026_10_09/valence_core_certificate_public020/pytest.log)
retains53 passes/3 missing-serial-runtime skips; its
[runtime complement](atomic_pci_sd_native_2026_10_09/valence_core_certificate_public020/serial_runtime_gates.log)
passes those three checks. The later021 run includes the scalar-identity guard
and supplies both real serial and OpenMP runtimes.
The [current-source private GPU replay](atomic_pci_sd_native_2026_10_09/public_valence011/result.json)
passes6 radial columns/all16 magnetic states against independent long-double
actions, roots and analytic responses in the declared rotated C2+ space.
These finite source-bound gates do not certify production spectral accuracy or
an end-to-end GPU speedup.

The proposed GCROT inverse's [022 panel](atomic_pci_sd_native_2026_10_09/valence_recycling_public022/pytest.log)
passes60 finite checks without skips, but the independent strong-reference4s
[GPU012 run](atomic_pci_sd_native_2026_10_09/public_valence012/failure.json)
fails the unchanged raw gate. The option was withdrawn; current public source
and tests were restored exactly to021. Retain the [source restoration](atomic_pci_sd_native_2026_10_09/valence_recycling_public022/withdrawal.json)
and negative result; the passing candidate panel is not public adoption.

The [full-Yb additive CUDA control002](atomic_pci_sd_native_2026_10_09/cuda_valence002/result.json)
passes complete affine/homogeneous/value/analytic-response and mode recovery in
all five s/p/d columns with68core/414virtual, maximum signed error1.388e-17 Ha.
These are complete maps, not fresh full production solutions or controlled
performance. After current-source independent011 and56 no-skip public checks,
[fresh private CUDA production005](atomic_pci_sd_native_2026_10_09/yb_full_valence005/contract.json)
has completed all five original fixed-energy roots/responses and directed
screening/joint response with certified GMRES; its [complete result](atomic_pci_sd_native_2026_10_09/yb_full_valence005/result.json) retains the declared model and gates. A [full64-coordinate recycling diagnosis023](atomic_pci_sd_native_2026_10_09/valence_recycling_diagnosis023/result.json)
reproduces retained-image failures in both dense and native operators; refreshing
actual C=A U before each inverse passes this finite raw response gate. This is a
private numerical follow-up, without public or production adoption.

The private recycled inverse requires two complete MGS passes, after the
retained012/013 failures. [Finite027](atomic_pci_sd_native_2026_10_09/valence_recycling_reorthogonal027/result.json),
[all four backend modes028](atomic_pci_sd_native_2026_10_09/valence_reorthogonal_private028/pytest.log)
(25 passes, no skips), and [independent014](atomic_pci_sd_native_2026_10_09/public_valence014/result.json)
pass unchanged gates. [Handoff015](atomic_pci_sd_native_2026_10_09/public_valence015/result.json)
separately re-certifies each complete private guess with the actual public
GMRES solver; no private result is relabeled for Sigma2.
[GPU resident-vector002](atomic_pci_sd_native_2026_10_09/cuda_krylov002/result.json)
passes the retained complete64-dimensional matrix/root-response arithmetic
and caller allocator restoration. It is a finite matrix gate, with an owned
8GiB numerical pool plus the declared2GiB diagram pool.
[Resource/failure003](atomic_pci_sd_native_2026_10_09/cuda_krylov003/result.json)
rejects insufficient memory before pool allocations and propagates callback
failure while restoring caller allocator/default-pool state.
[Current-source independent017](atomic_pci_sd_native_2026_10_09/public_valence017/result.json)
passes all six radial/16 magnetic equations and actual public recertification.
Full Yb production008 and controlled end-to-end timing remain distinct
acceptance steps. The public021 solver remains unchanged.


The source-bound [finite stage019](atomic_pci_sd_native_2026_10_09/public_valence019/result.json)
retains all six radial/16 magnetic independent action/root/response gates;
complete homogeneous decomposition differs by at most3.968e-17Ha.
[Full stage003](atomic_pci_sd_native_2026_10_09/cuda_valence003/result.json)
re-certifies the current complete core then reconstructs all five original
68core/414virtual homogeneous columns with maximum6.133e-19Ha signed error.
It validates actual COMMON extents before reading arrays. Legacy component
response timings are individual observations, not additive value-map times.
[Exactly reachable GPU projection002](atomic_pci_sd_native_2026_10_09/batched_projection002/result.json)
passes retained original CPU full414virtual term2 values/analytic responses
and full/value/full recovery; this frozen-field replay is not a fresh solve.
The complete callback's [independent020](atomic_pci_sd_native_2026_10_09/public_valence020/result.json) passes all six radial/16 magnetic equations and public roots, with maximum signed response1.558e-12Ha against unchanged1e-11. All-projection001 is
retained but performs3.182 times the exact-reach projection multiplication work;
neither projection candidate has controlled timing or public adoption.
[Production005 6p3/2](atomic_pci_sd_native_2026_10_09/yb_full_valence005/valence_6_-2.json)
passes actual raw root/response3.070e-15/1.631e-15Ha in183 matvecs;
all five original columns and directed screening are now complete in005/006;
resident GPU008, physical convergence and controlled performance remain separate.


The private [term41 BLAS control021](atomic_pci_sd_native_2026_10_09/public_valence021/result.json)
retains all original angular/denominator rows and replaces only the terminal
complete value/response matrix sums. All six radial/16 magnetic independent
and public root gates pass. [Full same-input five-column comparison004](atomic_pci_sd_native_2026_10_09/cuda_valence004/result.json)
passes every original homogeneous component/map array, maximum5.319e-19Ha;
production roots and controlled timing remain separate gates.
Production005 now also certifies5d3/2 raw root/response1.519e-15/1.027e-15Ha,
95 actual matvecs. Final5d5/2 also passes raw root/response1.660e-15/1.372e-15Ha
with95 actual matvecs; all five columns and directed screening/joint response now pass in005 and006.
CPU-vector MGS006 produces numerical guesses then re-certifies every column
through actual public GMRES. The [complete comparison](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_006.json)
passes original signed gates: columns1.348e-12Ha maximum, Sigma2 value/joint
response1.123e-15/3.635e-14Ha. Different source epochs remain explicitly bound.


[Complete P/Q singles023](atomic_pci_sd_native_2026_10_09/public_valence023/result.json)
keeps original angular rows/native RINT, full P/Q and derivative intermediates,
and passes all six radial/16 magnetic independent action/root/response gates.
The22 source-boundary failure precedes candidate arithmetic and is preserved.
Four complete buffers add8,007,680B in full Yb, predeclared before loading.
[Full five-column comparison005](atomic_pci_sd_native_2026_10_09/cuda_valence005/result.json)
passes every retained original affine/homogeneous/value/analytic-response and
restoration array, maximum8.674e-17Ha. This is an arithmetic gate, without
controlled stage/production speedup or public adoption.

[Fused angular024](atomic_pci_sd_native_2026_10_09/public_valence024/result.json)
passes all six radial/16 magnetic independent action/root/response gates.
[Full frozen6s003](atomic_pci_sd_native_2026_10_09/batched_projection003/result.json)
retains the original CPU term2 value/derivative anchor and all22538 signed rows;
complete output disjointness is checked before CUDA accumulation. [All-five
frozen fields001](atomic_pci_sd_native_2026_10_09/fused_full_fields001/result.json)
passes original GPU values/analytic responses/mode restoration, maximum5.421e-20Ha,
using the full CPU/GPU map002 anchor transitively. Neither frozen arithmetic
nor the finite solve panel replaces fresh full-map, production or timing gates.

[Resident GPU production008](atomic_pci_sd_native_2026_10_09/yb_full_valence008/result.json)
and [fused/GEMM production012](atomic_pci_sd_native_2026_10_09/yb_full_valence012/result.json)
now complete every original fixed-energy column and directed screening/joint
response. Actual public GMRES independently certifies every private guess.
[Full005/008](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_008.json)
passes signed column maximum1.330e-12Ha, screening value/joint response
1.118e-15/3.626e-14Ha. [Full005/012](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_012.json)
has column maximum1.735e-17Ha and bitwise identical screening arrays.
The comparator now checks full root and screening archive extents before
allocation and immutable screening hashes; [resource/refusal controls](atomic_pci_sd_native_2026_10_09/comparison_archive_resource001/result.json)
and [complete005/006 replay](atomic_pci_sd_native_2026_10_09/yb_full_public_screening005_006_v2.json) pass.
[Transfer-inclusive term2 timing001](atomic_pci_sd_native_2026_10_09/fused_term2_pair001/result.json)
passes four complete original-CPU signed warmups at2.982e-19Ha; unrelated GPU
contexts fail the declared exclusive-device guard, so paired timings were
skipped and controlled speedup remains null. Physical space/metric/spectral
convergence, experiment and public GPU adoption remain unverified.

[Compact complete EXP-T block storage002](atomic_pci_sd_native_2026_10_09/expt_compact_index002/result.json)
and [exact nonempty-axis construction003](atomic_pci_sd_native_2026_10_09/expt_compact_index003/result.json)
pass all native/complex C1/axial fresh and seven-diagram restart gates with
bitwise original signed HEFF and independent three-electron FCI error<=4.45e-14Ha.
Duplicate keys, overflow and legacy files reject. The001 driver argument
failure remains archived. [Actual optional public CLI build](atomic_pci_sd_native_2026_10_09/expt_compact_public_build001_source_proof.json)
reproduces validated003 engine source. [Public source-bound002 panel](atomic_pci_sd_native_2026_10_09/expt_compact_public002/result.json)
passes12 checks without skips, preserving original physical/raw gates and
complete package/test/tool sources before and after. Default dense layout
is unchanged; experimental compact diagrams explicitly useATXDG002. This
adds verified storage coverage, without a large-space FS-CCSDT or timing claim.

The private [Numba dual-target controls](atomic_pci_sd_native_2026_10_09/README.md#numba-cpu与cuda共享数值核)
retain complete source-bound density arrays for all eight ranks, CPU1/CPU8 and
CUDA signed values, fixed-geometry density directions, resource/layout refusal,
caller thread-mask recovery and actual no-FMA/no-atomic PTX audits. The full
five-column frozen term2 adapter checks identical device pointers and explicit
nondefault-stream ordering through complete values/analytic responses/mode
recovery. The existing independent cell-feature panel passes22 checks without
skips; its native convolution/analytic references remain the finite physical
anchor. Density directions, full frozen diagram responses, new whole solves
and experimental spectra remain separate gates. Reproduction and caps are in
the owner; a failed exclusive-device guard prohibits a GPU timing ratio.

The [signed transverse panel004](atomic_pci_sd_native_2026_10_09/transverse_integrals004/result.json)
passes 81 checks without skips, including complete static/finite CI matrices,
frozen-core full-determinant projection, owned energy/orbital inputs, bounded
cache eviction, packed/stream replay and actual C1/axial FS-CCSDT. All eight
low model roots in the Z=4, ten-spinor, 120-determinant three-electron oracle
pass the original 1e-9 Ha FCI gate; saved reconstruction measures 8.729e-10 Ha.
The complete directed C1/axial effective matrices differ by 8.957e-21 Ha.
Residual bounds about 1.786e-12 Ha apply to retained amplitude equations,
not the full FCI root defect. The [110-check native DC panel005](atomic_pci_sd_native_2026_10_09/transverse_native_dc005/result.json)
also passes without skips. Producer/runtime failures001/002 and the003 skip
remain archived;004 supplies both actual executable environment variables.
These gates validate the fixed real transverse Hamiltonian, not its orbital
relaxation, large-space reduced SD, production FS-CCSDT or experimental quality.
The [paired Yb frozen-spinor factor003](atomic_pci_sd_native_2026_10_09/yb_dynamic_breit003/result.json)
passes numerical/static-zero/state gates but fails its original5% experimental
goal; its excitation increments are not added to archived DC-SD roots.

The subsequent [current strict-target panel009](atomic_pci_sd_native_2026_10_09/transverse_precision_gate009/result.json)
has six original passes, one strict expected failure and no actual skips.
The tighter 1e-10 Ha noncanonical 0h1p FCI target remains failed even after
[same-H raw tolerance control007](atomic_pci_sd_native_2026_10_09/transverse_precision007/result.json)
reduces the equation bound to2.035e-15Ha. [Consistent Fov interventions008](atomic_pci_sd_native_2026_10_09/transverse_fov008/result.json)
associate the discrepancy with these terms without identifying a specific
missing diagram. The two docstring changes after004 preserve every executable
AST node; the current test only appends the stricter target. Original source
archives and negative targets remain distinct; that original panel did not
certify a general noncanonical attachment repair.

The subsequent [complete amplitude audit and candidate](atomic_pci_sd_native_2026_10_09/parallel_fscc012/README.md)
identifies a duplicated open Fov×T2 permutation in `dgd2b_2`. Removing only
the extra permutation passes nine finite C1/axial, real/complex, DC/static/dynamic
and declared changed-Fov controls. For the original H, full HEFF error is
4.219e−15 Ha and complete 120×8 Bloch residual is 1.776e−15 Ha, with the
original 1e−10/1e−11 gates preserved. The optional public builder flag
`--noncanonical-attachment-fix` reproduces the actually validated C source;
the old backend remains a distinct failing reference. Large-space and other
sector validation remain required.

The [current correctness-first audit](atomic_physics_correctness_2026_10_11/README.md)
adds 12 public boundary checks in `test_fscc_backend_physics_capability.py`:
missing/wrong source capability rejects nonzero real or complex Fov before
export, including arbitrarily weak coupling; corrected capability, exact-zero
Fov and other sectors preserve their separate paths. The actual old binary is
also rejected on the original native input. A fresh current-wrapper run with
the source-bound repaired binary checks all 120×8 signed Bloch entries,
independent HEFF and all eight complete-FCI roots at the unchanged gates.
Its backend update bound is not promoted to an omitted-rank or experimental
error certificate. The same owner separately audits native SD signed actions
and both classes of the second-order attachment energy against fixed-N matrices.

The [SD transverse bridge](atomic_pci_sd_native_2026_10_09/parallel_sd_transverse001/README.md)
adds immutable complete signed J/parity pair blocks and a finite linked two-electron
SD anchor. Its 22 checks pass without skips; large native pCI core/valence/Sigma2
transverse diagrams were not implemented by that bridge. The subsequent
complete source/PP/HH subset is tested independently below. The [tail preflight](atomic_pci_sd_native_2026_10_09/parallel_tail007/README.md)
checks all five original Yb states in four complete radial bands; it reports
the original 64 MiB norm scratch failure and does not certify new roots.
The subsequent [bounded-cache adoption and fresh controls](atomic_pci_sd_native_2026_10_09/parallel_tail007/norm_integration001/README.md)
pass672 finite and168 multicore signed component/M gates; a separate104-case
complete s/p/d/f/g fermion oracle passes. All four complete bands now satisfy
the unchanged64MiB scratch cap with the adopted planner. At that stage full-Yb
norm consumption and tail root solving were unexecuted. Initial missing-native
and mixed-BLAS runtime rejections remain archived.
[Receipt-driven three-process execution](atomx_hourly_progress_2026_10_10/parallel001/README.md)
preserves each lane's source/input/output identity and original scientific gates.
The [public builder proof](atomx_hourly_progress_2026_10_10/parallel001/public_builder_proof.json)
also records a real build through the optional flag:222 compiled source files
match the validated isolated candidate, and fresh C1/axial actual-binary gates pass.

The [six-direction continuation](atomx_hourly_progress_2026_10_10/parallel_six002/README.md)
adds independent retained-versus-omitted four-core CCSDT rank tests, complete
finite correlated-wave metrics and native E1/E2/M1 signed properties, all
source/PP/normal-ordered HH diagram directions, and shared FP64 CPU/CUDA
contraction controls. The latter run actual device admission and retain failed
load guards; bitwise agreement does not certify a speedup. The new norm probe
actually recomputes every original five-state/all18M contraction with the adopted
bounded helper and a fresh 104-case fermion anchor. It does not solve new roots.
The NIST audit retains the original 1% failures, source/space/zero identities,
unmatched observations and unknown uncertainties; report correctness and
experimental accuracy are distinct gates. Six real OS child processes are
covered by a mutual-start barrier test. Receipt execution separately seals each
scientific array, source/input archive and process interval.
