# Integral-driven atomic CCSD and Fock-space CCSDT

Development follows the [Atomx native roadmap](atomic_native_roadmap_2026_10_11/README.md).
The EXP-T path documented here is an existing implementation and independent
comparator; its integration does not complete production native RCCSDT/FSCC.
Native finite CCSD/CCSDT, finite EOM-EA, the new bounded spin-orbital tensor
CCSDT/EA consumer and successful native Ne/Eu CI routes remain distinct
building blocks.

`methods.tensor_coupled_cluster.CCSDTTensorOperator` now consumes owned real
native packed integrals without enumerating full particle sectors. It solves
complete nonlinear SD/SDT core equations and applies the directed R3 EA
operator, with optional batches and an admitted dense projected matrix.
`resource_plan` estimates complete tensor workspaces before allocation;
the default is512MiB. Equation recipes and their Apache-2.0 attribution live
in `physics.csf.ccsdt_terms`; the generic contraction engine lives in numerics.
No external program or upstream numerical package is imported or run.
The [expanded native Yb qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md)
verifies every directed target-sector entry against independent fermion/expm
operators and computes both D lines, with196 live/353 saved checks and54
affected tests passing,2 external-only skips. It still freezes64 electrons,
including4f14. Dense orbital tensors are not J-reduced; no production full-core,
valence-universal FSCC, physical-state/result facade, experimental accuracy or
dressed-property certificate follows. Existing finite/public defaults remain.

The [complete4f continuation](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md) now correlates18 core electrons
and freezes50 in30 magnetic spinors, retaining the same12 virtuals. Every189834
SDT equation and all12 target magnetic partners pass; both lines rise relative
to the frozen4f case. The earlier64-frozen result above is historical.
The matching59PASS/2 external-only SKIP panel and96 saved checks qualify this
source epoch. Full-H defects are not available for the larger tensor model.

`CCSDTTensorOperator(..., projection_storage='canonical')` explicitly gathers
all independent output coordinates before permutation/accumulation. Full
contractions and nested signed subtractions remain; default `'dense'` and
conservative workspace admission are unchanged. `scalar_core_indices()` requires
complete occupied/virtual multiplets and exactly M/parity-conserving Fock/ERI
entries. `solve_core(scalar_sector=True, preconditioner='fock_diagonal',
initial=owned_result)` restricts only exact forbidden core coordinates, uses
unshifted finite/nonzero Fock denominators and checks every retained equation.
The optional initial result must bind this exact operator and have zero forbidden
amplitudes. Defaults are `False`, `None`, `None`. Scalar M/parity coordinates
alone do not enforce total J. Highest-weight Davidson/partner reconstruction is
currently an evidence driver, not a production eigensolver/result API.
A same-input ABBA experiment measures8.87→5.10s for the complete residual,
1.74×, with bitwise identical full outputs; no solver-wide speedup follows.

The [complete6p virtual continuation](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_virtual_p003/README.md) extends that same18-core model
to18 virtual spinors (36 total), with all689589 SDT equations and all12 target
partners checked. A separately qualified local Fock-Jacobi/DIIS iteration
converges the core in14 complete calls; it is not a production `solve_core`
option. Existing production source/defaults remain unchanged. A frozen4f
companion completes the matched four-model panel:4f line increments differ
by441.12/489.20cm⁻¹ between sd and sdp virtual spaces, so they are not portable
additive corrections. Both new D lines rise and experimental gaps grow.
The p-only rank1 angular census still omits allowed D→f virtual channels;
this is structural admission, not a numerical missing-f correction. The local
experiment does not establish full-core, paper-input or physical convergence.

The [paper-alignment core discriminator](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/paper_alignment004/README.md)
subsequently opens the complete5p3/2 quartet at the same18 virtual spinors:
22 correlated/46 frozen core electrons,40 total spinors.1171 off-shell
old-model recovery/sign gates and531 new live gates pass; all1292379 SDT
equations and12 target partners retain the original tolerances. Lines become
25979.234799/26667.252046cm^-1, worsening experimental errors to13.1460%/9.59434%.
This explicitly admitted16GiB local workspace leaves production defaults/API
unchanged. GTO/nuclear/cut inputs and f/g virtual coverage remain unmatched;
no full-paper, full-H or controlled performance claim follows.

`atomx.methods.fscc.compute` runs full nonlinear single/double or
single/double/triple amplitude equations through a pinned, locally built
EXP-T backend. This lower-level interface uses Atomx's signed native
Dirac–Coulomb integrals, optional static or real finite-frequency transverse
exchange, and an explicit closed-shell reference. It does not
enumerate the many-electron determinant space. The older
`methods.coupled_cluster.compute` is the separate finite determinant
CCSD/CCSDT oracle. It now solves connected T3 with T1/T2 and uses the exact
commuting-excitation similarity Jacobian, retaining disconnected products.
Its complete fixed-N determinant and memory caps remain; FSCC and ranks above
three reject. `methods.eom_attachment.compute` lifts the same core cluster
into the complete N+1 sector and diagonalizes the unsymmetrized finite
EOM-EA operator through explicitly declared 1p/2p1h/3p2h ranks. Both sectors
consume one owned native integral operator, including the frozen energy zero.
Projected amplitude/eigenstate residuals and physical full-H residuals are
reported separately; convergence of the former does not prove omitted-rank
completeness. These real finite-space interfaces do not invoke EXP-T and do
not establish valence-universal FS-RCCSDT or scalable 69-electron Yb physics.
See the [paper development evidence](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/README.md).

The [actual-Yb native finite pilot](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/yb_native_rccsdt001/README.md)
now measures nonzero iterative core T3 and attachment-rank effects separately
on one owned DC operator. Complete5s²/5p1/2² core and6s/5d3/2 virtual multiplets
give210/252 full determinant sectors;64 electrons, including4f14, are frozen.
CCSDT/3p2h differs from same-model full CI by0.001114cm⁻¹ on the D3 line,
while physical full-H defects remain nonzero and the experimental gap grows.
This bounded native calculation supplies no full-paper, scalable RCCSDT or
correlated-property certificate and does not replace the MCDHF→RCI main route.

Both finite native solvers accept the explicit
`action_representation="state_action"` option. The shared
`numerics.cluster_actions.ClusterAction` applies complete signed excitation
strings and exact nilpotent exponentials without a dense generator bank.
The default remains `"dense"`; full determinant sectors, dense Hamiltonians,
Jacobians or projected operators, and the original scientific gates remain.
`cluster_term_application_cap=2_000_000` bounds each scalar action, including
zero-amplitude coordinates, rather than the entire nonlinear solve.
Python dictionary overhead is additional to the NumPy admission estimate.
The [matched integration and measurement](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/state_action_integration001/README.md)
preserves nonzero triples and omitted-rank defects. Its small synthetic EA
case saves generator storage but runs slower; this option does not establish
production atomic scaling or a performance improvement.

`read_heff` and `FSCCResult.effective_matrices` preserve the raw formatted
EXP-T blocks. For **0h1p**, `model_space_matrix_Ha` now returns the separately
qualified **ket** matrix, equal to the raw matrix's conjugate transpose and
relative to the same correlated core. The
[direct complex non-Hermitian comparison](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/reference_tensor_compare004/RESULTS.md)
checks complete decoded tensor waves and every directed matrix entry; all four
candidate conventions have the same roots, but only the adjoint passes the
wave/Bloch/projector gates. This conversion performs no Hermitian averaging.
Every 0h1p request verifies the five qualified construction/writer/Slater/
eigensolver/model-vector source signatures before export; missing or changed
signatures reject. Provenance records both raw and physical conventions.
The 1h0p matrix retains its raw convention pending separate physical evidence.
No metric or full correlated property certificate is inferred from a matrix;
the finite omitted-rank residual remains nonzero. Earlier saved model matrices
retain their original source/storage convention. See the
[public interface correction and live checks](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/public_heff_convention001/README.md).

Here “all-order SD” means nonlinear CCSD: repeated single/double correlation
and disconnected higher excitations through the exponential ansatz. It is
not a new native implementation of the distinct linearized atomic SD method,
nor a general CI+all-order effective Hamiltonian for multiple valence electrons.
The separate [native linearized SD interface](ATOMIC_LINEARIZED_SD.md) now
provides a bounded closed-core J=0 DC/DCB correctness anchor and exact scalar
resource planning. Its separate `CoreSD` consumer now runs original reduced DC
core diagrams with exact native Coulomb quadrature and full noncanonical Fock
terms on complete virtual bands. Its fixed-energy CI+all-order valence and
screening interfaces have separate finite signed equation checks. Separate
finite and J-reduced standalone monovalent Bloch-feedback interfaces now exist;
large-space production physical CI consumption remains unverified. This
does not change the nonlinear FSCC equations or substitute a core solver for
an attachment calculation. Both consumers use `methods.atomic_space` for the
same complete native energy-band construction.

The current large-space Yb comparison consumes reduced linearized SD, while
this nonlinear EXP-T path is a separate method. Closed-shell/one-particle/
one-hole support does not imply a general open-f-shell solver. See the
[method-selection assessment](atomic_physics_correctness_2026_10_11/METHOD_STRATEGY.md)
for actual Ne/Eu scope, source correspondences and proposed multi-reference routes.

## Build and verify the external backend

Use the upstream EXP-T revision
`aa8d809ad5818d4aa618dcdadbd7f12257ecbc28`. The build tool copies a supplied
checkout to a fresh destination and records a complete patch. It does not
download sources, overwrite the reference checkout or install globally.
The current local build uses the isolated toolchain recorded in the
[reference-build topic](ci_fscc_reference_upgrade_2026_10_09/README.md).

```bash
.venv/bin/python tools/build_expt_backend.py \
  --source /path/to/pinned/EXP-T \
  --destination /path/to/new/EXP-T-atomx \
  --toolchain /path/to/toolchain \
  --fortran-compiler /path/to/gfortran
```

The toolchain prefix contains `usr/` libraries and includes; `/` can be used
with a system compiler and development libraries. CMake 3.31.6 is run through
`uv`. Build metadata binds the executable SHA, source SHA and interface patch.
The upstream LGPL-3.0 license applies to EXP-T and its modified source; retain
the original copyright notices and license on redistribution.

The patch repairs an incomplete upstream integral reader, zero-filled Fock
blocks, hard-coded two-electron files, one-block sorting and explicit-active
input parsing. It retains arbitrary complex signed integrals, all requested
tiles and the upstream tensor CC equations. Raw pre-DIIS amplitude differences
are recorded. Complex orbital rotations and signed non-Hermitian effective
matrices are validated, in addition to real atomic energies.

The optional `--axial-parity` build flag enables original Dinfh tensor blocks
with all spinors retained. `compute(..., tensor_symmetry="axial")` explicitly
requests them; the default is C1. Export checks every signed Fock/packed-ERI
coordinate against exact M/parity zeros, binds the sidecar to the integral
identity, and verifies actual backend consumption. Incompatible operators,
incomplete multiplets and inadequate rank-six irrep range reject. The pinned
default range permits `6*abs(2m) <= 62`. The explicit
`--axial-irrep-capacity 512` build expands the allocated table and permitted
range to126; the rank-six range gate itself remains unchanged. The enlarged
allocation is source-bound in the backend manifest; arbitrary capacities are
rejected. Mixed-parity model-matrix reconstruction uses that same table size,
including the offset between g/u blocks. An initial hard-coded62 offset
failure is preserved. Public finite native/complex CCSDT matrices, complete
three-electron FCI, closed and one-hole sectors pass in the
[current continuation](atomic_pci_sd_native_2026_10_09/README.md#exp-t完整三激发的精确轴向宇称分块控制).
Production tensor size, performance and spectroscopic precision remain
separate gates.

`resource_estimate(..., tensor_symmetry="axial", active_indices=...,
axial_max_omega_x2=...)` counts ordered M/parity support by exact convolution.
It retains redundant Pauli coordinates,32 amplitude-sized arrays and two
**all-partition** ordered quartic envelopes plus the complete packed ERIs.
`compute` uses this admission only for explicitly requested axial mode, with
the compiled range. C1 estimation/defaults remain unchanged. The estimate is
not a peak-RSS proof; backend memory limits remain enforced. Independent
all-coordinate enumerations check every supported sector/rank. Actual native
high-j CCSD/CCSDT C1 and512-axial full matrices agree within1.777e-15Ha;
two independent483/454-determinant three-electron FCI anchors agree with
CCSDT within1.819e-12Ha. The complete50-check live panel passes with the
new backend. These controls permit larger representations; they do not
establish basis convergence, a controlled speedup or experimental precision.

An independent `--compact-axial-eri` build option and explicit
`integral_storage="axial_compact"` in `compute`, `export_integrals` and
`resource_estimate` remove only exactly forbidden M/parity ERI coordinates.
The outer-p,q order, every allowed r>=p,s>r value, complex phase and original
Hermitian duplicate remain. A source-bound `ATOMX_ERI_COMPACT` sidecar selects
the layout; ordinary builds and packed defaults remain unchanged. Sorted
pair-group ranks and first-index prefixes give constant-time random access.
Every forbidden coordinate is checked before conversion, rather than silently
discarded. The resource estimate includes the compact bank and index envelope;
both all-partition quartic envelopes and32 amplitude arrays remain unchanged.
This is a storage bound, not a peak-RSS or spectroscopic accuracy certificate.

`load_integrals` restores the original owned packed object. Compact replay
therefore requires a separately admitted **full decompression** budget; a small
file does not authorize a large hidden allocation. Full sidecar/hash/count,
Fock/reference and duplicate-Hermiticity checks still apply. A mislabeled or
legacy binary rejects compact requests before export. The real backend also
rejects wrong counts, truncated payloads, missing symmetry and wrong versions.
The initial complete live panel passes80 checks. Actual complete native Sn
n4/n5/n6 integral and Fock arrays replay bitwise; n6 ERI storage changes from
1,440,854,800 to66,274,800bytes. Its complete solved CCSD Heff agrees with
the original C1 calculation within6.138e-16Ha, with identical core and endpoint
energies. These are correctness/storage measurements, not a controlled speedup.

The experimental `--compact-block-index` build option retains every allowed
block and the original CC equations. It stores sorted full-key/block-number
pairs and grows the original-order pointer list with actual blocks. Exact
nonempty axes use the pinned original zero-block predicate, whose separability
is checked against its source identity; full DPD, zero and permutation checks
remain in place. The default retains the original dense layout. Metadata records
the layout and constructor policy. Its explicit `ATXDG002` diagram files require
the same backend format; old diagram files are rejected and require a fresh run.
The signed integral ABI remains `ATXCC001`.

[Native and complex storage/restart controls](atomic_pci_sd_native_2026_10_09/expt_compact_index003/result.json)
pass unchanged full equations, matrix and three-electron FCI gates. These are
finite storage proofs. The [full public CLI build](atomic_pci_sd_native_2026_10_09/expt_compact_public_build001_source_proof.json)
reproduces the validated engine source, and [12 source-bound public checks](atomic_pci_sd_native_2026_10_09/expt_compact_public002/result.json)
pass without skips in native/complex closed/particle/hole sectors. This option
does not establish scalable triples contractions,
large-space memory convergence, production spectroscopy or a timing speedup.

### Required equation capability for noncanonical attachment

`--noncanonical-attachment-fix` explicitly enables `ATOMX_FOV_T2_001` in a
fresh backend checkout. The open Fov*T2 term in `sector01.c` already receives
antisymmetric T2; an extra P(34) doubles it. The option removes only that
permutation, preserving the full Fock/ERI operator and original precision gates.
The builder default still reconstructs the historical original equations;
the old binary and negative evidence are retained. The public `compute` now
rejects `sector="0h1p"` with any nonzero occupied–virtual Fock element unless
the backend manifest declares `ATOMX_FOV_T2_001` and binds the independently
validated `sector01.c` source SHA. There is no small-coupling exemption and no
Fock zeroing or orbital substitution. Exactly zero Fov and other sectors retain
their separate capability requirements. For a native noncanonical attachment
calculation, add `--noncanonical-attachment-fix` to the build command above.
These checks bind trusted build-tool provenance and the executable SHA; another
source implementation needs its own independent validation and capability pin.

[Actual amplitude audit and isolated candidate](atomic_pci_sd_native_2026_10_09/parallel_fscc012/README.md)
identify the complete signed Q1 residual, rather than infer a diagram from
root energies. In the unchanged ten-spinor full 120-row/eight-root Hamiltonian,
the candidate has complete Bloch residual 1.777e-15 Ha and effective-matrix
error 4.219e-15 Ha against the original 1e-10 Ha target. Public patch tests
reproduce the actually compiled candidate sector source byte for byte.
These finite gates do not establish production heavy-atom convergence,
general sector coverage or experimental precision; use the candidate evidence
and its declared controls. The [current public-entry audit](atomic_physics_correctness_2026_10_11/README.md)
re-runs all 120 signed rows/eight roots through the guarded wrapper and verifies
rejection of the actual original binary before export. Request provenance now
records the occupied–virtual Fock maximum, equation capability and residual scope.
The backend's raw Jacobi update bound checks its implemented equations;
independent retained/Bloch residuals, omitted-rank residuals and experimental
accuracy are separate validations.

## Native integral and space contracts

`physics.csf.spinor_integrals.build_spinor_integrals` accepts complete sets of
occupied, virtual and frozen **radial orbital keys**. Every radial key contributes
its full magnetic multiplet. The shared B-spline metric, nuclear quadrature,
Dirac operator, Coulomb radial kernels and Grant angular phases are retained.
Common orbitals must be orthonormal; invalid metrics or missing keys fail.

Frozen electrons contribute an energy constant and signed one-body contraction.
Correlated occupied electrons additionally enter the Fock matrix. The owned
one-body, Fock and antisymmetric ERI arrays are checked against one another and
the reference energy before export. Only bounded producer roundoff is projected
onto Hermitian operators; the projection is recorded. Dirac–Coulomb remains
the default. Explicit `include_breit=True` includes complete static Breit;
`breit_frequency="orbital_differences"` on `atomic_integrals` instead selects
the real finite-frequency transverse operator. QED and recoil are not included
by this factory.

The low-level `build_spinor_integrals` accepts `breit_orbital_energies` in
Hartree for **every** occupied, virtual and frozen radial orbital. Without a
mapping it uses static Breit. The complete signed operator enters the ERIs,
frozen constant, reference energy, effective one-body and Fock contractions
before solving; there is no scalar Breit correction to add afterward.
The finite-frequency operator replaces static Breit, using the symmetric
average of the two vertex **operators**, not an averaged frequency.
See the [operator and frequency contract](TRANSVERSE_INTERACTION.md).

`atomic_integrals` obtains frequencies from actual closed-core orbital
energies and the saved VN virtual eigenenergies. These are fixed across the
CC iterations; dressed Fock diagonals do not redefine photon frequencies.
The owned snapshot and energy-map identity survive caller mutation. This is
a fixed-spinor real one-photon effective Hamiltonian, without finite-frequency
orbital relaxation, absorptive exchange, screened or multiphoton QED.

Pass `representation="on_demand"` to the native factory or
`atomic_integrals` to retain the immutable radial/angular operator without a
quartic resident ERI array. Every requested signed ERI still uses the shared
native kernels. Closed spherical-core one-body contractions are reused across
complete magnetic multiplets; this representation rejects nonscalar matrices
or partially occupied multiplets. Arbitrary complex occupied/virtual rotations
remain supported by the separate explicit `SpinorIntegrals` representation.

`radial_pair_cache_size` declares the bounded pair-potential cache capacity
(default 1024). Diagnostics report its hits, misses, entry count and an array
payload bound; caches and Python/snapshot overhead are additional to the native
one-body payload cap. Changing the capacity does not select or discard any
physical orbitals. Complete-band memory and actual peak RSS are separate checks.

For transverse exchange, `transverse_cache_size` (default 128) bounds each
array-valued cache. The retained FP64 array payload is bounded by
`12 * transverse_cache_size * Gauss_node_count * 8` bytes and included in
`max_integral_bytes` preflight with the required resident integral payload.
Input snapshots, Python keys, scalar caches, active temporaries and backend
tensors remain separate allocations. Eviction only recomputes integrals;
it does not discard orbitals, ranks or small signed matrix elements.

`methods.fscc.atomic_integrals` builds **every** positive-energy VN complement
orbital below an explicit energy upper bound through an explicit `virtual_lmax`.
It retains bound and pseudo-continuum states without a per-kappa count cap.
Frozen core keys must be explicit and closed; all other core spinors enter the
amplitude equations. Radial, partial-wave and continuum convergence remain
separate from amplitude convergence.

```python
from atomx.methods.fscc import atomic_integrals, compute

integrals = atomic_integrals(
    converged_closed_shell_core,
    virtual_lmax=2,
    virtual_energy_max_Ha=1.0,
    representation="on_demand",
    frozen_keys=(),
    include_breit=True,
    breit_frequency="orbital_differences",
    transverse_cache_size=128,
    max_integral_bytes=2 * 1024**3,
)
result = compute(
    integrals,
    executable="/path/to/EXP-T-atomx-build/expt.x",
    directory="new_run_directory",
    model="CCSD",
    sector="0h0p",
    memory_bytes=32 * 1024**3,
    tolerance=1e-13,
)
```

For electron attachment, request `sector="0h1p"` and exact `active_indices`
from the virtual magnetic spinor space. For detachment, use `sector="1h0p"`
and indices from the occupied space. `model="CCSDT"` solves full connected
triples in the closed and requested valence sector. Atomic model spaces retain
complete magnetic multiplets; partial multiplets are rejected before creating
a run directory. Complementary virtuals
remain in the amplitudes when absent from the active model space. Model matrix
rows follow ascending global active indices in the C1 one-particle/hole basis.

The [81-check transverse panel](atomic_pci_sd_native_2026_10_09/transverse_integrals004/result.json)
passes signed CI matrices, frozen-core full-determinant projection, stream
replay and actual C1/axial FS-CCSDT. Its finite Z=4, ten-spinor three-electron
oracle retains all 120 determinants and all eight low model roots. The
[saved independent reconstruction](atomic_pci_sd_native_2026_10_09/transverse_integrals004/saved_confirmation.json)
measures maximum root error 8.729e-10 Ha against the original 1e-9 gate;
the complete directed C1/axial effective matrices differ by 8.957e-21 Ha.
Raw retained-equation residual bounds are about 1.786e-12 Ha; they do not
bound the full FCI root defect. This is finite validation, without production
Yb FS-CCSDT, physical-space convergence or experimental precision certification.
Native J-reduced `CoreSD`/valence diagrams still use their separate DC operator.

### Unfinished noncanonical attachment precision

The [same-H tighter solve007](atomic_pci_sd_native_2026_10_09/transverse_precision007/result.json)
lowers the raw threshold from 1e-12 to 1e-14 and the measured equation bound
to 2.035e-15 Ha, but the two even 2s attachment roots still differ from full
FCI by 8.729e-10 Ha. The closed two-electron core agrees within 1.777e-15 Ha;
the six odd roots agree within about 1.6e-14 Ha. Thus tightening this stopping
condition is not a sufficient repair. The stricter 1e-10 Ha target fails.

The original transverse Fock has occupied–virtual entries about 1.272e-4 Ha.
[Declared changed-H interventions008](atomic_pci_sd_native_2026_10_09/transverse_fov008/result.json)
zero or double only these entries while reconstructing the one-body operator
consistently and keeping the full signed ERIs. Each intervention compares its
own complete FCI: zero-Fov error is 2.309e-14 Ha; twice-Fov error is 1.721e-9 Ha.
This supports an association with noncanonical Fov terms in the attachment
equations, without identifying a missing diagram or providing a universal
error bound. Zeroing Fov changes H and is not an adopted production repair.
The [strict expected-failure gate009](atomic_pci_sd_native_2026_10_09/transverse_precision_gate009/result.json)
keeps this higher-precision target visible while the six original transverse
cases pass. The original backend retains this strict failure. The explicit optional repair
above now closes the declared finite target; production large-space and broader
noncanonical sector validation remain separate gates.

An independent complete determinant oracle now constructs the exact two-core
cluster, removes it with `exp(-Tcore)`, and fixes the attachment wave operator
by intermediate normalization. Its complete eight-column Bloch equation and
signed effective matrix are retained in
[same-H oracle011](atomic_pci_sd_native_2026_10_09/transverse_bloch011/result.json).
The Bloch residual is 7.189e-15 Ha; the backend's two even effective diagonal
entries still differ by 8.729e-10 Ha. This is an operator diagnosis, not a
determinant replacement advertised as scalable CCSDT or a diagram repair.
The [receipt-triggered adaptive iteration](atomx_hourly_progress_2026_10_10/README.md)
now stops low-information radial scans after the dynamic ns=42/60/80/100 defect
plateau. The complete signed one-/two-core-hole partition closes to 8.882e-15 Ha;
whole-transverse lambda=0.5/1.5 controls give defects 4.402e-10/1.304e-9 Ha,
each against its own full oracle. Their response selects the next equation
audit; it does not identify a missing diagram or repair the strict target.
Unavailable implementations remain an explicit frontier, not completed jobs.

## Results, residuals and replay

`FSCCResult` reports the closed-core energy, complete requested model-space
energies, signed effective matrices, raw amplitude update and a conservative
equation residual bound. No intermediate-Hamiltonian shifts, amplitude
selections, perturbative triples or experiment-fitted alignments are inserted.
`effective_matrices` retains original backend symmetry blocks;
`model_space_matrix_Ha` restores their complete directed matrix in ascending
global active-index order for particle and hole sectors, including off-diagonal
phases. Closed-sector results have no model-space matrix. Missing/extra blocks
or wrong dimensions reject. The owned matrix and global indices are saved in
`model_space_matrix.npz` and bound in provenance.

Both closed and requested valence equations must converge. For exact nonzero
numerator division and retained rank≤3 equations without denominator shifts,
the raw Jacobi update times the diagonal denominator is the equation residual.
Let `Dmax = 6 * max(abs(F_pp))`. The pinned legacy engine skips division when
the numerator is below`1e-14` and leaves that number as the new amplitude.
Its conservative bound in atomic units is therefore
`Dmax * max_raw_update + (Dmax + 1) * 1e-14`, including the skip floor.
Zero update alone does not certify a zero residual. Exact nonzero division
has no skip-floor term. These bounds concern retained amplitude equations,
not a full many-electron eigenstate residual or correlation-space error.
The residual bound has its own acceptance gate.
Full triples participate in DIIS, including the valence equations.

### Numerator division source contract

`tools/build_expt_backend.py --exact-nonzero-division` explicitly replaces the
four legacy numerator magnitude tests with exact complex-zero tests, retaining
every nonzero numerator and every original denominator/equation. Existing
builder defaults are unchanged. This option composes with the existing axial,
compact and noncanonical-attachment options; it adds no denominator floor,
IH shift or physical model. The patch accepts only the qualified original
source and verifies the resulting complete source SHA.

Every public `compute` now checks the denominator-engine source. Known legacy
manifests remain usable, with the positive skip floor in their reported bound.
Exact division requires its source SHA, manifest capability
`ATOMX_EXACT_NONZERO_DIVEPS_001` and executable marker. Unknown or mislabeled
engines reject before export. Provenance records the engine SHA, numerator
threshold and residual floor. Older saved bounds retain their historical
definition and must not be silently reinterpreted.

An analytic one-electron two-spinor control with denominator-0.001Ha and
numerator8e-15Ha exposes the legacy issue: decoded t1=+8e-15 versus exact
-8e-12, despite a zero final update. Its physical residual8.008e-15Ha is
covered by the corrected legacy bound; the exact-division candidate gives
the correct amplitude and zero measured residual. The numerator2e-14 control
also passes. Public tests additionally verify source rejection, byte-exact
patching and actual legacy/exact binary behavior. The affected public panel
passes74 tests with both binaries and no skips. A fresh actual public factory
build, including `--exact-nonzero-division`, independently repeats all74
live cases in22.73s with `--runxfail`; this is beyond patch-only replay.
Separate complex finite
FCI/Bloch and native Be/42-spinor C1/axial/gauge controls pass; full Sn model
and experimental convergence remain separate in the [EUV guide](EUV_SOURCE_MODEL.md).

Each run owns integral byte hashes, input, binary/source/patch identity, resource
request, log, effective matrices and saved backend amplitudes. Failed solves and
timeouts retain their logs and do not yield solved results. Directories must be
fresh. `export_integrals` / `load_integrals` preserve and verify the ATXCC001
signed input for exact frozen-input replay. Loading reconstructs the normal-
ordered one-body operator and rechecks Fock/reference consistency.
Axial exports additionally verify sidecar bytes, labels and all signed
integral selection rules on replay.

By default, `export_integrals` writes the unchanged dense ATXCC001 order in chunks (default
4 MiB) and hashes the files with bounded buffers. It checks the full logical
file size before creating a directory (`max_integral_file_bytes`, default
2 GiB). `compute(..., max_integral_file_bytes=...)` forwards that explicit
file budget independently of backend memory, retaining the same2GiB default.
Large callers must preflight construction, complete export and amplitude
storage before expensive generation, and persist reusable integral checkpoints.
The native n7-lmax3 diagnostic exposes the distinction: its integral/tensor
budgets pass but its2.942GB file exceeds the old export default; the failed
run is retained and no orbital is removed. On-demand generation reduces native
resident memory; it does not remove
the backend's integral or amplitude storage. Explicit compact axial ERI storage
is a separate source-qualified representation, as described above. Export diagnostics retain the
representation, chunk/file caps and final cache/projection measurements.

## Scaling and remaining limits

The integral producer uses exact total-M and parity zeros. The on-demand native
operator can represent the complete declared Yb band; the backend currently
uses general C1 or explicitly requested exact axial/parity tensor contractions
and a resident packed ERI array. This is not a fully J-reduced atomic tensor
engine. Resource estimates still include uncompressed
amplitudes and integral diagrams, and explicit runtime/backend memory caps still
apply. Over-budget requests reject the entire space; no orbitals are silently
removed. The current ABI is little-endian and the pinned build has a 2048-spinor
limit. Multi-valence production FSCC, general left-state response properties,
complex-scaled resonance certification and Atom facade dispatch are not provided.

The [six-direction controls](atomx_hourly_progress_2026_10_10/parallel_six002/README.md)
extend the actual optional repaired backend to four core electrons and twelve
spinors, retaining complete 495-row core and 792-row attachment spaces with
nonzero occupied-virtual Fock coupling. Retained CCSDT equations and the full
signed effective-matrix projection pass their original gates. Connected T4/S4
are omitted: their residuals and the complete-FCI energy differences are reported
separately and do not inherit the two-core finite exactness claim. Larger
production virtual spaces and other attachment/detachment sectors remain open.

`properties.transitions.fscc_effective.FiniteCorrelatedWave` supplies a bounded
property consumer for an explicitly supplied full finite wave, physical metric
and total model Hamiltonian. It requires intermediate normalization and the
complete full-H wave equation before forming Bloch or orthonormal signed
operators through the existing bound CI transition interface. Native two-core
E1/E2/M1 complete spherical-component controls pass against independent full CI.
Truncated production CC waves failing that full-H gate are rejected; this API
does not claim general large-space dressed rates, lifetimes or response.

The [electronic anchors and measured atomic workloads](atomic_allorder_fscc_2026_10_09/README.md)
separate implemented methods from experimental and basis-convergence gates.
The [streaming continuation](atomic_streaming_tail_2026_10_09/README.md) records
complete native Yb-band construction, matched exports and larger retained SD
virtual pools; full-band C1 solves remain over budget.

## Live validation

Set `ATOMX_EXPT_EXECUTABLE` to the patched executable and supply its dynamic
library environment before running:

```bash
.venv/bin/pytest -q tests/test_spinor_integrals_fscc.py
.venv/bin/pytest -q tests/test_streaming_spinor_integrals.py tests/test_ci_radial_snapshot.py
```

Without that variable the external electronic gate is explicitly skipped.
Native phase, metric, frozen-core, packing, replay, complete-band and resource
tests still run. A build or a `--help` invocation does not count as an electronic
FS-CCSDT validation.


Explicit energy-independent all-electron scalar additions use
`physics.csf.one_body_composition.add_bound_one_body` with a complete
point-bound `CIEffectiveOneBody`. It preserves signed ERIs and updates frozen,
reference, effective one-body and Fock ledgers consistently. Every retained
radial key is required; packed and native on-demand operators are supported.
The latter retains its complete bound two-body evaluator and allocates only
the scalar matrices; packed-file export and backend budgets stay separate.
Both phase-sensitive independent full-CI controls exercise both representations.
Mismatched native points, repeated operators and
correlated-core self-energy re-addition reject. This composes an externally
qualified matrix and does not generate QED, reopen frozen shells or change
Atom/correction defaults. The [native Sn model-QED qualification](EUV_SOURCE_MODEL.md#native-model-qed-operator-qualification)
separates component conventions, independent signed integration, numerical
refinement, complete amplitude solves and the still-unpassed experiment.
