# Native tensor CCSDT and both Yb⁺ D lines

The continuation implements native tensor CCSDT core equations and complete
1p/2p1h/3p2h electron-attachment actions in Atomx, then adds the entire
5d₅/₂ multiplet to the preceding actual-Yb model. The production tensor
consumer constructs no complete fixed-particle determinant sector or cluster
generator bank. Independent complete target-sector CI and matrix exponentials
verify the new results. Native MCDHF→RCI remains the main route.

## Result and physical scope

The [predeclared contract](contract.json) fixes the same empiricalA171 Fermi
nucleus, owned ns180/order9/R500/RKB radial bank, native Dirac–Coulomb operator
and64 frozen electrons as the [earlier ten-spinor pilot](../yb_native_rccsdt001/README.md).
The four correlated core electrons remain5s²+5p₁/₂². All4f¹⁴ and5p₃/₂⁴
electrons remain frozen. The virtual space now contains complete6s₁/₂,
5d₃/₂ and5d₅/₂ multiplets:12 virtual and16 total magnetic spinors.
Every retained old integral and the original common68-electron energy origin
matches the previous model. Breit, QED, recoil and high-l/radial convergence
remain absent. No experimental number chooses a state or enters the solver.

| Method | 6s→5d₃/₂ (cm⁻¹) | 6s→5d₅/₂ (cm⁻¹) |
|---|---:|---:|
| CCSD / R₂ | 24372.894276 | 24717.220962 |
| CCSD / R₃ | 24372.121429 | 24708.867556 |
| CCSDT / R₂ | 24372.999169 | 24717.263416 |
| CCSDT / R₃ | **24372.156946** | **24708.853564** |
| Independent same-model full CI | 24372.158302 | 24708.849040 |

R₂ means1p+2p1h; R₃ adds every3p2h coordinate. CCSDT/R₃ minus full CI is
−0.0013555/+0.0045241 cm⁻¹. Holding R₃ fixed isolates the core-T₃ line
changes,+0.0355176/−0.0139917 cm⁻¹. The combined CCSD/R₂→CCSDT/R₃ change
also includes the independently measured attachment-rank contribution;
these effects are not interchangeable. The connected T₃ norm is3.66883e−4.
The [live result](actual001/result.json) retains all individual state energies,
core energies, both rank controls and their full-H defects.

Adding5d₅/₂ shifts the prior CCSDT/R₃ D₃ line by−15.281041 cm⁻¹. Relative
to the earlier same-bank DHF lines, the current correlation shifts are
about+99.416/−43.470 cm⁻¹. This is a measured space dependence, with no
claim of converged atomic correlation. The experimental gap remains large;
agreement with full CI in this tiny model does not establish experimental
precision. Neither4f correlation nor additional virtual radial functions
have been calculated in this increment.

The paper's [Sahoo2025 common-core RCC ansatz](https://arxiv.org/html/2504.01438v1)
guides this comparison, but the computed method is projected **EA-CCSDT/R₃**.
It does not implement a valence-universal FS-RCCSDT wave operator or the
paper's full numerical input. The published RCCSDT, extra h/i and Breit
increments are not added to Atomx. The40-GTO exponent lists, detailed frozen/
correlated partition and numerical cuts remain unavailable in the checked
paper sources; no other paper's cuts substitute for them. Correlated
quadrupole, hyperfine and isotope-shift properties remain unfinished.

## Native implementation and independent equations

- [CCSDTTensorOperator](../../../../../src/atomx/methods/tensor_coupled_cluster.py)
  consumes one owned packed native integral operator. It solves simultaneous
  nonlinear SD or SDT equations with a fixed four-point coupling continuation
  and a Krylov root solver. It also exposes complete batched directed EA
  actions and an explicitly admitted projected matrix. Its first consumer
  accepts exactly real integrals and rejects complex input.
- [Shared tensor contraction engine](../../../../../src/atomx/numerics/tensor_recipes.py)
  evaluates signed recipes and explicit free-index permutations. Shape-only
  contraction paths are cached; no physical amplitudes or integrals are cached.
- [Physical equation recipes](../../../../../src/atomx/physics/csf/ccsdt_terms.py)
  retain all5 energy,15 single,37 double,174 triple and15/59/267 EA terms.
  They are adapted from the independently generated, pinned
  [pdaggerq reference](https://github.com/edeprince3/pdaggerq/tree/f6fb6dec2c0edd2a02180ee84f395d74eed1c0a1),
  with the original Apache-2.0 license retained adjacent to the recipes.
  Only symbolic equations are adapted. Atomx neither imports nor executes
  pdaggerq, PySCF, OpenFermion, EXP-T or another electronic-structure program.
  The [reference manifest](equation_reference.json) records exact upstream
  hashes and attribution. This is an independent equation reference, rather
  than the original Sahoo program.

The native tensor amplitudes use ordinary antisymmetric excitation operators.
The older finite oracle divides each excitation generator by its action's
reference-determinant sign. The replay explicitly converts core amplitudes
and both directed EA endpoints between these conventions. All entries of
the old186×186 pencil agree within the original1e−9 Ha gate, and a fresh
tensor core solve reproduces its energy. Diagonal energies alone would not
test that conversion.

The expanded core has1324 independent SDT amplitudes; its full tensor shape
contains112944 coordinates including antisymmetric copies and exact zeros.
EA through R₃ has1596 canonical coordinates. The live spectrum computes six
complete even-parity M sectors,2M=−5,−3,−1,+1,+3,+5. Their R₃ dimensions
are58/86/102/102/86/58. All2/4/6 target magnetic partners are present;
other M/parity spectra are not computed or claimed.

Independent diagnostics intentionally enumerate complete fixed-N target
symmetry sectors outside the production tensor solver. The complete core
M=0/even sector has190 determinants. The attached full-CI sector dimensions
are237/322/374/374/322/237. The total unprojected particle counts would be
1820/4368, but full all-M/all-parity matrices are never constructed. Each
selected sector is complete for its conserved quantum numbers and the
finite one-body pool.

The [occupation-list audit](audit.py) uses the full four-index1/4 two-body
operator, independent signed creation/annihilation, complete matrix
exponentials and explicit J± algebra. It verifies every directed tensor
H̄ entry in every computed R₃ sector; the largest discrepancy is
3.32445e−12 Ha. Both R₂ arms are separately checked principal restrictions
of those operators. Target identity uses normalized reconstructed physical
bare-state weights≥0.5, J² and complete-multiplet spreads. Neither sorted
energy labels nor nearest experimental lines select the targets.

Numerical projected equations and full-H defects stay separate. Core SDT
residual3.31e−14 Ha passes, while its physical full-H residual is9.496e−5 Ha.
The S/D₃/D₅ target full-H residual maxima are1.3234e−4/2.0014e−4/2.5845e−4 Ha.
Omitted connected core T₄ and attachment ranks4/5 are not certified. The
CCSDT core lies6.86415e−9 Ha below same-model CI; CC is nonvariational and
that sign is preserved. Physical EOM norms are checked without imposing
orthogonality on the non-Hermitian right eigensystem.

## Validation, resources and replay

All **196 live controls** and **353 saved-array checks** pass. The
[saved-data review](saved_data_review.json) checks every archived eigen-equation,
full-CI Gram, all physical norms, complete target partners and individually
recomputed nonzero full-H residuals. It also verifies all362 actual production
Python source files, complete archive members and bound input hashes. The
prior359 production files remain unchanged; the increment adds three native
modules and meaningful independent tests.

The isolated affected panel passes47 tests with2 external EXP-T-only skips;
the architecture-boundary panel passes7 more tests: **54 passed,2 skipped**.
The new tensor panel covers arbitrary signed off-shell amplitudes against
full exponentials, all directed EA entries and batches, FCI solving,
nonzero triples, omitted quadruple defects, phase/energy-zero covariance,
ownership and rejection before resource allocation. The first broader run
failed during shared Numba cache serialization with a vanished weak reference.
Its [original log](tests.log) is retained. The same panel passed with an
isolated generated cache; no physical model, gate or tolerance changed.
See [isolated results](isolated_tests.log) and [architecture results](architecture_tests.log).

The actual single-thread batch completed in48.607 s with peak RSS402.355 MiB,
under the predeclared8 GiB/1200 s process limits. It includes the old-model
replay, one expanded native integral operator, three tensor core solves and
all independent sector CI/exponential diagnostics. Prior DHF radial-bank
construction is excluded. This is an unpaired feasibility measurement,
not a controlled speedup. The tensor NumPy admission estimate is37,375,904
bytes at batch4; Python/SciPy overhead is additional and actual RSS is reported
separately. The tensor512 MiB default and all prior finite defaults remain.

[run.py](run.py), [audit.py](audit.py), [source.json](source.json) and
[source.tar.gz](source.tar.gz) preserve the execution and source identity.
Existing `actual001` prevents overwrite. Saved-array validation from the
repository root requires no new integral construction or root solve:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python docs/yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/verify_saved.py
```

The [next-space scalar plans](next_space_plan.json) contain no atomic run.
Adding the complete4f¹⁴ core to this pilot estimates2176 MiB of tensor
payload/workspace; the unchanged512 MiB default rejects that model. Even the
minimal12-virtual/all68-core model estimates about103.4 GiB, and its full
projected EA matrix about1.86 TiB. Core/virtual convergence needs measured
contraction planning, J/symmetry blocking and targeted matrix-free attachment
solves. The new tensor path removes complete-Fock generator enumeration;
it does not complete large-space atomic RCCSDT, physical stationarity,
experimental accuracy or the all-elements NIST database.
