# Native Yb⁺ connected-triples pilot

The user authorized an additional RCCSDT trial on 2026-10-14. The native
finite CCSDT/EOM-EA calculation now runs on actual Yb⁺ radial functions and
Dirac–Coulomb integrals. Connected core T₃ is nonzero. The calculated
6s→5d₃/₂ line differs from full CI on the identical finite operator by
−0.0011144 cm⁻¹. This establishes a small-space equation and integration
anchor; experimental accuracy and the full paper reproduction remain open.
Native MCDHF→RCI remains the main route, including its original failed gates.

## Fixed physical model

The [pre-execution contract](contract.json) binds the exact inputs, acceptance
rules, source and resource limits. The prior [native DHF bank](../yb_dhf_table001/README.md)
supplies a common 68-electron reference for 69-electron Yb⁺: Z=70, empirical
A171 Fermi nucleus, ns=180, spline order9, R=500 a.u. and RKB. No electronic
structure executable outside Atomx is used.

- Correlate four core electrons in complete 5s² and 5p₁/₂² multiplets.
- Freeze the other64 electrons, including 4f¹⁴ and 5p₃/₂⁴, retaining all
  frozen-core contractions and the scalar frozen energy.
- Retain complete 6s₁/₂ and 5d₃/₂ virtual multiplets: six virtual magnetic
  spinors, ten total. No magnetic partner is discarded.
- Use the same native Dirac–Coulomb operator in both particle sectors and
  all four CC/rank arms. Breit, QED and recoil are absent.
- Subtract exactly the original common68-electron HF energy from both
  sectors for numerical conditioning. The original origin is retained in
  [the operator ledger](actual001/operator.json); integrals are unchanged.

This explicit, noncontiguous core partition was chosen before execution to
permit parity-allowed scalar T₃ while retaining complete multiplets within
the finite engine's determinant cap. It is a diagnostic partition, with no
claim of correlation-space convergence. The complete core4e and attached5e
spaces contain210 and252 determinants. Core T₁/T₂/T₃ have24/90/80 coordinates,
respectively. The attachment projections contain66 determinants for
R₂=1p+2p1h and186 for R₃=1p+2p1h+3p2h. Core T₄ and the66 attachment
determinants of ranks4/5 are omitted from the CCSDT/R₃ equations.

The precise method label is **finite-EOM-EA[core=CCSDT,R=3p2h]**. Core
T₁/T₂/T₃ are solved iteratively together with the analytic similarity
Jacobian and exact nilpotent exponentials. This is connected T₃, distinct
from a perturbative (T) correction. The projected attachment operator stays
non-Hermitian. This interface is not a production valence-universal
FS-RCCSDT implementation of [Sahoo2025](https://arxiv.org/html/2504.01438v1).
The paper's published triples increments are not applied to this pilot.

## Same-operator results

Every entry below uses the same ten magnetic spinors, nucleus and integrals.
Root identity uses the complete2/4-member bare 6s/5d₃/₂ model projectors,
their weights, J² and multiplet spreads; experimental energies do not select
roots. Full CI retains every210/252 determinant.

| Core cluster | Attachment projection | 6s→5d₃/₂ (cm⁻¹) | Difference from full CI (cm⁻¹) |
|---|---|---:|---:|
| CCSD | R₂ | 24387.003859 | −0.435242 |
| CCSD | R₃ | 24387.410567 | −0.028534 |
| CCSDT | R₂ | 24387.097260 | −0.341842 |
| CCSDT | R₃ | 24387.437987 | −0.001114 |
| Full CI | Complete5e | 24387.439101 | 0 |

The combined CCSD/R₂→CCSDT/R₃ shift is+0.434128 cm⁻¹. Holding the
attachment projection atR₃ isolates the core-T₃ line effect,+0.027420 cm⁻¹;
holding core CCSDT fixed isolates the R₂→R₃ effect,+0.340727 cm⁻¹. AtR₂
the core-T₃ effect is+0.093401 cm⁻¹, so these effects are not additive
independent corrections. The actual T₃ norm is3.4904326e−4 and its attached
sector action norm is8.5497788e−4; triples are not Pauli-blocked.

The CCSDT core lies1.31056e−9 Ha below same-model full CI, consistent with
the nonvariational CC energy. The reconstructed69-electron S/D₃ energies
lie+0.006043/+0.004929 cm⁻¹ above CI. Attachment energies relative to each
method's own correlated core have separately recorded errors. See the
[live result](actual001/result.json) and [saved-array review](saved_data_review.json).

Numerical equation convergence does not remove omitted-rank defects. The
CCSDT core projected residual is2.7020e−18 Ha; the largest projected
attachment residual over all four arms is1.8208e−14 Ha. In CCSDT/R₃ the
physical full-H residuals for the target S/D₃ multiplets remain
7.6677e−5/1.21736e−4 Ha; the maximum over all186 roots is0.085419 Ha.
The core full-H residual also remains nonzero. These diagnostics were
preserved, rather than required to vanish or described as full-H convergence.
Energy agreement alone supplies no wavefunction or correlated-property
certificate. No correlated quadrupole or hyperfine result is produced here.

The previous same-bank DHF line is24272.741259 cm⁻¹; this restricted
correlation model shifts it upward by114.696728 cm⁻¹. The historical
experimental comparator is22960.80 cm⁻¹, so the discrepancy remains
about+1426.64 cm⁻¹ and has grown. That stored comparator is not a new NIST
retrieval. Within this fixed tiny model, excitation-rank convergence alone
cannot explain or remove the experimental gap. A correlated-core/virtual
space study is still needed; the pilot does not measure which missing
shell dominates. In particular, 4f correlation and 5d₅/₂ are absent.

## Independent validation and resources

All44 live controls and48 saved-array controls pass. The independent full
Hamiltonians use occupation-list creation/annihilation and the full1/4
four-index sum from the source-bound test oracle, independently of the
production pair-reduced builder. The core/attached matrix differences are
4.44e−15/3.55e−15 Ha. Separate signed occupation-list cluster lifts and
SciPy matrix exponentials reproduce every directed projected H̄ entry;
the maximum difference across four arms is3.55e−15 Ha. The CCSDT/R₃
non-Hermitian asymmetry is0.042297 Ha and is retained.

A whole-radial 5p₁/₂ sign change checks complete projected matrices,
spectra and reconstructed physical S/D₃ projectors, including all252
physical components. The saved review also reenumerates complete determinant
sectors, checks full-CI eigenpairs/Gram, physical norms, source/input hashes,
projected equations and independently recomputes nonzero full-H residuals.
The execution log retains the real-cast warning from the independent oracle;
readback verifies every stored operator entry has exactly zero imaginary
part. No nonzero imaginary operator term was discarded.

The existing finite-CCSD/CCSDT, EOM-attachment and native-integral panel
passes **26 tests**, with **2 external EXP-T live tests skipped** because
their executable variables are unset. No requested native pilot was skipped.
See [test output](tests.log) and [JUnit details](tests.xml).

The single-thread native batch completed in7.1072 s with peak RSS539.719 MiB.
It includes one new integral operator, the independent full-CI pair, four
factorial native solves and one phase-control solve. This unpaired feasibility
measurement is not a speedup or a full radial/atomic solve timing. The owned
DHF radial bank was previously computed. CPU27,8 GiB address-space and900 s
caps were predeclared; determinant cap256 remains unchanged. Amplitude cap
256 and generator/workspace cap512 MiB explicitly replace public defaults
128/128 MiB for this194-amplitude case. Dense action remains the default.
The largest generator estimate is167,001,408 bytes; logical Hamiltonian
construction upper982,212 remains below the default2,000,000 action cap.

## Replay and next implementation boundary

[run.py](run.py) and [independent.py](independent.py) preserve preparation and
execution. [source.json](source.json) and [source.tar.gz](source.tar.gz) bind
all359 actual production Python files; no production source or public
default changed in this increment. Original physical inputs, full matrices,
wavefunctions and amplitudes are retained under `actual001/`.
Preparation preceded execution. Existing `actual001` prevents overwriting
the run; a fresh scientific trial requires its own directory and contract.
Saved-data validation from the repository root is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python docs/yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/yb_native_rccsdt001/verify_saved.py
```

The [scalar-only larger-space estimate](larger_space_scalar_plan.json)
correlates all68 core electrons with only the complete6s/5d₃/₂/5d₅/₂
virtual multiplets. Even that80-spinor model has60,246,643,120,300 core
and10,477,677,064,400 attached determinants, and11,176,684 SDT amplitudes.
It exceeds the finite mask and determinant budgets; no such matrix was
allocated or atomic run attempted. Production RCCSDT requires linked
tensor/J-reduced contractions, followed by correlated-core, radial,
partial-wave and Hamiltonian convergence. Raising this finite engine's
caps is not a viable route to the full paper calculation.
