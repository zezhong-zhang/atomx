# Native Yb II same-orbital core-hole RCI discriminator

This experiment fixes native MCDHF [run005](../native_run/nuclear_eol005_projected_increment/result.json),
which passed the original five-state orbital stationarity gate
(`5.1430e-8 < 1e-7`) from the original initialization. It uses exactly its
24 saved radial spinors, ns123/ks9 Gauss workspace, A175 Fermi nucleus,
and frozen 353-file Atomx source. No radial cutoff, orbital, isotope,
experimental target, Hamiltonian convention, or fit is changed.

The [prospective contract](contract.json) adds complete configuration families:

| Family | Electrons | CSFs | Parity |
|---|---:|---:|---|
| `[Xe]4f14 6s`, `[Xe]4f14 5d`, `[Xe]4f14 6p` | 69 | 5 | Both |
| `[Xe]4f13 6s 5d` | 69 | 39 | Odd |
| `[Xe]4f13 6s 6p` | 69 | 24 | Even |

Integer [resource preflight](same005_orbitals001/preflight.json) found
13 complete `(2J,parity)` blocks, at most 8 CSFs and 36 raw determinants per
block. Public Atomx `ci_solve` and `build_ci_matrix` evaluated **all 68 roots**
for DC and DC plus complete static Gaunt/retardation Breit. Actual elapsed
time was 6.53 s, with peak RSS 778.09 MiB, one thread, and hard 4 GiB/900 s
limits. Timing is uncontrolled; it establishes no parallel speedup.

The [result](same005_orbitals001/result.json) passes 42 numerical consistency
gates. Both five-CSF principal Hamiltonians replay the source baseline to
the declared `1e-8` Ha gate. Full signed amplitudes and matrices are preserved,
with exact-symmetry leakage, Hermiticity, orthogonality, residual, and spectral
moment checks. This is a **fixed-orbital RCI diagnostic**; no 68-CSF MCDHF
optimization or expanded-model stationarity was performed. The source bare
stationarity pass is retained, while the expanded production and experimental
claims remain false.

## Complete state identity and measured shifts

Each bare signed five-state CI vector is embedded in the expanded basis.
The DC target maximizes its squared overlap within the original symmetry;
the Breit target follows a complete one-to-one maximum-overlap assignment
from DC, with a separately retained bare-reference alternative. All block
roots and overlap distributions remain in the
[binding ledger](same005_orbitals001/target_binding.json),
[DC roots](same005_orbitals001/DC_all_roots.json), and
[Breit roots](same005_orbitals001/static_Breit_all_roots.json).
Local-root zero is not the selection rule. In this measured case it coincides
with every tracked target, with no conflicting bare-reference assignment.

The following values use the tracked `6s1/2` lower state and unchanged NIST
nominals. CV increments compare **the same frozen run005 orbitals and operator**.

| Target | DC CV increment (cm⁻¹) | Static-Breit CV increment (cm⁻¹) | Final static-Breit excitation (cm⁻¹) | Static-Breit−NIST (cm⁻¹) | Static bare-reference weight |
|---|---:|---:|---:|---:|---:|
| `5d3/2` | −2.748 | −2.824 | 23754.467 | +793.667 | 99.9965% |
| `5d5/2` | −3.439 | −3.523 | 24222.023 | −110.667 | 99.9949% |
| `6p1/2` | −643.580 | −667.608 | 24315.991 | −2745.829 | 98.5636% |
| `6p3/2` | −575.924 | −597.984 | 27201.095 | −3191.135 | 98.6434% |

All tracked DC→Breit squared overlaps exceed 99.998%. The
[near-degeneracy audit](same005_orbitals001/degeneracy_audit.json) checks the
predeclared `1e-8` Ha energy window in every complete symmetry block and
reports each target's nearest same-symmetry root. No target lies in that
window. Global minimum search over every root finds the tracked `6s1/2`
as the finite-model ground for both Hamiltonians. Its bare weight is 100%
to numerical precision; the static hole-family weight is `2.78e-16`.
Excitations from the tracked lower state and from the independently found
finite-model global ground are separately archived.

Degeneracy-weighted multiplet centroids use `(2J+1)` weights `(4,6)` for D
and `(2,4)` for P. Splittings always retain the original fine-level identity.

| Static-Breit observable | Bare five-CSF value (cm⁻¹) | Expanded value (cm⁻¹) | CV increment (cm⁻¹) | NIST nominal (cm⁻¹) |
|---|---:|---:|---:|---:|
| D multiplet centroid | 24038.244 | 24035.001 | −3.243 | 23783.934 |
| D fine splitting | 468.255 | 467.557 | −0.699 | 1371.890 |
| P multiplet centroid | 26860.585 | 26239.393 | −621.192 | 29282.093 |
| P fine splitting | 2815.480 | 2885.104 | +69.624 | 3330.410 |

Thus the retained D fine splitting still lacks about 904.33 cm⁻¹. P mixing
improves its splitting by 69.62 cm⁻¹ while lowering its centroid by 621.19
cm⁻¹ and **worsening both excitation gaps**. A 1.36–1.44% hole weight already
moves the P transitions by roughly 600–670 cm⁻¹; small weights cannot justify
discarding a physical channel.

The complete bare-reference spectral measure also retains its centroid and
variance, independently checked against `bᵀHb` and `||(H−centroid)b||²`.
Its displayed width measures redistribution of one bare vector across CI
eigenstates; it is not an emission linewidth or lifetime. Adding off-diagonal
mixing leaves the bare-vector centroid equal to its principal diagonal
element even when its tracked eigenenergy moves substantially.

## Physical decision and next balanced model

This set is not balanced across the five references: the retained ground
almost cannot mix, D mixing is tiny, and P mixing is substantial. It measures
an important missing core-hole channel, but does not demonstrate that a
larger configuration list automatically closes the experimental gap. Its
limited correlation mechanism and the remaining D splitting deficit require
a physical model decision before larger n/ℓ scans. No QED model is assumed
to cancel the missing thousands of cm⁻¹, and the earlier
[radiative correction budget](../correction_budget/README.md) is not spliced
into these newly expanded states.

The next prospective balanced CV model should:

1. Generate configurations systematically from **every** bare reference in
   the same common active-orbital pool. Prioritize complete allowed channels
   involving `4f`, `5p`, and `5s` core holes, preserving all coupling trees,
   Pauli restrictions, symmetries, and signed angular terms.
2. Include core promotion and allowed valence replacements together, rather
   than require an unchanged spectator `6s`. For example, the present families
   omit `[Xe]4f13 5d 6p`, which can supply even-parity ground-sector channels;
   its actual contribution remains unmeasured. Generate the full allowed
   occupation patterns from each reference before choosing an example as a
   production model.
3. Before any solve, count each complete J/parity CSF span and raw determinant
   space, estimate angular-cache/radial storage and work, and retain a bounded
   contract. The present resource caps must not be met by silently omitting
   configurations. Start with the existing radial pool, without an automatic
   larger cutoff.
4. Use the balanced common state objective for a new MCDHF solve and then
   same-state RCI; pass the unchanged orbital, metric, CI, and experimental
   gates. Distinguish fixed-orbital mixing from additional common-orbital
   relaxation, rather than borrowing the bare five-CSF stationarity pass.

No new optimization run or source change is part of this increment.

## Replay

[Raw CI matrices/amplitudes](same005_orbitals001/ci_arrays.npz),
[signed overlaps](same005_orbitals001/overlap_arrays.npz),
[complete basis identities](same005_orbitals001/csf_basis.json), and
[verification](verification.json) bind the measured result to its exact source
and inputs. The separate verifier reruns complete-H `eigvalsh`, signed closure,
matrix reconstruction, interlacing, ledger and state-binding identities.
These are independent numerical checks, not a second atomic Hamiltonian
implementation or a certificate of complete correlation.
All 866 archive replay checks pass, alongside the 42 execution gates.

```sh
.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/cv_discriminator/verify.py
```
