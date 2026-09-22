# `correlation="s"` / `correlation="sd"` — first-principles audit

**Date:** 2026-09-20
**Scope:** the `Atom(..., correlation="s" | "sd")` feature and the MCDHF/CI
matrix that backs it (`src/atomx/methods/mcdhf/{excitations,config,matrix}.py`,
`src/atomx/methods/{ci,solver}.py`).
**Trigger:** open finding **N1** of
[`PHYSICS_NUMERICS_REVIEW_2026-09-19.md`](PHYSICS_NUMERICS_REVIEW_2026-09-19.md)
— "`correlation="sd"` is not variational".
**Status:** N1 root-caused, characterised, and **fixed** (2026-09-21,
§8 option A: exact matrix elements by m-scheme projection). This document
remains the research deliverable and supersedes the first-pass diagnosis
recorded in
[`PHYSICS_NUMERICS_REVIEW_2026-09-19_FIXES.md`](PHYSICS_NUMERICS_REVIEW_2026-09-19_FIXES.md)
("the CI matrix is not the suspect" — retracted). See
[§11](#11-fix-landed-2026-09-21) for what the fix changed, including a bug
it exposed in the independent oracle and the one basis-dependence it does
*not* remove.

---

## 0. Verdict

`correlation="s"` and `correlation="sd"` do **not** produce a variational
configuration-interaction energy. They produce the lowest eigenvalue of a
matrix that is **not a representation of the Hamiltonian** on the CSF basis.
For Be the error is **0.43 Ha (11.7 eV)** at the default `correlation="sd"`
protocol and **1.70 Ha (46 eV)** under `radial_optimization="fac_frozen"`. The
`"s"` variant is wrong too — by 0.081 Ha at the default protocol — even though
its headline number happens to land near the exact energy.

The defect is *not* the CSF generator and *not* a linearly dependent basis
(both were tested and are clean). It is that `build_ci_matrix` evaluates a
**configuration-average** diagonal plus two hand-rolled "J-correction" patches
instead of the exact matrix element ⟨CSF_r|Ĥ|CSF_s⟩. The root enabler is
structural: `CSF` carries only `total_J2` and a subshell occupation list —
**no coupling tree (parentage)** — so the exact jj-coupled matrix element is
not computable from the object the code passes around.

The orbital set is broken too, but that is a *second, independent* defect
(§3.7): low-weight correlation orbitals are solved on the wrong branch. Fixing
only the CI would leave a bad calculation; fixing only the orbitals would leave
a non-variational one.

**Correction 2026-09-22 (oracle independence).**  The oracle described below as
"independent" is **not** independent of the implementation: `exact_ci.two_body_angular`
is a line-by-line transcription of `mscheme_ci_reference._angular_terms`, the
determinant sign conventions are shared by design, the CSF patterns come from the
repo's own `enumerate_jj_csfs`, and `_h_one_body` / `_norm_sq` / `_rk` are shared.
It gates the *assembly* (patterns, fermion signs, CSF orthogonality, the
`C† H_det C` reduction) faithfully, but it cannot arbitrate the angular algebra.
The independent evidence for the CI matrix is the Hylleraas–Undheim–MacDonald
bound, the exact hydrogenic `F^k`/`G^k` rationals, the analytic Slater–Condon
ratio, and the external NIST/DBSR_HF comparisons.  Every "independent oracle"
claim below should be read with that qualification.

**Correction 2026-09-20 (oracle bug).**  The oracle
`benchmarks/mscheme_ci_reference.py` originally read `if v == 0.0: continue`
on the two-body term, dropping the *exchange* integral whenever the *direct*
one vanished by angular selection.  Proven wrong by rotational invariance:
Be `1s2 2s1 2p1` J=1 came out -14.490934 from the 2M=0 block but -14.530753
from 2M=2, while a scalar Hamiltonian cannot depend on M.  Fixed to
`if v == 0.0 and v2 == 0.0`.  **The oracle-derived targets quoted below for
the singles and SD spaces (-14.595092, -14.654637) were artefacts of that
bug and are superseded**; the corrected values are -14.575893 and -14.632497.
The conclusion of the audit is unaffected - it rested on the variational
bound and on the orbital-free Slater-Condon identity, not on the oracle -
but the target numbers were wrong and this note replaces them.

Evidence chain, all reproducible (§3):

| # | system / space | atomx | correct value in the *same* space and orbitals | Δ |
|---|---|---|---|---|
| 1 | Be 1s²2s² (1 CSF) | −14.575892 | −14.575892 | 0 |
| 2 | Be 1s²2s²+1s²2p² (6 CSF) | −14.616689 | −14.614096 | −0.0026 |
| 3 | Be 1s²2s2p, J=1 block | −14.711459 | −14.490933 / −14.468254 | −0.22 … −0.24 |
| 4 | Be **singles** (29 CSF, `fac_frozen`) | −15.400463 | −14.575893 | **−0.825** |
| 5 | Be **singles** (29 CSF, `mcdhf`) | −14.659475 | −14.578383 | **−0.081** |
| 6 | Be **SD** (399 CSF, `fac_frozen`) | −16.358105 | −14.632497 | **−1.726** |
| 7 | Be **SD** (399 CSF, `mcdhf`) | −15.052683 | −14.619112 | **−0.434** |

Exact non-relativistic Be ground state: **−14.66736 Ha** (Drake); with the
one-electron relativistic shift the Dirac–Coulomb reference is ≈ −14.670 Ha.
Rows 4–7 are *below the exact ground state of the Hamiltonian being solved*,
which no correct CI in any orbital basis can be.

---

## 1. What the switch actually does

`Atom.__init__` (`src/atomx/atom.py:354-376`):

```
virtuals = suggest_virtual_orbitals(Z, n_virtual=1)   # Be → ['2p','3s','3p','3d']
max_exc  = 2 if correlation == "sd" else 1
core     = None if Z <= 4 else (['1s'] if Z <= 10 else ['1s','2s','2p'])
self._config_strings = generate_sd_excitations(base, virtuals, core, max_exc)
```

`Atom.solve` then routes to `MCDHFSolver` with
`csfs = enumerate_jj_csfs_many(self._config_strings)`
(`atom.py:522-544`), i.e. **every NR configuration is expanded into its full
jj-coupled J-multiplet**, and the solver iterates

```
Guerra weights ϖᵢ = Σⱼ cⱼ² nᵢʲ   →  DFSolver on ⟨n,κ⟩ subshells
                                 →  build_ci_matrix + ci_solve
                                 →  c = ground eigenvector, repeat
```

with `total_energy = min_eig(H)` (+ Breit/QED on the final orbitals).

For Be the resulting basis is 9 configurations / 29 CSFs for `"s"` and
39 configurations / 399 CSFs for `"sd"`. The `core=None` branch matters: for
Z ≤ 4 **nothing is frozen**, so the SD list contains hollow (1s⁰)
configurations such as `2s2 2p2`, `2s2 3d2`, `2s2 3p1 3d1`.

---

## 2. The variational contract

A CI energy is meaningful only if three things hold. This is textbook
(Hylleraas 1928; Undheim 1930; MacDonald 1933 — the
[Hylleraas–Undheim–MacDonald theorem](https://ar5iv.labs.arxiv.org/html/physics/0411204),
Appendix A.2 — the sentence following Eq. (102) in the ar5iv HTML; the paper's
own appendix-local numbering makes that equation (A2), with the intertwining
inequalities at (103)), but it is worth stating because atomx violates all
three in different ways.

1. **Exact matrix elements.** `H_rs = ⟨CSF_r|Ĥ|CSF_s⟩` for the *same*
   Hamiltonian, the *same* orbitals, the *same* nuclear model and the *same*
   two-electron integrals as the SCF energy functional. Any term dropped,
   added twice, or approximated inconsistently destroys the bound.
2. **A basis, not a spanning set.** The trial functions must be linearly
   independent (or the generalised problem `Hc = ESc` must be solved with the
   correct `S`). Solving `Hc = Ec` on a rank-deficient basis is the classic
   *variational collapse*.
3. **Symmetry blocking on exact quantum numbers.** Only CSFs with identical
   `(J, parity)` may mix; a stray inter-`J` element lets the diagonalisation
   reach states that do not exist.

Two corollaries used repeatedly below:

* **Monotonicity.** If 𝒮 ⊂ 𝒮′, then `E(𝒮′) ≤ E(𝒮) ≤ E_HF`.
* **Brillouin's theorem.** For orbitals that are self-consistent for the
  reference determinant, `⟨Φ₀|Ĥ|Φ_single⟩ = 0` *exactly*. A singles-only
  expansion on HF orbitals therefore buys **nothing**; any large "singles
  correlation energy" is a diagnostic of either orbital relaxation (legitimate
  in MCDHF) or of a wrong matrix element (not legitimate).

---

## 3. Measurements

All runs: `Z=4`, `ns=113`, `ks=9`, `breit=False`, `qed=False`, so the energies
are pure Dirac–Coulomb and directly comparable with the non-relativistic exact
value modulo the ~2–3 mHa relativistic shift.

### 3.1 Reproduction

```
Atom(Z=4, correlation=None)  → −14.573390
Atom(Z=4, correlation="s")   → −14.656986
Atom(Z=4, correlation="sd")  → −15.053912     ← 0.386 Ha below exact
```

### 3.2 Bisect: one configuration is enough

Holding the orbitals fixed (`radial_optimization="fac_frozen"`, so every run
below uses *the same* orbital set) and pairing the reference with each of the
38 double-excitation configurations one at a time:

* 32 of the 38 pairs leave the energy within 0.2 mHa of the reference
  (−14.5758916);
* 5 move it further, all in the physically expected direction and all safely
  above the exact ground state — `1s2 2p2` (−40.8 mHa, the classic Be
  near-degeneracy pair, landing on the literature 2-configuration MCHF value),
  `1s2 2s1 3s1` (−36.0 mHa), `1s2 2p1 3p1` (−9.0 mHa), `1s2 3p2` (−0.43 mHa),
  `1s1 2s2 3s1` (−0.19 mHa);
* **exactly one breaks the bound**: `["1s2 2s2", "1s2 2s1 2p1"]` gives
  **−14.708567 Ha**, 0.133 Ha below the reference and 0.041 Ha below the exact
  ground state.

That configuration is a **single** excitation (2s → 2p), so it is present in
the `"s"` space as well. It is the J-multiplet structure of an open shell that
breaks, not the doubles — and the closed-shell ↔ closed-shell coupling (row 2
above) is handled correctly, which is why the existing MCDHF tests pass.

### 3.3 The failing block, element by element

For `["1s2 2s1 2p1"]` alone the CSF list is

| i | CSF | 2J | block |
|---|---|---|---|
| 0 | 1s² 2s₁/₂ 2p₃/₂ | 2 | J=1 |
| 1 | 1s² 2s₁/₂ 2p₃/₂ | 4 | J=2 |
| 2 | 1s² 2s₁/₂ 2p₁/₂ | 0 | J=0 |
| 3 | 1s² 2s₁/₂ 2p₁/₂ | 2 | J=1 |

and `build_ci_matrix` returns

```
H[0,0] = −14.517037   H[0,3] = +0.237425
H[3,3] = −14.421520
```

⇒ J=1 block eigenvalues **−14.711459** and −14.227098.

The same code also returns **non-zero elements between different J** —
`H[0,2] = −0.193857` (J=1 ↔ J=0), `H[2,3] = −14.479364` (J=0 ↔ J=1). Those are
discarded by `ci._solve_blocked` so they do not affect production numbers, but
they show the builder is not angular-momentum aware.

Breaking the contributions apart (instrumenting the four helpers):

```
_diagonal_element            1s²2s 2p₃/₂ [J=1]   −14.479534   (config average)
_diagonal_J_correction       "                   −0.037502
_one_diff_element            J=1(2p₃/₂) || J=1(2p₁/₂)  −0.193857
_off_diagonal_J_correction   "                          +0.431282   ← 2.2× the base
```

`_off_diagonal_J_correction` contributes more than twice the physical
off-diagonal it is supposed to correct.

### 3.4 An independent arbiter: m-scheme Slater–Condon CI

To avoid arguing from atomx against atomx, an independent CI was written and
promoted to [`benchmarks/mscheme_ci_reference.py`](../benchmarks/mscheme_ci_reference.py).
It deliberately shares **no angular code** with the repo:

* two-electron elements from the Wigner–Eckart theorem,
  `⟨ab|r₁₂⁻¹|cd⟩ = Σ_k R^k(ac,bd) (−1)^{j_a−m_a+j_b−m_b}
   Σ_q (−1)^q (j_a k j_c; −m_a q m_c)(j_b k j_d; −m_b −q m_d)
   ⟨κ_a‖C^k‖κ_c⟩⟨κ_b‖C^k‖κ_d⟩`,
  with every 3j from `sympy.physics.wigner`;
* the Hamiltonian is built in the **m-scheme determinant basis** by explicit
  Slater–Condon rules, then restricted to the same determinant space as the
  atomx CSF list;
* only the radial integrals (`_h_one_body`, `_rk`) and the orbitals are shared
  with atomx — those are separately validated by the FAC/DBSR_HF harnesses.

Validation of the arbiter:

| check | result |
|---|---|
| He 1s² (1 determinant) | matches atomx to 1.8e−15 |
| Be 1s²2s² (1 determinant) | matches atomx to 7.1e−15 |
| 2s2p J=0/1 degeneracy with only 2p₁/₂ present | exact (no fine structure without spin–orbit) ✔ |
| closed-shell energy reproduces `2h + F⁰` analytically | ✔ |

Results on **identical orbitals and identical spaces**:

| space | protocol | N | atomx | independent | Δ (Ha) |
|---|---|---|---|---|---|
| 1s²2s² | fac_frozen | 1 | −14.575892 | −14.575892 | 0 |
| +1s²2p² | fac_frozen | 6 | −14.616689 | −14.614096 | −0.0026 |
| singles | fac_frozen | 29 | −15.400463 | −14.575893 | **−0.825** |
| singles | mcdhf | 29 | −14.659475 | −14.578383 | **−0.081** |
| SD | fac_frozen | 399 | −16.358105 | −14.632497 | **−1.726** |
| SD | mcdhf | 399 | −15.052683 | −14.619112 | **−0.434** |

Note row 6: the correct SD energy in atomx's own (bad, `fac_frozen`) orbital
basis is −14.632497 — a physically sensible correlation energy (0.057 Ha below
HF and 0.013 Ha above the exact NR value). atomx reports −16.358105.

Note row 7: the correct SD energy in atomx's own `mcdhf` orbitals is
−14.620800, i.e. *higher* than in the `fac_frozen` orbitals — the Guerra
average over 399 CSFs produces a radial compromise that is worse for the
ground state than orbitals optimised for the reference alone. The correct
answer is still 0.047 Ha above exact. atomx reports −15.052683.

Note also that in the `ref + 2s2p` space the independent CI puts the ground
state exactly at the reference energy: the correct
`⟨1s²2s²|Ĥ|1s²2s¹2p¹⟩` is ≈ 0 (Brillouin). **atomx's reference row and its
ref↔single elements are all exactly 0.000000** — the error is entirely inside
the open-shell blocks.

### 3.5 The basis is not the problem

Hypothesis tested and rejected: "the 399 CSFs are linearly dependent, so
solving `Hc = Ec` collapses." For each of the 39 configurations the number of
jj-CSFs emitted by `enumerate_jj_csfs` was compared with the exact dimension of
the corresponding m-scheme determinant space (M = 0):

```
configs = 39   Σ n_CSF = 399   Σ dim(space) = 399   mismatches = 0
```

So the CSF list is a complete, minimally-sized basis of exactly the intended
space. The defect is in `H`, not in the basis.

A related ratio is also a red herring rather than a defect: the SD list has
**399 CSFs over 99 distinct occupation vectors** (up to 16 CSFs sharing one
vector). That is *normal* for jj-coupled coupling trees — GRASP's CSFs have the
same structure — and it is exactly what the `#1`, `#2`, … labels in
`enumerate_jj_csfs` are for. The dimension check above proves the multiplicities
are right.

### 3.6 Cross-check against FAC

FAC (`sfac`, live binary) on the same configuration strings, all
configurations in **one** radial group:

| FAC deck | energy (Ha) |
|---|---|
| `1s2 2s2` alone | −14.570703 |
| `1s2 2s2` + `1s2 2p2` | −14.604106 |
| `2s2 2p2` (hollow 1s⁰) alone | −4.795798 |
| singles (9 cfg) | −14.529486 |
| SD (39 cfg, one group) | −14.485198 |

FAC is not an apples-to-apples reference here because it optimises its own
radial functions and a 39-configuration average (including the 1s⁰ hollow
configurations) compromises them badly for the ground state — the one-group SD
energy is *above* the singles energy for that reason. FAC is therefore used
only as (a) a sanity anchor for the reference and 2-configuration numbers and
(b) corroboration that a hollow `2s2 2p2` is ~10 Ha above the ground state.

**Cautionary data point, not a claim about FAC:** issuing the 39
configurations as 39 *separate* FAC groups (i.e. 39 independent orbital sets in
one CI) returns −14.716975 Ha, below the exact ground state — the same failure
signature as atomx. Any CI over non-orthogonal orbital sets that is not handled
through the proper biorthogonal transformation collapses this way. This is
recorded because it is the closest external analogue of the atomx behaviour,
and because it is a trap for anyone building the cross-check deck.

### 3.7 The orbital side is broken too (a distinct defect)

`radial_optimization="fac_frozen"` builds the radial set from the *first*
configuration with `0.001` occupations for every correlation orbital
(`solver._fac_frozen_subshells`). Measured on Be `correlation="s"`:

| orbital | protocol | ⟨r⟩ (a₀) | hydrogenic ⟨r⟩ for Z=4 |
|---|---|---|---|
| 2p₁/₂, 2p₃/₂ | fac_frozen | 2.995 / 3.054 | 1.25 (n=2) — screened, plausible |
| 3s | fac_frozen | 11.923 | 3.375 |
| 3p | fac_frozen | 19.520 / 20.335 | 3.125 |
| 3d | fac_frozen | 25.036 / 24.963 | 2.625 |

and for the default `mcdhf` protocol:

| orbital | ⟨r⟩ (a₀) | verdict |
|---|---|---|
| 1s | 0.415 | correct |
| 2s | 2.592 | correct |
| **2p₁/₂, 2p₃/₂** | **1.250** | **exactly the bare-nucleus hydrogenic 2p** |
| 3p₃/₂ | 3.125 | exactly hydrogenic 3p |
| 3d₃/₂, 3d₅/₂ | 2.625 | exactly hydrogenic 3d |
| 3s | 7.894 | screened (not hydrogenic) |

This is the collapse already documented in `_floored_occ_map`'s own docstring
("an orbital dives to a hydrogenic bare-nucleus state … Pb 6p₃/₂ → −95.6 Ha").
Its trigger is a Guerra/`fac_frozen` weight near zero: the orbital loses the
electron-repulsion term of its Fock operator and falls to the bare-nucleus
solution (or, in the diffuse case, to a near-unbound higher-`n` branch). It is
independent of the CI defect and must be fixed separately; a correct CI built
on hydrogenic correlation orbitals is still a bad calculation.

He `1s2 2p1` is the same problem in its acute form: the "2p" comes out with
⟨r⟩ = 12.4 a₀ and ε = −0.0544 Ha (a Z=1 `n≈3` solution) instead of the HF 2p
(ε ≈ −0.124 Ha), so the 1s2p energy is reported at −2.8448 Ha instead of
−2.133 Ha.

**R7 root cause located (2026-09-20).** The default SCF mode is
`mode="batch"` with `exchange="hf"`, which builds the Fock matrix through
`_build_hf_matrix_dbsr` → `_precompute_angular_coefficients` →
`DFSolver._get_direct_coef` / `_get_exchange_coef`. `_get_direct_coef(i, j, k)`
returns `self._get_WW(i, j)` for `i != j` (k = 0 only) and `WW(i, i)/2` for
`i == j`, where `WW(i, j) = Σ_c w_c ñ_c(i) ñ_c(j)` is the Guerra product.

For a target orbital whose Guerra weight `ϖᵢ = Σ_c w_c ñ_c(i)` is ~0 — an
orbital that appears only in CSFs belonging to a different `(J, parity)` block,
so its ground-state amplitudes are *exactly* zero — `WW(i, j) ≈ 0` for every
`j`. The whole two-electron part of the Fock matrix then vanishes and the
orbital is solved with the **bare-nucleus Dirac operator**. Measured on Be
`correlation="s"`: 2p at `⟨r⟩ = 1.250` a₀ and `ε = −2.0005` Ha, i.e. *exactly*
the hydrogenic Z=4 2p; 3p at 3.124 and 3d at 2.625 are likewise the exact
hydrogenic values, while 3s is properly screened (7.877) because 3s appears in
even-parity `J=0` CSFs that do mix with the reference.

The *other* potential path (`mode="orbital"`: `_build_direct_potential` +
`_build_exchange_matrix_hf`) collapses too, for a different reason: it divides
the unclamped `compute_WW` numerator by the **0.001-clamped**
`subshells[i].occupation`, scaling the mean field by `ϖᵢ / 0.001`. Both paths
therefore lose the screening; a consistent denominator alone is not enough,
because in this space `ϖ_2p` is structurally zero and the ratio is genuinely
0/0.

This also explains why `correlation="sd"` looked *better* than `"s"`: the SD
space contains even-parity `1s²2p²` (`J=0`) CSFs that mix with the reference, so
`ϖ_2p ≠ 0` there.

**R7 fix status 2026-09-20 (in flight).** An asymmetric mean field
(`MCDHFSolver._mean_field_matrices`: a spectator virtual is solved in the
reference configuration's frozen potential while the core is not re-screened
by it) is implemented in `dirac_fock.py` / `mcdhf/solver.py`. Verified so far:

* **Fixed for Be** — `correlation="s"`, default protocol: 2p₁/₂ −2.000533 →
  **−0.228585**, 2p₃/₂ −2.000107 → **−0.216499**, 3p/3d off the hydrogenic
  −0.8889. `Atom(Z=4, correlation="sd")` = −14.616668 (was −15.052683).
* **Closed shells bit-identical** — Ne −128.691925807 and Be −14.575891586
  are unchanged, which is the invariant that protects the DBSR_HF 2e−10
  agreement.
* **REGRESSION, open** — the He-like anchor
  `Atom(Z=2, configs=["1s2", "1s1 2p1"])` breaks:
  `tests/test_ci_analytic_slater_condon.py::test_helike_1s2p_j1_splitting_is_two_thirds_g1[2]`
  goes from ratio **1.0000** to **15.054**, because `G¹(1s,2p)` collapses from
  0.1405 Ha to 0.000186 Ha — the He 2p is left so diffuse that its overlap
  with 1s essentially vanishes. Z = 6 and Z = 10 still pass. The anchoring
  rule must distinguish a spectator the reference field *can* bind from one it
  cannot; the analytic gate is the guard and must return to ratio ≈ 1.0 with
  `G¹ ≈ 0.14` at Z = 2 before R7 can be called fixed.

**Decisive evidence (2026-09-20): every l>0 correlation orbital is solved
against a bare nucleus.** `MCDHFResult.orbital_energies` for the Be
`correlation="s"` space (29 CSFs, `1s2 2s2` + singles into `2p 3s 3p 3d`),
compared with the hydrogenic `−Z²/2n²` for Z=4:

| orbital | Guerra `mcdhf` | `fac_frozen` | hydrogenic Z=4 | physical |
|---|---|---|---|---|
| 1s₁/₂ | −4.736777 | −4.732549 | — | ≈ −4.73 |
| 2s₁/₂ | −0.310490 | −0.308476 | — | ≈ −0.31 |
| **2p₁/₂** | **−2.000533** | −0.228040 | **−2.0** | ≈ −0.22 |
| **2p₃/₂** | **−2.000107** | −0.215960 | **−2.0** | |
| 3s₁/₂ | −0.097224 | −0.061601 | −0.8889 | |
| **3p₁/₂** | **−0.889078** | −0.029556 | **−0.8889** | |
| **3p₃/₂** | **−0.888952** | −0.028225 | **−0.8889** | |
| **3d₃/₂** | **−0.888952** | −0.025331 | **−0.8889** | |
| **3d₅/₂** | **−0.888910** | −0.025550 | **−0.8889** | |

The Guerra protocol reproduces the hydrogenic energies *exactly* for 2p, 3p and
3d — the Fock operator those orbitals are solved with has no two-electron
term at all. `fac_frozen` gets 2p right (−0.228) but leaves 3p/3d nearly
unbound (≈ −0.03), so it has its own version of the same defect.

**The wrong-physics comment that causes it.** `_floored_occ_map`'s docstring
states the failure mode correctly — "its Fock operator loses the entire
electron-repulsion term, and the orbital dives to a hydrogenic bare-nucleus
state" — and then deliberately declines to generalise the `0.001` floor:

> "The floor is deliberately restricted to jj partners: orbitals from *other*
> NR shells (pure correlation orbitals, e.g. 2s/2p in a He-like
> `1s² + 2s² + 2p²` expansion) keep hard zeros so their shapes — to which
> correlation-sensitive observables are tuned — are not re-screened by phantom
> occupation."

That is the wrong physics: a correlation orbital **must** be screened by the
occupied electrons. What must *not* happen is the reverse — the core must not
be screened by the phantom correlation orbital. The correct mean field is
therefore **asymmetric**, and the symmetric Guerra product `WW(i,j)` cannot
express it. `C[corr, core] = q_core^ref` while `C[core, corr] = 0` is GRASP's
layer protocol, and it is the fix being implemented (see the `mean_field_coeffs`
seam described below).

**Quantified (2026-09-20, runtime instrumentation of
`_build_hf_matrix_dbsr`).** The batch path feeds the *raw* Guerra product
`WW(i,j) = Σ_c w_c ñ_c(i) ñ_c(j)` into the Fock matrix as the direct
coefficient, with no normalisation by the target's own weight. Measured
`a_direct[target, :, 0]` for the Be `correlation="s"` space:

| protocol | target 2p₃/₂ vs 1s | vs 2s | total screening | resulting ⟨r⟩(2p) |
|---|---|---|---|---|
| Guerra `mcdhf` | 0.207103 | 0.207103 | 0.414 | **1.250** (hydrogenic) |
| `fac_frozen` | 0.002 | 0.002 | 0.004 | 2.995 |

The physical screening coefficient is ≈ 2 per electron. Because the
coefficient is proportional to `ϖᵢ`, an orbital whose CI population collapses
gets a proportionally vanishing mean field. Note the ratio `WW(2p,1s)/ϖ_2p` is
1.5 in both protocols — the *mean* screening is right, but the *scale* is set
by `ϖᵢ`, so the potential collapses with the population. For a closed shell
`ϖᵢ = qᵢ` exactly, which is why the closed-shell totals match DBSR_HF to
2e-10 and why this never showed up there.

So the fix is a **normalisation**, not a new fallback: the coefficient must be
`qᵢ^eff · (WW(i,j)/ϖᵢ)` with `qᵢ^eff = ϖᵢ` when the weight is physical and
`qᵢ^eff = qᵢ^ref` (1 for a spectator virtual) when it has collapsed — which
is exactly GRASP's layer protocol. The `fac_frozen` row shows the same defect
in its own way (0.002 vs ≈2), so *both* protocols under-screen correlation
orbitals; that is the loose end behind the differing 2p radii and it needs to
be resolved together with the normalisation.

**Prototype attempt — null result, recorded so it is not repeated.** A
runtime monkey-patch of `DFSolver._get_WW` that returns the reference
product `ref[i]·ref[j]` whenever the target's Guerra weight is below `1e-9`
changed *nothing*: energies and every `⟨r⟩` came out bit-identical to the
unpatched run. So either the phantom target's coefficients are not consumed
through `_get_WW` on the path that actually solves it, or the attach point
was wrong — `_precompute_angular_coefficients(orbital_schedule)` is called
from `solve()` (line 1598), not from `__init__`, so timing is not the
explanation and the call graph needs tracing rather than guessing. Two
concrete open questions for the next attempt:

1. Which of `_get_direct_coef` / `_get_exchange_coef` /
   `_build_intra_shell_potential` actually determines the phantom orbital's
   equation, and with which `(i, j)` index roles (target first or source
   first) at each call site.
2. What normalisation DBSR_HF's `av_energy_coef` uses, given that
   `_get_direct_coef(i, j, 0)` returns `WW(i,j) = q_i q_j` — carrying the
   target's own occupation — while the closed-shell total matches DBSR_HF to
   2e-10. Any change here must leave that agreement intact.

**The fix (designed, not yet implemented).** GRASP's layer protocol: when `ϖᵢ`
is below a threshold, build target *i*'s Fock operator from the **reference
configuration's** occupations — the orbital is a spectator virtual there, and
that is the potential of the frozen lower layers. Empirically this is the
protocol `fac_frozen` already implements, and it produces a physical 2p
(`⟨r⟩ = 3.0`). Implementation must cover **both** paths (`_get_direct_coef` /
`_get_exchange_coef` for `mode="batch"`, `_build_direct_potential` /
`_build_exchange_matrix_hf` for `mode="orbital"`), and must leave the
closed-shell case bit-identical — that path is validated against DBSR_HF to
2e-10 and must not regress. A `mean_field_coeffs` seam on `DFSolver` is the
right shape but has to be threaded into the batch angular-coefficient
precomputation, not only into the orbital path.

### 3.8 External acceptance anchors for the fix

Two anchors that do not depend on any atomx internal, measured on the broken
code so the fix can be scored against them.

**NIST ASD, Be I.** `https://data.nublado.org/nist/asd_v5.12/levels/asd.004000`
(Be I, ground 1s²2s² ¹S₀ = 0; 1 Ha = 219474.6313632 cm⁻¹; exact NR ground
−14.66736 Ha):

| level | NIST (cm⁻¹) | E (Ha) |
|---|---|---|
| 1s²2s² ¹S₀ | 0 | −14.66736 |
| 1s²2s2p ³P°₀ | 21978.31 | −14.567219 |
| 1s²2s2p ³P°₁ | 21978.925 | −14.567191 |
| 1s²2s2p ³P°₂ | 21981.26 | −14.567180 |
| 1s²2s2p ¹P°₁ | 42565.4502 | −14.473418 |

With `configs=["1s2 2s2", "1s2 2s1 2p1"]`, `breit=False, qed=False`, default
`mcdhf` orbitals, the broken code returns

```
levels = [-14.709218, -14.574294, -14.513401, -14.402886, -14.230968]
```

i.e. a "ground state" 0.042 Ha *below* the exact ground state, with the level
ordering scrambled and the excited levels off by **+7,600 to +83,000 cm⁻¹**.
A correct CI must order them ³P°₀ < ³P°₁ ≈ ³P°₂ < ¹P°₁ and land within a few
thousand cm⁻¹ of NIST — the residual being the orbital basis, not the CI.

**FAC, Fe XVII.** Live `sfac` on the 7-configuration L-shell panel used by
`tests/test_ci_J_block.py`, comparing the first 37 level energies:

```
per-level |atomx − FAC| : 0 … 135.0 eV      rms 73.2 eV
```

That is consistent with the 200 eV tolerance the existing gate carries — a
tolerance that exists precisely because the kernel is approximate.

**Analytic Slater–Condon identity, He-like 1s2p.** The cleanest of the three,
because it is orbital-independent. In the non-relativistic limit the two J=1
levels of an ns n′p configuration split by exactly `2·G¹(ns,n′p)/3`; at Z ≤ 26
the relativistic correction to that relation is a fraction of a percent. With
`configs=["1s2", "1s1 2p1"]` (the 5-CSF space — note `"1s2 2p1"` is 1s²2p, a
one-electron configuration with no ³P/¹P splitting), `breit=False, qed=False`,
default `mcdhf` orbitals, the broken code gives

| Z | J=1 splitting (Ha) | `2G¹/3` (Ha) | ratio |
|---|---|---|---|
| 2 | 0.330881 | 0.093654 | **3.533** |
| 6 | 0.805684 | 0.228037 | **3.533** |
| 10 | 1.284557 | 0.363898 | **3.530** |

A **constant factor of 3.53 across the whole isoelectronic sequence** is the
signature of a pure angular-coefficient error — the radial integrals cancel in
the ratio. The exact path must return ratio 1.000.

**Separate, out of scope for the CI fix.** `Atom(Z=10, configs=["1s2", "1s2 2p1"])`
currently returns the 1s²2p levels at −102.17 Ha and the 1s² ground state at
−93.98 Ha — the excited configuration 8.2 Ha *below* the ground state. That is
§3.7 (the 2p virtual dives to the bare-nucleus hydrogenic 2p, ≈ −12.5 Ha at
Z=10), not the CI matrix. It must be fixed separately, and it means absolute
level energies will stay wrong until it is.

---

## 4. Root cause, itemised

**R1 — the diagonal is a configuration average, not a CSF expectation value.**
`matrix._diagonal_element` applies the average-of-configuration coefficients
`q_a q_b /(g_a g_b) λ^K`. That number *is* the configuration average (and it is
consistently used by `DFSolver._compute_total_energy`), but it is **not**
`⟨CSF|Ĥ|CSF⟩` for any individual CSF. Configuration average is the right
concept for *orbital optimisation* (the Guerra weights) and the wrong concept
for the *CI matrix*.

**R2 — a `CSF` has no coupling tree.** `CSF` stores subshells + `total_J2` +
`parity`. In jj coupling the exact matrix element depends on the intermediate
couplings (parentage), which are not recoverable from `total_J2` alone. The
code says so itself (`_diagonal_J_correction` docstring: "more than two open
subshells … requires a full parentage chain that is not inferable from a bare
CSF object today"). R2 is why R3 exists.

**R3 — the patches that stand in for the exact elements are wrong.**
`_diagonal_J_correction` and `_off_diagonal_J_correction` are applied only in a
narrow envelope (exactly two open subshells, each a single electron or single
hole); outside it the configuration average is used unchanged, so the diagonal
is a *mixture* of two different physical quantities depending on the CSF. Even
inside the envelope they are quantitatively wrong: for the He-like 1s2p J=1
anchor the measured error is 0.22–0.24 Ha (§3.3), and
`_off_diagonal_J_correction` is 2.2× its own base element.

**R4 — inter-`J` elements are returned non-zero.** §3.3. Currently harmless
because `ci._solve_blocked` filters them, but it means any future caller that
diagonalises `H` directly (or any change to the blocking) silently mixes
symmetries.

**R5 — genuine couplings are zeroed.** `_ci_matrix_element` returns `0.0` for
two CSFs with identical occupations and identical `(J, parity)`
("orthogonal-parentage approximation"). Such CSFs (repeated parentage, e.g.
`(j)^3` or `j⊗j'` with two coupling paths) are *not* orthogonal under `H`;
their true coupling is generally non-zero. This is a second, quieter source of
error that survives in exactly the configurations the generator emits with
`#1`/`#2` labels.

**R6 — documented missing physics.** `tests/test_ci_J_block.py` states that
"the diagonal kernel currently only includes `K=0` direct Coulomb contributions
between distinct open subshells; the `K>0` direct piece (Grant Eq. 8.3.17) is
left for follow-up", and sets a **200 eV** tolerance against FAC for Fe XVII.
The CI is knowingly incomplete.

**R7 — correlation orbitals are not solved on a physical branch** (§3.7).

**R8 — the SD expansion itself is unbalanced for Z ≤ 4.** `core=None` leaves 1s
active, so the list includes hollow 1s⁰ configurations. That is legitimate SD
space, but combined with Guerra weighting it distorts the average potential;
standard MCDHF practice is an explicit active-space/layer construction with a
frozen core (§5).

**R9 — nothing tests the invariant.** No test asserts `E(space) ≥ E_exact`,
`E(space′) ≤ E(space)` for nested spaces, `H == Hᵀ`, or `⟨ref|H|single⟩ ≈ 0`.
The existing gates are structural ("≥ 30 distinct levels", "37 CSFs") or
extremely loose (200 eV). `tests/test_sd_space_variational.py` is the only
place the bound is even mentioned, and two of its four cases are strict xfail.

**R10 — the module documents itself as an unfinished scaffold, and the
docstrings are stale.** `_off_diagonal_J_correction` still carries

> "**Current status — scaffold only.** … Always ``0.0`` in the current
> scaffold." (`matrix.py:1506`, `:1544`)

while its body returns a fully populated non-zero value — measured
**+0.431282** for the He-like 1s2p J=1 anchor (§3.3). Whatever wrote that body
did not update the contract text. Alongside it, three helpers in the
production file are pure stubs:

* `_fac_one_diff_open_spectator_exchange_coef` — always returns `0.0`
  (`matrix.py:1729`);
* `_chain_one_diff_spectator_exact_scalar` — always returns `None`
  (`matrix.py:1752`);
* `_fac_two_diff_branch9_element` — a "reachability probe … while the full FAC
  `SumCoeff` recoupling is being ported" (`matrix.py:1757`).

A reader (or an automated reviewer) cannot tell from the module whether the
angular algebra is complete. That ambiguity is itself a defect: it is why N1
survived several rounds of review.

---

## 5. What other packages do

Full primary-source survey, with quotes, section/equation numbers and
`file:line` evidence: [`SD_EXPANSION_SURVEY.md`](SD_EXPANSION_SURVEY.md).
The comparison that matters for N1:

| package | SD/active space specified by | list closed under de-excitation? | coupling trees / exact matrix elements | orthonormality | how the bound is protected |
|---|---|---|---|---|---|
| **GRASP2018 / GRASP2K** | Active set: orbital classes `c`/`i`/`*`/`m` + per-`l` limit `n`, added layer by layer; `rcsfgenerate` takes the MR, the AS, the J range and "number of excitations" (2 = S+D) | **yes, explicit** — manual §4.8 fn. 2: de-excited CSFs "should also be part of the list"; closed iff the MR is | **yes** — each CSF is three lines: configuration, per-subshell J, and how those J "are coupled together from left to right" (manual §5.1); exact angular integration in `rangular` | Lagrange constraint `C_ab = ∫[P_aP_b+Q_aQ_b]dr − δ_ab = 0` **and** `orthy.f90` Schmidt orthogonalisation; ASFs orthonormal | `Hc = Ec` is legitimate *only* because of the two orthonormality guarantees; `maneig.f90` calls `DSPEVX` on a real symmetric packed matrix, blocked by (parity, J) |
| **ATSP2K / MCHF** | Same active-set/layer concept (§4.3–4.4); `gencl` rule `s`/`d`/`sd` + virtual set | not stated; the monograph does concede that limiting SD CSFs to interacting ones needs coupling information, "**something not done in the present code**" (§4.3) | **yes** — `gencl.f` "generates all unique, possible configurations **and their couplings**" | Eq. (4.18) orthonormal CSFs; one shared orthonormal radial basis (§4.6–4.7); `orthog.f` Gram–Schmidt | same as GRASP |
| **FAC** | **no SD generator at all** — `ConfigList` does not exist and `SDConfig()` has zero callers; configurations are authored by hand, via wildcards (`2*8`) or `pfac.add_excited` | no rule documented | **yes** — `Couple()` enumerates every CSF of a configuration; exact second-quantised recoupling + Slater `R^k` | one global κ/n-indexed radial table; explicit `Orthogonalize()` Gram–Schmidt | real symmetric `H` via `DSBEV`, blocked by `DecodePJ`. **Caveat:** `ConfigEnergy(1)` applies a diagonal shift "to remove some of the errors … introduced by using a single central potential", which *does* break the bound |
| **Cowan RCN/RCG** | no automatic SD generation — a hand-authored "calculational deck" of configuration cards | n/a (explicit list) | **yes** — `cfp`/`cfgp` decks read by `CUVFD`; Slater–Condon tables `PFGD`/`PRK` with the eleven CI classes; `CALCFC` builds the LS→jj transformation per J | radials are **frozen input**, so the radial integrals are data | HUM applies to `ENERGY`; note `RCE` least-squares fitting deliberately abandons it |
| **BSR / pCI** | explicit user CSF list; **no ORMAS** | not stated | **yes** — "each configuration state function, CSF, is defined by its configuration and the coupling scheme" (CPC §12.5) | one orthonormal set per state; `PREP` rejects overlaps above `eps_ovl = 1e-8` | genuinely generalised: `diag.f90` solves `Hx = E Sx` (`DSYGV`/`DSYGVX`) with a **Cholesky** reduction `S = LLᵀ` |
| **DIRAC `krci`** (no `mrci` module) | Generalized Active Space, 1–7 subspaces with accumulated min/max electron counts | not stated | **no coupling trees — determinant-based**: elements come from determinants | one orthonormal Kramers-paired set from the MOLTRA integrals | orthonormal basis + determinant CI, blocked by double-group irrep and Ω; no non-orthogonal CI support found |
| **AMBiT / CI+MBPT** | `LeadingConfigurations` + a basis string (`10spdf`); `GenerateExcitations` moves every particle into every valence orbital | not stated | **yes** — CSFs orthonormalised by diagonalising Ĵ², then `AᵀHA` symmetrised | Ĵ² eigenbasis; virtuals admitted only "as long as they are orthogonal" | the CI matrix is bounded but the final answer is not — Eq. (13) carries `E` in the denominator (Brillouin–Wigner), hermiticity only by freezing it |

Cross-cutting conclusions:

1. **No production code computes a configuration average as a CI diagonal.**
   Configuration averaging appears in all of them, but only ever to build the
   *radial* Fock operator, never the CI matrix. atomx is the outlier.
2. **The one axis on which atomx diverges from all of them is the coupling
   tree.** GRASP, ATSP2K, FAC, Cowan, BSR and AMBiT all carry one explicitly
   (or implicitly, in Cowan's `cfp` decks). The codes that do not — DIRAC
   `krci`, and partly Cowan — avoid needing one by being **determinant-based**,
   which is exactly the route §8 Option A proposes. atomx is the only code in
   the set that is CSF-based *and* coupling-tree-free.
3. **`Hc = Ec` without an overlap matrix is a privilege, not a default.**
   GRASP earns it twice over — the Lagrange-multiplier constraint Eq. (45)–(46)
   on the *orbitals* plus Eq. (38) orthonormality of the *ASFs* (not the CSFs);
   BSR, by contrast, solves the generalised problem with a Cholesky reduction.
   atomx inherits the shortcut with neither guarantee asserted in code.
4. **The S-only expansion is understood everywhere as adding nothing for HF
   orbitals** — MCHF monograph §3.7.2 gives Brillouin's theorem and its
   corollary. That is the textbook reason atomx's `"s"` run gaining 81 mHa
   should have been treated as a red flag rather than a result.
5. **Orbital collapse is a named, documented failure mode** with a standard
   remedy chain (MCHF §3.9: "When the probability distribution of the lowest
   eigenfunction changes rapidly as a function of increasing nuclear charge
   from the outer well to the inner well, orbital collapse is said to occur").
   GRASP screens it by requiring orbitals to "have the same number of nodes as
   the corresponding hydrogen like orbitals" (manual §4.4) and by optimising one
   layer at a time with everything else frozen (§4.5) — exactly what §8's R7 fix
   recommends and what atomx does not do. `dbsr_hf` dispenses with node counting
   in favour of projection operators (manual §4.3). No source states the
   *zero-occupation* trigger explicitly; that mechanism is an original
   observation of this audit, not literature.

---

## 6. Why `"s"` looked fine and `"sd"` did not

Both are broken; `"s"` hides it:

* Be `"s"` (mcdhf) → −14.659475 vs exact −14.6674: 8 mHa high. That looks like
  a good correlation energy — but the correct answer *in the same 29-CSF space
  on the same orbitals* is −14.578383. The apparent agreement is the
  coincidence of a spurious 81 mHa gain landing just short of the exact value.
* `"sd"` adds the J-multiplets of many more open shells, so many more broken
  blocks contribute coherently in the same direction and the spurious gain
  overshoots the exact value: by 0.385 Ha at the default `mcdhf` protocol and
  by 1.69 Ha under `fac_frozen`.
* `fac_frozen` is worse than `mcdhf` in both cases because the dived
  correlation orbitals (§3.7) enlarge the wrong matrix elements.

The lesson for the gate design: **a correlation energy that is "close to
exact" is not evidence of correctness.** The invariants that catch this are
(a) `E ≥ E_exact`, (b) nested-space monotonicity, (c) agreement with an
independent matrix-element implementation on identical orbitals.

---

## 7. What a correct implementation should deliver

With the arbiter in place the target numbers are already measurable:

| case | correct value in atomx's own space | note |
|---|---|---|
| Be singles, `fac_frozen` orbitals | −14.575893 | equals the reference HF energy to 1 µHa — Brillouin's theorem, as it must |
| Be SD, `fac_frozen` orbitals | −14.632497 | 35 mHa above exact NR |
| Be singles, `mcdhf` orbitals | −14.578383 | |
| Be SD, `mcdhf` orbitals | −14.619112 | 48 mHa above exact NR |
| exact NR Be 1s² ¹S | −14.66736 | Drake |
| Dirac–Coulomb Be 1s² ¹S | ≈ −14.670 | NR exact + one-electron relativistic shift |

A correct SD-MCDHF with properly optimised correlation orbitals (i.e. after R7
is also fixed) should approach −14.666 for Be. The 2-configuration MCHF anchor
(−14.6166, literature and reproduced by atomx row 1) is already correct because
a closed-shell-plus-closed-shell space has no open-shell J-blocks — which is
also why the existing MCDHF tests pass.

---

## 8. Fix options

**Option A — exact matrix elements via the m-scheme projection (recommended).**
For each (J, parity) block: build the determinant basis of the block with
M = J, express each CSF as a vector in it, and evaluate `H` by Slater–Condon
(the oracle's algorithm, which is already validated).  This is exact,
parentage-free, and directly testable against the oracle.  It is also the route
two of the surveyed packages take: DIRAC `krci` is determinant-based for
exactly this reason, and Cowan's code is effectively determinant-based behind
its `cfp` decks (§5).  Cost: the block determinant space can be larger than the
CSF space; needs the M = J reduction and blocking.  It fixes R3, R4, R5 and R6
at once.

**Option B — thread coupling trees through `CSF` and implement the Grant §8.3
recoupling coefficients directly.** Preserves the current data flow and is
O(N_CSF²) rather than determinant-based, but requires a full CFP/recoupling
implementation and a rewrite of the CSF generator.  This is the GRASP/ATSP2K/FAC
route (§5) and is what the existing "reachability probe … while the full FAC
`SumCoeff` recoupling is being ported" comment was reaching for.

**Option C — restrict the feature honestly.** While A or B is developed, make
`correlation="s" | "sd"` raise (or emit a loud `provenance` warning and a
documented accuracy caveat) and remove/replace the numbers it currently feeds
into `docs/VERIFICATION.md`, `benchmarks/REPORT.md` and any property
calculations.

Independent of the choice, the following are cheap and should land first:

* **R7 fix:** solve correlation orbitals as the next layer in the frozen
  potential of the previous one (GRASP's layer protocol), or enforce the
  hydrogenic node count and reject branch-jumped solutions — a diagnostic
  already exists in the shooting engine's `ier_max`/ceiling guard.
* **R8 fix:** expose `core` explicitly and default to freezing `1s` for all Z.
* **R9 fix:** add invariant gates — `E ≥ E_exact`; monotonicity across nested
  spaces; `H - Hᵀ = 0`; zero inter-(J, parity) elements; `⟨ref|H|single⟩ ≈ 0`
  on HF orbitals.

---

## 9. Verification assets produced by this audit

Promoted into the repo:

* `benchmarks/mscheme_ci_reference.py` — the independent m-scheme
  Slater–Condon CI.  Angular algebra from sympy 3j; only the radial integrals
  (`matrix._h_one_body` / `matrix._rk`) and the orbitals are shared with atomx.
  Validated to 1e−15 on closed shells (He 1s², Be 1s²2s²) and against FAC on
  the singles space.
* `tests/test_ci_matrix_vs_mscheme_reference.py` — **3 passing invariants**
  (closed-shell equality; per-configuration CSF count == determinant-space
  dimension; `H − Hᵀ = 0`) and **5 strict `xfail`s** carrying the measured
  numbers (2-open-subshell block, single-config J-block, `"s"`,
  inter-`J` elements, `"sd"`).  Strict xfail means the day the kernel is fixed
  these become XPASS failures and must be flipped to assertions.
* `docs/VERIFICATION.md` — the "Known limitations" entry was rewritten (the
  old text repeated the retracted "the CI matrix is not the suspect" guess)
  and a new gate table row was added.
* `docs/PHYSICS_NUMERICS_REVIEW_2026-09-19_FIXES.md` — the N1 entry now
  records the corrected diagnosis, the measurement table and the retraction of
  the original guess.
* `CHANGELOG.md` — entry under *Changed — validation honesty*.
* [`SD_EXPANSION_SURVEY.md`](SD_EXPANSION_SURVEY.md) — the primary-source
  survey of GRASP2018/GRASP2K, ATSP2K/MCHF, FAC, Cowan RCN/RCG, BSR/pCI,
  DIRAC `krci` and AMBiT/CI+MBPT behind §5, with a quote/`file:line` evidence
  appendix and an explicit "could not verify" list.  Its citation-integrity
  header also corrects four wrong arXiv IDs that had been circulating in the
  review brief.

Reproduction (from the repo root; interpreter `.venv/bin/python`;
`MPLCONFIGDIR` set to a writable temp dir):

```
# headline reproduction
python -c "from atomx import Atom; a=Atom(Z=4, correlation='sd'); a.solve(); print(a._result.total_energy)"

# the gate module
python -m pytest tests/test_ci_matrix_vs_mscheme_reference.py -q -p no:randomly

# auxiliary probes used for the tables above
python /tmp/atomx_n1/bisect2.py     # per-config scan, fac_frozen  (§3.2)
python /tmp/atomx_n1/parts.py       # element-by-element breakdown (§3.3)
python /tmp/atomx_n1/compare.py     # atomx vs oracle, same orbitals (§3.4)
python /tmp/atomx_n1/scheck.py      # correlation='s'  (§3.4)
python /tmp/atomx_n1/sdcheck.py     # correlation='sd' (§3.4)
python /tmp/atomx_n1/orb_probe.py   # orbital branch census (§3.7)
```

## 10. What is *not* claimed here

* The exact Dirac–Coulomb reference for Be is quoted as ≈ −14.670 Ha, i.e. the
  non-relativistic exact value plus the ~2.5 mHa one-electron relativistic
  shift; no independent relativistic CI was run to pin it more tightly.  Every
  inequality used in this document needs only the *non-relativistic* exact
  value −14.66736 Ha as a lower bound on the true energy, so the conclusion is
  insensitive to this number.
* FAC enters only as a corroborating anchor (§3.6).  The authoritative
  comparison is the oracle, because it runs on atomx's own orbitals and the
  same space; FAC optimises its own radial functions and its answer moves by
  0.23 Ha depending on the group structure of the deck.
* No claim is made about the size of the error outside the light-element cases
  measured here.  The Fe XVII row is *not* re-measured against the oracle in
  this audit — the 200 eV FAC tolerance in `tests/test_ci_J_block.py` is quoted
  as the code's own admission, not as a measurement made here.
* ~~The fix is not implemented.~~ **Superseded 2026-09-21: the fix landed
  (§11) as §8 option A.**  The remaining open defects are R7 (correlation
  orbitals on the wrong branch — still open, and now the *dominant*
  residual: it is what keeps the Fe XVII f-ratio and the He-like Z=2 gauge
  ratio out of band) and R8 (unbalanced SD expansion for Z ≤ 4).
* **Degenerate subshell parentage is a basis choice, not a seniority
  label.** The highest-weight subshell states come from
  ``atomx.angular.mscheme.highest_weight_states``, whose null-space basis
  inside a *degenerate* subshell J (e.g. ``(7/2)^4`` has J = 2 and J = 4
  twice each) is an arbitrary SVD choice. Individual ``H_rs`` elements
  between CSFs that differ only in that index therefore depend on the
  basis within the degenerate subspace; the block eigenvalues and the
  trace do not. The basis is deterministic for a fixed input (verified
  across fresh processes by ``scripts/check_determinism.py``) and the CSFs
  are orthonormal, so every energy quoted here is well defined — but the
  ``#1``/``#2`` labels in ``CSF.label`` must not be read as seniority
  quantum numbers. Light-element benchmark cases are unaffected (they have
  no degenerate repeated parentage).

## 11. Fix landed (2026-09-21)

`build_ci_matrix` now evaluates ``H_rs = <CSF_r|H|CSF_s>`` exactly, by
expanding every CSF over the Slater determinants of its own configuration
and projecting (``src/atomx/methods/mcdhf/exact_ci.py``; §8 option A). The
old assembly survives as ``matrix._build_ci_matrix_approximate`` for
comparison only — there is no fallback to it.

| case (breit/qed off, ns=113, ks=9) | before | after | m-scheme oracle, same orbitals |
|---|---|---|---|
| Be `["1s2 2s2", "1s2 2s1 2p1"]`, fac_frozen | −14.708567 | −14.5758915 | −14.5758915 (2e−15) |
| Be singles (29 CSF), fac_frozen | −15.400463 | −14.5758932 | −14.5758932 (5e−15) |
| Be SD (399 CSF), fac_frozen | −16.358105 | −14.6324974 | −14.6324974 (2e−14) |
| Be SD (399 CSF), mcdhf protocol | −15.052683 | −14.6191124 | −14.6191124 (2e−14) |
| exact NR Be ground state (Drake) | −14.66736 | −14.66736 | — |

Both the singles and the SD numbers are now ≥ the exact ground state, and
the singles space lands *on* the reference energy: Brillouin's theorem for
a singles expansion on (near-)HF orbitals, which §6 identified as the
textbook signature the old numbers violated.

Two findings from the fix that change how the §3 numbers should be read:

* **The oracle had a bug.** `benchmarks/mscheme_ci_reference.py` skipped the
  exchange integral whenever the *direct* one vanished by angular selection
  rules (``if v == 0.0: continue``). That is not a symmetry, and it broke
  rotational invariance: the Be `1s2 2s1 2p1` J=1 level came out at
  −14.490934 Ha from the 2M=0 block and −14.530753 Ha from the 2M=2 block,
  where the exact (and the one-determinant 2M=4) value is −14.513834 Ha.
  Every open-shell off-diagonal number in §3.4 comes from the pre-fix
  oracle and is 19–22 mHa too low: Be singles −14.595092 → −14.575893, Be SD
  fac_frozen −14.654637 → −14.632497, Be SD mcdhf −14.620800 → the loop
  converges to different orbitals (see below). The oracle is fixed and now
  carries an explicit rotational-invariance invariant.
* **The mcdhf-protocol SD number moved for an orbital reason.** The Guerra
  weights *are* the CI ground eigenvector, so a different (correct) matrix
  changes the orbitals the outer loop converges to. Be SD under the default
  protocol now gives −14.6191124 Ha (oracle agreement 1.6e−14 at those
  orbitals), 1.7 mHa from the pre-fix target of −14.620800 — which was the
  *buggy* oracle's value in the *buggy* loop's orbitals. Re-measured on
  identical orbitals: the old loop's orbitals with the exact matrix give
  −14.6177230 Ha (oracle agreement 1.2e−14), so the 1.7 mHa is the orbital
  fixed point moving, not the matrix. `Atom(Z=4, correlation="sd")` with
  the default flags (Breit + QED on) returns −14.616668 Ha, against
  −15.05 Ha before the fix.
