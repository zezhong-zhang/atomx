# Σ⁽²⁾ implementation spec — two-body core-polarisation screening

Status: spec only (2026-10-01).  `methods/ci_mbpt.py` raises until this
lands.  Target: the multiplet-splitting error that survives full-SD CI —
measured on Eu³⁺ 4f⁶ as ⁵D₀−⁷F₀ = +4727 cm⁻¹ (sparse full-SD, protocol
orbitals pending v5) vs NIST 17270.

## Physics

H_eff = H_CI + Σ⁽¹⁾ + Σ⁽²⁾ (Dzuba–Flambaum–Kozlov).  Σ⁽²⁾ corrects the
two-electron matrix elements between valence CSFs at second order in the
Coulomb interaction over a closed core.  Three diagram families:

1. **Bubble (screening)** — the dominant one for 4fⁿ term splittings: a
   valence electron polarises the core, the other feels the induced field.
   RPA-resumable; first order in the expansion:

       δg(abcd) = − Σ_{m∈core} Σ_{r∈virt}
                    g(ab,mr) g(mr,cd) / (ε_m − ε_r)
                 × (two time orderings; J-coupled sum below)

2. **Hole-line** — valence pair virtually excites a core electron:

       δg(abcd) = + Σ_{m∈core} Σ_{r∈virt} [g(ab,mr) g(mr,cd) + exch.]
                    / (ε_a + ε_b − ε_m − ε_r + Δ)

3. **Particle-particle** — both valence electrons virtually in the
   continuum pair; small for term splittings, include for completeness.

In J-coupled form every product runs through the same V_J(abrs) machinery
`methods/mbpt.py` already computes for the closed-shell E⁽²⁾; the core is
Eu⁹⁺ [Xe] (solve separately, `V^(N−6)` reference), virtuals from
`build_virtual_basis` (bound + positive-energy box states — keep the
existing energy-window discipline).

## Injection point

Mirror the Σ⁽¹⁾ pattern exactly: a `two_body_shifts` callable passed to
`build_ci_matrix` that wraps the `("r", k, a, c, b, d)` radial key,

    R^k(ac,bd) → R^k(ac,bd) + δR^k(ac,bd)   (per k, per orbital 4-tuple)

where δRᵏ is the J-coupled second-order sum for that tuple.  The
determinant/CSF projection then places the screened integrals with the
correct angular weights, on- and off-diagonal, with no further changes.

Computed once per orbital 4-tuple and cached — the count is the same
order as the exact engine's radial cache, not the determinant count.

## Validation protocol (gate before any production use)

1. **Analytic:** closed shell + one raised pair — Σ⁽²⁾ must reduce to the
   second-order energy correction already computed by `mbpt2_energy`
   (cross-check against the existing tested code path).
2. **Sign/magnitude:** on Eu³⁺ 4f⁶, Σ⁽²⁾ must LOWER the ⁵D₀−⁷F₀ gap
   (screening reduces effective Gᵏ) by an amount within a factor ~2 of
   the measured residual; if it raises the gap the bubble sign/denominator
   is wrong — do not ship.
3. **Gauge:** with Σ⁽¹⁾+Σ⁽²⁾ on, the A-value gauge ratio must not degrade
   (transition operators unchanged, states improve).
4. Convergence: virtual energy window and κ range sweeps must move ⁵D₀ by
   less than the claimed error bar.

## Provenance discipline

`mbpt_provenance` records the Σ⁽²⁾ diagram set, window, and reference
core; CI+Σ⁽¹⁾+Σ⁽²⁾ is the first configuration allowed to be labelled
"CI+MBPT" (see the guard in ci_mbpt.py).

## High-performance architecture (design requirement, 2026-10-01)

The naive cost model — δR^k per orbital 4-tuple with an (m, r) core×virtual
loop inside — is O(#tuples × #core × #virt) Python-level V_J evaluations:
for Eu9+ [Xe] (~20 core shells) and a window of ~200 virtuals that is
10^4 tuples × 4·10^3 pairs = 10^7-10^8 evaluations.  Unacceptable.  The
design below is the performance contract for the implementation.

**Key structural fact:** at fixed total pair angular momentum J, the
bubble term is a matrix product in pair space,

    δG^{(J)} = - S^{(J)} · D^{-1} · S^{(J)T}

with S^{(J)}[(ab), (mr)] = ⟨ab; J|g|mr; J⟩ (valence-pair × core-virtual-pair
supermatrix) and D the energy denominator diagonal.  Same for the
hole-line family with core-pair × valence-virtual-pair blocks.  So:

1. **Reduce Σ⁽²⁾ to dense linear algebra per J channel.**  Build the
   S^{(J)} blocks once (this is the only place radial integrals appear),
   then one `dgemm` per J per diagram family.  Pair-space dimensions are
   10^2-10^3 per block — laptop BLAS territory, and the 5090 makes it
   trivial via CuPy later with zero algorithm change.

2. **Batched radial precompute.**  Every distinct R^k the S-blocks need is
   enumerated up front and evaluated in one batched pass over the numba
   convol kernels (the same discipline as the sparse-CI phase-1 value
   table: enumerate keys once, compute once, never inside a hot loop).

3. **prange over J channels** — blocks are independent; reuse the
   block-partition pattern from the sparse assembler.

4. **Compute-once, reuse-everywhere caching.**  The δR^k table depends
   only on (core reference, virtual window), never on the CI space: build
   it once, cache on disk (npz keyed by a provenance hash), and let every
   subsequent CI diagonalization — every M sector, every J extraction,
   every active-space variant — ride the same table through the radial
   key.  One Eu9+ build amortises across the whole lanthanide campaign.

5. **Numerics discipline:** denominators floored away from zero (quasi-
   degenerate core/virtual crossings clipped and counted); float64
   throughout the pair-space algebra (the dgemm path is not where to
   economise); the analytic gate (reduction to mbpt2_energy) doubles as
   the numerical-accuracy check at machine precision.

Budget target: Eu9+ Σ⁽²⁾ table build in ≤ 5 min single-node CPU
(≈ S-block assembly dominates; the dgemms are milliseconds), so the full
CI+MBPT loop for one ion stays in "minutes per configuration" territory
end to end alongside the sparse engine.

## Measured verdicts (2026-10-01, Eu3+ anchor run)

**Same-J bubble: NOT SHIPPED (gate 2 verdict).**  Full-SD sparse CI,
clean workspace ([Xe]+statistical 4f, 14 genuinely-open shells, 8 J
channels all attractive, screening -13..-18 mHa per pair diagonal):

    7F0: -10846.319581 -> -10846.322445 Ha  (-630 cm-1, common-mode)
    5D0: 20376 -> 20417 cm-1   =>  Delta(5D0-7F0) = +41 cm-1

The same-J restriction captures the diagonal (common-mode polarisation)
part only; the differential physics lives in the cross-J-channel induced
interaction (independent vertex J's, 6j-recoupled) and the hole-line
family.  ci_mbpt keeps raising; the machine (S-blocks, dgemm, CG
recoupling, hook, gates) is retained and tested.

**Reframing of the +3195 residual:** it is not "screening" alone — it is
ALL correlation beyond the 12 discrete SD virtual shells (continuum,
higher n, higher sectors).  The full Sigma(1)+Sigma(2) operator set
should be understood as completing the virtual spectrum.

## Physics completeness audit for the Eu3+ free-ion anchor

| physics | in model? | expected size (5D-7F) | machinery |
|---|---|---|---|
| Dirac-Coulomb, SD valence CI | yes | — | sparse engine |
| same-J bubble | yes (measured) | +41 (common-mode -630) | mbpt_sigma2 |
| full Sigma (multi-J induced + hole-line + Sigma1) | NO | main dish (~3000) | next increment |
| Breit (Gaunt+retardation) in sparse CI | **yes (today)** | ~50-200 | include_breit, gated vs exact engine |
| triples sector | no | sub-leading | absorbed partly in Sigma |
| QED (SE/VP) on 4f | no | ~cm-1 | qed modules, needed at CF stage |
| term-dependent orbital relaxation | no (one orbital set) | second order | EOL variants |
| frequency-dependent Breit, two-loop | no | negligible here | — |

Breit reached the sparse engine through the same antisymmetrised-additive
pattern as two_body_delta (phase-1 value table); parity test asserts
sparse-with-Breit == reference-with-Breit == build_ci_matrix(
include_breit=True) on Be.
