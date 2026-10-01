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
