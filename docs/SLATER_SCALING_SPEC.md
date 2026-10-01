# Slater-Scaling Spec (F^k calibration, R4 route)

Opened 2026-10-01.  Resolves the F^k anchor problem documented in
CI_MBPT_SIGMA2_SPEC.md (the Sigma2 arc verdict: no second-order
valence-pair route closes the Eu 5D0 gap).

## Measured evidence

| ion | observable | bare CI | scaled | NIST |
|---|---|---|---|---|
| Pr³⁺ (fit source, 11 levels, no ¹S₀) | RMS | 6078 | **210** | — |
| Eu³⁺ (transfer) | ⁵D₀ | +3164 | +1499 (ratio-loose) / −257 (ratio-anchored) | 17270 |
| Nd³⁺ (transfer) | ⁴F₃/₂ | +34.1% | **−1.5%** | 11699 (Wyart 2006) |
| Nd³⁺ (transfer) | ⁴I₁₁/₂ | +2.5% | −0.6% | 1897 |

Fit (s₂,s₄,s₆) = (−0.273, −0.080, −0.119); magnitudes match the
textbook correlation-screening (Cowan κ) reduction.

## Protocol (two tiers)

**Tier B — series transfer (ions without data).**  Fit once on a
source ion's NIST levels; FREEZE; apply unchanged across the series.
Robust for F²-class observables (~200 cm⁻¹).  LIMITATION (measured,
gate G3): the s₄/s₆ RATIO is only weakly pinned by low levels —
pure G^k-ratio probes (Eu ⁵D₀) swing ~±1500 cm⁻¹ with the anchoring.
The Pr ¹S₀ (highest, most F^k-sensitive term) pins the ratio — free
-ion value uncertain (hosts 46-50k); fitting with a guessed ¹S₀ gave
the −257 transfer but that anchor is not citable yet.

**Tier A — anchor-ion calibration (the pipeline default).**  For the
ion being anchored, fit its OWN NIST levels.  FIT ONLY F^k-CLASS
OBSERVABLES (inter-multiplet gaps, e.g. ⁷F₀→⁵D_J): mixing in fine
-structure spacings (ζ-class residuals 10-540 cm⁻¹ that s cannot fix)
wrecks the least squares (measured: a ζ-polluted fit takes s₆ = −0.64
and doubles the RMS).  ζ stays ab initio (2-11% measured); refine
later with a ζ-scaling hook if needed.

## Module (methods/csf/slater_scaling.py)

- `make_slater_delta(od, bsp, Z, {k: s})` — antisymmetrised per-k
  scaling increment Σ_k s_k[term(ab,cd) − term(ab,dc)] (scaling only
  the first ordering breaks antisymmetrisation).
- `free_ion_levels(od, bsp, Z, ns, pats, scale)` — pure-J ladder by
  M-subtraction; **M ≥ 0 sectors only** (negative-M inclusion is
  reversed and exhausts the subtraction); the TOP sector carries the
  stretched J = M_max manifold; full (dense) sector spectra required.
- `fit_slater_scaling(residuals_fn)` — Gauss-Newton with a numerical
  Jacobian (energies are near-linear in s; one step gets most of the
  way — gate G1).

Gates (tests/test_slater_scaling.py): G1 Pr fit + linearity,
G2 scaled-H Hermiticity, G3 Eu transfer HONEST BOUND (ratio
sensitivity documented, not hidden), G4 Nd transfer regression.

## Perf note

`two_body_delta` runs on the numba phase-1 path (value computed once
per distinct quadruple): 0.3 s per scaled 4f² ladder vs 86 s with
`method="python"` — a 280× trap.

## Follow-ups

1. Pr³⁺ free-ion ¹S₀ reference (NIST ASD lookup) → refit with the
   ratio anchored; re-measure the Eu transfer.
2. Eu Tier-A anchor: fit ⁵D₀/₅D₁/₅D₂ gaps (refs to verify) with the
   Pr s as prior (minimum-norm correction); then wire into the CF
   anchor pipeline.
3. ζ-scaling hook (one-body SO scaling) — the ⁷F/³H/⁴I fine structure
   runs 2-11% high; a 4th series parameter would close it.
4. Sm³⁺ 4f⁵ / Pm³⁺ transfer validation (series breadth).
5. Public per-k two-body term iterator in the MAT layer (this module
   and the assembler both reach into private internals).
6. External review of this module (workflow) before the anchor
   pipeline adopts it.
