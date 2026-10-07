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

Gates (tests/test_slater_scaling_transfer.py, slow): G1 Pr fit + linearity,
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

## Root-cause roadmap (user direction, 2026-10-01) — "让电子云不再虚胖"

The fitted (s2,s4,s6) = (−27%, −8%, −12%) are not free parameters —
they are the MEASURE of what the frozen-statistical-DF machinery is
missing, dominated by semi-core correlation.  Physics reading
(literature-consistent magnitudes): 4d semi-core correlation screens
F^k(4f,4f) by ~10-20%; the rest is 5s/5p screening + orbital
relaxation lost to the statistical average.  The fundamental fix is to
put that physics back:

1. **解冻 4d 半芯** — CI/MBPT with 4d⁻¹ sectors active (4d↔4f dynamic
   correlation tightens the 4f cloud).  Cost estimate with the sparse
   machinery: core-hole patterns multiply the 4f⁶ space by ~50-100×
   (hours-scale anchor runs; staged: single-4d-hole subspace first).
2. **全电子 DKH3/四分量变分自洽** — term-specific (EOL) 4f spinors
   instead of the statistical average; the repo's MCDHF/
   radial_optimization machinery is the entry point.
3. **NEVPT2/CASPT2 级半芯相关 (4d,5s,5p)** — beyond the CI+MBPT we
   measured as insufficient (the Sigma2 arc: second-order pair
   channels into high virtuals are orthogonality-suppressed — but
   SEMI-CORE holes are a different, unsuppressed channel; that is
   exactly why the fitted s_k are as large as they are).

Success criterion: ⁵D₀ walks to ~17270 ab initio (no anchor); until
then the two-tier calibration is "the honest thermometer".  Fallback
(per user): per-ion Cowan fits — Tier A already supports them (fit
more than 3 parameters on the anchor ion's own levels).

Measured partial support already in hand: the bubble arc measured
core-polarisation as COMMON-MODE (−9300 abs / −53 cm⁻¹ differential
5D-7F) — the 4d MONOPOLE screening is inside the DF; what it cannot
give is the 4d-hole EXCHANGE screening of F^k, which is the missing
piece the slater_scaling s_k currently absorb.

## Rejected alternative: single-parameter uniform alpha (2026-10-01)

Proposal (user): all F^k share one radial scale, F^k -> F^k/alpha,
alpha ~ 1.183 from 20434/17270; shape corrections <2%; zeta untouched.
TESTED AND REFUTED on three independent measurements:

1. Pr constrained uniform fit: RMS 460 vs 210 (3-param) — per-k
   differentiation carries real information (F^2-dominated terms want
   -27%, F^4/F^6-dominated -8..-12%).
2. Eu 5D0 under alpha=1.183: +3164 -> **+6735** (WORSE), and the 7F_J
   ladder degrades at high J (+10.9% -> +14.1%) — opposite of the
   predicted 5-8% compression.
3. Nd 4F3/2: +34% -> +54% (worse); 3-param transfer: -1.5%.

PHYSICS OF THE FAILURE: the derivation assumed Delta-E(5D0) =
c2 F2 + c4 F4 + c6 F6 with positive coefficients.  The MEASURED
Jacobian row is (-4216, +11995, +15194) — SIGN-ALTERNATING: the
5D-7F splitting is G^k-EXCHANGE-dominated, and 4f^6 multiplet gaps
are not same-sign combinations of the F^k.  Uniform F^k scaling
therefore cannot translate into uniform gap scaling.  Additionally
alpha = 20434/17270 treats the whole gap (including its F^k-independent
part) as proportional to F^k — circular.

VALID parts of the proposal, already adopted: zeta must NOT be scaled
(measured: 7F fine structure is F^k-orthogonal and stays at its zeta
level under the 3-param scaling); the semi-core-correlation origin;
constraint-disciplined fitting.
