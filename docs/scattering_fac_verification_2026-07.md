# Scattering & spectrum verification against FAC (2026-07, updated 07-14)

Goal: verify atomx's scattering (electron-impact excitation/ionization,
dielectronic recombination, Auger) and spectrum against FAC
(`~/code/fac`, FAC 1.1.5, both `sfac` and `pfac` runnable here), and fix
deviations with **exact physics**. This documents the step-by-step physics +
implementation comparison and the fixes applied.

## Status after the 2026-07-14 deep physics pass

| Process | Before | After | Anchor |
|---|---|---|---|
| DW excitation Ω | 5–50× off, erratic in Z/E | **rms 9.4%, max 13%** (Z=10/26/54 panel) | live FAC CE, per-partial-wave |
| DW EII σ | ~4 orders erratic in E | **2–25% at E/th≥2**; ≤2× at E/th=1.5 | live FAC CI `SetCIQkMode('dw')` |
| Auger per-level Γ | 1.2–15× (config-avg proxy) | 0.5–4× (exact jj-level amplitude) | FAC AITable KLL panel |
| DR Burgess kernel | <1% (FAC widths in) | unchanged, passes | FAC AI+TR tables |

## The DW excitation fix — exact formula, derived and cross-validated

The 2026-07-13 investigation traced the deviations to the B-spline box
continuum. The 07-14 pass went further: even with correct δ(E)-normalized
shooting waves, atomx's collision strengths were ~17× low with the *right
per-l shape* — a wrong constant + missing exchange + one-of-two jj pairs.
The exact collision strength, derived independently from S-matrix theory
(`S = 1 − 2πiT`, `|S−1|² = 4π²|⟨f|V|i⟩|²`, electron-spin average ½) and
verified against FAC's internal factor chain
(`Ω = 8·Σ_k ang²/(2k+1)·Σ(sd+se)²` with 1/√k-amplitude waves,
faclib/orbital.c:1988 — algebraically identical):

    Ω = 2π² Σ_pairs pop · Σ_{J,κin,κout} (2J+1) |D_J − E_J|²

with δ(E_Hartree)-normalized continuum waves. Implementation
(`dw_excitation_cross_section`, rewritten):

- **D_J, E_J by explicit m-summation** (`_cc_direct`, `_cc_exchange`,
  lru-cached). The collapsed-k form (FAC's) hides a J-alternating
  recoupling phase in the exchange cross-term; a first attempt with 6j
  formulas flipped the interference sign at odd l — found by comparing
  per-l with FAC's `SetCEPWFile` dump, which shows the exchange/direct
  ratio per partial wave. m-summation has no phase-convention risk.
- **Exchange**: destructive 20–30% at low E (FAC per-l: 0.67–0.99),
  including pure-exchange k channels (dqk=0 blocks in the pw dump).
- **All jj destination pairs** (e.g. 2p₁/₂ AND 2p₃/₂) with vacancy
  population factors; Ω is the multiplet-summed collision strength (the
  J-sum already counts the full source line — pop = vacancy only; the
  q_a scaling cancels against the M-degeneracy count, verified H-like
  q=1 vs He-like q=2 against FAC).
- **Continuum**: δ(E)-normalized adaptive-Numerov shooting solver in
  FAC's mean potential `−Z/r + ((N−1)/N)V_Hartree` (bare −Z/r for
  H-like, exactly as FAC), Coulombic extrapolation beyond the B-spline
  box, energy-scaled grids (shooting h·p ≤ 0.1, Slater fine grid
  h·p ≤ 0.2; the shooting amplitude was verified against the analytic
  free solution to 0.3% — an apparent 20% "error" was a phase-drift
  artifact of a naive fit, relativistic vs nonrelativistic dispersion).
- **High-l tail**: geometric-ratio top-up per FAC's TopUpQk, with the
  ratio capped at 0.85 (the raw formula diverges when the profile has
  not entered geometric decay — He⁺ showed 50× spikes).

Validation vs live FAC 1.1.5:
- H-like Ne 1s→2p per-level: 0.96–1.06 over E/th = 1.05–9.
- H-like Ne 1s→3d (pure k=2): 1.05–1.15.
- He-like multiplet sum: 1.09–1.15.
- Panel Z=10/26/54 × E/th=1.05/2/5: **rms 9.4%** (gate ≤10%, passing).

Key FAC facts established along the way (for future work):
- FAC CE defaults: DW partial waves to l=60 on a sparse grid with cubic
  log-l interpolation (`EXCLMAX/EXCLCB=60`), Coulomb-Bethe ratio tail
  (`coulomb.c PrepCoulombBethe/CoulombBetheTail`), Bethe `b·ln(x)`
  asymptote blended at high E. The sfac wrapper argument order is
  `SetCEPWOptions(tol, max, qr, kl_cb)` — different from the C API.
  `SetOption('excitation:topup', 0)` disables the tail.
- ~99% of the He-like 1s→2p Ω at E/th≤2 comes from l≤8; the Bethe/CB
  completion dominates only at E/th ≳ 8 (up to ~46% at E_s=8×th).

## The DW EII fix

`_eii_dw_total` rewritten on the same machinery — ionization treated as
excitation into the continuum:

    σ(E) = π/p² · q_a/(2j_a+1) · ∫₀^{Tmax/2} dE₁ · 2π² Σ_{J,κin,κ1,κ2} (2J+1)|T_J|²

with three δ(E)-normalized shooting waves, exact m-summation angular
coefficients, coherent antisymmetrized `T_J = D_J − E_J`, E₁ integral to
the symmetric split, Y^k caching, geometric l-tail (capped). The old
box-continuum kernel (4 orders of magnitude swings across 200 eV) and its
K-shell BEB hybrid hack are gone.

Parity vs live FAC CI in **pure-DW mode** (`SetCIQkMode('dw')` — FAC's
*default* CI is QK_BED, a different binary-encounter-dipole-scaled model):
2–25% at E/th ≥ 2 across Z=10/26/54 He-like K-shell; up to ~2× at
E/th=1.5 where FAC-CI's ejected-wave treatment (`max_kl_eject`
restriction, log-y ejected grid) differs from straight DWBA — a
documented model difference. Note: the old benchmark passed absolute
energies into FAC's threshold-relative CI grid (mismatched energies) and
the frozen CSV was fabricated; both fixed.

He⁺ vs Younger 1981: the coherent |D−E|² prescription runs up to ~2×
hotter near threshold than Younger's exchange treatment — the well-known
post/prior/Peterkop EII exchange-model spread. FAC-DW parity is the
exactness anchor.

## Auger per-level amplitudes — determinant CI, traces exact

Two new modules:

1. `atomx/scattering/autoionization/jj_level.py`: exact amplitude for a
   *pure* jj-coupled basis state by explicit m-summation — hole
   conjugation `h†_{JM} = (−1)^{j−M}c_{j,−M}`, sequential
   `((h⊗e₁)J_a⊗e₂)J` and equivalent-pair `(h⊗(e²)J_p)J` schemes,
   coherent direct−exchange, δ(E) shooting continuum,
   `Γ = 2π|A|²` (≡ FAC's `4Σp²/(2J+1)` under their 1/√k convention,
   recombination.c:2168). Verified exactly M-independent.

2. `atomx/scattering/autoionization/mscheme_ci.py`
   (**ResonanceManifold**): a self-contained bitmask-determinant CI over
   the resonance manifold. Hamiltonian (orbital energies +
   all active-active Coulomb via Slater–Condon), J² classification, and
   the Auger decay amplitude are all evaluated over the *same*
   determinants — fermionic signs consistent by construction. Supports
   multi-configuration manifolds (e.g. 2p⁵{3s3p + 3p3d} for a given
   (J, parity)). Yields level energies *and* widths.

Key validation steps (Fe XVII KLL vs FAC AITable):
- With the **FAC-like orbital basis** (ground-config solve + frozen-core
  virtuals from `mbpt._build_virtual_basis`), the **mixing-invariant
  manifold traces match FAC**: J=7/2 2p⁵3d² trace ratio **0.984**;
  J=1/2 2p⁵3s3p **0.878** (FAC lists only above-cut levels) — the
  amplitude machinery is exact.
- Per-level after CI: strongest-level widths at 1.078 / 0.56 / 0.73 /
  1.149 / 1.066 of FAC across the five (J, parity) blocks (was 1.2–15×
  with the config-average proxy). Pinned by
  `tests/test_auger_mscheme_ci.py` (trace ≤20%, top level ≤35%).
- The pure-basis-state panel with average-level orbitals was 0.49–4.2;
  switching to ground-optimized orbitals moved every d-state into range
  (ILEV 190: 4.2 → 0.77), confirming the orbital-basis diagnosis.

Remaining residual (the 3p²-dominated blocks at 0.56–0.73 with strength
spread over neighbors): the virtual-orbital potential. atomx's V^{N−1}
virtuals give ε(3s) = −18.0 Ha where FAC's optimized DFS potential gives
≈ −13.7 (full-N inner screening) → larger Slater integrals → stretched
multiplet splittings → different mixing. Completion path: generate the
3l orbitals in atomx's `exchange='fac'` local DFS potential (already
validated to 1e-3 against sfac for structure) and/or include the
dr34/2s-hole configurations in the manifold union.

FAC's own machinery for reference: `recombination.c AutoionizeRate`
(2021–2170), spectator 6j in `structure.c:6421` (AngularZxZFreeBound),
`ReducedCL` (angular.c:529) ≡ atomx `reduced_cl_j2`.

## Remaining follow-ups

- CE at very high E (E/th ≳ 8): add the Bethe `b·ln x` completion beyond
  the geometric tail (FAC blends Born there; atomx is ~4% low at E/th=9).
- EII near threshold: FAC-CI's ejected-wave conventions if sub-2×
  near-threshold parity is needed.
- Auger: CI-mixed level amplitudes + ground-config orbital basis (above).
- DR end-to-end: re-run `bench_fac_dr.py` with jj_level widths once the
  CI-mixing wiring lands (the Burgess kernel already passes at <1% with
  FAC widths).
