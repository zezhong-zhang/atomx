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

## DR end-to-end — ab-initio, 1.0–2.0× of FAC (was 10³–10⁴×)

`ResonanceManifold.dr_levels` extends the determinant CI with ab-initio
resonance energies (level minus closed-shell-ion energy in the same
frozen-orbital accounting) and E1 stabilizing widths (one-orbital
Slater–Condon dipole amplitudes, coherent over the CI vector, summed
over all hole-filling final states with per-final ω³).  All 237 dr33
resonances (2p⁵3l3l′ + 2s-hole 2s¹2p⁶3l3l′, every (J, parity) manifold)
build in ~5 s; the Burgess kernel then gives

| T (MK) | α_atomx/α_FAC(dr33) |
|---|---|
| 1 | 1.01 |
| 5 | 1.77 |
| 10 | 1.91 |
| 50 | 2.05 |

Pinned by `tests/test_dr_manifold_e2e.py` (ratios ∈ [0.5, 2.5]).
Decomposition of the high-T excess: ~25% is *real* — FAC's reference
drops levels below `SetAICut(1e-4)` (75 listed of 237); the rest is the
orbital-basis/energy chain: resonance energies ~60 eV high
(frozen-orbital, no relaxation) inflate ω³ in A_r, and the compact
V^{N−1} 3l radials inflate the dipole integrals.  The low-T agreement
(1.01) benefits from partial cancellation between the energy shift
(Boltzmann suppression) and the width excess.

## FAC-equivalent orbital basis — built and validated; H accounting isolated

The shooting solver now exports the converged DFS potential and waves
(`ShootingResult._rv/_pa/_qa`), and `dfs_frozen_orbital` solves excited
orbitals in that frozen potential — exactly FAC's OptimizeRadial scheme.
Validated against FAC 1.1.5 directly (`tests/test_dfs_frozen_orbitals.py`):
occupied ε match `GetPotential` to ≤0.3%, and the frozen 3s/3p/3d
eigenvalues match `WaveFuncTable` to 0.1–0.4%
(−19.44/−18.24/−16.38 vs FAC −19.46/−18.26/−16.44 Ha).

Feeding this exactly-FAC basis into the manifold CI *flipped* the
residuals (traces 0.88–0.98 → 1.3–1.6; E_res +60 eV → −60 eV), which
isolates the true remaining error: the manifold Hamiltonian's one-body
accounting.  Using dressed mean-field eigenvalues ε *plus* explicit
active–active v_as double-counts within the active space; the earlier
V^{N−1} numbers benefited from partial error cancellation.  The exact
scheme is standard active-space CI bookkeeping: core-folded one-body
integrals ⟨a|h + V_core|b⟩ (including off-diagonal same-κ elements)
with the FAC-basis orbitals.  That is the single remaining item for
per-level exactness in Auger/DR.

## Auger exactness campaign — verified-exact inventory (2026-07-15)

Head-to-head component verification against FAC 1.1.5, all PASSING:

| component | test | agreement |
|---|---|---|
| DFS potential r·V(r) | `WaveFuncTable` cols 2+3 | 0.8% max |
| continuum wave (s, 300 eV) | inner-region P vs shooting solve | **0.02% amplitude, 0.2% shape** (convention √(2/π) exact) |
| bound 3s/3p/3d shapes | `WaveFuncTable` | 0.2–0.6% |
| orbital eigenvalues | `GetPotential`/`WaveFuncTable` | 0.1–0.4% |
| Slater–Condon amplitude formula | brute-force second quantization | exact (6/6) |
| golden-rule constant | 2-state J=11/2 manifold trace | **1.01** |
| manifold dimensions | level counts per (J, parity) | exact (all 11 blocks) |
| weak-level Coulomb CI | vs no-QED FAC (`SetSE(0)` etc.) | 1.01–1.10 |

Clean-room comparison protocol established: FAC with `SetAICut(0.0)`,
`SetSE(0)/SetVP(0)/SetMS(0,0)/SetBreit(0)`, dr33 = 2p-hole configs only —
the exact counterpart of the atomx manifold model (`/tmp` drivers,
reference `d0.lev/d0.ai`).

**Localized remaining deviation**: manifold traces are channel-class
dependent — pure k=3 channels exact (J=11/2: 1.01), 3s3p (k=1 direct +
k=0 exchange) 1.18, 3p3d (k=2 direct − k=1 exchange) 1.72, J=9/2-odd 0.82.
Since the trace is unitarily invariant (= Σ_det |A_det|², independent of
H/mixing) and every wave ingredient is verified, the deviation sits in
specific multipole-class radial channels of the Auger matrix element.
QED (on by default in FAC) accounts for −8 eV of E_res and up to 6× on
individual weak levels; the remaining constant E_res offset vs no-QED
FAC is −75 eV.

## 2026-07-15: THE quadrature bug — found via instrumented sfac, fixed

Instrumenting `AIRadialPk` (env-gated printf of per-channel sd/se,
rebuild, restored afterwards) gave FAC's raw radial integrals.  Direct
comparison: my integrals were uniformly 1.14–1.16× FAC's — while a
dense-window uniform grid reproduced FAC to 0.02%.  Root causes, both
fixed in `distorted_wave/solver.py`:

1. **`_make_fine_grid` was uniform**, sized only by the continuum
   wavelength: 37 points below r=0.4 — ~5 points across the Fe 1s peak.
   Now a composite grid with local step
   `dr ≤ min(0.15/p_loc(r), r/48)`, `p_loc = √(k² + 2 z_inner/r)`,
   resolving both the inner bound structure and the oscillation.
2. **Rectangle-rule cumulative sums** in the Y^k kernels — O(h) biased
   on non-uniform grids (hidden by cancellation on uniform ones).  Both
   fused kernels upgraded to trapezoidal cumulatives (2nd order on any
   grid).

The same bug had been corrupting the manifold Hamiltonian's v_as —
fixing it repaired the resonance energies too.

**Results after the fix** (vs the clean-room no-QED 2p-hole-only FAC):
- Li-like Fe KLL: per-level Auger widths **1.005 / 1.023 / 1.006 /
  1.003 / 1.013** of FAC; traces 0.986 / 1.007; E_res within 2–6 eV.
- Ne-like Fe dr33: E_res within **0.1–0.4 eV** (was −30); even-parity
  traces 1.02–1.08; J=11/2 1.002; low-energy levels 0.95–1.15.
- CE benchmark: **rms 9.4% → 6.4%**, 8 of 9 points ≤ 5.2% (the
  quadrature bias had been degrading CE too).
- EII rms 0.33 → 0.29.

## 2026-07-17: J-resolved radiative line strengths — determinant-CI lines

The transitions calculator distributed configuration-average strength
with FAC's UTA population factor: multiplet totals correct, individual
lines inside multi-open-shell multiplets not (measured 0.37–14× off per
line on Ne-like Fe 2p⁵3l → 2p⁶).  New engine
`atomx/properties/transitions/ci_lines.py` reuses the FAC-verified
m-scheme machinery (`manifold_from_configs` determinants + Hamiltonian +
J²): both configurations are enumerated over one shared orbital basis,
each (J, M) block is diagonalized (J-purified inside degenerate
clusters), and the exact reduced E1 element comes from the coherent
Slater–Condon one-body amplitude at fixed M_i = J_i:

    S = |<f||d||i>|² = (2J_i+1) · Σ_{M_f} |amp(M_f, q=M_f−M_i)|²

(3j orthogonality; only one q per (M_i, M_f), so eigenvector signs
cancel per term — no phase matching across M blocks).  Exact sum rule
pinned at 1e-10: Σ_lines S = Σ_{det_i,det_f,q} |⟨det_f|d_q|det_i⟩|², the
mixing-invariant configuration-pair total.

Validation vs live clean-room FAC 1.1.5 TRTable (same protocol; TR
PrintTable verbose=1 segfaults — verbose=0 stores the rme with
gf = ω·rme²/3, A = 2α³ω²gf/g_u, S_std = rme²/2, verified exact in the
A_r audit).  Fe XVII 2p⁵3s → 2p⁶ and 2p⁵3d → 2p⁶ (the 3C/3D multiplet),
`exchange='gu'` DFS basis:

| line | A ratio atomx/FAC | in-multiplet fraction ratio |
|---|---|---|
| 3s (2p₃/₂ hole) J=1 | 1.021 | 1.004 |
| 3s (2p₁/₂ hole) J=1 | 1.011 | 0.994 |
| 3d weak J=1 | 1.034 | 1.031 |
| 3D | 1.033 | 1.031 |
| 3C | 0.994 | 0.992 |

3C/3D line-strength ratio 3.729 vs FAC 3.876 (0.962); transition
energies +2.4 to +2.9 eV of 730–830 eV.  Reference + provenance:
`tests/data/fe17_tr_fac_cleanroom.json`; gates in
`tests/test_ci_line_strengths.py` (per-line A ∈ [0.85, 1.20], relative
intensities ≤5%, sum rule 1e-10, one-electron consistency: multiplet
total ≡ Σ|⟨κ_h‖C¹‖κ_e⟩|²R² exactly, per-line redistribution =
physical 2p₁/₂/2p₃/₂ hole mixing, CI/pure = 0.820/1.397).

Wiring: `compute_transitions(..., line_resolution='ci')` (also
`Atom.transitions`) routes two-configuration single-hole-pair atoms
through the CI engine (determinant cap 2000/block; closed-shell ground
re-solved internally in the gu-DFS pure-Coulomb basis);
`TransitionResult.resolution` records `'ci'`/`'uta'` per row.  Anything
else — >2 configs, multi-hole configs, above-cap blocks, velocity
gauge, magnetic multipoles — falls back to UTA.  E2 implemented via the
generic r^L C^L element (sum rule verified); **M1 and velocity gauge
not implemented in the CI engine**.  Default NOT flipped: on 2-config
atoms the flip would replace the MCDHF/UTA values (energies ~70–95 eV
higher than FAC, per-line A 0.37–14× off) with the CI values wholesale —
a large intentional physics change that also swaps level labels
(`State_n` → jj-determinant labels) and silently ignores the atom's
Breit/QED flags (CI manifold is pure Coulomb by design).

## Remaining follow-ups

- Bethe/Born completion at very high E: CE Z=54 E/th=5 (−16%) and EII
  E/th=5 (−25..−35%, shown NOT to be tail truncation by raising
  kappa_max) — FAC blends Born there.
- Ne-like odd-parity manifolds: traces 1.1–1.22 and high-energy-level
  clusters at 1.3–1.9 (J=9/2-odd 0.83) — level-position/mixing fine
  structure (candidates: FAC's ResidualPotential one-body diagonal
  corrections, bound-wave linear interpolation, E-order pairing in the
  comparison).
- E_res: QED contributes −8 eV (FAC includes self-energy by default;
  atomx manifolds are pure Coulomb by design so far).
- CE at very high E (E/th ≳ 8): add the Bethe `b·ln x` completion beyond
  the geometric tail (FAC blends Born there; atomx is ~4% low at E/th=9).
- EII near threshold: FAC-CI's ejected-wave conventions if sub-2×
  near-threshold parity is needed.
- DR: dr34 (3l4l′) manifolds need n=4 virtuals — same machinery.
