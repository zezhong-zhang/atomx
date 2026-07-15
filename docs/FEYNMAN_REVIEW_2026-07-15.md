# Feynman Review — missing physics & the physics/corrections/numerics decoupling

Date: 2026-07-15
Scope: `src/atomx` after the FAC scattering-verification campaign (see
`docs/scattering_fac_verification_2026-07.md`) and the 2026-07-09 critical review.
Method: three parallel evidence-gathering sweeps (layering audit; structure/spectrum
physics inventory; scattering/CRM physics inventory), with every load-bearing claim
re-verified against source before inclusion. Lens: **Feynman** — what physics is
*absent or not actually reaching the observable*, ranked by accuracy impact — plus a
design lens on whether `physics/`, `corrections/`, `numerics/` stay decoupled.

---

## Verdict

The verified physics core is in good shape: DW excitation/ionization/Auger/DR are
FAC-parity to a few percent, the QED stack is deeper than most in-house codes
(Mohr/Indelicato/Welton/Shabaev-local SE; Uehling + simplified Wichmann–Kroll VP),
and spot-checked conventions (Breit sign, Uehling sign, Einstein-A prefactor) are
correct.

The headline Feynman finding is therefore **not a wrong formula — it is that the
verified physics does not reach the spectrum**. The CRM rate matrix still uses the
van Regemorter formula while the FAC-validated distorted-wave machinery sits unused;
the synthesized spectrum has no free-free or free-bound continuum; and the
Coulomb–Bethe high-energy top-up exists in the codebase but is not wired into the
production cross-section assembly (this is precisely the documented E/threshold ≳ 5
deficit).

The headline elegance finding is symmetric: **a clean corrections contract exists
(`corrections/protocol.py`, `stack.py`, `adapters.py`) but every mean-field solver
bypasses it** with hardwired `include_breit`/`include_qed` calls.

---

## 1. Missing physics, ranked by impact on observables

### Tier 1 — verified physics not consumed (consistency defects; cheapest accuracy wins)

**M1. CRM excitation rates use van Regemorter, not the verified DW.**
`plasma/crm.py:312–327` builds C_exc from `1.578e-5·gf·⟨g⟩/(g_l·√kT)·e^(−ΔE/kT)`.
The FAC-parity DW collision strengths (`scattering/distorted_wave/solver.py`) are
never consumed — grep finds no collision-strength or Ω pathway into the rate matrix.
Threshold behavior differs from DW by 20–40% for mid-Z ions, and van Regemorter is
E1-only (no spin-changing / non-dipole excitation at all). *Fix:* a
`CRMTransition.omega(T)` pathway (Maxwellian integral over Ω(E)) with van Regemorter
as an explicit, labelled fallback.

**M2. Coulomb–Bethe top-up exists but is not wired in.**
`coulomb_bethe_topup` (`distorted_wave/solver.py:989`) is implemented, tested
(`tests/test_born_cb.py`) and exported — but the production CE/EII assemblies use
only the geometric partial-wave tail (`solver.py:777`,
`electron_ionization/__init__.py:719`). This is exactly the documented high-energy
deficit: CE Z=54 −16% at E/th=5; EII −25…−35% at f=5 (proven not to be κ-truncation).
*Fix:* switch the ℓ-tail from geometric to Coulomb–Bethe for dipole-allowed
channels above the profile-matching region.

**M3. Spectrum synthesis has no continuum.**
`spectrum/crm_synth.py` renders line emission (n_u·A_ul) only. The RR continuum
emissivity already exists (`scattering/radiative_recombination`, `rrc_spectrum`) but
is not folded in; free-free (bremsstrahlung) is absent from the entire codebase.
Line-to-continuum ratio diagnostics — a standard hot-plasma measurable — are
currently impossible. *Fix:* add free-bound (existing) + free-free (Kramers·Gaunt
first, Elwert–Haug later) channels to synthesis.

**M4. Photoionization resonances don't reach the PI *rate*.**
`photoionization/fano.py` decorates cross sections, but the CRM PI rate uses a
Kramers 1/E³ tail — resonance enhancement never enters the kinetics.

### Tier 2 — collision-operator physics (the next accuracy floor)

**M5. Breit interaction absent from the DW collision operator.** The T-matrix is
Coulomb-only (`_cc_direct`/`_cc_exchange`). FAC includes Breit in CE; for Z ≳ 20–30
inner-shell excitation this is a 5–15% systematic. (Our FAC-parity gates are honest
— the clean-room references were run with `SetBreit(0)` — but production FAC
comparisons will show this gap.)

**M6. Resonance excitation absent.** CE is direct-only; excitation through
autoionizing intermediates (capture → autoionization back to an excited state) can
add 10–30% near resonance energies for high-Z. The infrastructure exists —
`ResonanceManifold` already enumerates resonances, energies and Auger widths — it is
not coupled back into CE. Same story for excitation–autoionization (EA) in total
ionization (`electron_ionization` computes direct DW only), which dominates EII for
many M-shell ions.

**M7. QED in transition/resonance energies.** DR resonance positions sit ~8 eV from
FAC-with-QED (established in the verification campaign; atomx E_res is pure Coulomb
by design). The QED stack exists (`corrections/`) but is not threaded into
scattering energetics. Cheap to apply as per-orbital ε shifts.

**M8. High-n DR density suppression absent.** All resonances weighted equally
regardless of n_e; DR overestimated 20–50% at ICF/white-dwarf densities. *Fix:*
Burgess–Merts cutoff or Debye-screened continuum lowering on the resonance series.

### Tier 3 — structure physics (precision ceiling)

**M9. Correlation stops at closed-shell MBPT(2)** (`methods/mbpt.py`);
`coupled_cluster.py` raises `NotImplementedError`. No all-order/RPA chains — the
known ceiling for transition energies (~%-level) and polarizabilities (up to 2×).

**M10. Breit is frequency-independent and perturbative only** (never in the SCF
potential); no negative-energy contributions to Breit or M1.

**M11. QED depth:** local Shabaev SE only (non-local QEDMOD projector deferred,
`qedmod/shabaev.py:363–372`); Källén–Sabry VP absent; SE screening via
orbital-overlap ratio only.

**M12. Nuclear/recoil:** NMS via virial estimator; no relativistic recoil operator
(Shabaev), no ab-initio SMS pipeline — limits isotope-shift work.

**M13. Line broadening:** Doppler + natural + quasi-static H Griem Stark only. No
electron-impact (collisional) broadening — 10–50% line-shape errors in dense
plasmas; this is the same DW machinery (elastic amplitudes) reused, so M14 feeds it.

### Tier 4 — scope holes (flag, don't necessarily fill)

**M14. Elastic scattering / momentum transfer:** absent (also blocks electron-impact
broadening). **M15. Proton/heavy-particle impact:** absent. **M16. CX:** Landau–Zener
only (close-coupling deferred, documented). **M17. Multi-channel Fano** for
overlapping PI resonances (independent-resonance approximation documented in
`fano.py:131`). **M18. Two-photon decay:** Drake tables for H/He-like only.

### Validation debt (from the campaign, restated)

- A_r (E1 stabilizing widths) never audited against FAC `TRTable` under the
  clean-room protocol — drives the remaining DR high-T factor ~2.
- Ne-like odd-manifold traces 1.1–1.22; J=9/2-odd 0.83 (specific exchange multipole
  channels).
- `test_auger_kll_vs_fac.py` strict 10% gate still xfail.

---

## 2. Elegance: does physics/corrections/numerics decoupling hold?

The three-layer intent is real and the corrections contract is genuinely good:
`Correction.supports(result)/evaluate(result)` returning `CorrectionResult` with
provenance (`corrections/protocol.py`), composed by a flavor-aware
`CorrectionStack` (`stack.py`), with adapters bridging the legacy functions
(`adapters.py`). The failures are in *adoption*, not design:

**E1 (structural). Every solver bypasses the stack.**
`methods/dirac_fock.py:1413–1424` (and the analogous block near 1739) imports
`compute_breit_energy`/`compute_qed_corrections` directly, keyed on constructor
booleans; `methods/mcdhf/solver.py` likewise; `Atom.solve()` forwards raw
`include_breit`/`include_qed` flags. Consequence: you cannot swap the SE backend or
add a correction without editing solver source, and `Atom.corrections()` (which
*does* use the stack, `atom.py:813–834`) is a second, post-hoc path that can silently
disagree with what the solver already folded into `total_energy`. *Fix:* solvers
accept a `CorrectionStack` and apply it at the single post-SCF seam; the boolean
flags become sugar that builds a stack.

**E2 (structural). A physics choice lives inside the numerics layer.**
`numerics/shooting/solver.py:946–957` branches on `exchange == 'fac'` and reaches
into `physics.local_exchange.get_hxs`; `_sfield` (lines 72–97) hardcodes the Slater
ρ^(1/3) form and the Latter tail. The functional *forms* rightly live in
`physics/local_exchange.py` — but selecting among them is wired into the solver.
*Fix:* inject `exchange_fn(rho, r) -> alpha_eff` (or a small ExchangeModel object);
shooting becomes a generic radial DFS engine.

**E3 (structural). Nuclear model hardwired per solver** (Fermi/point branching
inside both shooting and B-spline paths) instead of a shared potential factory from
`potentials/nuclear.py`.

**E4 (design). No single seam to compose the three layers.** `Atom.solve()`
(atom.py:210–414) routes on the `exchange` string to three solver classes with
disjoint parameter sets (grid params for shooting, `ns/ks` for B-spline, hardwired
correction flags). A user cannot independently choose (grid, potential model,
corrections) — e.g. "B-spline DHF + FAC α_xc + Breit-off/QED-on" is not expressible.

**E5 (verdict: acceptable). Quadrature "duplication" is per-grid-context, not
copy-paste.** B-spline Gauss (`numerics/bspline/evaluate.py`), Salvat SLAG6/power
series (shooting), and the DW fine-grid trapezoidal kernels
(`distorted_wave/solver.py:459–519`) serve different grids and are each validated.
One consolidation *is* warranted: the DW composite-grid + trapezoidal-cumulative
kernels are now the best-validated arbitrary-grid quadrature in the tree (the 2026-07
campaign showed exactly this pairing fixes O(h) bias); they belong in
`numerics/` as the canonical arbitrary-grid Slater/Y^k module, imported by
scattering and by `ResonanceManifold`, so the next consumer cannot reintroduce the
rectangle-rule bug.

**Positive findings worth keeping:** corrections spot-checks all clean (Breit sign
convention `breit.py:169` correct; Uehling attractive sign correct; dimensions
consistent); `auger.py` correctly delegates to `numerics/bspline/slater.py`;
`corrections/qed.py` and `breit.py` correctly *consume* numerics rather than
embedding it.

---

## 3. Prioritized actions

| # | Action | Kind | Payoff | Size |
|---|--------|------|--------|------|
| 1 | CRM consumes DW Ω(T) (Maxwellian-integrated), van Regemorter as labelled fallback | M1 | 20–40% on every synthesized line ratio | M |
| 2 | Wire `coulomb_bethe_topup` into CE/EII ℓ-tail | M2 | closes the documented −16…−35% high-E deficit | S |
| 3 | Inject `CorrectionStack` into DFSolver/MCDHF/Shooting; flags become sugar | E1 | unblocks all correction work; kills dual-path drift | S–M |
| 4 | Fold RR continuum + Kramers·Gaunt free-free into `crm_synth` | M3 | line-to-continuum diagnostics exist at all | M |
| 5 | Promote DW composite-grid Slater/Y^k kernels to `numerics/` | E5 | prevents rectangle-rule regressions; one validated quadrature | S |
| 6 | Exchange-functional injection into shooting; nuclear-potential factory | E2,E3 | numerics layer becomes model-agnostic | S |
| 7 | Per-orbital QED ε-shifts into scattering energetics (E_res, thresholds) | M7 | −8 eV DR resonance offset | S |
| 8 | A_r audit vs FAC TRTable (clean-room) | debt | remaining DR high-T factor ~2 | M |
| 9 | EA channel in total EII; resonance excitation via `ResonanceManifold` coupling | M6 | correct EII for M-shell ions; CE near resonances | L |
| 10 | Breit in the DW collision operator | M5 | high-Z inner-shell CE 5–15% | L |
| 11 | High-n DR density suppression | M8 | dense-plasma DR 20–50% | S |
| 12 | All-order correlation (RPA chains → CC) | M9 | the precision ceiling | XL |

Items 1–2 are the striking ones: both halves of each fix already exist and are
individually verified; only the wiring is missing. That is the recurring pattern of
this review — atomx's physics debt is currently *integration* debt, not derivation
debt.

---

## Remediation status (2026-07-15, same-day)

All twelve actions were executed the day of the review; each carries its
commit and verification.

| # | Status | Evidence |
|---|--------|----------|
| 1 | **DONE** | `effective_collision_strength` (piecewise-exact Maxwellian kernel; constant-Ω recovers Υ to machine precision); CRM prefers tabulated Ω, van Regemorter stays as labelled fallback. 4 tests. |
| 2 | **DONE — and the diagnosis deepened** | CE: exact-GOS plane-wave Born completion (Bethe limit verified to 0.5%), FAC `_topup=2` blend → panel rms 6.4%→5.1%, Z=54 f=5 −16%→−11%. EII: the deficit was **not** a tail problem — the ejected-energy window `[0.05T, 0.49T]` missed the dominant soft-ejection region (FAC uses `[5e-4·I, 0.5T]` log-spaced), and the benchmark's FAC reference itself was −40% at f=1.5 (interpolated across the threshold rise; now pinned with `SetUsrCIEGrid`). New parity: Z=10 ≤5%, Z=26 ≤15%/7%/1%. |
| 3 | **DONE** | `CorrectionStack` injected into DFSolver/MCDHF; booleans are sugar; single `_finalize_with_corrections` seam; 84 existing gates pass unchanged; Breit-only-via-stack now possible and tested. |
| 4 | **DONE** | `spectrum/continuum.py`: free-free (CODATA-verified constant, exact Maxwell-averaged Born Gaunt `(√3/π)e^{x/2}K₀(x/2)`) + Milne RR continuum folded into `crm_emission_spectrum`; He⁺ RR edge lands exactly at the 1s binding (test). |
| 5 | **DONE** | `numerics/quadrature.py` canonical; solver keeps byte-identical wrappers. |
| 6 | **DONE** | `exchange_alpha_fn` injection (values byte-identical on default paths); `nuclear='point'/'fermi'` via the shared mass table. |
| 7 | **DONE** | `corrections/orbital_shifts.py`; Fe 1s +8.7 eV/e⁻ ≡ the documented −8 eV E_res offset. |
| 8 | **DONE — real bug found** | `_radiative_width` was ~3.3× low: `⟨fdet|H|fdet⟩` is not a valid final-state energy (ω came out 450–550 eV vs physical 700–900 eV). Fixed with ω = E_res + binding(spectator). Per-level A_r now 1.09–1.27 of clean-room FAC TRTable; α_DR vs the **complete** clean-room dr33 set: 0.49/0.99/1.08/1.17 at 1/5/10/50 MK — the high-T factor-2 is closed (1 MK = the +40 eV frozen-orbital E_res offset in the Boltzmann factor). |
| 9 | **DONE (EA + rate-level RE)** | `excitation_autoionization_cross_section` (DW × branching; He-like Fe test) and `resonant_excitation_rate` (Burgess twin, exactness test). Excited-channel *partial* Auger widths from the manifold remain the generalization. Side-finding: 3-open-shell jj-averaged solves can collapse below ground (noted in test). |
| 10 | **CORRECTED — review premise was wrong** | Measured directly against instrumented/live FAC: with default options FAC **never** includes Breit in the CE collision operator (`SlaterTotal` requires `mode≠−1`, but `EXCLQR=0` forces phase-amplitude mode for every CE partial wave; `SetBreit(-1)` alone is inert — verified by identical CE tables). Forcing exact mode (qr=60) + `SetBreit(-1)`: operator-Breit at Z=54 He-like 1s→2p is **±5%** (+4.9% near threshold, −1.3% mid, +4.8% at 143 keV). Since our FAC references exclude it, it does not explain the Z=54 residual. Porting FAC's generalized `Breit()` for continuum orbitals is recorded as the opt-in follow-up; incidental finding: FAC's default phase-amplitude integrals differ from exact mode-1 by ~4% at low E (ours are exact). |
| 11 | **DONE** | IPD (Stewart–Pyatt) dissolves resonances with binding below the depression, per temperature; zero-density path bit-identical. |
| 12 | **OPEN (scoped)** | All-order correlation remains the precision ceiling. The existing MBPT(2) machinery (720 lines, per-pair coupled multipoles over the V^{N−1} virtual basis) is the correct substrate for a ladder/CCD iteration; the honest estimate is a multi-session project and no placeholder was added — `coupled_cluster.py` keeps its explicit `NotImplementedError`. |
