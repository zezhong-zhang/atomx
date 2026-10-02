# XFEL Capability Gap Analysis — atomx vs XATOM / FAC / Cowan (2026-10-02)

Goal: enable atomx for XFEL (x-ray free-electron laser) calculations —
sequential multiphoton ionization of atoms under intense pulsed x-ray
irradiation, transient-configuration spectroscopy and scattering, and
the observables of single-particle imaging.

Grounding: the vendored XATOM source (xraypac, /tmp build) is read at
module level; atomx claims are taken from `src/` and the 2026-10
validation campaign (see trace.md and the gates).

## 1. What XFEL physics needs

1. **Sequential multiphoton ionization cascade** — during one pulse an
   atom absorbs N photons; after every absorption/decay event the
   electronic structure is re-solved for the new configuration and all
   rates (PI, Auger, fluorescence) recomputed.  This is XATOM's
   `track_elec_config` core (Young et al., Nature 466, 56 (2010);
   Son–Young–Santra, PRA 83, 033402).
2. **Electron-by-electron Monte Carlo** over the pulse for fluence-
   dependent charge-state/spectral yields (`xatom_mc`, `xatom_mc_fly`).
3. **Rate-equation occupation dynamics** over the configuration graph
   (`yes_occupation` mode).
4. **Secondary channels**: direct double ionization / shakeoff
   (`xatom_shakeoff` + fitted data), electron-impact ionization by
   photoelectrons (`impact_ionization`), Compton (`compton.f90`),
   two-step / REXMI resonant multiphoton absorption (`xatom_twostep`).
5. **Transient-config imaging observables**: elastic scattering
   (f0(q), anomalous f1/f2) and spectra per transient charge state
   (`xatom_imaging`, `spectra.f90`).
6. **Pulse model**: spectral photon fluence, temporal envelope.

## 2. Capability matrix

| Capability | XATOM | FAC | Cowan | atomx today | Gap |
|---|---|---|---|---|---|
| Structure (avg config) | HFS | DFS/DHFR | HFS+fit | **DHF/MCDHF/CI** (stronger) | — |
| Subshell PI σ(ω) | ✓ | ✓ | — | ✓ validated vs FAC 4-23% (E1+finite-ω) | near-edge resonances (Fano) partial (`fano.py` exists) |
| Auger rates | LS chain | jj/J-resolved | UTA tools | **3 tiers** (LS-exact 1.0000 vs XATOM; UTA-0 vs FAC; determinant-CI 0.89-1.11 of FAC) | manifold-H dressing (10%) |
| Fluorescence A | ✓ | ✓ | ✓ | E1 validated; M1/E2 partial | M1/E2 vs FAC panel |
| RR / DR | — | ✓ | — | ✓ validated | — |
| EII / CE | ✓ | ✓ | — | DW + BEB | high-Z DW basis limits |
| Shakeoff / direct double | ✓ (fitted data) | — | — | **missing** | new |
| Compton profile | ✓ | — | — | planned (matrix TODO) | new |
| Two-photon / REXMI | ✓ | — | — | two_photon tests exist (bound-bound) | XFEL sequential-resonance driver |
| **Sequential-PI cascade (on-the-fly re-solve)** | ✓ `track_elec_config` | — | — | **missing** — `plasma/cascade.py` is a DECAY cascade (FAC SetCascade analog), not a photoionization cascade | **the core gap** |
| MC over pulse | ✓ `xatom_mc(_fly)` | — | — | missing | new |
| Occupation rate equations | ✓ | CRM (Te/ne) | — | CRM chassis ✓; **PI rates use Kramers/hydrogenic placeholders** — the real `scattering/photoionization` σ(ω) is not wired in | wiring |
| Charge-state/scd tables | ✓ | ✓ | — | CRM produces them | wire real rates |
| Elastic scattering f0/S(q) | ✓ | spinor integral | — | ✓ (in flight, Waller-Hartree) | per-transient-config |
| Anomalous f1/f2 | — | via PI (same formalism) | — | ✓ **2026-10-02**: f2=ωσ/(4πα), f1 KK, TRK gates | discrete lines into tabulation |
| Imaging (per-config diffraction) | ✓ | — | — | ingredients only | driver |
| Speed per configuration | ~ms (HFS+analytic) | s | fast | **seconds-minutes** (B-spline SCF) | fast path + caching |

## 3. The gap, ranked

1. **Photoionization cascade driver** (the XATOM core): configuration
   graph, per-configuration rate table (PI per subshell + Auger +
   fluorescence), on-the-fly structure re-solve with memoisation,
   pumped by a photon fluence spectrum; outputs = charge-state
   distribution, electron/photon spectra, transient populations.
   Everything atomx needs for the RATES already exists and is
   validated — this is an integration increment, not new physics.
2. **Wire real cross sections into the CRM chassis** (replace
   `_kramers_photoionization_sigma_cm2`-class placeholders with
   `scattering/photoionization` + today's Auger tiers + einstein_A).
3. **Fast configuration-solve path**: engine="shooting" + local
   exchange is the natural XATOM-speed analog; needs a per-config
   timing benchmark vs XATOM (target: ≤10× XATOM for the same config).
4. **Shakeoff / direct double ionization** (XATOM ships fitted
   per-shell data — transcribe + gate against their published values).
5. **MC over pulse + occupation dynamics** on the cached graph.
6. **Transient-config observables**: per-config f0/f1/f2 (both modules
   already parametrised by orbital sets — plug the cascade's transient
   configs in) → imaging/diffraction signals.
7. **REXMI / two-step resonance absorption** (uses the DR/manifold
   machinery from today's campaign).
8. Cowan: no gap for XFEL per se — atomx's CI/spectroscopy subsumes
   Cowan's level/rates role with relativistic physics; keep Cowan as
   an external cross-check anchor only.

## 4. Proposed first increment (one session)

`src/atomx/xfel/cascade.py` — deterministic (rate-equation) sequential-
PI cascade over an explicit configuration graph:

* input: element/config, photon fluence Φ(ω) (integrated over the
  pulse), channel rate providers (PI per subshell σ(ω); Auger LS-chain
  tier; E1 fluorescence);
* loop: populate configuration graph breadth-first — each config:
  solve structure ONCE (memoised), tabulate rates, branch on
  (photoabsorption | Auger | fluorescence);
* outputs: charge-state distribution vs fluence, total electrons,
  per-config Auger/photon spectra, transient config table;
* validation anchor: reproduce XATOM's published Ne charge-state
  yields vs fluence (Young 2010 Fig. 1 shape; regenerate numbers with
  the vendored binary `-occupation`-style run), gated at O(20%)
  initially (orbital-model difference), tightening later.

Defer to later increments: MC trajectories, shakeoff, REXMI, pulse
time structure, imaging observables.

## 5. Honest risks

* **Per-config cost**: atomx SCF ≈ 10²-10⁴× XATOM's HFS.  Mitigations:
  shooting-engine fast path; frozen-core Δ-SCF after the first solve
  of a charge state; graph caching (many configs share the solve);
  subshell-rate reuse across configurations that differ by one
  electron.
* **Near-edge resonances**: XFEL pulses sit ON resonances (REXMI);
  the smooth-σ cascade underestimates there — the resonance-aware
  increment is required for photon energies near thresholds.
* The Kramers placeholders in `crm.py` must not silently coexist with
  real rates — route the CRM through the same providers.
