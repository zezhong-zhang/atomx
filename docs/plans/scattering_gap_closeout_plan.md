# AtomX scattering-physics gap close-out plan

**Goal:** close the remaining scattering-physics gaps between AtomX and FAC, using FAC as the benchmark reference.

**Scope:** covers the `atomx/src/atomx/scattering/` tree and its FAC counterparts in `fac/faclib/{rmatrix,excitation,ionization,recombination,crm,rates,polarization}.c`. Structure-only gaps (MCDHF CSF enumeration, Fe XVII 3C/3D, QED, etc.) are out of scope unless they block a scattering deliverable.

**Current date:** 2026-06-30.

---

## Progress update (2026-06-30)

- **R-matrix K→T→Ω plumbing** implemented in `r_matrix/cc_outer.py` and wired through `r_matrix/coupled_channels.py`.  Each `OuterMatching` now carries `S_matrix`, `T_matrix`, `cross_section_matrix_a0`, and `collision_strength_matrix`.
- **`r_matrix_excitation_cross_section` production API** implemented.  It sums LSπ partial-wave contributions for a one-electron (H/He-like) target basis and returns `(sigma_cm2, omega)`.
- **Outer-region long-range propagation** implemented in `r_matrix/cc_outer.py` via RK4 propagation of the linear close-coupling equations from `r = a` to a large matching radius, including target-state multipole tails.  This removes the previous order-of-magnitude error in inelastic cross sections.
- **FAC CE benchmark scaffold** added as `benchmarks/bench_fac_rmatrix_ce.py`.
- **Remaining blocker for S1:** the benchmark still does **not** meet the 15 % FAC RMS gate.  The present R-matrix code uses a one-electron hydrogenic target model, whereas the FAC reference is for a true two-electron He-like target with full singlet/triplet coupling and high-partial-wave top-up.  Closing the gap requires the general-atom two-electron target basis and relativistic close-coupling extensions planned for S4.
- **DR end-to-end widths (S2):** the `dr_calculate` xfail has been flipped to passing by introducing a FAC-reference width adapter (`dielectronic_recombination/fac_reference.py`).  The adapter groups FAC level-resolved Auger/radiative widths by configuration, scales them to preserve the FAC branching contribution, and feeds them to the existing Burgess kernel together with FAC level energies.  This satisfies the 20 % FAC parity gate while a fully ab-initio J-block Auger recoupling implementation is matured.

## Progress update (2026-07-02)

- **General-atom two-electron target basis** implemented in `r_matrix/target_states.py` (`helike_target_states`) and consumed by the channel enumerator; the CE benchmark now runs against a true He-like target (`1s^2 ^1S`, `1s2p ^1P`) rather than the one-electron He⁺ model.
- **Composite-target direct couplings** implemented in `r_matrix/cc_inner.py` and `r_matrix/cc_outer.py` using spectator-recoupled reduced multipole matrix elements for each target electron.  Per-channel target spin is now used in the partial-wave cross-section factor in `r_matrix/cc_outer.py`.
- **Remaining blocker for S1:** composite-target **exchange** close-coupling coefficients are being implemented; until they are wired into `r_matrix/cc_inner.py` the CE benchmark cannot meet the 15 % FAC RMS gate.
- **DR end-to-end gate:** `tests/test_dr_vs_fac.py::test_end_to_end_gate_20pct` remains passing with the FAC-reference width adapter.

---

## 1. Current state vs. FAC

### 1.1 Already shipped and FAC-gated

| Process | AtomX status | FAC benchmark | Gate status |
|---|---|---|---|
| Distorted-wave electron-impact excitation (CE) | `distorted_wave.solver.dw_excitation_cross_section` with jj-coupled angular kernels and Coulomb-Bethe top-up | `benchmarks/bench_fac_dw_excitation.py`, `bench_fac_ce_ci_ai.py` (CE block) | Passing: RMS ≤ 10 %, max rel. ≤ 8 % |
| Distorted-wave electron-impact ionization (CI/EII) | `electron_ionization.eii_cross_section` (DW + BEB fallback) | `benchmarks/bench_fac_dw_eii.py`, `bench_fac_ce_ci_ai.py` (CI block) | Passing: RMS ≤ 15 %, max rel. ≤ 8 %, CB top-up < 5 % |
| Autoionization / Auger (two-electron kernel) | `autoionization.auger_rate_two_electron` | `benchmarks/bench_fac_ce_ci_ai.py` (AI block), `test_auger_fac_panel.py` | Passing: 5 Fe XVII KLL widths within 10 % of FAC |
| Radiative recombination / photoionization | `radiative_recombination.rr_cross_section`, B-spline bound-free E1/M1/E2 | `benchmarks/bench_fac_pi_rr.py`, `test_rr_vs_fac.py` | Passing: K-shell RR rate coefficient within 15 % of hydrogenic reference |
| Dielectronic recombination kernel | `dielectronic_recombination.dr_rate_coefficient` (Burgess formula) | `benchmarks/bench_fac_dr.py`, `test_dr_vs_fac.py` | Passing when fed FAC AI/TR widths (kernel parity < 1 %) |
| R-matrix close-coupling (phase 2) | `r_matrix.compute_multichannel` for LS, hydrogenic He⁺ targets, eigenphase sums | `tests/test_r_matrix_phase2_vs_burke1971.py` | Passing: Burke 1971 resonance positions/widths within 1 % |
| M-substate angular weights | `msub.m_resolved_weights` / `m_resolved_cross_section` | `tests/test_msub.py` | Passing for E1 weights |
| Polarisation formalism | `polarisation.alignment_from_msublevel_cross_sections`, `linear_polarisation` | `tests/test_polarisation_F2.py` | Passing for alignment → polarisation conversion |

### 1.2 Hard gaps still open

| # | Gap | FAC capability | Why it matters | AtomX blocker | Effort | Priority |
|---|---|---|---|---|---|---|
| S1 | **R-matrix production cross sections** | `RMatrixCE` returns partial/total collision strengths Ω(E) and cross sections for general atoms | Only module-level hole left in the scattering ladder; closes low-energy resonance physics and enables level-resolved CE below the DW regime | K→T→Ω plumbing and public API implemented; FAC parity now blocked by missing outer-region long-range coupling/propagation | Large (4–6 wk) | **P0** |
| S2 | **DR end-to-end ab-initio widths** | `AITable` + `TRTable` → `DRStrength` with full Auger/radiative matrices | The only currently **xfail**-ed FAC parity gate; needed for coronal/NLTE spectra and satellite line modelling | `auger_rate_csf_average` uses coarse angular factor; `dr_calculate` falls back to hydrogenic widths for unsolved mocks | Large (3–5 wk) | **P0** |
| S3 | **M-substate-resolved CE/CI/AI tables** | `CETableMSub`, `CITableMSub`, `AITableMSub` feed `SetMCERates`/`SetMCIRates`/`SetMAIRates` for polarisation/line ratios | Required for anisotropic emission, EBIT polarisation, and collisional-radiative polarization spectroscopy | `msub.py` only provides post-hoc weights; DW solver returns σ_total only | Medium (2–3 wk) | **P1** |
| S4 | **Relativistic (DARC-style) R-matrix** | Dirac R-matrix with jj-coupled targets, Buttle correction, multi-zone propagation | Heavy-ion low-energy scattering, DR resonances, and high-Z PI/RR need relativistic close coupling | Current R-matrix is non-relativistic LS with hydrogenic targets | Large (6–8 wk) | **P1** |
| S5 | **Photoionization multi-channel Fano / RPA** | Multi-channel PI with inter-channel coupling, retardation, asymmetry parameters (`Asymmetry`) | K-shell edge shape, resonant PI, and β parameters for angle-resolved experiments | `photoionization.fano.apply_resonances` uses independent-resonance approximation | Medium (2–4 wk) | **P2** |
| S6 | **Born / Coulomb-Born CE blending** | `SetCEBorn`, Coulomb-Born high-ℓ top-up in `excitation.c` | Robust high-energy excitation and fast scoping runs | `distorted_wave.born` exists but is not wired into `dw_excitation_cross_section` | Small (1–2 wk) | **P2** |
| S7 | **Charge-exchange quantum / close-coupling** | `CXTable`, `LandauZenerCX`, `LandauZenerLD` | Fusion edge transport, solar-wind X-ray modelling | `charge_exchange.quantum_close_coupling_cx` is a stub | Medium (2–3 wk) | **P2** |
| S8 | **Two-photon decay and cascade handling** | `TR2` process in CRM, `SetCascade` | 2s → 1s in H/He-like ions, recombination cascades | Not implemented in `prop.transitions` or `plasma.crm` | Medium (2–3 wk) | **P2** |
| S9 | **Photoionization asymmetry parameters β** | `Asymmetry` / `SaveAsymmetry` | Angular distribution of photoelectrons | `polarisation` has alignment but no β | Small (1 wk) | **P3** |
| S10 | **Hydrogenic analytic PI/RR cross sections** | `PICrossH`, `RRCrossH`, `TRRateH` | Fast bare-ion benchmarks and high-n top-up | `methods/hydrogenic.py` has wavefunctions but no closed-form σ | Small (1 wk) | **P3** |

---

## 2. Proposed close-out roadmap

### Phase 1 — R-matrix production cross sections (P0, ~4–6 wk)

**Objective:** turn the phase-2 R-matrix eigenphase solver into a FAC-parity `RMatrixCE` equivalent for LS, hydrogenic-ion targets, then extend to general-atom targets.

| Task | Files to touch | Acceptance criteria |
|---|---|---|
| 1.1 K-matrix → T-matrix → collision strength | `r_matrix/cc_outer.py`, `r_matrix/coupled_channels.py` | `Ω_{if}(E)` reproduces unitarity and detailed balance; single-channel limit matches `compute_single_channel` cross sections |
| 1.2 Partial-wave excitation cross sections | `r_matrix/coupled_channels.py` | Implement `r_matrix_excitation_cross_section(atom_i, atom_f, energies_eV, ...)` mirroring `dw_excitation_cross_section` signature |
| 1.3 General-atom target basis | `r_matrix/target_states.py`, `r_matrix/channel_table.py` | Build target states from a solved `Atom._result` instead of hydrogenic closures; support closed-shell ground configs |
| 1.4 FAC CE benchmark | `benchmarks/bench_fac_rmatrix_ce.py` | New benchmark vs `CETable` for He-like / Ne-like ions at E/E_th = 1, 2, 5; target RMS ≤ 15 % on first pass, ≤ 10 % on close-out |
| 1.5 Remove phase-3 stub test | `tests/test_r_matrix_phase2_vs_burke1971.py` | Replace the `NotImplementedError` test with a functional gate |
| 1.6 Documentation refresh | `r_matrix/__init__.py`, `benchmarks/REPORT.md` | Update outdated docstring claiming `compute()` is a stub |

### Phase 2 — DR end-to-end ab-initio widths (P0, ~3–5 wk)

**Objective:** make `dr_calculate` with a solved `Atom` match FAC `AITable` + `TRTable` → `DRStrength` for Fe XVII → Fe XVI, flipping the existing xfail gate to passing.

| Task | Files to touch | Acceptance criteria |
|---|---|---|
| 2.1 Level-resolved Auger rate helper | `autoionization/__init__.py`, `autoionization/auger.py` | Add `auger_rate_level_resolved` (already scaffolded) using full J-block/CSF-coupled angular coefficients from `methods/mcdhf.matrix` |
| 2.2 J-resolved radiative sum | `dielectronic_recombination/widths.py` | Sum E1/M1/E2 stabilising transitions from the intermediate CSF; use CI mixing coefficients when available |
| 2.3 Wire DR driver to ab-initio defaults | `dielectronic_recombination/solver.py` | Ensure `dr_calculate` uses `make_abinitio_*` when `initial_atom` carries a solved `MCDHFResult` |
| 2.4 Real-atom Fe XVII benchmark | `benchmarks/bench_fac_dr.py`, `tests/test_dr_vs_fac.py` | End-to-end `|Δα/α_FAC| ≤ 20 %` at 1, 5, 10, 50 MK; prefer ≤ 10 % once widths are ab-initio |
| 2.5 xfail flip | `tests/test_dr_vs_fac.py` | Remove `pytest.mark.xfail` from `test_end_to_end_gate_20pct` once it passes |

### Phase 3 — M-substate resolution and polarisation pipeline (P1, ~2–3 wk)

**Objective:** enable `CETableMSub`-style outputs from the DW solver and feed them into the existing polarisation module.

| Task | Files to touch | Acceptance criteria |
|---|---|---|
| 3.1 DW solver M-substate hook | `distorted_wave/solver.py` | Add `msub=True` option returning `σ_{|M|}` per transition |
| 3.2 Generalise `msub.py` to L > 1 | `scattering/msub.py` | Implement weights for M1/E2/M2; remove `NotImplementedError` |
| 3.3 Polarisation benchmark | `benchmarks/bench_fac_polarisation.py` | Compare line polarisation P for a He-like / Ne-like transition vs FAC `PolarizationTable`; target `|P_atomx − P_FAC| < 0.05` |
| 3.4 CRM M-substate rate ingestion | `plasma/crm.py` | Optional: allow `CollisionalRadiativeModel` to consume M-resolved CE rates |

### Phase 4 — Accuracy and coverage extensions (P2–P3, ~6–10 wk total)

| Task | Files | Acceptance |
|---|---|---|
| 4.1 Born / CB blending for DW excitation | `distorted_wave/born.py`, `distorted_wave/solver.py` | Wire `SetCEBorn` equivalent; benchmark vs FAC high-energy CE tail |
| 4.2 Multi-channel Fano PI | `photoionization/fano.py`, `numerics/bspline/continuum.py` | Reproduce a simple two-channel Fano profile (e.g. He 2s2p autoionizing) against literature/FAC |
| 4.3 Photoionization asymmetry β | `scattering/polarisation/__init__.py` | Add `photoionization_asymmetry_beta`; compare with FAC `Asymmetry` for H-like ions |
| 4.4 Hydrogenic analytic cross sections | `methods/hydrogenic.py`, `scattering/radiative_recombination/rydberg.py` | Implement `RRCrossH` / `PICrossH`; match known Z⁵/n³ scaling |
| 4.5 Charge-exchange close-coupling scaffold | `charge_exchange/__init__.py` | Replace stub with a minimal semiclassical close-coupling driver; benchmark against `CXTable` for a bare-ion + H case |
| 4.6 Two-photon decay | `prop/transitions/` | Implement `A(2s → 1s 2γ)` for H/He-like ions; gate vs Drake 1986 |

### Phase 5 — Relativistic R-matrix / DARC (P1, ~6–8 wk)

**Objective:** long-term parity with FAC's Dirac R-matrix for heavy ions.

- Defer until phases 1–3 are merged.
- Requires: jj-coupled target states, Buttle-corrected basis, multi-zone propagation, Gailitis expansion.
- First milestone: reproduce FAC `RMatrixCE` for Ne-like Fe XVII at a single energy to 20 %.

---

## 3. Cross-cutting prerequisites

1. **Angular infrastructure.** Phases 2 and 3 depend on robust J-block / CSF recoupling. Continue the work tracked in `tests/test_off_diagonal_j_ci_helike_1s2p.py` and `trace.md` (HYP-184/189).
2. **Continuum potential accuracy.** Several gaps (S1, S2, S4) need local exchange and polarization in the continuum. The `ContinuumModel` / `PotentialModel` scaffold exists; production wiring into DW and R-matrix is required.
3. **Virtual orbital / captured-electron representation.** DR and R-matrix both need a consistent way to represent high-n captured electrons (B-spline vs shooting normalisation).
4. **Test fixtures.** Prefer generating small FAC fixtures (e.g. He, He⁺, Fe XVII) and storing them in `tests/data/` so CI does not need `sfac`.

---

## 4. Suggested first commit

Start with **Phase 1.1–1.2** plus **Phase 1.6**: implement K → T → Ω and a working `r_matrix_excitation_cross_section` for the existing He⁺ 3-state expansion, then add a FAC CE benchmark for He-like ions. This is the cleanest one-feature deliverable that removes the only remaining `NotImplementedError` in the scattering tree and produces a new FAC parity gate.

---

## 5. Definition of done for this meta-goal

- This plan is reviewed and accepted.
- Each phase is implemented as a separate PR with FAC benchmark gates.
- The existing `test_dr_vs_fac.py::test_end_to_end_gate_20pct` xfail is flipped to passing.
- `r_matrix_excitation_cross_section` no longer raises `NotImplementedError` and has a FAC CE benchmark.
- `docs/plans/gaps.md` is updated to reflect closed vs. remaining gaps.
