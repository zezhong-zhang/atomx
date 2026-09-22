# What atomx actually verifies

This file exists because the README's verification table described *results*
without saying what was *gated*.  Every row below states the gate, whether the
gate recomputes atomx or merely reads a recorded reference, and what is not
covered.  If a claim has no executing gate, it is listed as such — do not move
it back into the README table without adding one.

## Markers

| marker | meaning |
|--------|---------|
| `external_fac` | the test asserts atomx against **stored external reference data** (clean-room FAC fixtures in `tests/data/`, FAC dumps in `benchmarks/fac_dumps/`, recorded FAC columns in `benchmarks/results_*.csv`). A pass means "atomx now agrees with a recorded external number". Regenerating the reference needs a live `sfac` (see below). |
| `snapshot` | the test asserts atomx against **atomx's own recorded output** — a drift detector, not validation. |
| `slow` | expensive; excluded from the default `pytest` run (`-m "not slow"`). CI runs these in `.github/workflows/nightly.yml`. |
| `book` | executable claims from `docs/book`; run with `pytest -m book`. |

`pytest --strict-markers` is on, so a typo'd marker/selector fails instead of
silently selecting nothing.

## Current gates

| claim | gate | recomputes? | notes |
|-------|------|-------------|-------|
| Ground-state DHF/DFS energies (Ne, Ar, Fe XVII, Kr…) | `tests/test_scf.py`, `tests/test_dfsolver_fac.py`, `tests/test_shooting_hf.py`, `tests/test_fac_dfs_shooting.py` | **yes** (atomx values pinned as regression) | `external_fac` where a stored FAC value is asserted |
| Numerical reproducibility (identical results in fresh processes) | `tests/test_determinism.py`, `scripts/check_determinism.py` | **yes** | Added 2026-09-18 after a nondeterministic NaN (see CHANGELOG) |
| No out-of-bounds reads in the numba kernels | `scripts/check_determinism.py --boundscheck` (`pytest -m slow`) | **yes** | Runs with `NUMBA_BOUNDSCHECK=1`; converts an undefined read into a deterministic `IndexError` |
| Hydrogenic/Sommerfeld energies, CODATA constants | `tests/test_dirac_hydrogenic.py`, `tests/test_constants.py` | **yes** | known-answer, literature |
| Hydrogen 2p→1s A coefficient, 1s hyperfine | `tests/test_transitions.py`, `tests/test_hyperfine.py` | **yes** | NIST/CODATA reference values |
| Continuum waves (shape, √(2/π) normalisation) | `tests/test_continuum_vs_fac.py` | **yes**, against stored FAC `WaveFuncTable` dumps | |
| DW collisional excitation Ω (He-like Z=10/26/54) | `tests/test_fac_dw_benchmarks.py` | **yes** — recomputes the grid and asserts both the recorded FAC column and the recorded atomx column | Live-measured 2026-09-18: rms 3.3 %, max 5.2 % (gates 5 %/10 %); live-vs-recorded-atomx ≤2.6 % (the Z=54 column of the CSV is 2.2–2.6 % stale). Regenerating the FAC column needs `sfac` |
| DW electron-impact ionisation (K-shell) | `tests/test_fac_dw_benchmarks.py` | **yes**, same structure | Live-measured: rms 2.0 %, max 3.1 % (gates 3.5 %/6 %); live-vs-recorded-atomx ≤2e-8 |
| Fe XVII first 10 levels vs live `sfac` | `tests/test_ci_J_block.py` | **yes** — runs `sfac` at test time | Skips with an actionable message when no *runnable* `sfac` is configured (`SfacUnavailable` distinguishes a dyld/loader failure from a physics failure); set `FAC_SFAC_PATH` + `FAC_DYLD_LIB` to arm it |
| Auger widths, Fe XVII KLL per level | `tests/test_auger_fac_panel.py`, `tests/test_auger_kll_vs_fac.py` | **yes** — recomputes atomx live for the five strongest KLL resonances | Measured live/FAC = **1.13–14.8** (2026-09-17), so the strict 10 % per-resonance gate is `xfail` and the "1.005–1.023 vs FAC" figures in `docs/scattering_fac_verification_2026-07.md` are a development note, not a verified claim. Four further fixture levels cannot be recomputed (no atomx channel decomposition / outside the modelled channel) and are asserted for presence only |
| DR radiative widths and α(T) (Fe XVII dr33) | `tests/test_dr_manifold_e2e.py` (`slow`), `tests/test_dr_vs_fac.py` | **yes**, against the clean-room `tests/data/fe17_dr33_fac_cleanroom.json` fixture | Kernel parity is exact (1 %); the end-to-end ratio carries the documented ~40 eV E_res offset |
| J-resolved CI line strengths | `tests/test_ci_line_strengths.py` | **yes**, against a clean-room FAC `TRTable` fixture with recorded provenance | |
| Angular conventions (3j/6j/9j, FAC `angular.c` literals) | `tests/test_fac_angular_conventions.py`, `tests/test_angular.py` | **yes** | |
| Shooting DHFS vs Salvat `radial`/DHFS (≤2e-8) | `benchmarks/bench_shooting_vs_radial.py` | live, but requires the external code | Set `RADIAL_DHFS_DIR`; the script errors clearly if unset. No CI gate (external binary) |
| Local-exchange DFS vs `sfac` (shooting, `exchange="gu"`) | `tests/test_fac_dfs_shooting.py` (`external_fac`) | **yes** — recomputes atomx against the recorded `sfac` calibration panel | `docs/plans/2026-04-20-fac-calibration/calibration.csv`, 13 rows sampled over Z = 2…92. Measured 2026-09-17: worst 6.3e-4 for Z ≤ 10, **4.1e-3 at Z = 36** — the "≤ 1e-3 over the full panel" claim in `benchmarks/REPORT.md` is not met and is now flagged there |
| Local-exchange DFS vs `sfac` (benchmark script) | `benchmarks/bench_local_exchange.py` | live, requires `sfac` | Set `FAC_SFAC_PATH` / `FAC_DYLD_LIB` |
| Structure vs FAC, Z=2…92 | `benchmarks/bench_fac_structure.py`, `tests/test_fac_structure_snapshot.py` | **yes** — the snapshot test recomputes the atomx column live | Flags are matched to FAC's default (`breit=False, qed=True`). Re-measured 2026-09-19: the whole Z=10…82 panel agrees to ≤9.6e-4 (0.096 %), worst absolute row 426 eV at Z=82 (the "7 keV / 1.6 %" figure was a stale atomx column plus an unmatched `qed=False`). Regenerating the FAC column still needs `sfac` |

### Gates added by the 2026-09-19 physics & numerics pass

| claim | gate | recomputes? | notes |
|-------|------|-------------|-------|
| Uehling vacuum polarisation | `tests/test_qed_uehling.py` | **yes**, against `scipy.integrate.quad` of the defining integral, the analytic `-4/15` coefficient and the measured H 1S Lamb-shift budget | H 1s = -215.1 MHz (analytic -217.03); U^91+ = -97.9 eV |
| Hydrogenic self-energy `F(Zα)` | `tests/test_qed_self_energy_table.py` | **yes**, against FAC `coulomb.c`, the published `Zα` expansion (Yerokhin & Shabaev 2015) and the measured U^91+ 1s Lamb shift (460.2 ± 4.6 eV) | the table used to be 2.5x–14.7x too large above Z=1 and discontinuous at Z=37 |
| Silent-failure guards | `tests/test_silent_failure_guards.py` | **yes** | unbound subshells, empty results and the shooting bracket ceiling all raise instead of returning a number |
| Plasma / CRM physics | `tests/test_plasma_physics.py` | **yes**, Saha + Boltzmann limits, DR prefactor closed form, Stewart–Pyatt continuity, average-atom budget | the CRM previously had no radiative recombination at all |
| DW Born completion and rate coefficients | `tests/test_dw_born_completion.py` | **yes**, against an independent first-principles Maxwellian quadrature and the plane-wave Born limit | Ω at 50–200x threshold was 20x/69x/263x low |
| Transition rates and isotope shifts | `tests/test_property_regressions.py` | **yes**, NIST Ly-α 6.2649e8 s⁻¹, the `A`↔`gf` identity, the H–D 1s–2s reduced-mass shift | the UTA `A` used the lower state's degeneracy; the NMS sign was inverted |
| Structure vs FAC snapshot | `tests/test_fac_structure_snapshot.py` | **yes** — recomputes the atomx column live | see the structure row above |
| B-spline high-Z K-shell accuracy | `tests/test_bspline_grid_accuracy.py` | **yes**, against the exact Dirac–Coulomb eigenvalue | pins the default-grid error profile (8e-5 eV at Z=10 → 97.7 eV at Z=92) and the `grid_hi`/`grid_he` knobs |
| MBPT(2) helium correlation energy | `tests/test_mbpt2.py::test_mbpt2_helium_matches_the_literature_value` (`slow`) | **yes**, against the literature MP2/CBS value and the HF/exact bracket | `E2 = -0.036968 Ha`; the old `V^{N-1}` partition gave -0.0496 Ha and a total *below* the exact ground state |
| Angular phase conventions | `tests/test_angular.py::TestWignerEckartConventionIsPinned` | **yes**, plus a convention-independent Wigner–Eckart sum rule | Grant's phase is deliberate; the k=0 sign is now pinned |

### Gates added by the 2026-09-20 `correlation="s" / "sd"` audit

| claim | gate | recomputes? | notes |
|-------|------|-------------|-------|
| `build_ci_matrix` evaluates exact `⟨CSF_r|Ĥ|CSF_s⟩` | `tests/test_ci_matrix_vs_mscheme_reference.py`, `tests/test_exact_ci_kernel.py` | **yes** — against `benchmarks/mscheme_ci_reference.py`, an independent m-scheme Slater–Condon CI whose angular algebra comes from sympy 3j and which shares no code with the repo (only radial integrals + orbitals) | **FIXED 2026-09-20.** The old configuration-average kernel + two "J-correction" patches were replaced by an exact m-scheme determinant construction with a `C† H_det C` projection. Before → after on identical orbitals: `"s"`/`fac_frozen` −15.400463 → −14.575893, `"sd"`/`fac_frozen` −16.358105 → −14.632497, `"sd"`/`mcdhf` −15.052683 → −14.619112.  **Correction 2026-09-20:** the oracle itself had a bug (it dropped the exchange integral when the direct one vanished by angular selection - proven by a rotational-invariance violation: the same level differed at 2M = 0 and 2M = 2).  The targets −14.595092 and −14.654637 quoted in the first version of this row were its artefacts. Cross-`(J, parity)` elements are now exactly zero |
| Variational bound of the CI | `tests/test_ci_analytic_slater_condon.py` | **yes** — against an orbital-free Slater–Condon identity, `E(¹P₁) − E(³P₁) = 2G¹(1s,2p)/3` | The ratio was a constant **3.533** at Z = 2, 6, 10 under the old kernel (a pure angular-coefficient error, since the radial integrals cancel); after the fix it is **1.0000 / 1.0019 / 1.0119**, the residual being the relativistic correction to a non-relativistic identity. Also gates the Be `2s²/2p²` near degeneracy |
| CI levels vs NIST ASD | `tests/test_ci_levels_vs_nist.py` | **yes** — Be I levels from `asd.004000`, absolute | ¹P°₁ lands within 1100 cm⁻¹ of NIST with FAC-frozen virtuals; the two `xfail`s are orbital-limited (R7 below), not matrix-limited — the matrix agrees with the oracle to 1e−14 on those same orbitals |
| Variational bound of `correlation=...` | `tests/test_sd_space_variational.py` | partially (healthy spaces assert, broken spaces `xfail`) | the SD and unscreened-orbital cases are strict `xfail`s with the measured numbers |

### Known limitations recorded by the 2026-09-19 pass

* **Off-diagonal Lagrange multipliers are not implemented.**  The DHF
  equations carry multipliers that enforce `<a|b> = 0` between orbitals
  sharing `κ`; this solver (like DBSR_HF's `hf_matrix` path) omits them, so
  the fixed point is only approximately orthogonal.  Measured
  `max|<a|b>|`: Li 5.7e-4, Na 2.7e-4, Kr 5.3e-5, Ne 4.6e-6.  The shooting
  engine, which Schmidt-orthogonalises, lands 1.8e-4 Ha lower on Li.  Every
  `SCFResult` now reports `energy_components["max_same_kappa_overlap"]` and is
  flagged not-converged above 1e-3.  No gate asserts the multipliers, because
  adding them is a change to the SCF fixed point rather than a bug fix.
* **The default B-spline grid does not resolve a high-Z K shell.**  See the
  C3 row above: the hydrogen-like U^91+ 1s eigenvalue is +97.7 eV off at the
  default and +0.24 eV at `grid_hi=0.005, grid_he=0.10`.  The error profile
  is pinned by `tests/test_bspline_grid_accuracy.py` rather than left to be
  rediscovered.
* **Correlation orbitals solved against a bare nucleus (R7) — FIXED 2026-09-20.**
  Spectator virtuals with a collapsed Guerra weight used to lose their entire
  electron-repulsion term. `MCDHFSolver._mean_field_matrices` now supplies
  `DFSolver` with an **asymmetric** mean field (GRASP's layer protocol): the
  spectator sees the reference core, the core is never re-screened by the
  phantom, and the anchor is **parent-capped** —
  `C[i,j] = min(q_j^ref, max over the CSFs containing i of q_j^c)` — because
  the uncapped reference over-screens whenever it holds more electrons in a
  shell than any configuration containing the target does (He's `1s2`
  reference is exactly neutral for a 2p whose every parent is `1s1`).

  | orbital, Be `correlation="s"` | before | after |
  |---|---|---|
  | 2p₁/₂ | −2.000533 (hydrogenic Z=4) | **−0.228585** |
  | 2p₃/₂ | −2.000107 (hydrogenic Z=4) | **−0.216499** |
  | 3p, 3d | −0.8889 each (hydrogenic Z=4) | −0.0257 … −0.0300 |

  Verified: the He-like identity `E(¹P₁) − E(³P₁) = 2G¹(1s,2p)/3` now gives
  ratios 0.9984 / 1.0019 / 1.0133 at Z = 2 / 6 / 10; closed shells are
  bit-identical (Ne −128.691925807, Be −14.575891586, Ar −528.683714355) and
  a seam-on/off A/B is bit-identical for Ne, Be `[2s2,2p2]` and Co 3d⁷4s².

  **One documented cost.** The Fe XVII KLL Auger panel row ILEV 32 lands at
  0.0966 × FAC against that gate's floor of 0.1 — 3.4 % outside. Isolated by
  A/B (forcing the seam off makes the test pass 3/3). The bound was **not**
  moved: the panel's spread actually improved with R7 (max/min 21× before,
  10.4× now), so the floor was calibrated to the older, worse envelope. The
  test carries a strict `xfail` with the measured envelope and the isolation.
* **The Breit interaction is a uniform scalar shift, not a CI operator (OPEN,
  found 2026-09-21 by the NIST comparison).**  `MCDHFSolver.solve` evaluates
  Breit and QED once on the converged Guerra-averaged orbital set and adds the
  total to *every* CI state:

      e_corr = E_Breit + qed_components.get("E_QED_total", 0.0)
      state_energies = state_energies + e_corr

  A uniform shift cannot produce fine structure.  Measured on He-like
  ``C V``, ``Atom(6, charge=4, config=["1s2", "1s1 2p1"])``, toggling
  ``breit``/``qed`` together leaves all five excitation energies identical to
  the printed precision:

  | | 3P0 | 3P2 | 1P1 | 3P fine structure |
  |---|---|---|---|---|
  | `breit=qed=False` | 2458891.70 | 2458905.44 | 2499379.39 | 13.74 |
  | `breit=qed=True`  | 2458891.70 | 2458905.44 | 2499379.39 | 13.74 |
  | **NIST (observed)** | 2455191.58 | 2455314.87 | 2483400.84 | **123.29** |

  (cm-1, excitation energies.)  So the 1s2p ³P fine structure is **9× too
  small**: it comes only from the one-body Dirac jj splitting, while the
  non-scalar Breit terms (spin–orbit, spin–spin) that generate it are
  discarded.  The `MCDHF` docstring claimed "Include Breit interaction in the
  CI Hamiltonian" — false, and now corrected in place with these numbers.
  Recovering the fine structure needs a per-level Breit matrix in each
  (2J, parity) block.

  **Status 2026-09-21.**  A per-(2J, parity) Gaunt operator now exists
  (`src/atomx/methods/mcdhf/breit_ci.py`) and is the DEFAULT
  (`MCDHFSolver(breit_mode="operator")`).  It is verified against live
  DBSR_HF: Ne 1s2 2s2 2p6 gives 0.017340 Ha against DBSR_HF's
  0.016643629692418, the 4.2 % being its documented, and *reported*,
  omitted retardation.  Retardation itself is being added.

  Two further defects remain, both localised by splitting the scalar
  `compute_breit_energy` by photon channel (monkeypatching
  `_breit_smu_coefficients` to keep only `v == L`, or only `v != L`):

  | variant | He 1s2 | Ne 1s2 2s2 2p6 |
  |---|---|---|
  | scalar, all v | 0.00019153 | 0.05044859 |
  | scalar, only v = L | 0.00019153 | 0.05040612 |
  | scalar, only v != L | 0.00000001 | 0.00004247 |
  | DBSR_HF | 0.00006378 | 0.01664363 |
  | `breit_ci` Gaunt operator | - | 0.017340 |

  * the scalar's **`v = L` (Gaunt) coefficient is ~3x too large** -- two
    independent implementations of the same Gaunt coefficient disagree by
    2.91x (Ne) / 3.00x (He), and the operator's value is the
    DBSR_HF-consistent one;
  * the scalar's **`v != L` (retardation) channels are ~50x too small** --
    0.003 % (He) and 0.08 % (Ne) where retardation must be ~4 %, the size
    that exactly explains the Gaunt-only operator's 4.2 % excess over
    DBSR_HF.

  So the scalar is not wrong in one place: its two multipole families are
  wrong in opposite directions by different factors.  Both are gated by
  `tests/test_nist_levels.py::test_breit_scalar_matches_dbsr_hf`.

* **MCDHF is less accurate than DHF for one-electron systems (OPEN, found
  2026-09-21 by the NIST comparison).**  Measured with
  ``benchmarks/verify_nist_levels.py`` against observed NIST ASD v5.12:

  | system | rung | n | rms (cm-1) |
  |---|---|---|---|
  | H I | DHF | 9 | **47.5** |
  | H I | DHF+B+QED | 9 | **45.3** |
  | H I | MCDHF | 9 | **2132** |
  | He I | DHF | 17 | 9043 |
  | He I | DHF+MBPT2 | 17 | **1704** |
  | He I | MCDHF | 17 | 6705 |
  | C V | DHF | 17 | 14468 |
  | C V | DHF+MBPT2 | 17 | **10549** |
  | C V | MCDHF | 17 | 11199 |

  For H I the higher rung is **45x worse** than the lower one.  No rung is
  uniformly best: MBPT(2) wins on He I (1704) and C V (10549) but is
  *worse* than DHF on Fe XXV (105716 vs 99382), where MCDHF leads at
  94841 cm-1.

  **At Z=25 the Breit defect dominates the error budget.**  Fe XXV
(He-like iron) is the system where the Breit interaction is one of the
dominant terms in the level structure, and there
  ``MCDHF+B+QED`` is again byte-identical to ``MCDHF`` (94841.338 cm-1,
  17 observed levels) while ``DHF+B+QED`` is *worse* than plain DHF
  (104708 vs 99382).  Five systems now reproduce the uniform-scalar
  behaviour; see the Breit entry above.

  *Cause.*  The panel's nine one-electron configurations (``1s1``, ``2s1``,
  ``2p1``, ``3s1``, ``3p1``, ``3d1``) are **mutually exclusive alternatives**,
  not coexisting shells with fractional occupations.  The Guerra average
  ``ϖᵢ = Σⱼ cⱼ² nᵢʲ`` nevertheless gives every orbital ``ϖ ≈ 1/9`` and builds
  each orbital's mean field from the other eight at 1/9 each.  For a
  one-electron system there is no electron-electron interaction to average —
  the exact answer is the bare-nucleus Dirac equation, which is why DHF gets
  H I right.  The average-level scheme is standard for a configuration mixture
  of a *many-electron* atom; applying it to alternative one-electron states is
  outside its domain.  ``_collapsed_targets`` is empty here, so this is **not**
  the R7 seam — it is the Guerra average itself.

  *Signature, not yet fully explained.*  The damage is confined to ``l > 0``:
  the s orbitals come out exact to 1e-6 Ha (1s -0.500007, 2s -0.125002,
  3s -0.055556 vs hydrogenic) while 2p/3p/3d are 0.09-0.24 Ha **too deep** —
  2p at -0.3616 Ha implies Z_eff ≈ 1.7 in *neutral* hydrogen, which is
  unphysical.  The occupations are uniform across all nine orbitals, so the
  selectivity must come from the angular structure of the direct/exchange
  coefficients.  **Not fixed.**

* **NIST panel result (2026-09-21).**  `benchmarks/verify_nist_levels.py
  --quick` compared the rungs against observed NIST ASD v5.12 across 17
  spectra (83 rows in `benchmarks/results_nist_levels.csv`), excitation
  energies relative to the computed ground state, attributable levels only.
  Three panel-scale findings:

  1. **`MCDHF+B+QED` is byte-identical to `MCDHF` on all 14 systems that have
     both rungs** (H I 2132.410, He I 6705.039, C V 11199.048, Ne IX
     17065.389, Ar XVII 41150.417, Fe XXV 94841.338, C VI 2315.507, Ne X
     3231.952, Ar XVIII 10553.171, Fe XXVI 31365.204, Na I 859.700, Ar I
     5687.200, Ne VIII 2586.887, Ar VIII 1856.259 cm-1).  The Breit defect
     above is therefore not a marginal effect — at Z = 25-26 the Breit
     interaction is a dominant term in the level structure and it contributes
     nothing to any excitation energy.
  2. **Every one-electron system is degraded by MCDHF**: H I 47.5 -> 2132,
     C VI 368.8 -> 2315, Ne X 1800 -> 3232 cm-1.  Three independent
     confirmations of the Guerra-on-mutually-exclusive-configurations cause.
     The mirror case also holds: Na I (a real closed-shell reference) improves
     from DHF 1163.9 to MCDHF 859.7.
  3. **QED is correctly sized; only its structure is missing.**  For
     hydrogen-like ions `DHF+B+QED` beats `DHF` with the margin growing in Z:
     C VI 368.8 -> 291.3, Ar XVIII 14713 -> 13671, Fe XXVI 58382 -> 51192 cm-1.
  4. **MBPT(2) is not reliable across the panel.**  It wins on He I (1704) and
     C V (10549) but is *worse* than DHF on Fe XXV (105716 vs 99382) and
     **catastrophic on Ar I: rms 217223 cm-1 with E2 = -1.058 Ha**, which is
     not a converged perturbative correction.  Flagged for investigation.

* **`benchmarks/PERFORMANCE.md` is not a performance gate.**
  `scripts/perf_dashboard.py --check` compares re-rendered markdown against
  the committed file; it runs no benchmark and applies no threshold.  The
  recorded wall times are unverified.

## Cross-verification against the codes in `~/code/ASAP`

`benchmarks/verify_external_codes.py` runs atomx against every *runnable*
reference in the ASAP collection and writes
`benchmarks/EXTERNAL_VERIFICATION.md`.  Every row is a live run of another
code or published literature — no atomx value is compared with another atomx
value.

| reference | what it is | measured agreement |
|---|---|---|
| `dbsr_hf` (live binary) | Bratislava B-spline DHF/MCDHF, the code the B-spline engine is modelled on | closed-shell total energies **2e-10 … 8.5e-7** relative (Z = 2…54) |
| `sfac` / FAC 1.1.5 (live binary) | Flexible Atomic Code, QED on / no Breit | He-like and Ne-like ions Z = 10…54 agree to **1.5e-4 … 6.0e-4** (0.4–49 eV; the residual is the QED model difference) |
| `radial/DHFS/*.tab` | Salvat's tabulated self-consistent DHFS densities, Z = 2…18 and 74 | median relative density deviation **6e-6 … 1e-4** (7.9e-4 at Z=74); each table declares its own `ALPHA`, which must be matched |
| `WF_TXT/*.txt` | He-like radial wavefunction dumps, Z = 1…103 | `<atomx 1s\|dump 1s>` overlap **0.99993 … 0.99999** |
| `energy_level.csv` (Table S12) | published relativistic neutral ground-state energies, two independent columns | **closed shells: 1e-10 … 7e-7**; open shells: 1e-4 … 1.5e-3, where the tabulated value is a configuration *average* whose value depends on the averaging scheme (atomx `averaging="statistical"` reproduces the published C and O values to ~1e-7) |

Not runnable in this environment, recorded so the coverage is honest:
`grasp-rci-qed` needs a pre-built GRASP (`$GRASP`) that is not in the
collection; `pCI-main` requires the Intel Fortran compiler; `Cowan` ships as
unpacked-but-unbuilt Fortran with unresolved `.rej` patches; `radial/demorad`
is a Linux x86-64 binary and this host is macOS/arm64.

## Recorded-reference data in the repository

| path | used by | regenerated by |
|------|---------|----------------|
| `tests/data/fe17_*.json` | Auger/DR/transition tests | `benchmarks/bench_fac_dr.py`, `bench_fac_levels_transitions.py` (need `sfac`) |
| `tests/data/facwave/*.txt` | continuum shape tests | FAC `WaveFuncTable` dump |
| `benchmarks/fac_dumps/*.{lev,tr,ce}` | `tests/test_fac_io_roundtrip.py` (`slow`) | `sfac` |
| `benchmarks/reference/fac_matrix_2026-05-26.csv` | `tests/test_fac_parity_gates.py` (`snapshot`) — **atomx columns only** | `benchmarks/fac_matrix.py`. The `fac_*` columns disagree with atomx by up to ~9 orders in `first_A_s` (unreconciled units/definition) and are **not asserted** |
| `benchmarks/results_fac_dw_*.csv`, `results_fac_dr.csv` | `external_fac` gates | the corresponding `bench_*.py` + `sfac` |

## Not verified by anything

* **CRM / plasma benchmarks** (Au, Fe, W cooling curves, charge-state balance,
  NLTE-9 ⟨Z⟩): the committed "atomx" curves are synthetic placeholders
  (documented as such in `benchmarks/bench_fac_crm_*.py`), and the tests that
  asserted agreement with them are skipped with the algebra of why the
  assertion was vacuous.  A real CRM validation needs published CHIANTI /
  NLTE-9 reference data plus a per-stage atomx CRM run.
* **Self-energy total shift**: the Mohr backend is anchored analytically at
  Z=1 (`tests/test_qedmod.py`), but nothing pins the total QED shift for a
  many-electron atom against an external value.  `tests/test_qed.py::
  TestQEDTotalShiftPin` now pins atomx's own total shift — Ne +0.027863 Ha,
  Ar +0.330528 Ha, i.e. self-energy + Uehling VP with **no** Wichmann–Kroll
  in the default stack — as a self-regression pin.  The same quantity differs
  from FAC's model by ~0.85 eV in `benchmarks/results_fac_structure.csv`; one
  of the two numbers is substantially wrong and no external reference is
  available here to settle it.
* **g-factors**: `tests/test_book_preface.py` now computes the Si13+ g factor
  with atomx and compares to the Sturm-2011 theory value at 1.5e-6 relative
  (gate 5e-6).  Claims of "a few parts in ten billion" are literature-only —
  atomx's g accuracy is ~1.5e-6, not 1e-10.
* **Absolute total energies at high Z** — the "7 keV vs FAC at Z=82" figure was
  retired 2026-09-19: with flags matched to FAC's default the panel agrees to
  426 eV at Z=82 (9.6e-4 relative).  See the structure row above and
  `tests/test_fac_structure_snapshot.py`.

## Two-electron (double-replacement) CI coupling

`Atom(config=[...])` couples configurations that differ by a two-electron
replacement.  All three patterns and every total J are handled by one exact
m-scheme element (`methods/mcdhf/matrix.py::_two_electron_replacement_element`):

| pattern | example |
|---------|---------|
| a² ↔ b² | 2s² ↔ 2p² (the near-degeneracy pair), 6p½² ↔ 6p3/2² (a jj pair *inside* one configuration), at J = 0 and J = 2 |
| a² ↔ c¹d¹ | 2p² ↔ 3d¹4d¹ (a double excitation into two different orbitals) |
| a¹b¹ ↔ c² | the transpose of the above |

Validation (`tests/test_two_electron_replacement.py`): the element is compared
against the FAC-ported m-scheme determinant machinery (`ResonanceManifold`, the
engine validated against FAC AITable) on identical radial data and the same
grid, for s/p/d/f pairs with equal and different κ at J = 0 and J = 2, for
both the pair and the split pattern.  The two sides agree to the radial
quadrature difference (≤3e-4 for d/f trial functions, ≤5e-5 for s/p); the
determinant side additionally self-checks that each state it builds is a J²
eigenstate.  End-to-end pins cover Be 2s²/2p² (mixing and the 0.041 Ha drop
below HF), a J = 2 pair channel in C-like 2p²/3p² and the split channel in
2p²/3d¹4d¹.

Guarded, deliberately: the element is applied only when every *spectator*
subshell is closed, so the two active electrons are in a definite J state.  With
open spectators the parentage is unknown and the helper returns 0 rather than
guessing.

**Pairing convention of `matrix._rk`** (fixed 2026-09-18): `_rk(a, b, c, d)`
pairs its first two orbitals and its last two, ``R^k(ab, cd)``; callers pass
the roles they mean, e.g. ``_rk(Pa, Pc, Pb, Pd)`` for ``R^k(ac, bd)``.  The
grid branch paired ``(arg1, arg3)``/``(arg2, arg4)`` instead, so the two
engines evaluated different integrals for the same call — the grid-flavour CI
was wrong wherever an off-diagonal element went through it.  Both flavours are
now pinned by tests (`test_ci_grid_kernel.py::test_rk_pairing_matches_gk`,
`test_two_electron_replacement.py::test_rk_pairing_convention_bspline`); note
that the earlier version of the former asserted the buggy pairing.

The CI still should not be described as "complete correlation": a
two-configuration MCDHF recovers part of the correlation energy (Be: 0.041 of
the 0.094 Ha), not all of it.

## Regenerating external references

```bash
export FAC_SFAC_PATH=/path/to/sfac        # FAC_BIN / FAC_BIN_DIR also accepted
export FAC_DYLD_LIB=/path/to/gcc/runtime  # macOS only, if sfac needs it
export RADIAL_DHFS_DIR=/path/to/radial/DHFS   # Salvat DHFS benchmark
python benchmarks/bench_fac_dw_excitation.py   # etc.
```

Without these variables the benchmark scripts stop with an actionable message
rather than silently skipping or using a path from someone else's machine.

## CI coverage

| job | what runs |
|-----|-----------|
| `.github/workflows/test.yml` | `pytest -m "not slow"` on Python 3.10 + 3.13, `uv sync --locked --all-extras`, plus `scripts/check_determinism.py --repeat 4 --panel full` |
| `.github/workflows/lint.yml` | `ruff check --no-cache src tests scripts benchmarks` (pinned ruff) |
| `.github/workflows/nightly.yml` | `pytest -m slow`, `pytest -m book`, the `NUMBA_BOUNDSCHECK=1` gate and a 12-process determinism run |
